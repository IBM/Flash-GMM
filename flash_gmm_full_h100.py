"""
Flash-GMM E-step (full covariance) — two-launch atomic kernel.

A correct, conservatively-sized full-covariance variant. The N×K×D×D
sufficient-statistic stream is processed without ever materializing the
N×K responsibility matrix.

Per-component covariance is full (D x D). Input is the lower-triangular
Cholesky factor `prec_chol[k] = L_k` of the precision matrix:

    Σ_k^{-1} = L_k L_k^T,   L_k lower-triangular

so the squared Mahalanobis distance has the cheap form

    || L_k^T (x - μ_k) ||² = (x - μ_k)^T Σ_k^{-1} (x - μ_k)

and the half-log-determinant of the precision is Σ_d log L_kdd.

Two Triton launches:
  Pass 1: tiled online log-sum-exp over K   →  writes logZ to HBM (log2 space)
  Pass 2: re-reads X, computes responsibilities, atomic-adds Nk / mu_acc /
          sig_outer into O(K + KD + KD²) global buffers.

The N×K responsibility matrix is never materialized.

Design notes — why this kernel is intentionally conservative:

  * **Per-d2 runtime loop** (not `tl.static_range`). Avoids exploding the
    SMEM budget under multi-stage pipelining when BLOCK_D is large
    (BLOCK_D=128 unrolled × num_stages=3 stages of `LT_d2` panels would
    request ~3 MB SMEM, well over H100's ~228 KB / SM).
  * **Single autotune config.** Triton autotune over many configs adds
    minutes of compile time on first call. This kernel uses one fixed,
    safe (BLOCK_N, BLOCK_K) tile so the first call is fast.
  * **Standard atomic-add accumulation** in pass 2 (no split-K). For
    K=1024, D=128 the (K, D, D) sig_outer buffer has 16M cells; atomic
    contention is bounded.

This kernel prioritizes correctness and cold-start latency over peak
throughput. It's the right starting point for an agent to optimize.

Entry point (DO NOT RENAME):
    flash_gmm_estep_full(X, mu, prec_chol, log_pi)
        -> (logZ, Nk, mu_acc, sig_outer)
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_LOG_2PI = 1.8378770664093453
_LOG2_E  = 1.4426950408889634

# NOTE: Triton @jit kernel bodies CANNOT access these module-level globals.
# Inside @triton.jit functions we inline the float literal `1.4426950408889634`
# (which equals log2(e)) directly. Do NOT extract it to a module constant —
# Triton will raise NameError("Cannot access global variable ... from within
# @jit'ed function"). If you really want a named constant, pass it as a
# kernel argument or use `tl.constexpr` typing on a *kernel parameter*.


# --------------------------------------------------------------------
# Pass 1: per-N-tile online log-sum-exp over K
# --------------------------------------------------------------------

@triton.jit
def _full_lse_kernel(
    X_ptr, LT_ptr, LT_mu_ptr, c0_ptr,
    logZ2_ptr,
    N, K, D,
    stride_xn, stride_xd,
    stride_LTk, stride_LTd2, stride_LTd1,
    stride_LTmk, stride_LTmd,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """For each (n, k) compute
        v_d2[n, k] = (LT[k, d2, :] · x[n, :]) - LT_mu[k, d2]
        maha[n, k] = Σ_{d2} v_d2²
        ll2[n, k]  = c0[k] - 0.5·log2_e · maha[n, k]   (log2 units)
    accumulating ll2 into an online log-sum-exp over K, writing logZ2[n].
    """
    pid = tl.program_id(0)
    n_off = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_off < N
    d_off = tl.arange(0, BLOCK_D)

    x = tl.load(
        X_ptr + n_off[:, None] * stride_xn + d_off[None, :] * stride_xd,
        mask=n_mask[:, None], other=0.0,
    )                                                            # (BLOCK_N, BLOCK_D)
    x_bf = x.to(tl.bfloat16)

    running_max = tl.full([BLOCK_N], float('-inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_off < K

        c0 = tl.load(c0_ptr + k_off, mask=k_mask, other=float('-inf'))

        maha = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
        # Runtime d2 loop — does NOT unroll; bounds SMEM under pipelining.
        for d2 in range(BLOCK_D):
            LT_d2 = tl.load(
                LT_ptr
                + k_off[:, None] * stride_LTk
                + d2 * stride_LTd2
                + d_off[None, :] * stride_LTd1,
                mask=k_mask[:, None], other=0.0,
            )                                                    # (BLOCK_K, BLOCK_D) BF16
            LT_mu_d2 = tl.load(
                LT_mu_ptr + k_off * stride_LTmk + d2 * stride_LTmd,
                mask=k_mask, other=0.0,
            )                                                    # (BLOCK_K,) FP32

            # v_d2 = x · LT_d2.T - LT_mu_d2     (BLOCK_N, BLOCK_K)
            v_d2 = tl.dot(x_bf, tl.trans(LT_d2), out_dtype=tl.float32)
            v_d2 = v_d2 - LT_mu_d2[None, :]
            maha += v_d2 * v_d2

        ll2 = c0[None, :] - (0.5 * 1.4426950408889634) * maha  # 0.5 · log2(e)
        ll2 = tl.where(k_mask[None, :], ll2, float('-inf'))

        block_max = tl.max(ll2, axis=1)
        new_max = tl.maximum(running_max, block_max)
        running_sum = (running_sum * tl.exp2(running_max - new_max)
                       + tl.sum(tl.exp2(ll2 - new_max[:, None]), axis=1))
        running_max = new_max

    log_Z2 = running_max + tl.log2(running_sum)
    tl.store(logZ2_ptr + n_off, log_Z2, mask=n_mask)


# --------------------------------------------------------------------
# Pass 2: per-(N-tile, K-tile) atomic accumulation
# --------------------------------------------------------------------

@triton.jit
def _full_atomic_accum(
    X_ptr, LT_ptr, LT_mu_ptr, c0_ptr, logZ2_ptr,
    Nk_ptr, mu_acc_ptr, sig_outer_ptr,
    N, K, D,
    stride_xn, stride_xd,
    stride_LTk, stride_LTd2, stride_LTd1,
    stride_LTmk, stride_LTmd,
    stride_ma_k, stride_ma_d,
    stride_so_k, stride_so_d1, stride_so_d2,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Re-reads X, recomputes maha for THIS K-tile, computes responsibilities,
    accumulates into Nk / mu_acc / sig_outer via atomic_add."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    n_off = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_off < N
    k_off = (pid_k * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
    k_mask = k_off < K
    d_off = tl.arange(0, BLOCK_D)

    x = tl.load(
        X_ptr + n_off[:, None] * stride_xn + d_off[None, :] * stride_xd,
        mask=n_mask[:, None], other=0.0,
    )
    log_Z2 = tl.load(logZ2_ptr + n_off, mask=n_mask, other=0.0)
    x_bf = x.to(tl.bfloat16)
    c0   = tl.load(c0_ptr + k_off, mask=k_mask, other=float('-inf'))

    maha = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    for d2 in range(BLOCK_D):
        LT_d2 = tl.load(
            LT_ptr
            + k_off[:, None] * stride_LTk
            + d2 * stride_LTd2
            + d_off[None, :] * stride_LTd1,
            mask=k_mask[:, None], other=0.0,
        )
        LT_mu_d2 = tl.load(
            LT_mu_ptr + k_off * stride_LTmk + d2 * stride_LTmd,
            mask=k_mask, other=0.0,
        )
        v_d2 = tl.dot(x_bf, tl.trans(LT_d2), out_dtype=tl.float32)
        v_d2 = v_d2 - LT_mu_d2[None, :]
        maha += v_d2 * v_d2

    ll2 = c0[None, :] - (0.5 * 1.4426950408889634) * maha  # 0.5 · log2(e)
    ll2 = tl.where(n_mask[:, None] & k_mask[None, :], ll2, float('-inf'))
    r = tl.exp2(ll2 - log_Z2[:, None])                         # (BLOCK_N, BLOCK_K) FP32
    r = tl.where(n_mask[:, None] & k_mask[None, :], r, 0.0)
    r_bf = r.to(tl.bfloat16)

    # Nk += sum over n
    nk_block = tl.sum(r, axis=0)                                # (BLOCK_K,)
    tl.atomic_add(Nk_ptr + k_off, nk_block, mask=k_mask)

    # mu_acc += r.T @ x        (BLOCK_K, BLOCK_D)
    mu_block = tl.dot(tl.trans(r_bf), x_bf, out_dtype=tl.float32)
    tl.atomic_add(
        mu_acc_ptr + k_off[:, None] * stride_ma_k + d_off[None, :] * stride_ma_d,
        mu_block, mask=k_mask[:, None],
    )

    # sig_outer[k, d1, d2] += Σ_n r[n, k] · x[n, d1] · x[n, d2]
    # Stream over d2 with runtime loop; per-d2 is a (BLOCK_K, BLOCK_D) row
    # written via one atomic_add of size (BLOCK_K, BLOCK_D).
    for d2 in range(BLOCK_D):
        x_d2 = tl.load(
            X_ptr + n_off * stride_xn + d2 * stride_xd,
            mask=n_mask, other=0.0,
        )                                                       # (BLOCK_N,)
        rx_d2_bf = (r * x_d2[:, None]).to(tl.bfloat16)          # (BLOCK_N, BLOCK_K)
        sig_d2_row = tl.dot(tl.trans(rx_d2_bf), x_bf, out_dtype=tl.float32)
        # (BLOCK_K, BLOCK_D)

        so_off = (k_off[:, None] * stride_so_k
                  + d2 * stride_so_d1
                  + d_off[None, :] * stride_so_d2)
        tl.atomic_add(sig_outer_ptr + so_off, sig_d2_row, mask=k_mask[:, None])


# --------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------

def flash_gmm_estep_full(
    X: torch.Tensor,
    mu: torch.Tensor,
    prec_chol: torch.Tensor,    # (K, D, D) — lower-triangular Cholesky factor of Σ⁻¹
    log_pi: torch.Tensor,        # (K,)
):
    """
    Flash-GMM E-step with full covariance.

    Args:
        X:         (N, D)    float32 GPU — data
        mu:        (K, D)    float32 GPU — component means
        prec_chol: (K, D, D) float32 GPU — lower-triangular Cholesky factor of
                   the precision matrix Σ_k^{-1}; prec_chol[k] @ prec_chol[k].T = Σ_k^{-1}
        log_pi:    (K,)      float32 GPU — log mixture weights

    Returns:
        logZ:      (N,)        float32 — per-sample log-partition function
        Nk:        (K,)        float32 — Σ_i r_ik
        mu_acc:    (K, D)      float32 — Σ_i r_ik · x_i
        sig_outer: (K, D, D)   float32 — Σ_i r_ik · x_i x_i^T  (uncentered second moment)

    M-step recovery:
        π_k = Nk[k] / N
        μ_k = mu_acc[k] / Nk[k]
        Σ_k = sig_outer[k] / Nk[k]  -  μ_k μ_k^T
    """
    N, D = X.shape
    K = mu.shape[0]
    assert prec_chol.shape == (K, D, D), \
        f"prec_chol must be (K, D, D)=({K}, {D}, {D}), got {tuple(prec_chol.shape)}"
    assert X.is_cuda and mu.is_cuda

    BLOCK_D = max(16, triton.next_power_of_2(D) if D > 0 else 16)
    # LT (K x D x D) is re-streamed once per N-tile in BOTH passes. Larger
    # BLOCK_N amortizes those re-reads (the dominant L2/HBM traffic). 256
    # rows = one full LT sweep per 256 points instead of per 64.
    BLOCK_N = 256
    BLOCK_K = 64

    # ---- Host-side pad to BLOCK_D when D is not a power of 2 ----
    if D == BLOCK_D:
        X_pad = X
        mu_pad = mu
        L_pad  = prec_chol
    else:
        X_pad = torch.zeros(N, BLOCK_D, device=X.device, dtype=X.dtype)
        X_pad[:, :D] = X
        mu_pad = torch.zeros(K, BLOCK_D, device=mu.device, dtype=mu.dtype)
        mu_pad[:, :D] = mu
        L_pad = torch.zeros(K, BLOCK_D, BLOCK_D, device=prec_chol.device, dtype=prec_chol.dtype)
        L_pad[:, :D, :D] = prec_chol

    # ---- Host-side precomputes ----
    LT    = L_pad.transpose(-1, -2).contiguous()                  # (K, BLOCK_D, BLOCK_D)
    LT_mu = torch.einsum("kij,kj->ki", LT, mu_pad).contiguous()    # (K, BLOCK_D)

    # log|Σ⁻¹|^{1/2} = Σ_d log L_kdd. Use unpadded prec_chol diagonal so
    # padded-zero rows don't NaN under log.
    log_diag_sum = torch.log(prec_chol.diagonal(dim1=-2, dim2=-1).abs() + 1e-30).sum(dim=-1)
    # c0 in log2 space (the -0.5·log2_e·maha term is added in the kernel):
    c0 = ((log_pi + log_diag_sum - 0.5 * D * _LOG_2PI) * _LOG2_E).contiguous()

    LT_bf = LT.to(torch.bfloat16).contiguous()                    # (K, BLOCK_D, BLOCK_D)

    # ---- Outputs ----
    logZ2 = torch.empty(N, device=X.device, dtype=torch.float32)
    Nk            = torch.zeros(K,                    device=X.device, dtype=torch.float32)
    mu_acc_pad    = torch.zeros(K, BLOCK_D,           device=X.device, dtype=torch.float32)
    sig_outer_pad = torch.zeros(K, BLOCK_D, BLOCK_D,  device=X.device, dtype=torch.float32)

    # ---- Pass 1: per-N-tile online LSE ----
    grid1 = (triton.cdiv(N, BLOCK_N),)
    _full_lse_kernel[grid1](
        X_pad, LT_bf, LT_mu, c0, logZ2,
        N, K, D,
        X_pad.stride(0), X_pad.stride(1),
        LT_bf.stride(0), LT_bf.stride(1), LT_bf.stride(2),
        LT_mu.stride(0), LT_mu.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=3,
    )

    # ---- Pass 2: 2-D grid over (N-tile, K-tile) ----
    grid2 = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _full_atomic_accum[grid2](
        X_pad, LT_bf, LT_mu, c0, logZ2,
        Nk, mu_acc_pad, sig_outer_pad,
        N, K, D,
        X_pad.stride(0), X_pad.stride(1),
        LT_bf.stride(0), LT_bf.stride(1), LT_bf.stride(2),
        LT_mu.stride(0), LT_mu.stride(1),
        mu_acc_pad.stride(0), mu_acc_pad.stride(1),
        sig_outer_pad.stride(0), sig_outer_pad.stride(1), sig_outer_pad.stride(2),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=3,
    )

    mu_acc    = mu_acc_pad[:, :D].contiguous()
    sig_outer = sig_outer_pad[:, :D, :D].contiguous()
    logZ = logZ2 * (1.0 / _LOG2_E)
    return logZ, Nk, mu_acc, sig_outer
