"""
Flash-GMM E-step (full covariance) — H100-tuned with split-K persistent pass 2.

Two Triton launches:
  Pass 1: tiled online log-sum-exp over K  -> writes logZ to HBM (in log2 space)
  Pass 2: split-K persistent — responsibilities + 3 atomic_adds per CTA

The N x K responsibility matrix is never materialized.

Per-component covariance is full (D x D). The kernel takes the Cholesky factor
L_k of the *precision* matrix:

    Σ_k^{-1} = L_k L_k^T,   L_k lower-triangular

Math, derived for WGMMA-friendly inner loops:

    || L_k^T (x - μ_k) ||²
        = (x - μ_k)^T Σ_k^{-1} (x - μ_k)
        = x^T Σ_k^{-1} x  -  2 μ_k^T Σ_k^{-1} x  +  μ_k^T Σ_k^{-1} μ_k

    Define:
        prec_flat[k, d1*D + d2] = (Σ_k^{-1})[d1, d2]            (K, D²) — host-precomp
        mu_w[k, d]              = (Σ_k^{-1} μ_k)[d]              (K, D)  — host-precomp
        mu_quad[k]              = μ_k^T Σ_k^{-1} μ_k             (K,)    — host-precomp

    Then for each (n, k):
        quad[n, k]   = Σ_{d1,d2} x[n,d1] · x[n,d2] · prec_flat[k, d1*D+d2]
        linear[n, k] = Σ_d x[n, d] · mu_w[k, d]
        maha[n, k]   = quad[n, k]  -  2 · linear[n, k]  +  mu_quad[k]

    The TWO matmuls are clean WGMMA:
        (BLOCK_N, D²)   @   (D², BLOCK_K)   →   (BLOCK_N, BLOCK_K)   for quad
        (BLOCK_N, D)    @   (D,  BLOCK_K)   →   (BLOCK_N, BLOCK_K)   for linear

    The constant μ_k^T Σ_k^{-1} μ_k folds into c0[k].

Optimizations vs the previous version:

  1. Pre-transpose prec_flat → (D², K) and mu_w → (D, K). Inner-loop matmuls
     become direct WGMMA-friendly contractions.
  2. Split-K persistent pass 2:
       Grid = (cdiv(K, BLOCK_K), N_SPLIT)
       Each CTA owns one K-block and a contiguous N-slab. It iterates over
       its N rows in chunks of BLOCK_N, accumulating nk_local / mu_local /
       sig_local in REGISTERS. After the entire N-slab is processed it does
       exactly THREE atomic_adds per CTA (Nk, mu_acc, sig_outer).
     vs the prior cdiv(N,BN) * cdiv(K,BK) atomic groups. For Deep10M roughly
     78k * 8 atomic groups → ~256 groups, a >2000x reduction in atomic
     contention on the (K, D, D) sig_outer buffer.
  3. K_EXACT constexpr drops K-mask paths when K%256==0 (always true here).
  4. exp2/log2 — H100 SFU has dedicated exp2 instruction.

Same math as the previous full-cov kernel; same numerical envelope. Output
tensors are re-allocated fresh on every call.

Entry point (DO NOT RENAME):
    flash_gmm_estep_full(X, mu, prec_chol, log_pi)
        -> (logZ, Nk, mu_acc, sig_outer)

where prec_chol[k] is the lower-triangular Cholesky factor of Σ_k^{-1}.
The reconstructed precision matrix Σ⁻¹ = L Lᵀ is what enters the kernel.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------
# Pass 1 autotune (grid over N)
# --------------------------------------------------------------------

def _pass1_configs():
    cfgs = [
        # (BLOCK_N, BLOCK_K, num_warps, num_stages)
        (64,  64, 4, 3),
        (64, 128, 4, 3),
        (128, 64, 4, 3),  (128, 64, 8, 3),
        (128,128, 8, 3),  (128,128, 8, 4),
        (128,256, 8, 3),
        (256, 64, 8, 3),  (256, 64, 8, 4),
        (256,128, 8, 3),  (256,128, 8, 4),
    ]
    return [
        triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
        for (bn, bk, nw, ns) in cfgs
    ]


@triton.autotune(configs=_pass1_configs(), key=["N", "K", "D", "BLOCK_D", "DD", "K_EXACT"])
@triton.jit
def _full_lse_kernel(
    X_ptr, prec_flatT_ptr, mu_wT_ptr, c0_ptr,
    logZ2_ptr,
    N, K, D,
    stride_xn, stride_xd,
    stride_pdT_dd, stride_pdT_k,
    stride_mwT_d, stride_mwT_k,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DD: tl.constexpr,        # = BLOCK_D * BLOCK_D — flattened precision width
    K_EXACT: tl.constexpr,
):
    """
    Pass 1: tiled online log-sum-exp over K.

    Inputs (host-precomputed):
      prec_flatT[d1*D+d2, k] = (Σ_k^{-1})[d1, d2]                (D², K) bf16
      mu_wT[d, k]            = (Σ_k^{-1} μ_k)[d]                  (D,  K) bf16
      c0[k]                  = ( log_pi + Σ log L_kdd
                                 - 0.5·D·log(2π)
                                 - 0.5 · μ_k^T Σ^{-1} μ_k ) · log2_e   (K,) fp32

    For each (n, k) the log2-likelihood is
        ll2[n,k] = c0[k]
                   - 0.5 · log2_e · ( quad[n,k] - 2 · linear[n,k] )
        quad[n,k]   = Σ_{d1,d2} (x[n,d1]·x[n,d2]) · prec_flat[k, d1·D+d2]
        linear[n,k] = Σ_d x[n,d] · mu_w[k,d]
    """
    pid = tl.program_id(0)
    n_off = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_off < N
    d_off  = tl.arange(0, BLOCK_D)
    dd_off = tl.arange(0, DD)

    # Load X tile (BLOCK_N, BLOCK_D); padded host-side so no d-mask needed
    x = tl.load(
        X_ptr + n_off[:, None] * stride_xn + d_off[None, :] * stride_xd,
        mask=n_mask[:, None], other=0.0,
    )
    x_bf = x.to(tl.bfloat16)

    # Outer product per row: xx[n, d1*D + d2] = x[n, d1] * x[n, d2]
    # Materialize as (BLOCK_N, BLOCK_D, BLOCK_D) then reshape — one register tile.
    xx = x[:, :, None] * x[:, None, :]                          # (BLOCK_N, BLOCK_D, BLOCK_D) fp32
    xx_flat = tl.reshape(xx, (BLOCK_N, DD)).to(tl.bfloat16)     # (BLOCK_N, DD) bf16

    running_max = tl.full([BLOCK_N], float('-inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)

        if K_EXACT:
            # Two BF16 matmuls — both clean WGMMA on H100
            prec_blk = tl.load(
                prec_flatT_ptr + dd_off[:, None] * stride_pdT_dd + k_off[None, :] * stride_pdT_k
            )
            mu_w_blk = tl.load(
                mu_wT_ptr + d_off[:, None] * stride_mwT_d + k_off[None, :] * stride_mwT_k
            )
            c0 = tl.load(c0_ptr + k_off)
        else:
            k_mask = k_off < K
            prec_blk = tl.load(
                prec_flatT_ptr + dd_off[:, None] * stride_pdT_dd + k_off[None, :] * stride_pdT_k,
                mask=k_mask[None, :], other=0.0,
            )
            mu_w_blk = tl.load(
                mu_wT_ptr + d_off[:, None] * stride_mwT_d + k_off[None, :] * stride_mwT_k,
                mask=k_mask[None, :], other=0.0,
            )
            c0 = tl.load(c0_ptr + k_off, mask=k_mask, other=float('-inf'))

        quad   = tl.dot(xx_flat, prec_blk, out_dtype=tl.float32)   # (BLOCK_N, BLOCK_K)
        linear = tl.dot(x_bf,    mu_w_blk, out_dtype=tl.float32)   # (BLOCK_N, BLOCK_K)
        # ll2 in log2 space: c0 already includes the 0.5·log2_e·μ^TΣ⁻¹μ correction
        ll2 = c0[None, :] - (0.5 * 1.4426950408889634) * (quad - 2.0 * linear)

        if not K_EXACT:
            ll2 = tl.where(k_mask[None, :], ll2, float('-inf'))

        block_max = tl.max(ll2, axis=1)
        new_max = tl.maximum(running_max, block_max)
        running_sum = (running_sum * tl.exp2(running_max - new_max)
                       + tl.sum(tl.exp2(ll2 - new_max[:, None]), axis=1))
        running_max = new_max

    log_Z2 = running_max + tl.log2(running_sum)
    tl.store(logZ2_ptr + n_off, log_Z2, mask=n_mask)


# --------------------------------------------------------------------
# Pass 2: split-K persistent kernel
#   Grid = (n_k_blocks, n_splits)
#   Each CTA owns one K-block and a contiguous N-slab.
# --------------------------------------------------------------------

def _pass2_configs():
    # sig_local is (BLOCK_K, BLOCK_D, BLOCK_D) fp32 — biggest live value.
    # At BLOCK_K=8, BLOCK_D=128: 8 * 128 * 128 * 4 = 512 KB → too big for regs.
    # H100 register file is 256 KB / SM. Keep BLOCK_K small for full-cov.
    cfgs = [
        # (BLOCK_N, BLOCK_K, num_warps, num_stages)
        (64,  4, 4, 3),  (64,  4, 8, 3),
        (64,  8, 4, 3),  (64,  8, 8, 3),
        (128, 4, 4, 3),  (128, 4, 8, 3),  (128, 4, 8, 4),
        (128, 8, 8, 3),  (128, 8, 8, 4),
        (256, 4, 8, 3),  (256, 4, 8, 4),
    ]
    return [
        triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
        for (bn, bk, nw, ns) in cfgs
    ]


@triton.autotune(
    configs=_pass2_configs(),
    key=["N", "K", "D", "BLOCK_D", "DD", "K_EXACT", "N_SPLIT"],
    reset_to_zero=["Nk_ptr", "mu_acc_ptr", "sig_outer_ptr"],
)
@triton.jit
def _full_split_k_accum(
    X_ptr, prec_flatT_ptr, mu_wT_ptr, c0_ptr, logZ2_ptr,
    Nk_ptr, mu_acc_ptr, sig_outer_ptr,
    N, K, D, N_SPLIT,
    stride_xn, stride_xd,
    stride_pdT_dd, stride_pdT_k,
    stride_mwT_d, stride_mwT_k,
    stride_ma_k, stride_ma_d,
    stride_so_k, stride_so_d1, stride_so_d2,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DD: tl.constexpr,
    K_EXACT: tl.constexpr,
):
    k_pid = tl.program_id(0)
    n_pid = tl.program_id(1)

    k_off  = (k_pid * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
    d_off  = tl.arange(0, BLOCK_D)
    dd_off = tl.arange(0, DD)

    if K_EXACT:
        prec_blk = tl.load(
            prec_flatT_ptr + dd_off[:, None] * stride_pdT_dd + k_off[None, :] * stride_pdT_k
        )
        mu_w_blk = tl.load(
            mu_wT_ptr + d_off[:, None] * stride_mwT_d + k_off[None, :] * stride_mwT_k
        )
        c0 = tl.load(c0_ptr + k_off)
    else:
        k_mask_active = k_off < K
        prec_blk = tl.load(
            prec_flatT_ptr + dd_off[:, None] * stride_pdT_dd + k_off[None, :] * stride_pdT_k,
            mask=k_mask_active[None, :], other=0.0,
        )
        mu_w_blk = tl.load(
            mu_wT_ptr + d_off[:, None] * stride_mwT_d + k_off[None, :] * stride_mwT_k,
            mask=k_mask_active[None, :], other=0.0,
        )
        c0 = tl.load(c0_ptr + k_off, mask=k_mask_active, other=float('-inf'))

    chunk = (N + N_SPLIT - 1) // N_SPLIT
    n_start = n_pid * chunk
    n_end   = n_start + chunk

    # Local accumulators in fp32 — kept in registers across the inner N-loop.
    nk_local  = tl.zeros([BLOCK_K], dtype=tl.float32)
    mu_local  = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)
    # sig_local is the SECOND MOMENT (uncentered):  Σ_i r_ik · x_i x_i^T
    # Stored flat (BLOCK_K, DD) and unflattened on store. Same memory either way.
    sig_local = tl.zeros([BLOCK_K, DD], dtype=tl.float32)

    for n0 in range(n_start, n_end, BLOCK_N):
        n_off = (n0 + tl.arange(0, BLOCK_N)).to(tl.int64)
        n_mask = n_off < N

        x = tl.load(
            X_ptr + n_off[:, None] * stride_xn + d_off[None, :] * stride_xd,
            mask=n_mask[:, None], other=0.0,
        )
        log_Z2 = tl.load(logZ2_ptr + n_off, mask=n_mask, other=0.0)
        x_bf = x.to(tl.bfloat16)

        # Same outer product as pass 1
        xx = x[:, :, None] * x[:, None, :]
        xx_flat = tl.reshape(xx, (BLOCK_N, DD)).to(tl.bfloat16)

        # Recompute ll2 for THIS K-block
        quad   = tl.dot(xx_flat, prec_blk, out_dtype=tl.float32)
        linear = tl.dot(x_bf,    mu_w_blk, out_dtype=tl.float32)
        ll2 = c0[None, :] - (0.5 * 1.4426950408889634) * (quad - 2.0 * linear)
        ll2 = tl.where(n_mask[:, None], ll2, float('-inf'))

        # Responsibilities
        r = tl.exp2(ll2 - log_Z2[:, None])             # (BLOCK_N, BLOCK_K) fp32
        r_bf = r.to(tl.bfloat16)

        # Accumulate
        nk_local  += tl.sum(r, axis=0)                                          # (BLOCK_K,)
        mu_local  += tl.dot(tl.trans(r_bf), x_bf,    out_dtype=tl.float32)      # (BLOCK_K, BLOCK_D)
        # sig_local[k, d1*D+d2] += Σ_n r[n,k] · xx_flat[n, d1*D+d2]
        sig_local += tl.dot(tl.trans(r_bf), xx_flat, out_dtype=tl.float32)      # (BLOCK_K, DD)

    # Atomic-add the accumulators back to global. Three atomic groups per CTA total.
    if K_EXACT:
        tl.atomic_add(Nk_ptr + k_off, nk_local)
        tl.atomic_add(
            mu_acc_ptr + k_off[:, None] * stride_ma_k + d_off[None, :] * stride_ma_d,
            mu_local,
        )
        # Unflatten sig_local (BLOCK_K, DD) → (BLOCK_K, BLOCK_D, BLOCK_D) for the store.
        sig_3d = tl.reshape(sig_local, (BLOCK_K, BLOCK_D, BLOCK_D))
        d1_idx = tl.arange(0, BLOCK_D)
        d2_idx = tl.arange(0, BLOCK_D)
        sig_off = (k_off[:, None, None] * stride_so_k
                   + d1_idx[None, :, None] * stride_so_d1
                   + d2_idx[None, None, :] * stride_so_d2)
        tl.atomic_add(sig_outer_ptr + sig_off, sig_3d)
    else:
        tl.atomic_add(Nk_ptr + k_off, nk_local, mask=k_mask_active)
        tl.atomic_add(
            mu_acc_ptr + k_off[:, None] * stride_ma_k + d_off[None, :] * stride_ma_d,
            mu_local, mask=k_mask_active[:, None],
        )
        sig_3d = tl.reshape(sig_local, (BLOCK_K, BLOCK_D, BLOCK_D))
        d1_idx = tl.arange(0, BLOCK_D)
        d2_idx = tl.arange(0, BLOCK_D)
        sig_off = (k_off[:, None, None] * stride_so_k
                   + d1_idx[None, :, None] * stride_so_d1
                   + d2_idx[None, None, :] * stride_so_d2)
        tl.atomic_add(sig_outer_ptr + sig_off, sig_3d, mask=k_mask_active[:, None, None])


# --------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------

_LOG_2PI = 1.8378770664093453
_LOG2_E  = 1.4426950408889634


def _pick_n_split(N: int) -> int:
    """Choose N-splits so the (k_blocks × N_SPLIT) total CTA count lands
    around ~128–512 on H100 (132 SMs)."""
    if N >= 5_000_000:
        return 32          # Deep10M
    if N >= 1_000_000:
        return 16          # SIFT1M, GloVe
    return 8


def flash_gmm_estep_full(
    X: torch.Tensor,
    mu: torch.Tensor,
    prec_chol: torch.Tensor,    # (K, D, D) — lower-triangular Cholesky factor of Σ⁻¹
    log_pi: torch.Tensor,        # (K,)
):
    """
    Flash-GMM E-step with full covariance.

    Args:
        X:         (N, D)    float32 GPU tensor — data
        mu:        (K, D)    float32 GPU tensor — component means
        prec_chol: (K, D, D) float32 GPU tensor — lower-triangular Cholesky
                   factor of the precision matrix Σ_k^{-1}, i.e.
                       prec_chol[k] @ prec_chol[k].T = Σ_k^{-1}
        log_pi:    (K,)      float32 GPU tensor — log mixture weights

    Returns:
        logZ:           (N,)        float32 — per-sample log-partition function
        Nk:             (K,)        float32 — Σ_i r_ik
        mu_acc:         (K, D)      float32 — Σ_i r_ik · x_i
        sig_outer:      (K, D, D)   float32 — Σ_i r_ik · x_i x_i^T  (uncentered)

    M-step recovery:
        π_k    = Nk[k] / N
        μ_k    = mu_acc[k] / Nk[k]
        Σ_k    = sig_outer[k] / Nk[k]  -  μ_k μ_k^T
        Apply your preferred regularization (e.g. Σ_k += ε I) before
        re-Cholesky-factorizing for the next iteration.
    """
    N, D = X.shape
    K = mu.shape[0]
    assert prec_chol.shape == (K, D, D), \
        f"prec_chol must be (K, D, D)=({K}, {D}, {D}), got {tuple(prec_chol.shape)}"
    assert X.is_cuda and mu.is_cuda

    BLOCK_D = max(16, triton.next_power_of_2(D) if D > 0 else 16)
    DD = BLOCK_D * BLOCK_D

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

    # ---- Host-side precomputes (run once, reused by both passes) ----
    # Precision matrix Σ⁻¹_k = L_k L_k^T   (K, BLOCK_D, BLOCK_D)
    prec = torch.bmm(L_pad, L_pad.transpose(-1, -2))                    # (K, BLOCK_D, BLOCK_D)
    # Flatten to (K, DD) row-major (d1·BLOCK_D + d2)
    prec_flat = prec.reshape(K, DD)                                     # (K, DD)
    # mu_w[k] = Σ⁻¹_k μ_k    (K, BLOCK_D)
    mu_w = torch.einsum("kij,kj->ki", prec, mu_pad)                     # (K, BLOCK_D)
    # μ_k^T Σ⁻¹_k μ_k    (K,)
    mu_quad = (mu_w * mu_pad).sum(dim=1)                                # (K,)

    # log|Σ⁻¹|^{1/2} = Σ_d log L_kdd. Use the unpadded diagonal (padded rows are zero).
    log_diag_sum = torch.log(prec_chol.diagonal(dim1=-2, dim2=-1).abs() + 1e-30).sum(dim=-1)  # (K,)
    # c0 in log2 space: log_pi + log|Σ⁻¹|^{1/2} - 0.5·D·log(2π) - 0.5·μ^TΣ⁻¹μ
    c0 = ((log_pi + log_diag_sum - 0.5 * D * _LOG_2PI - 0.5 * mu_quad) * _LOG2_E).contiguous()

    # Pre-transpose for WGMMA-friendly inner matmuls
    prec_flatT_bf = prec_flat.to(torch.bfloat16).t().contiguous()       # (DD, K)
    mu_wT_bf      = mu_w.to(torch.bfloat16).t().contiguous()            # (BLOCK_D, K)

    # ---- Outputs ----
    logZ2 = torch.empty(N, device=X.device, dtype=torch.float32)
    Nk = torch.zeros(K, device=X.device, dtype=torch.float32)
    mu_acc_pad    = torch.zeros(K, BLOCK_D, device=X.device, dtype=torch.float32)
    sig_outer_pad = torch.zeros(K, BLOCK_D, BLOCK_D, device=X.device, dtype=torch.float32)

    K_EXACT = (K % 256 == 0)
    N_SPLIT = _pick_n_split(N)

    # ---- Pass 1: per-N-tile online LSE ----
    pass1_grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    _full_lse_kernel[pass1_grid](
        X_pad, prec_flatT_bf, mu_wT_bf, c0, logZ2,
        N, K, D,
        X_pad.stride(0), X_pad.stride(1),
        prec_flatT_bf.stride(0), prec_flatT_bf.stride(1),
        mu_wT_bf.stride(0), mu_wT_bf.stride(1),
        BLOCK_D=BLOCK_D, DD=DD, K_EXACT=K_EXACT,
    )

    # ---- Pass 2: split-K persistent ----
    pass2_grid = lambda META: (triton.cdiv(K, META["BLOCK_K"]), N_SPLIT)
    _full_split_k_accum[pass2_grid](
        X_pad, prec_flatT_bf, mu_wT_bf, c0, logZ2,
        Nk, mu_acc_pad, sig_outer_pad,
        N, K, D, N_SPLIT,
        X_pad.stride(0), X_pad.stride(1),
        prec_flatT_bf.stride(0), prec_flatT_bf.stride(1),
        mu_wT_bf.stride(0), mu_wT_bf.stride(1),
        mu_acc_pad.stride(0), mu_acc_pad.stride(1),
        sig_outer_pad.stride(0), sig_outer_pad.stride(1), sig_outer_pad.stride(2),
        BLOCK_D=BLOCK_D, DD=DD, K_EXACT=K_EXACT,
    )

    mu_acc    = mu_acc_pad[:, :D].contiguous()
    sig_outer = sig_outer_pad[:, :D, :D].contiguous()
    logZ = logZ2 * (1.0 / _LOG2_E)
    return logZ, Nk, mu_acc, sig_outer
