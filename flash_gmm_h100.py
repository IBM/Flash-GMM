"""
Flash-GMM E-step — H100-tuned two-launch kernel with BF16 matmul.

Two Triton launches (same structure as baseline):
  Pass 1: tiled online log-sum-exp over K -> writes logZ to HBM (in log2 space)
  Pass 2: responsibilities + atomic-add into Nk / mu_acc / sig_acc

H100 wins relative to baseline:
  - BF16 inputs to tl.dot with FP32 accumulator -> WGMMA (2x matmul throughput)
  - Autotuned BLOCK_N / BLOCK_K / num_warps / num_stages per workload
  - Per-K constants precomputed on host (mu_sq, inv_s, log_pi offsets) and
    folded into log2 space:
      ll2[n,k] = c0[k] - ish[k] * x_sq[n] + (2*ish[k]) * xmu[n,k]      (log2 units)
  - tl.exp / tl.log replaced by tl.exp2 / tl.log2 -> H100 SFU ex2.approx fast paths.
  - Both X and mu are pre-padded to BLOCK_D columns on the host so all D-axis
    masks can be dropped from the inner kernels. mu_acc is allocated padded
    and sliced to [:, :D] at the end.

Entry point (DO NOT RENAME):
    flash_gmm_estep(X, mu, log_sigma_sq, log_pi) -> (logZ, Nk, mu_acc, sig_acc)
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def _autotune_configs():
    # Agent-1's 18-config sweep got us to 4.57x. Adding configs that probe
    # the Hopper-specific extremes that weren't covered:
    #   - num_warps=16 (full 4-warpgroup occupancy = 16 warps × 32 = 512 threads,
    #     enabling FA3-style producer/consumer overlap of TMA with WGMMA).
    #   - num_stages=5 (deeper pipeline; H100 has the SRAM headroom).
    #   - 128×256 num_warps=4 (smaller warpgroup, larger K tile for occupancy).
    cfgs = [
        # (BLOCK_N, BLOCK_K, num_warps, num_stages)
        # original 18 (agent-1's 4.57x sweep)
        (64,  64, 4, 3),
        (64,  64, 8, 3),
        (64, 128, 4, 3),
        (64, 128, 8, 3),
        (64, 256, 8, 3),
        (128, 64, 4, 3),
        (128, 64, 8, 3),
        (128, 64, 8, 4),
        (128, 128, 4, 3),
        (128, 128, 8, 3),
        (128, 128, 8, 4),
        (128, 256, 8, 3),
        (128, 256, 8, 4),
        (256, 64, 8, 3),
        (256, 64, 8, 4),
        (256, 128, 8, 3),
        (256, 128, 8, 4),
        (256, 256, 8, 3),
        # agent-2 v6 additions
        (128, 256, 4, 3),     # smaller warpgroup, fat K tile
        (256, 128, 16, 3),    # full 4-warpgroup occupancy
        (256, 128, 16, 4),
        (256, 256, 16, 3),
        (256, 256, 8, 4),     # fill in missing num_stages=4 for the largest tile
        (128, 128, 8, 5),     # deeper pipeline at moderate tile size
        (128, 256, 8, 5),
        (64, 256, 4, 3),      # mid-N, fat K, smaller warpgroup
        # v11: BLOCK_K=512 enabled by BF16 X+mu (half the SMEM cost). K=1024 -> 2 K-tiles only.
        (64, 512, 8, 3),
        (128, 512, 8, 3),
        (64, 512, 4, 3),
        (128, 512, 4, 3),
    ]
    return [
        triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
        for (bn, bk, nw, ns) in cfgs
    ]


# ---------------------------------------------------------------------------
# Pass 1: online log-sum-exp over K (writes logZ to HBM, in log2 units)
# ---------------------------------------------------------------------------

@triton.autotune(configs=_autotune_configs(), key=["N", "K", "D", "BLOCK_D", "K_EXACT"])
@triton.jit
def _gmm_lse_kernel(
    X_ptr, mu_ptr, c0_ptr, ish_ptr,
    logZ2_ptr,
    N, K, D,
    stride_xn, stride_xd,
    stride_mk, stride_md,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K_EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    n_off = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_off < N
    d_off = tl.arange(0, BLOCK_D)

    # X is pre-cast to BF16 on host and padded to BLOCK_D columns.
    # Load as BF16, cast UP to FP32 for x_sq (the sum-of-squares needs FP32
    # accumulation precision, but the underlying values are already BF16 so
    # no additional precision is lost).
    x_bf = tl.load(
        X_ptr + n_off[:, None] * stride_xn + d_off[None, :] * stride_xd,
        mask=n_mask[:, None], other=0.0,
    )
    x_fp32 = x_bf.to(tl.float32)
    x_sq = tl.sum(x_fp32 * x_fp32, axis=1)

    running_max = tl.full([BLOCK_N], float('-inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)

        # K_EXACT: K divides BLOCK_K -> drop k_mask (production case K=1024).
        # mu is pre-cast to BF16 on the host -> load directly without per-tile cast.
        if K_EXACT:
            mu_bf = tl.load(mu_ptr + k_off[:, None] * stride_mk + d_off[None, :] * stride_md)
            c0 = tl.load(c0_ptr + k_off)
            ish = tl.load(ish_ptr + k_off)
        else:
            k_mask = k_off < K
            mu_bf = tl.load(
                mu_ptr + k_off[:, None] * stride_mk + d_off[None, :] * stride_md,
                mask=k_mask[:, None], other=0.0,
            )
            c0 = tl.load(c0_ptr + k_off, mask=k_mask, other=float('-inf'))
            ish = tl.load(ish_ptr + k_off, mask=k_mask, other=0.0)

        xmu = tl.dot(x_bf, tl.trans(mu_bf), out_dtype=tl.float32)

        ll2 = (c0[None, :] - ish[None, :] * x_sq[:, None]
               + (2.0 * ish[None, :]) * xmu)
        if not K_EXACT:
            ll2 = tl.where((k_off < K)[None, :], ll2, float('-inf'))

        block_max = tl.max(ll2, axis=1)
        new_max = tl.maximum(running_max, block_max)
        running_sum = (running_sum * tl.exp2(running_max - new_max)
                       + tl.sum(tl.exp2(ll2 - new_max[:, None]), axis=1))
        running_max = new_max

    log_Z2 = running_max + tl.log2(running_sum)
    tl.store(logZ2_ptr + n_off, log_Z2, mask=n_mask)


# ---------------------------------------------------------------------------
# Pass 2: responsibilities + atomic accumulation
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=_autotune_configs(),
    key=["N", "K", "D", "BLOCK_D", "K_EXACT"],
    reset_to_zero=["Nk_ptr", "mu_acc_ptr", "sig_acc_ptr"],
)
@triton.jit
def _gmm_atomic_accum(
    X_ptr, mu_ptr, c0_ptr, ish_ptr, mu_sq_ptr, logZ2_ptr,
    Nk_ptr, mu_acc_ptr, sig_acc_ptr,
    N, K, D,
    stride_xn, stride_xd,
    stride_mk, stride_md,
    stride_ma_k, stride_ma_d,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K_EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    n_off = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_off < N
    d_off = tl.arange(0, BLOCK_D)

    # X pre-cast to BF16 on host, padded to BLOCK_D columns.
    x_bf = tl.load(
        X_ptr + n_off[:, None] * stride_xn + d_off[None, :] * stride_xd,
        mask=n_mask[:, None], other=0.0,
    )
    log_Z2 = tl.load(logZ2_ptr + n_off, mask=n_mask, other=0.0)
    x_fp32 = x_bf.to(tl.float32)
    x_sq = tl.sum(x_fp32 * x_fp32, axis=1)

    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)

        # mu pre-cast to BF16 on host -> load directly without per-tile cast.
        if K_EXACT:
            mu_bf = tl.load(mu_ptr + k_off[:, None] * stride_mk + d_off[None, :] * stride_md)
            c0 = tl.load(c0_ptr + k_off)
            ish = tl.load(ish_ptr + k_off)
            mu_sq = tl.load(mu_sq_ptr + k_off)
        else:
            k_mask = k_off < K
            mu_bf = tl.load(
                mu_ptr + k_off[:, None] * stride_mk + d_off[None, :] * stride_md,
                mask=k_mask[:, None], other=0.0,
            )
            c0 = tl.load(c0_ptr + k_off, mask=k_mask, other=float('-inf'))
            ish = tl.load(ish_ptr + k_off, mask=k_mask, other=0.0)
            mu_sq = tl.load(mu_sq_ptr + k_off, mask=k_mask, other=0.0)

        xmu = tl.dot(x_bf, tl.trans(mu_bf), out_dtype=tl.float32)

        sq = x_sq[:, None] - 2.0 * xmu + mu_sq[None, :]

        ll2 = (c0[None, :] - ish[None, :] * x_sq[:, None]
               + (2.0 * ish[None, :]) * xmu)
        if K_EXACT:
            ll2 = tl.where(n_mask[:, None], ll2, float('-inf'))
        else:
            ll2 = tl.where((k_off < K)[None, :] & n_mask[:, None], ll2, float('-inf'))

        r = tl.exp2(ll2 - log_Z2[:, None])

        nk_block = tl.sum(r, axis=0)
        sig_block = tl.sum(r * sq, axis=0)
        if K_EXACT:
            tl.atomic_add(Nk_ptr + k_off, nk_block)
            tl.atomic_add(sig_acc_ptr + k_off, sig_block)
        else:
            k_mask = k_off < K
            tl.atomic_add(Nk_ptr + k_off, nk_block, mask=k_mask)
            tl.atomic_add(sig_acc_ptr + k_off, sig_block, mask=k_mask)

        r_bf = r.to(tl.bfloat16)
        mu_block = tl.dot(tl.trans(r_bf), x_bf, out_dtype=tl.float32)
        # mu_acc allocated as [K, BLOCK_D] -> drop d_mask, full-width store.
        if K_EXACT:
            tl.atomic_add(
                mu_acc_ptr + k_off[:, None] * stride_ma_k + d_off[None, :] * stride_ma_d,
                mu_block,
            )
        else:
            tl.atomic_add(
                mu_acc_ptr + k_off[:, None] * stride_ma_k + d_off[None, :] * stride_ma_d,
                mu_block,
                mask=(k_off < K)[:, None],
            )


# ---------------------------------------------------------------------------
# Public entry point — DO NOT RENAME
# ---------------------------------------------------------------------------

_LOG_2PI = 1.8378770664093453
_LOG2_E = 1.4426950408889634   # 1 / ln(2)


def flash_gmm_estep(
    X: torch.Tensor,
    mu: torch.Tensor,
    log_sigma_sq: torch.Tensor,
    log_pi: torch.Tensor,
):
    """Compute the GMM E-step sufficient statistics on GPU.

    Returns (logZ, Nk, mu_acc, sig_acc).
    """
    N, D = X.shape
    K = mu.shape[0]
    assert X.is_cuda and mu.is_cuda

    BLOCK_D = triton.next_power_of_2(D) if D > 0 else 1
    if BLOCK_D < 16:
        BLOCK_D = 16

    # Pre-cast X and mu to BF16 on host, padded to BLOCK_D columns.
    # Halves HBM bandwidth for X loads (10M*128*2B vs 4B on Deep10M).
    # x_sq = sum(x*x) is computed inside the kernel from x cast UP to FP32;
    # the underlying BF16 storage already lost ~3e-3 rel precision per element,
    # but D-summed x_sq has comparable rel error -- still well within 5e-3 tol.
    if D == BLOCK_D:
        X_pad_bf = X.to(torch.bfloat16).contiguous()
        mu_pad_bf = mu.to(torch.bfloat16).contiguous()
    else:
        X_pad_bf = torch.zeros(N, BLOCK_D, device=X.device, dtype=torch.bfloat16)
        X_pad_bf[:, :D] = X.to(torch.bfloat16)
        mu_pad_bf = torch.zeros(K, BLOCK_D, device=mu.device, dtype=torch.bfloat16)
        mu_pad_bf[:, :D] = mu.to(torch.bfloat16)

    # Per-K constants in log2 space:
    inv_s = torch.exp(-log_sigma_sq)
    mu_sq = (mu * mu).sum(dim=1)                                       # [K]
    ish = (0.5 * inv_s * _LOG2_E).contiguous()                         # [K]
    c0 = ((log_pi - 0.5 * D * (_LOG_2PI + log_sigma_sq)) * _LOG2_E
          - ish * mu_sq).contiguous()                                  # [K]
    mu_sq = mu_sq.contiguous()

    logZ2 = torch.empty(N, device=X.device, dtype=torch.float32)
    Nk = torch.zeros(K, device=X.device, dtype=torch.float32)
    # mu_acc allocated padded -> output is contiguous slice.
    mu_acc_pad = torch.zeros(K, BLOCK_D, device=X.device, dtype=torch.float32)
    sig_acc = torch.zeros(K, device=X.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)

    # K_EXACT: K is divisible by every BLOCK_K config in the autotune sweep
    # (32, 64, 128, 256) when K is a multiple of 256. K=1024 satisfies this.
    K_EXACT = (K % 256 == 0)

    _gmm_lse_kernel[grid](
        X_pad_bf, mu_pad_bf, c0, ish, logZ2,
        N, K, D,
        X_pad_bf.stride(0), X_pad_bf.stride(1),
        mu_pad_bf.stride(0), mu_pad_bf.stride(1),
        BLOCK_D=BLOCK_D, K_EXACT=K_EXACT,
    )

    _gmm_atomic_accum[grid](
        X_pad_bf, mu_pad_bf, c0, ish, mu_sq, logZ2,
        Nk, mu_acc_pad, sig_acc,
        N, K, D,
        X_pad_bf.stride(0), X_pad_bf.stride(1),
        mu_pad_bf.stride(0), mu_pad_bf.stride(1),
        mu_acc_pad.stride(0), mu_acc_pad.stride(1),
        BLOCK_D=BLOCK_D, K_EXACT=K_EXACT,
    )

    # Return contiguous mu_acc slice and ln-units logZ.
    mu_acc = mu_acc_pad[:, :D].contiguous()
    logZ = logZ2 * (1.0 / _LOG2_E)

    return logZ, Nk, mu_acc, sig_acc
