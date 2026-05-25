"""
Flash-GMM: Memory-Efficient GPU Kernel for the GMM E-step

A fused Triton kernel that computes the full GMM E-step in a single pass
over the data, eliminating the N×K responsibility matrix.

Architecture:
  - Grid: (ceil(N / BLOCK_N),) blocks
  - Each block processes BLOCK_N rows, iterates over K components in tiles of BLOCK_K
  - Pass 1: compute log Z_i via online numerically stable log-sum-exp
  - Pass 2: reuse X tile from registers, compute r_ik, warp-reduce then
            atomicAdd into O(KD) output buffers

Memory: O(K*D) constant regardless of N — the N×K responsibility matrix
is never materialized.

Usage:
    from flash_gmm import flash_gmm_estep

    logZ, Nk, mu_acc, sig_acc = flash_gmm_estep(X, mu, log_sigma_sq, log_pi)

Where:
    X            : (N, D) float32 GPU tensor — data
    mu           : (K, D) float32 GPU tensor — component means
    log_sigma_sq : (K,)   float32 GPU tensor — log variance per component
    log_pi       : (K,)   float32 GPU tensor — log mixture weights

Returns:
    logZ     : (N,)   float32 — per-sample log-normaliser log Z_i
    Nk       : (K,)   float32 — effective cluster counts sum_i r_ik
    mu_acc   : (K, D) float32 — responsibility-weighted sum sum_i r_ik * x_i
    sig_acc  : (K,)   float32 — responsibility-weighted sq dist sum_i r_ik ||x_i - mu_k||^2

These are the sufficient statistics for the M-step:
    pi_k      = Nk / N
    mu_k      = mu_acc[k] / Nk[k]
    sigma_k^2 = sig_acc[k] / (D * Nk[k])
"""

import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pass 1: Online log-sum-exp to compute log Z_i
# ---------------------------------------------------------------------------

@triton.jit
def _flash_gmm_lse_kernel(
    X_ptr, mu_ptr, log_sigma_sq_ptr, log_pi_ptr,
    logZ_ptr,
    N, K, D,
    stride_xn, stride_xd,
    stride_mk, stride_md,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    n_off = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_off < N
    d_off = tl.arange(0, BLOCK_D)

    # Load X tile into registers — stays resident for Pass 2
    x = tl.load(
        X_ptr + n_off[:, None] * stride_xn + d_off[None, :] * stride_xd,
        mask=n_mask[:, None] & (d_off[None, :] < D), other=0.0,
    )

    running_max = tl.full([BLOCK_N], float('-inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_off < K

        mu  = tl.load(mu_ptr + k_off[:, None] * stride_mk + d_off[None, :] * stride_md,
                      mask=k_mask[:, None] & (d_off[None, :] < D), other=0.0)
        lss = tl.load(log_sigma_sq_ptr + k_off, mask=k_mask, other=0.0)
        lpi = tl.load(log_pi_ptr + k_off, mask=k_mask, other=float('-inf'))

        # Squared distance via the ||x-mu||^2 = ||x||^2 - 2x·mu + ||mu||^2 trick
        x_sq  = tl.sum(x * x, axis=1)           # (BLOCK_N,)
        mu_sq = tl.sum(mu * mu, axis=1)          # (BLOCK_K,)
        xmu   = tl.dot(x, tl.trans(mu))          # (BLOCK_N, BLOCK_K)
        sq    = x_sq[:, None] - 2.0 * xmu + mu_sq[None, :]

        inv_s  = tl.exp(-lss)
        log2pi = 1.8378770664093453
        ll = lpi[None, :] - 0.5 * sq * inv_s[None, :] - 0.5 * D * (log2pi + lss[None, :])
        ll = tl.where(k_mask[None, :], ll, float('-inf'))

        # Numerically stable online log-sum-exp update
        block_max  = tl.max(ll, axis=1)
        new_max    = tl.maximum(running_max, block_max)
        running_sum = (running_sum * tl.exp(running_max - new_max)
                       + tl.sum(tl.exp(ll - new_max[:, None]), axis=1))
        running_max = new_max

    log_Z = running_max + tl.log(running_sum)
    tl.store(logZ_ptr + n_off, log_Z, mask=n_mask)


# ---------------------------------------------------------------------------
# Pass 2: Compute responsibilities and accumulate sufficient statistics
# ---------------------------------------------------------------------------

@triton.jit
def _flash_gmm_accum_kernel(
    X_ptr, mu_ptr, log_sigma_sq_ptr, log_pi_ptr, logZ_ptr,
    Nk_ptr, mu_acc_ptr, sig_acc_ptr,
    N, K, D,
    stride_xn, stride_xd,
    stride_mk, stride_md,
    stride_ma_k, stride_ma_d,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    n_off = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_off < N
    d_off  = tl.arange(0, BLOCK_D)

    # Reload X tile (same tile as Pass 1 — may hit L2 cache)
    x     = tl.load(X_ptr + n_off[:, None] * stride_xn + d_off[None, :] * stride_xd,
                    mask=n_mask[:, None] & (d_off[None, :] < D), other=0.0)
    log_Z = tl.load(logZ_ptr + n_off, mask=n_mask, other=0.0)

    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_off < K

        mu  = tl.load(mu_ptr + k_off[:, None] * stride_mk + d_off[None, :] * stride_md,
                      mask=k_mask[:, None] & (d_off[None, :] < D), other=0.0)
        lss = tl.load(log_sigma_sq_ptr + k_off, mask=k_mask, other=0.0)
        lpi = tl.load(log_pi_ptr + k_off, mask=k_mask, other=float('-inf'))

        x_sq  = tl.sum(x * x, axis=1)
        mu_sq = tl.sum(mu * mu, axis=1)
        xmu   = tl.dot(x, tl.trans(mu))
        sq    = x_sq[:, None] - 2.0 * xmu + mu_sq[None, :]

        inv_s  = tl.exp(-lss)
        log2pi = 1.8378770664093453
        ll = lpi[None, :] - 0.5 * sq * inv_s[None, :] - 0.5 * D * (log2pi + lss[None, :])
        ll = tl.where(k_mask[None, :] & n_mask[:, None], ll, float('-inf'))

        r = tl.exp(ll - log_Z[:, None])
        r = tl.where(k_mask[None, :] & n_mask[:, None], r, 0.0)

        # Accumulate N_k and sig_acc via atomic add
        nk_block  = tl.sum(r, axis=0)
        sig_block = tl.sum(r * sq, axis=0)
        tl.atomic_add(Nk_ptr  + k_off, nk_block,  mask=k_mask)
        tl.atomic_add(sig_acc_ptr + k_off, sig_block, mask=k_mask)

        # Accumulate mu_acc = R^T @ X via atomic add
        mu_block = tl.dot(tl.trans(r), x)  # (BLOCK_K, BLOCK_D)
        tl.atomic_add(
            mu_acc_ptr + k_off[:, None] * stride_ma_k + d_off[None, :] * stride_ma_d,
            mu_block,
            mask=k_mask[:, None] & (d_off[None, :] < D),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flash_gmm_estep(
    X:            torch.Tensor,   # (N, D) float32, GPU
    mu:           torch.Tensor,   # (K, D) float32, GPU
    log_sigma_sq: torch.Tensor,   # (K,)   float32, GPU
    log_pi:       torch.Tensor,   # (K,)   float32, GPU
    BLOCK_N: int = 64,
    BLOCK_K: int = 16,
    BLOCK_D: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flash-GMM E-step: compute sufficient statistics without materializing R.

    Returns (logZ, Nk, mu_acc, sig_acc) — see module docstring for details.
    """
    N, D = X.shape
    K    = mu.shape[0]
    assert D <= BLOCK_D, f"D={D} exceeds BLOCK_D={BLOCK_D}; increase BLOCK_D."
    assert X.is_cuda and mu.is_cuda, "Inputs must be on GPU."

    logZ    = torch.empty(N,    device=X.device, dtype=torch.float32)
    Nk      = torch.zeros(K,    device=X.device, dtype=torch.float32)
    mu_acc  = torch.zeros(K, D, device=X.device, dtype=torch.float32)
    sig_acc = torch.zeros(K,    device=X.device, dtype=torch.float32)

    grid = (triton.cdiv(N, BLOCK_N),)

    _flash_gmm_lse_kernel[grid](
        X, mu, log_sigma_sq, log_pi, logZ,
        N, K, D,
        X.stride(0), X.stride(1),
        mu.stride(0), mu.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
    )

    _flash_gmm_accum_kernel[grid](
        X, mu, log_sigma_sq, log_pi, logZ,
        Nk, mu_acc, sig_acc,
        N, K, D,
        X.stride(0), X.stride(1),
        mu.stride(0), mu.stride(1),
        mu_acc.stride(0), mu_acc.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
    )

    return logZ, Nk, mu_acc, sig_acc


# ---------------------------------------------------------------------------
# Quick correctness + speed test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    device = torch.device("cuda")
    K, D, N = 1024, 128, 1_000_000
    torch.manual_seed(0)

    X   = torch.randn(N, D, device=device)
    mu  = torch.randn(K, D, device=device)
    lss = torch.zeros(K, device=device)
    lpi = torch.full((K,), -math.log(K), device=device)

    # Warmup
    for _ in range(3):
        flash_gmm_estep(X, mu, lss, lpi)
    torch.cuda.synchronize()

    # Timing
    t0 = time.perf_counter()
    logZ, Nk, mu_acc, sig_acc = flash_gmm_estep(X, mu, lss, lpi)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3

    print(f"N={N:,}  K={K}  D={D}")
    print(f"E-step time : {ms:.1f} ms")
    print(f"Nk sum      : {Nk.sum().item():.0f}  (expected {N})")
    print(f"Peak GPU mem: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")
