# Flash-GMM

A memory-efficient, IO-aware Triton GPU kernel for the Gaussian Mixture Model (GMM) E-step at arbitrary scale.

## Overview

Computing GMM responsibilities for large datasets is memory-bound: a naive implementation materializes an N×K responsibility matrix that exhausts GPU memory well before the scales where modern applications operate. Flash-GMM eliminates this matrix entirely.

Inspired by the IO-aware tiling strategy of [FlashAttention](https://arxiv.org/abs/2205.14135), Flash-GMM computes the full GMM E-step in a **single pass** over the data, accumulating only O(KD) sufficient statistics. The N×K responsibility matrix is never written to memory.

## Key Properties

| Property | Value |
|---|---|
| Kernel memory (N=1M, K=1024, D=128) | **4.5 MB** |
| TorchGMM memory (same config) | 21,006 MB |
| Memory reduction | **4,668×** |
| Speedup vs SciPy (CPU) | **766–1,740×** |
| Speedup vs TorchGMM (GPU) | **19–32×** |
| Max N on A100 80GB (streaming) | **1B+** |
| Validated on | A100 80GB, H100, RTX 5080 |

## How It Works

The kernel processes data in tiles of `BLOCK_N` rows. For each tile:

1. **Pass 1 (log-sum-exp)**: Loads the tile into registers and computes the per-sample log-normaliser `log Z_i` via a numerically stable online log-sum-exp, iterating over all K components in tiles of `BLOCK_K`.

2. **Pass 2 (accumulation)**: Reuses the tile from registers (no second HBM read), computes responsibilities `r_ik = exp(log z_ik - log Z_i)`, and atomically accumulates the sufficient statistics `N_k`, `mu_acc`, `sig_acc` into O(KD) global buffers.

The tile never leaves on-chip memory between the two passes, giving a **single HBM read of X** per iteration. The N×K responsibility matrix is discarded immediately after each tile.

## Installation

```bash
pip install torch triton
```

No other dependencies required for the kernel itself.

## Usage

```python
import math
import torch
from flash_gmm import flash_gmm_estep

# Inputs (all GPU tensors, float32)
N, K, D = 1_000_000, 1024, 128
X            = torch.randn(N, D, device='cuda')        # data
mu           = torch.randn(K, D, device='cuda')        # component means
log_sigma_sq = torch.zeros(K, device='cuda')           # log variances
log_pi       = torch.full((K,), -math.log(K), device='cuda')  # log weights

# E-step
logZ, Nk, mu_acc, sig_acc = flash_gmm_estep(X, mu, log_sigma_sq, log_pi)

# M-step
pi_new    = Nk / N
mu_new    = mu_acc / Nk[:, None]
sigma_sq  = sig_acc / (D * Nk)
```

### Outputs

| Tensor | Shape | Description |
|---|---|---|
| `logZ` | (N,) | Per-sample log-normaliser log Z_i |
| `Nk` | (K,) | Effective cluster counts Σ_i r_ik |
| `mu_acc` | (K, D) | Weighted sum Σ_i r_ik x_i |
| `sig_acc` | (K,) | Weighted sq. dist. Σ_i r_ik ‖x_i − μ_k‖² |

### Block sizes

The defaults `BLOCK_N=64`, `BLOCK_K=16`, `BLOCK_D=128` work for D≤128. For larger D, increase `BLOCK_D` to the next power of two ≥ D:

```python
logZ, Nk, mu_acc, sig_acc = flash_gmm_estep(X, mu, lss, lpi, BLOCK_D=256)
```

### Streaming for N > GPU memory

For datasets larger than GPU memory, feed data in chunks — the O(KD) accumulators are simply summed across chunks:

```python
Nk_total = torch.zeros(K, device='cuda')
mu_total = torch.zeros(K, D, device='cuda')
sig_total = torch.zeros(K, device='cuda')

for chunk in dataloader:  # chunks loaded from CPU/SSD
    X_chunk = chunk.cuda()
    _, Nk_c, mu_c, sig_c = flash_gmm_estep(X_chunk, mu, lss, lpi)
    Nk_total  += Nk_c
    mu_total  += mu_c
    sig_total += sig_c
    del X_chunk

# M-step on aggregated statistics
mu_new   = mu_total / Nk_total[:, None]
sigma_sq = sig_total / (D * Nk_total)
```

This was validated at **N=1B** vectors (512 GB of data) on a single A100 80GB, completing in ~28 minutes with **1,548 MB peak GPU memory**.

## Performance

Runtime of a single E-step (K=1024, D=128, A100 80GB):

| N | Flash-GMM | vs SciPy (CPU) | vs TorchGMM (GPU) |
|---|---|---|---|
| 10K | 3 ms | 766× | 32× |
| 50K | 9 ms | 1,260× | 20× |
| 100K | 18 ms | 1,458× | 23× |
| 250K | 46 ms | 1,597× | 19× |
| 500K | 84 ms | 1,571× | 20× |
| 1M | 152 ms | 1,738× | 22× |
| 10M | 1,519 ms | 1,740× | OOM |
| 50M | 35,510 ms | 1,752× | OOM |

TorchGMM runs out of memory beyond N≈1M. Flash-GMM scales to N=10⁸ on the same device.

## H100-Tuned Variants

Three additional kernels target the paper benchmark workloads (K=1024, D=96–128) on H100 with BF16 / WGMMA tensor cores. They trade some of the original kernel's flexibility for substantial speedups on those specific shapes; outside that envelope they may regress or fail to compile.

| File | Covariance | Geomean speedup over the same-form atomic baseline | Implied speedup over TorchGMM (GPU) |
|---|---|---|---|
| [`flash_gmm_h100.py`](flash_gmm_h100.py) | Isotropic — `σ_k²` scalar per cluster | **5.14×** | **~100×** |
| [`flash_gmm_diag_h100.py`](flash_gmm_diag_h100.py) | Diagonal — `σ_k²[d]` per dim per cluster | **5.61×** | **~110×** |
| [`flash_gmm_full_h100.py`](flash_gmm_full_h100.py) | Full — `Σ_k` Cholesky factor of precision | **2.41×** | n/a (TorchGMM full-cov OOMs on these shapes) |

Speedups are geomeans across SIFT1M (N=1M, D=128), GloVe (N=1.18M, D=100), and Deep10M (N=9.99M, D=96) at K=1024 on H100 80GB, measured against a same-form two-launch atomic baseline (the iso baseline is `flash_gmm.py`; the diag and full baselines are simple per-cluster Triton kernels with the same algorithmic structure as `flash_gmm.py` but extended to richer covariance forms — included as `_h100.py` siblings' code paths' starting point).

Implied vs.-TorchGMM speedups multiply each kernel's measured speedup by `flash_gmm.py`'s original ~20× advantage over TorchGMM (isotropic case, paper Table 1). They apply only when an apples-to-apples TorchGMM comparison exists — i.e. for iso and diag where TorchGMM has matching covariance support and stays within memory; full covariance in TorchGMM OOMs at these shapes, so the comparison there is not meaningful.

### Per-workload results (Full-cov on H100)

| Workload | N | D | Atomic baseline | H100 candidate | Speedup |
|---|---|---|---|---|---|
| SIFT1M  | 1,000,000 | 128 | 1,498 ms | 622 ms | 2.41× |
| GloVe   | 1,180,000 | 100 | 1,767 ms | 732 ms | 2.41× |
| Deep10M | 9,990,000 |  96 | 14,937 ms | 6,178 ms | 2.42× |

`max_rel_err` ≤ 5×10⁻³ vs. CPU float64 reference on three seeds.

### When to use which kernel

- **`flash_gmm.py`** — portable across CUDA GPUs (A100, H100, L4, RTX). Any K ≥ 1, D ≤ 128 (or override `BLOCK_D`). Fastest cold start (~1 sec compile). Use this unless you specifically have an H100.
- **`flash_gmm_h100.py`** — H100 only. Isotropic covariance. K ≥ 64, D ≤ 128 for best perf; K%256==0 unlocks a faster path. Cold autotune is 30–180 sec; subsequent calls reuse the Triton cache.
- **`flash_gmm_diag_h100.py`** — H100. Diagonal covariance (per-component per-dimension variance). Same K/D constraints.
- **`flash_gmm_full_h100.py`** — H100. Full covariance via Cholesky factor of precision. Slower per-call than diag/iso because of the K·D² state, but the same 2.4× geomean speedup over the equivalent original-style baseline.

The H100 variants use BF16 inputs to `tl.dot` with FP32 accumulators (WGMMA tensor cores), per-K-constant precompute on the host, `exp2`/`log2` for the H100 SFU, and autotuned tile sizes. The diagonal and full variants additionally use a split-K persistent pass-2 layout that reduces atomic-add contention by ~3 orders of magnitude vs. the dense-grid baseline.

## Authors

- Gal Bloch (gal.bloch@ibm.com)
- Assaf Toledo (assaf.toledo@ibm.com)
- Ohad Eytan (ohad.eytan1@ibm.com)
- Ariel Gera (ariel.gera1@ibm.com)
- Matan Orbach (matano@il.ibm.com)

IBM Research

## Citation

If you use Flash-GMM in your research, please cite:

```bibtex
@article{bloch2026flashgmm,
  title     = {Flash-GMM: Breaking the Memory Barrier for Large-Scale
               Gaussian Mixture Models, with Applications to IVF Indexing},
  author    = {Bloch, Gal and Toledo, Assaf and Eytan, Ohad and Gera, Ariel and Orbach, Matan},
  journal   = {arXiv preprint},
  year      = {2026}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
