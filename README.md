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

## Citation

If you use Flash-GMM in your research, please cite:

```bibtex
@article{bloch2026flashgmm,
  title     = {Flash-GMM: Breaking the Memory Barrier for Large-Scale
               Gaussian Mixture Models, with Applications to IVF Indexing},
  author    = {Bloch, Gal},
  journal   = {arXiv preprint},
  year      = {2026}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
