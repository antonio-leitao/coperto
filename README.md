# Coperto

Fast and exact persistent homology via quotient Vietoris-Rips filtrations.

## Installation

```bash
pip install coperto
```

## Usage

```python
import numpy as np
import coperto
# Generate a simple point cloud (e.g. 100 points on a noisy circle)
theta = np.linspace(0, 2 * np.pi, 100)
data = np.column_stack([np.cos(theta), np.sin(theta)]) + 0.05 * np.random.randn(100, 2)
# Compute persistent homology up to dimension 1 (H0 = components, H1 = loops)
barcodes = coperto.persistent_homology(data, max_dim=1)
# Each entry is (dimension, birth, death)
for dim, birth, death in barcodes:
    print(f"H{dim}: [{birth:.3f}, {death:.3f})")
```

**Parameters for `persistent_homology`:**
| Parameter | Default | Description |
| ------------ | ------- | --------------------------------------------------------------- |
| `data` | — | NumPy array of shape `(n_points, n_dims)` |
| `max_dim` | `1` | Maximum homological dimension to compute |
| `greedy` | `True` | Use greedy linkage (faster); set `False` for exact tie-breaking |
| `use_128bit` | `False` | Use 128-bit arithmetic in the persistence computation |
**Returns:** list of `(dim, birth, death)` tuples.

> **Note:** `use_128bit=True` is only available on Linux and macOS. Windows does not support 128-bit integers (`__int128`). Passing `use_128bit=True` on Windows will raise a `RuntimeError`.

## Reproducing the paper benchmarks

Clone the repository and run the benchmark script from the examples directory:

```bash
git clone https://github.com/antonio-leitao/coperto.git
cd coperto
uv run python examples/main.py
```

Results are saved to `examples/results/main_benchmark.csv`.
The benchmark compares Coperto against [Ripser](https://github.com/Ripser/ripser) on ten datasets: C. elegans, Vicsek, Klein bottle (400 and 900 points), Dragon (1k and 2k), HIV1, O(3) (1024 and 2048), and PBMC3k.
