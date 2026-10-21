# Input data

All inputs are generated deterministically and are not versioned (`*.bin` and
`*.csr` are gitignored). `sparse/manifest.json` records the generation
parameters of every sparse input that was produced.

## Dense inputs (`generate.py`, copied from Assignment 1)

| Property | Value |
|---|---|
| Files | `A_n{n}.bin`, `B_n{n}.bin` |
| Format | raw little-endian IEEE-754 float64, row-major, n*n values, no header |
| Generator | `numpy.random.default_rng(seed)` (PCG64), `uniform(-1.0, 1.0, size=(n, n))` |
| Seeds | A: 42, B: 43 (a fresh generator per file and per n) |

```powershell
.venv\Scripts\python data\generate.py 64 128 256   # add --force to regenerate
```

## Sparse inputs (`generate_sparse.py`)

Definitions: **density** = percentage of non-zero entries (100 * nnz / n^2);
**sparsity** = percentage of zero entries (100 - density).

| Property | Value |
|---|---|
| Files | `sparse/{A,B}_n{n}_d{density}pct.csr` and, for the sizes compared with dense, `sparse/{A,B}_n{n}_d{density}pct.bin` |
| nnz | exactly `round(density/100 * n^2)` per matrix |
| Positions | uniform without replacement over the n^2 cells (`Generator.choice(n*n, nnz, replace=False)`), sorted |
| Values | uniform in [-1, 1); values that are exactly 0.0 are redrawn, so every stored entry is non-zero |
| Seeds | A: 42, B: 43 (fresh `default_rng` per matrix; positions are drawn first, then values) |
| CSR format | little-endian `int64 n_rows, int64 n_cols, int64 nnz`, then `int64 indptr[n_rows+1]`, `int32 indices[nnz]`, `float64 data[nnz]`; column indices strictly increasing within each row |
| Dense twin | the same logical matrix in the dense format above, so the dense and CSR methods multiply identical inputs |

```powershell
.venv\Scripts\python data\generate_sparse.py --sizes 1000 2000 4000 --densities 0.1 1 5 10 25 50 --with-dense
.venv\Scripts\python data\generate_sparse.py --sizes 10000 --densities 0.1 1 5 10 25 50
```

The benchmark harness generates missing files automatically. The full sparse
set uses about 4.5 GB of disk (the 10000 x 50 % pair alone is 1.2 GB).
