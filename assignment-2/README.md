# Assignment 2 — Optimised Dense and Sparse Matrix Multiplication

Sequential (single-thread) C study of two questions:

1. **Dense**: how much do loop reordering (`ikj`) and cache blocking
   (`blocked`) improve on the naive `ijk` baseline of Assignment 1, and how does
   the tile size relate to the L1d / L2 caches?
2. **Sparse**: when does an own CSR implementation (dense->CSR conversion plus
   Gustavson SpGEMM) beat the dense blocked kernel on the same matrices, as a
   function of size and density, and how much fill-in does the product have?

## Structure

| Path | Content |
|---|---|
| `c/matmul.c`, `c/matmul.h` | dense kernels `matmul_ijk`, `matmul_ikj`, `matmul_blocked(a, b, n, tile)`, `checksum` |
| `c/csr.c`, `c/csr.h` | own CSR: `csr_from_dense`, `csr_spgemm` (Gustavson), `csr_to_dense`, `csr_checksum`, `csr_read`/`csr_write` |
| `c/runner.c` | bench/verify runner for dense, sparse and conversion modes |
| `c/test_matmul.c` | C unit tests (dense and CSR) |
| `data/generate.py` | dense inputs (copied from Assignment 1) |
| `data/generate_sparse.py` | sparse inputs (CSR + matching dense twin + `manifest.json`) |
| `data/README.md` | seeds, formats, definitions |
| `bench_config_dense.json`, `bench_config_tiles.json`, `bench_config_sparse.json` | experiment configurations (each with a `smoke` override block) |
| `benchmark.py`, `bench_common.py` | harness (build, generate, run, budgets, CSV) |
| `build.py`, `run_tests.py`, `cross_check.py` | build, all correctness checks, comparison with NumPy/SciPy |
| `tests/` | pytest tests of the generators, harness helpers and analysis |
| `results/analyze.py` | summaries (median, IQR) and plots; raw CSVs go to `results/{dense,tiles,sparse}/` |
| `tools/` | `cpu_topology.c`, `cache_info.c` (Windows topology / caches), recorded JSON, `capture_environment.py` |

## Requirements

- Windows 11, Python 3.13 (`py` launcher), GCC from MSYS2 UCRT64
  (`C:\msys64\ucrt64\bin` on `PATH`).
- Run every command from **PowerShell or cmd** inside `assignment-2/` (in the
  materials zip this is the folder `Assignment02_LopezGuillen_Materials/`). As in
  Assignment 1, the MSYS2 `gcc` exits with status 1 and no diagnostics when it
  is launched from Git Bash.

## Setup

```powershell
cd assignment-2
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Build and test

```powershell
.venv\Scripts\python build.py           # c\build\runner.exe, c\build\test_matmul.exe, tools\build\*.exe
.venv\Scripts\python run_tests.py       # C unit tests + pytest + cross_check.py
```

Individual suites:

```powershell
c\build\test_matmul.exe c\build\test_tmp.csr
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python cross_check.py --sizes 64 100 130 --sparse-sizes 50 200
```

Manual build commands (what `build.py` runs):

```powershell
gcc -O2 -std=c11 -Wall -Wextra -o c\build\runner.exe c\runner.c c\matmul.c c\csr.c
gcc -O2 -std=c11 -Wall -Wextra -o c\build\test_matmul.exe c\test_matmul.c c\matmul.c c\csr.c
gcc -O2 -std=c11 -Wall -Wextra -o tools\build\cpu_topology.exe tools\cpu_topology.c
gcc -O2 -std=c11 -Wall -Wextra -o tools\build\cache_info.exe tools\cache_info.c
```

## Generate data (optional; the harness generates missing inputs itself)

```powershell
.venv\Scripts\python data\generate.py 64 128 256 512 768 1024 1536 2048 3072 4096
.venv\Scripts\python data\generate_sparse.py --sizes 1000 2000 4000 --densities 0.1 1 5 10 25 50 --with-dense
.venv\Scripts\python data\generate_sparse.py --sizes 10000 --densities 0.1 1 5 10 25 50
```

## Run the benchmarks

Rows are **appended** to `results/<experiment>/raw_measurements.csv`; delete
the file before a fresh run. Recommended order:

```powershell
# 1. Tile sweep, then set "blocked_tiles" in bench_config_dense.json and
#    "dense_tile" in bench_config_sparse.json to the chosen tile (both are set
#    to T = 128, chosen from the recorded sweep).
.venv\Scripts\python benchmark.py --config bench_config_tiles.json
.venv\Scripts\python results\analyze.py --input results\tiles\raw_measurements.csv

# 2. Dense methods
.venv\Scripts\python benchmark.py --config bench_config_dense.json

# 3. Sparse vs dense
.venv\Scripts\python benchmark.py --config bench_config_sparse.json

# 4. Summaries and plots for every experiment found under results\
.venv\Scripts\python results\analyze.py
```

Smoke test of the whole pipeline (tiny sizes, 1 repetition, output in the
gitignored `results\smoke\`):

```powershell
.venv\Scripts\python benchmark.py --config bench_config_tiles.json --smoke
.venv\Scripts\python benchmark.py --config bench_config_dense.json --smoke
.venv\Scripts\python benchmark.py --config bench_config_sparse.json --smoke
.venv\Scripts\python results\analyze.py --input results\smoke\tiles\raw_measurements.csv results\smoke\dense\raw_measurements.csv results\smoke\sparse\raw_measurements.csv
```

Useful options: `--sizes`, `--methods ijk ikj`, `--tiles 32 64`,
`--densities 1 5`, `--repetitions`, `--timeout`, `--memory-budget-gib`,
`--output-dir`, `--skip-build`. `tools\capture_environment.py` can also be
run on its own; the harness writes `environment.json` next to each CSV.

## Experiment design

### Common settings

| Setting | Value |
|---|---|
| Language / type | C (`gcc -O2 -std=c11 -Wall -Wextra`, no `-march=native`), float64 |
| Threads | 1 (sequential code); process pinned to P-core logical processors 0-11, `HIGH_PRIORITY_CLASS` (detected with `tools/cpu_topology.c`) |
| Repetitions | 1 warm-up (excluded) + 5 measured repetitions, one child process per configuration |
| Timer | `QueryPerformanceCounter` (100 ns resolution on this machine) |
| Result consumption | checksum (sum of all stored values of C) printed for every repetition |
| Memory metric | peak working set (`psutil` `peak_wset`) of the child process, sampled while it waits on stdin after printing its results; includes the inputs loaded in the process |
| Time budget | **60 s per kernel execution**: the runner reports an event after every warm-up/repetition and the harness restarts a 60 s deadline at each one; if any single execution exceeds 60 s the process is killed and the configuration is recorded as `status=timeout` with no time. Loading the inputs has its own 300 s limit. |
| Memory budget | **4 GiB peak working set**: before launching, the estimate (below) is compared with the budget (`status=over_memory_budget`, not run); while running, the working set is polled every 0.25 s and the process is killed above the budget (`status=over_memory_budget`, peak recorded). |
| Failure propagation | after a non-ok status at size n, every larger n of the same series (method + tile, or method + density) is recorded as `skipped_after_<status>`; allocation failures in C are `status=oom` (exit code 3). No time is ever invented. |

### Part A: dense methods

| Method | Description |
|---|---|
| `ijk` | the validated Assignment 1 kernel (tag `assignment-1`), copied verbatim and renamed |
| `ikj` | loop reordering: `a = A[i][k]` hoisted, inner loop over j walks rows of B and C with unit stride |
| `blocked` | cache blocking with square tiles T: ii-kk-jj tile loops, i-k-j micro-kernel inside each tile, shorter edge tiles when T does not divide n |

- Sizes: 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096 (shared
  `A_n{n}.bin`/`B_n{n}.bin`, seeds 42/43, as in Assignment 1).
- Timed region: one kernel call **including `calloc` of C** (allocation and
  zero-initialisation), identical for the three methods and for Assignment 1.
- Memory estimate: `3 * 8 * n^2` bytes (A, B, C).
- Tile sweep: T = 16, 32, 64, 128, 256 at n = 1024 and 2048. The three tiles
  touched by the micro-kernel occupy `3 * 8 * T^2` bytes, which equals the
  48 KiB L1d at T ~ 45 and the 1.25 MiB L2 at T ~ 234 (P-core caches from
  `tools/cache_info.json`); `tile_sweep.png` marks both. T = 128 was the
  fastest at both sizes, so the dense and sparse configs use T = 128.

### Part B: sparse matrices (own CSR code)

- Definitions: **density** = % of non-zero entries (100 * nnz / n^2);
  **sparsity** = % of zero entries (100 - density).
- Densities: 0.1, 1, 5, 10, 25, 50 %. Sizes: n = 1000, 2000, 4000 (CSR and
  dense blocked on the same matrices) and n = 10000 (CSR only; the dense
  comparison and the conversion are recorded as `status=not_run` by design,
  because no dense input is generated at that size).
- Inputs (`data/generate_sparse.py`): exact nnz = round(density * n^2),
  positions uniform without replacement, values uniform in [-1, 1) without
  exact zeros, seeds 42 (A) / 43 (B); written as CSR and, for the dense
  comparison sizes, as the matching dense row-major file. Format and parameters
  in `data/README.md` and `data/sparse/manifest.json`.
- CSR layout: `int64 indptr[n+1]`, `int32 indices[nnz]`, `double data[nnz]`.
- Measurements, each its own series:
  - `dense_to_csr` (phase `convert`): `csr_from_dense(A)` + `csr_from_dense(B)`
    from the dense twins (two passes: count, exact allocation, fill).
  - `csr_gustavson` (phase `multiply`): `csr_spgemm(A, B)` = symbolic pass
    (row nnz of C with a marker array), exact allocation of C, numeric pass with
    a dense accumulator. Output column indices are sorted per row (insertion
    sort / `qsort`, or a linear scan of the marker array when a row fills at
    least 1/64 of the columns). Allocation of C is inside the timed region, the
    same boundary as the dense kernels.
  - `blocked` (dense, configured tile) on the same matrices.
  - End-to-end = median conversion + median SpGEMM, **derived** in the analysis
    (not a separate measured run).
- Reported: nnz of A and B, nnz of C (structural), output density, times,
  peak working set.
- Structural non-zeros: C keeps every entry that receives at least one product,
  even if it cancels to exactly 0.0. SciPy drops such entries; the cross-check
  accounts for that (see below).
- Memory estimates: SpGEMM = CSR(A) + CSR(B) + CSR(C with the expected nnz
  `n^2 (1 - (1 - dA dB)^n)`) + `12 n` bytes of workspace; conversion =
  `2 * 8 n^2 + CSR(A) + CSR(B)`. The 10000 x 25-50 % cases are expected to
  exceed the 60 s budget and are recorded as such.

### Own code vs libraries

All multiplication, conversion and CSR code that is timed is our own C code.
**NumPy and SciPy are used only as correctness references** in
`cross_check.py` and in the Python tests, and NumPy for input generation and
analysis. No library kernel is benchmarked.

## Correctness

- `c/test_matmul.c`: for every dense method, identity, zero, known 1x1/2x2/3x3
  products; all methods agree with `ijk` for n = 1, 7, 63, 64, 65, 130 and
  tiles 16/64/128 (edge tiles); `tile = 0` is rejected. CSR: known conversion
  with an empty row, dense->CSR->dense round trip, all-zero matrix, identity,
  known product, structural zero from cancellation, shape mismatch, SpGEMM vs
  dense `ijk` on random matrices (density 1-100 %), checksum, file round trip.
- `cross_check.py`: every dense method (tiles 16/64/128) vs NumPy `A @ B` for
  n = 64, 100, 130; for n = 50 and 200 and all six densities, own conversion vs
  the generated CSR (exact equality), own SpGEMM vs `scipy.sparse` `A @ B`
  (values, sorted CSR structure, structural nnz equal to the nnz of the boolean
  product pattern, and numerically non-zero entries equal to SciPy's except for
  entries whose magnitude is <= atol).
- Tolerances: `np.allclose(rtol=1e-9, atol=1e-12)`, as in Assignment 1. The
  C kernels and the references sum the same products in possibly different
  orders; for inputs in [-1, 1) and n <= a few hundred the float64 rounding
  error is around n * 2^-53 * max|partial sum| ~ 1e-13 relative, far below
  rtol = 1e-9, while a wrong index produces O(1) errors. atol = 1e-12 only
  covers entries that are close to zero by cancellation.

## CSV columns

`experiment, method, representation, phase, n, numeric_type, threads, warmups,
repetition, tile, density_percent, sparsity_percent, nnz_a, nnz_b, nnz_c,
output_density_percent, status, time_ns, checksum, peak_working_set_bytes,
memory_estimate_bytes, timestamp, note` (one row per measured repetition;
failed, skipped and not-run configurations have one row with no time).

## Report

The research report is `report/report.md`, rendered to
`report/Assignment02_LopezGuillen_Report.pdf` (methodology, results,
discussion, references and the generative AI disclosure). To rebuild the PDF
(needs Microsoft Edge or Google Chrome for headless printing):

```powershell
.venv\Scripts\python report\build_pdf.py
```
