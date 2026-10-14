# Basic Dense Matrix Multiplication in Python, Java and C: How Matrix Size Affects Runtime and Memory

**Author:** Joel López Guillén
**Assignment:** 1 — Basic Matrix Multiplication in Different Languages
**Course:** BIG DATA, Grado en Ciencia e Ingeniería de Datos, ULPGC, 2026-2027
**Repository:** https://github.com/JoelLopezGuillen/big-data-matrix-multiplication (tag `assignment-1`)

> Skeleton. Lines starting with `TODO` describe what to write; replace them with your own text.
> Bullets marked `FACT` are values already measured or recorded in the repository.

---

## Abstract

TODO (4-6 sentences, write it LAST): the question, the method (naive i-j-k triple loop, same inputs,
5 repetitions, median), the main result, and the scope of the conclusion (this machine, these settings).

## 1. Introduction and objective

TODO: why matrix multiplication matters as a benchmark (cubic work, memory-bound behaviour).

- **Research question:** How does matrix size *n* affect the runtime and peak memory of the same
  naive triple-loop dense matrix multiplication implemented in Python, Java and C, on one fixed machine?
- **Hypotheses (written BEFORE seeing results):**
  - TODO H1: runtime grows approximately as n³ (×8 per doubling of n).
  - TODO H2: C and Java (compiled to native code) are much faster than CPython (interpreted).
  - TODO H3: in Python, interpreter overhead dominates, so memory-hierarchy effects should be smaller
    (this is the prediction you made; the results partially refute it — say so in the Discussion).

## 2. Background

TODO: short explanations, each with a citation (IEEE style):

- Naive matrix multiplication and its O(n³) cost.
- Row-major storage and why `B[k][j]` is accessed with stride n.
- Memory hierarchy (L1/L2/L3/RAM) and cache lines.
- Execution models: ahead-of-time compilation (C/gcc), bytecode + JIT (Java HotSpot), interpretation (CPython).
- Benchmarking pitfalls: warm-up, dead-code elimination, monotonic timers.

## 3. Methodology

### 3.1 Implementations

TODO: describe the shared i-j-k algorithm and the per-language differences.

- FACT: C and Java use a flat row-major `double` array of length n²; Python uses a list of lists of `float`.
- FACT: the kernel accumulates the dot product in a local `sum` and writes `C[i][j] = sum`.
- FACT: Python hoists `a_row = a[i]` out of the inner loops (gcc -O2 performs equivalent hoisting automatically).
- FACT: no library matmul is used in any implementation; NumPy is used only as a correctness reference.

### 3.2 Inputs

- FACT: A and B generated once per n with NumPy `default_rng` (PCG64), seeds 42 (A) and 43 (B),
  uniform values in [-1, 1), float64, stored as little-endian row-major binary files read by all three languages.
- TODO: explain why a shared file is better than three random generators (cite the assignment protocol).

### 3.3 Environment

| Item | Value |
|---|---|
| CPU | Intel Core i7-12650H — 10 physical cores (6 P-cores with Hyper-Threading + 4 E-cores), 16 logical processors |
| Cache sizes | TODO: take L1/L2/L3 from Intel ARK and cite it |
| RAM | 15.7 GiB (16,868,974,592 bytes) |
| OS | Windows 11 (10.0.26200) |
| C | gcc 16.1.0 (MSYS2 UCRT64), flags `-O2 -std=c11 -Wall -Wextra` |
| Java | Oracle JDK 21.0.10 (HotSpot), `javac --release 21`, run with `-Xms4g -Xmx4g -XX:+UseSerialGC` |
| Python | CPython 3.13.7 (base interpreter, not the venv launcher) |
| Affinity | Each benchmark process pinned to P-core logical processors 0–11, HIGH priority class |

TODO: explain why pinning to P-cores matters on a hybrid CPU.

### 3.4 Measurement protocol

- FACT: timed region = kernel only (allocation and zero-initialisation of C included; file loading excluded).
- FACT: monotonic timers — `QueryPerformanceCounter` (C, 100 ns resolution), `System.nanoTime` (Java), `time.perf_counter_ns` (Python).
- FACT: one process per (language, n): warm-ups (C 1, Java 5, Python 1, excluded from samples), then 5 measured repetitions.
- FACT: result consumed via a checksum (sum of all elements of C) to prevent dead-code elimination.
- FACT: memory metric = peak working set (`psutil` `peak_wset`) of the whole process, read by the harness
  while the child is blocked on stdin after reporting its results.
- TODO: state the limitation — peak working set includes runtime overhead (JVM, interpreter) and the loaded inputs.
- FACT: summary statistic = median; variability = interquartile range (IQR).

### 3.5 Correctness

- FACT: unit tests in each language — identity, zero matrix (both sides), known 2×2, known 3×3 with non-integer values, checksum.
- FACT: cross-check against NumPy `A @ B` with `rtol=1e-9, atol=1e-12`; max abs error 5.3e-15 and max rel error 1.9e-12 (n = 64, 128).
- TODO: justify the tolerance (different summation orders give small floating-point differences).

### 3.6 Sizes and budget

- FACT: declared sequence n = 64, 128, 256, 512, 1024, 2048, 4096.
- FACT: budget = 900 s per process (warm-ups + 5 repetitions); on timeout, larger sizes for that language are skipped.

## 4. Results

TODO: insert the summary table and plots produced by `results/analyze.py` (median, IQR, n reps).

- FACT (main run, medians): C 0.27 s / 2.2 s / 114 s at n = 512 / 1024 / 2048; Java 0.28 s / 3.8 s at n = 512 / 1024;
  Python 9.6 s / 114 s at n = 512 / 1024. Use the exact values from `summary.md`.
- FACT: largest feasible tested size — C 2048, Java 1024, Python 1024.
- FACT: timeouts at 900 s — C n=4096, Java n=2048, Python n=2048; skipped — Java n=4096, Python n=4096.
- TODO: table of growth ratios T(2n)/T(n) per language vs the theoretical ×8.
- TODO: peak working set per language and n (note the fixed JVM baseline of ~47 MiB).

### 4.1 Supplementary experiment: Java at n = 2048 with one warm-up

- TODO: explain the bias found — the per-process budget included warm-ups, so Java (5 warm-ups + 5 reps = 10 multiplications)
  had a stricter effective limit than C (6 multiplications).
- TODO: method — same inputs and flags, 1 warm-up (matching C), 2700 s budget; results in `results/supplementary/`.
- TODO: results and comparison with C at n = 2048. Keep this table SEPARATE from the main results.
- FACT: Java n=2048, 1 warm-up: median 129.2 s, IQR 9.8 s, peak working set 303 MiB (C n=2048: median 114.5 s, IQR 5.5 s, 99.8 MiB).
- FACT: the whole process took ~6 × 128 s ≈ 13 min, so with 1 warm-up it would have fit the original 900 s budget;
  with 5 warm-ups (10 multiplications) it would not — this confirms the budget-design bias.
- FACT: Java and C checksums at n=2048 are bit-identical (-23415.104678236432) — strong correctness evidence (same IEEE-754 operations in the same order).
- FACT: Java/C runtime ratio drops from 1.73 (n=1024) to 1.13 (n=2048); hypothesis: both become RAM-bound.
- FACT: Java times rise across repetitions (120 → 139 s); a warm-up effect would make them fall. Possible cause: thermal throttling (not measured — state as limitation).
- FACT: run at a different time (2026-10-14 21:45) than the main run, so thermal/background conditions may differ.

## 5. Discussion

TODO: answer the research question with evidence. Suggested points:

- Growth vs n³: ratios close to ×8 within one cache level (C 512→1024 ×8) and much larger when the working set
  crosses a cache level (C 256→512 ×32, 1024→2048 ×52). Relate to the size of B (8n² bytes) vs L2/L3.
  Present this as an explanation consistent with the data, not as proof.
- Alternative/complementary explanation: power-of-two strides causing cache-set conflicts (could be tested with n = 1000 vs 1024).
- Python: ×7.9 and ×8.8 for small n, but ×18 at 256→512 — hypothesis H3 only partially holds; discuss scattered
  float objects (~32 bytes per element vs 8 bytes).
- Java vs C: similar up to n = 512, Java slower at 1024 — hypotheses (bounds checks, JIT code generation); do not overclaim.
- Memory: fixed runtime baselines vs growth with n².
- Limitations: single machine, Windows scheduling noise, peak working set includes runtime overhead, timer resolution,
  budget design issue (Section 4.1), only powers of two tested.

## 6. Conclusions

TODO: what the study establishes within its tested scope (this machine, these settings, these sizes), and what remains
unresolved. Avoid "language X is faster" in general.

## 7. References

TODO: IEEE style. Verify every source exists and supports the claim. Candidates to look up:
the assignment statement; Intel ARK page for i7-12650H; a computer architecture textbook (memory hierarchy);
Oracle HotSpot/JIT documentation; Python `time.perf_counter_ns` docs; psutil docs (`peak_wset`).

## 8. Reproducibility materials

- Code, raw CSV, environment metadata and analysis scripts: repository above, tag `assignment-1`.
- Exact commands: see `assignment-1/README.md`.

## Generative AI disclosure

TODO (required by the course policy): tool used, purpose, and which parts of the work it affected.
State that you validated the measurements and that you can explain the code.
