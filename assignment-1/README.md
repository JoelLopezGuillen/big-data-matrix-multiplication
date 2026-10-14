# Assignment 1 — Basic Matrix Multiplication in Different Languages

Comparison of the naive triple-loop dense matrix multiplication implemented in
Python, Java and C, studying how matrix size affects runtime and memory.

## Structure

| Path | Content |
|---|---|
| `python/` | `matmul.py` (pure-Python kernel), `runner.py` (bench/verify runner), `test_matmul.py` (pytest) |
| `java/` | `MatMul.java` (kernel), `MatMulRunner.java` (bench/verify runner), `MatMulTest.java` (tests) |
| `c/` | `matmul.c`/`matmul.h` (kernel), `runner.c` (bench/verify runner), `test_matmul.c` (tests) |
| `data/` | `generate.py` (input generator) and `README.md` (seeds, format) |
| `results/` | `raw_measurements.csv` (written by the harness), `environment.json`, `analyze.py` |
| `tools/` | `cpu_topology.c` (P/E-core detection), `cpu_topology.json` (recorded result), `capture_environment.py` |
| `bench_config.json` | sizes, repetitions, warm-ups, timeout, compiler and JVM flags |
| `benchmark.py` | top-level harness (build, generate, run all configurations, write CSV) |
| `build.py`, `run_tests.py`, `cross_check.py` | build, correctness tests, cross-language check against NumPy |
| `report/` | research report |

## Requirements (verified versions)

- Windows 11, Python 3.13 (`py` launcher), JDK 21 (`javac`/`java` on `PATH`),
  GCC 16.1 from MSYS2 UCRT64 (`C:\msys64\ucrt64\bin` on `PATH`).
- Run all commands from **PowerShell or cmd** in `assignment-1/`. Do not build
  from Git Bash: in this environment the MSYS2 `gcc` exits with status 1 and
  no diagnostics when launched from Git Bash.

## Setup

```powershell
cd assignment-1
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Build

```powershell
.venv\Scripts\python build.py
```

Equivalent manual commands:

```powershell
gcc -O2 -std=c11 -Wall -Wextra -o c\build\runner.exe c\runner.c c\matmul.c
gcc -O2 -std=c11 -Wall -Wextra -o c\build\test_matmul.exe c\test_matmul.c c\matmul.c
gcc -O2 -std=c11 -Wall -Wextra -o tools\build\cpu_topology.exe tools\cpu_topology.c
javac --release 21 -d java\out java\MatMul.java java\MatMulRunner.java java\MatMulTest.java
```

## Test

```powershell
.venv\Scripts\python run_tests.py                 # all three suites
.venv\Scripts\python -m pytest python -q          # Python only
java -cp java\out MatMulTest                      # Java only
c\build\test_matmul.exe                           # C only
.venv\Scripts\python cross_check.py --sizes 64 128
```

Each suite checks identity (A·I = A), the zero matrix, a known 2x2 product, a
known 3x3 product with non-integer values, and the checksum. `cross_check.py`
runs every implementation in `verify` mode and compares its product with NumPy
`A @ B` using `np.allclose(rtol=1e-9, atol=1e-12)`, reporting max abs/rel error.

## Run the benchmark

```powershell
.venv\Scripts\python benchmark.py                 # full run (several hours)
.venv\Scripts\python results\analyze.py           # summary.csv, summary.md, PNG plots
```

Useful options: `--sizes 64 128`, `--languages c java`, `--repetitions 5`,
`--timeout 900`, `--output-dir results/smoke` (gitignored), `--config other.json`.
Rows are appended to `results/raw_measurements.csv`, so delete it before a fresh
full run. `tools\capture_environment.py` can also be run on its own.

For a quick smoke test:

```powershell
.venv\Scripts\python benchmark.py --sizes 64 128 --output-dir results/smoke
.venv\Scripts\python results\analyze.py --input results/smoke/raw_measurements.csv --output-dir results/smoke
```

## Design decisions

- **Algorithm**: classic triple loop in i-j-k order, `C[i][j] = sum_k A[i][k]*B[k][j]`
  accumulated in a local `sum`, identical in all three languages. No library matmul.
- **Numeric type**: float64 (`double`, Python `float`).
- **Storage**: C and Java use a flat row-major 1D array of length n*n. Python uses
  an idiomatic list of lists of floats and a pure-Python kernel (no NumPy). This
  representation difference is part of what is being compared.
- **Inputs**: generated once per n by `data/generate.py` (PCG64 `default_rng`,
  seed 42 for A and 43 for B, uniform in [-1, 1)), stored as raw little-endian
  float64 row-major files. See `data/README.md`.
- **Timed region**: the kernel call only. File loading is outside. Allocation and
  zero-initialisation of the output matrix happen inside the kernel and **are
  included** (`calloc` in C, `new double[n*n]` in Java, `[[0.0]*n ...]` in Python).
- **Timers** (monotonic): C `QueryPerformanceCounter` (10 MHz on this machine,
  i.e. 100 ns resolution), Java `System.nanoTime`, Python `time.perf_counter_ns`.
- **Result consumption**: after each multiplication the checksum (sum of all
  elements of C) is computed and printed, so the work cannot be eliminated.
  Warm-up checksums are accumulated and printed as `warmup_sink`.
- **Repetitions**: warm-ups excluded from the samples (Java 5, C 1, Python 1),
  then 5 measured repetitions. Each (language, n) configuration is ONE process
  that prints a single JSON line with per-repetition times (ns) and checksums.
- **Memory metric**: peak working set (`psutil` `memory_info().peak_wset`) of the
  child process, the same tool for all languages. After printing its results the
  child blocks reading one line from stdin; the harness samples `peak_wset` while
  it is still alive and then releases it. The value includes runtime overhead
  (JVM, interpreter) and the input matrices loaded in the process; it is the
  same for every repetition row of a configuration.
- **Real executables**: the harness launches the base Python interpreter
  (`sys._base_executable`, the runner uses only the standard library) and
  `<java.home>\bin\java.exe`, because the venv `python.exe` and the Oracle
  `javapath\java.exe` are launchers that spawn a second process, which would
  hide the real working set and escape the CPU pinning.
- **CPU affinity and priority**: the i7-12650H is hybrid. `tools/cpu_topology.c`
  calls `GetSystemCpuSetInformation`; logical processors 0-11 report
  EfficiencyClass 1 (6 P-cores with Hyper-Threading) and 12-15 report
  EfficiencyClass 0 (4 E-cores). Result recorded in `tools/cpu_topology.json`.
  Every child is pinned with `psutil` `cpu_affinity` to logical processors 0-11
  and set to `HIGH_PRIORITY_CLASS`.
- **Flags**: C `gcc -O2 -std=c11 -Wall -Wextra` (no `-march=native`; the warning
  flags do not affect code generation). Java `javac --release 21`, run with
  `java -Xms4g -Xmx4g -XX:+UseSerialGC`.
- **Sizes and budget**: 64, 128, 256, 512, 1024, 2048, 4096 with a 15-minute
  timeout per configuration process (all configurable in `bench_config.json` or
  via CLI). A timeout or failure is recorded with `status=timeout|oom|error`, no
  time, and the limit in the `note` column; larger sizes of that language are
  then recorded as `status=skipped`.
- **CSV columns**: `language, n, numeric_type, repetition, time_ns, checksum,
  peak_wset_bytes, warmups, status, timestamp, note` (one row per measured repetition).
- **Environment**: `results/environment.json` records CPU model, cores, logical
  processors, P/E-core mapping, RAM, OS, Python/Java/GCC versions and flags,
  NumPy/psutil versions and a timestamp.

## Generative AI disclosure

_To be completed by the student: describe which generative AI tools were used,
for which parts of the work, and how their output was reviewed and validated._
