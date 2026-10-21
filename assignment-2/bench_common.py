"""Shared helpers for the build, test, cross-check and benchmark scripts.

All child processes run with the assignment directory as working directory and
receive relative paths, which keeps non-ASCII characters in the absolute path
away from the native tools.
"""

import json
import math
import os
import queue
import subprocess
import threading
import time

import psutil

ROOT = os.path.dirname(os.path.abspath(__file__))
DENSE_CONFIG = os.path.join(ROOT, "bench_config_dense.json")
TILES_CONFIG = os.path.join(ROOT, "bench_config_tiles.json")
SPARSE_CONFIG = os.path.join(ROOT, "bench_config_sparse.json")
DATA_DIR_REL = "data"
TOPOLOGY_JSON = os.path.join(ROOT, "tools", "cpu_topology.json")
CACHE_JSON = os.path.join(ROOT, "tools", "cache_info.json")

C_BUILD_DIR_REL = os.path.join("c", "build")
C_RUNNER_REL = os.path.join(C_BUILD_DIR_REL, "runner.exe")
C_TEST_REL = os.path.join(C_BUILD_DIR_REL, "test_matmul.exe")
TOPOLOGY_EXE_REL = os.path.join("tools", "build", "cpu_topology.exe")
CACHE_EXE_REL = os.path.join("tools", "build", "cache_info.exe")

C_SOURCES = ["c/matmul.c", "c/csr.c"]
DENSE_METHODS = ("ijk", "ikj", "blocked")
GCC_FLAGS = ["-O2", "-std=c11", "-Wall", "-Wextra"]


def load_config(path=DENSE_CONFIG):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run_checked(cmd):
    """Run a build/tool command in ROOT and raise with its output on failure."""
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}{proc.stderr}"
        )
    return proc.stdout


# --------------------------------------------------------------------------- build


def build_c(flags=GCC_FLAGS):
    os.makedirs(os.path.join(ROOT, C_BUILD_DIR_REL), exist_ok=True)
    run_checked(["gcc", *flags, "-o", C_RUNNER_REL, "c/runner.c", *C_SOURCES])
    run_checked(["gcc", *flags, "-o", C_TEST_REL, "c/test_matmul.c", *C_SOURCES])


def build_tools(flags=GCC_FLAGS):
    os.makedirs(os.path.join(ROOT, "tools", "build"), exist_ok=True)
    run_checked(["gcc", *flags, "-o", TOPOLOGY_EXE_REL, "tools/cpu_topology.c"])
    run_checked(["gcc", *flags, "-o", CACHE_EXE_REL, "tools/cache_info.c"])


def build_all(flags=GCC_FLAGS):
    build_c(flags)
    build_tools(flags)


def runner_command(*args):
    return [os.path.join(ROOT, C_RUNNER_REL), *[str(arg) for arg in args]]


def rel(path):
    """Path relative to ROOT (child processes run with cwd=ROOT)."""
    return os.path.relpath(os.fspath(path), ROOT)


# --------------------------------------------------------------------- topology


def cpu_topology(refresh=False):
    """Return the logical-processor topology, running tools/cpu_topology.exe if needed."""
    if refresh or not os.path.exists(TOPOLOGY_JSON):
        if not os.path.exists(os.path.join(ROOT, TOPOLOGY_EXE_REL)):
            build_tools()
        output = run_checked([os.path.join(ROOT, TOPOLOGY_EXE_REL)])
        topology = json.loads(output)
        with open(TOPOLOGY_JSON, "w", encoding="utf-8") as handle:
            json.dump(topology, handle, indent=2)
            handle.write("\n")
    with open(TOPOLOGY_JSON, encoding="utf-8") as handle:
        return json.load(handle)


def performance_cores(topology):
    """Logical processors with the highest EfficiencyClass (the P-cores on hybrid CPUs)."""
    processors = topology["logical_processors"]
    best = max(p["efficiency_class"] for p in processors)
    return sorted(p["logical_processor_index"] for p in processors if p["efficiency_class"] == best)


def efficiency_cores(topology):
    processors = topology["logical_processors"]
    best = max(p["efficiency_class"] for p in processors)
    return sorted(p["logical_processor_index"] for p in processors if p["efficiency_class"] != best)


# ---------------------------------------------------------------- memory budget


def dense_memory_estimate_bytes(n):
    """A, B and C as n x n float64 matrices."""
    return 3 * 8 * n * n


def csr_bytes(n, nnz):
    """Bytes of an n x n CSR matrix: header, int64 indptr, int32 indices, float64 data."""
    return 3 * 8 + (n + 1) * 8 + nnz * 4 + nnz * 8


def expected_output_nnz(n, density_a, density_b):
    """Expected nnz of C = A*B for independent uniformly random sparsity patterns.

    C[i][j] is structurally zero iff none of the n products A[i][k]*B[k][j] has
    two non-zero factors, which happens with probability (1 - da*db)^n."""
    return n * n * (1.0 - (1.0 - density_a * density_b) ** n)


def spgemm_memory_estimate_bytes(n, nnz_a, nnz_b):
    """Inputs + expected output (CSR) + Gustavson workspace (int32 marker, float64 acc)."""
    est_c = expected_output_nnz(n, nnz_a / (n * n), nnz_b / (n * n))
    return (csr_bytes(n, nnz_a) + csr_bytes(n, nnz_b) + csr_bytes(n, math.ceil(est_c))
            + n * (4 + 8))


def conversion_memory_estimate_bytes(n, nnz_a, nnz_b):
    """Two dense float64 inputs plus their two CSR conversions."""
    return 2 * 8 * n * n + csr_bytes(n, nnz_a) + csr_bytes(n, nnz_b)


class SeriesTracker:
    """Remembers the first failure of each measurement series.

    A series is e.g. (method, tile) for dense or (method, density) for sparse.
    Sizes are visited in increasing order, so after a failure at n every larger
    n of that series is skipped with status `skipped_after_<status>`."""

    def __init__(self):
        self._failures = {}

    def record(self, key, n, status):
        if status != "ok" and key not in self._failures:
            self._failures[key] = (n, status)

    def skip_status(self, key):
        if key not in self._failures:
            return None
        n, status = self._failures[key]
        return f"skipped_after_{status}", n


# ------------------------------------------------------------------- child runs

PRIORITY_CLASSES = {
    "normal": getattr(psutil, "NORMAL_PRIORITY_CLASS", None),
    "above_normal": getattr(psutil, "ABOVE_NORMAL_PRIORITY_CLASS", None),
    "high": getattr(psutil, "HIGH_PRIORITY_CLASS", None),
}
POLL_SECONDS = 0.25


def _pump(stream, sink):
    for line in stream:
        sink.put(line)
    sink.put(None)


def _peak_wset(proc):
    try:
        return proc.memory_info().peak_wset
    except (psutil.Error, AttributeError):
        return None


def run_benchmark_child(cmd, timeout_s, memory_budget_bytes, load_timeout_s=300,
                        affinity=None, priority=None):
    """Run one benchmark process and sample its peak working set.

    Protocol (see c/runner.c): the child prints {"event": "loaded"} after
    reading its inputs, {"event": "done", ...} after every kernel execution
    (warm-up or repetition), then one final result JSON line, and blocks on
    stdin. While it is still alive the harness reads `peak_wset`, then writes a
    newline to release it.

    Budgets:
      * time: each single kernel execution must finish within timeout_s
        (the deadline restarts at every event); loading has load_timeout_s.
      * memory: the working set is polled every POLL_SECONDS; above
        memory_budget_bytes the child is killed (over_memory_budget).

    Returns a dict with keys: status (ok|timeout|over_memory_budget|oom|error),
    result, peak_wset_bytes, returncode, stderr, setup_warnings, progress
    (number of completed kernel executions), note.
    """
    proc = psutil.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, bufsize=1)
    setup_warnings = []
    if affinity:
        try:
            proc.cpu_affinity(affinity)
        except (psutil.Error, ValueError, OSError) as exc:
            setup_warnings.append(f"affinity not applied ({type(exc).__name__})")
    if priority and PRIORITY_CLASSES.get(priority) is not None:
        try:
            proc.nice(PRIORITY_CLASSES[priority])
        except (psutil.Error, ValueError, OSError) as exc:
            setup_warnings.append(f"priority not applied ({type(exc).__name__})")

    lines = queue.Queue()
    stderr_lines = queue.Queue()
    threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True).start()
    threading.Thread(target=_pump, args=(proc.stderr, stderr_lines), daemon=True).start()

    deadline = time.monotonic() + load_timeout_s
    result = None
    status = None
    note = ""
    progress = 0
    loading = True
    peak = None
    while status is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            status = "timeout"
            if loading:
                note = f"loading inputs exceeded {load_timeout_s:g} s"
            else:
                note = (f"a single kernel execution exceeded {timeout_s:g} s "
                        f"(after {progress} completed executions)")
            break
        try:
            line = lines.get(timeout=min(POLL_SECONDS, remaining))
        except queue.Empty:
            try:
                wset = proc.memory_info().wset
            except (psutil.Error, AttributeError):
                wset = 0
            if memory_budget_bytes and wset > memory_budget_bytes:
                status = "over_memory_budget"
                note = (f"measured working set {wset / 2**30:.2f} GiB exceeded budget "
                        f"{memory_budget_bytes / 2**30:.2f} GiB")
            continue
        if line is None:
            break
        line = line.strip()
        if not line.startswith("{"):
            continue
        message = json.loads(line)
        if message.get("event") == "loaded":
            loading = False
            deadline = time.monotonic() + timeout_s
        elif message.get("event") == "done":
            progress += 1
            deadline = time.monotonic() + timeout_s
        else:
            result = message
            break

    if status in ("timeout", "over_memory_budget"):
        peak = _peak_wset(proc)
        proc.kill()
        proc.wait()
    elif result is not None and result.get("status") == "ok":
        peak = _peak_wset(proc)
        try:
            proc.stdin.write("\n")
            proc.stdin.flush()
        except OSError:
            pass
        proc.wait(timeout=120)
        status = "ok"
    else:
        proc.wait(timeout=120)
        status = "oom" if (result or {}).get("status") == "oom" else "error"

    stderr_text = []
    while True:
        try:
            item = stderr_lines.get(timeout=1)
        except queue.Empty:
            break
        if item is None:
            break
        stderr_text.append(item)
    return {
        "status": status,
        "result": result,
        "peak_wset_bytes": peak,
        "returncode": proc.returncode,
        "stderr": "".join(stderr_text),
        "setup_warnings": setup_warnings,
        "progress": progress,
        "note": note,
    }

