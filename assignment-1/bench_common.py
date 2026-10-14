"""Shared helpers for the build, cross-check, environment and benchmark scripts.

All child processes run with the assignment directory as working directory and
receive relative paths, which keeps non-ASCII characters in the absolute path
away from the native tools.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

import psutil

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(ROOT, "bench_config.json")
DATA_DIR_REL = "data"
TOPOLOGY_JSON = os.path.join(ROOT, "tools", "cpu_topology.json")

C_BUILD_DIR_REL = os.path.join("c", "build")
C_RUNNER_REL = os.path.join(C_BUILD_DIR_REL, "runner.exe")
C_TEST_REL = os.path.join(C_BUILD_DIR_REL, "test_matmul.exe")
TOPOLOGY_EXE_REL = os.path.join("tools", "build", "cpu_topology.exe")
JAVA_OUT_REL = os.path.join("java", "out")
PYTHON_RUNNER_REL = os.path.join("python", "runner.py")

LANGUAGES = ("c", "java", "python")


def load_config(path=DEFAULT_CONFIG):
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


def build_c(config):
    os.makedirs(os.path.join(ROOT, C_BUILD_DIR_REL), exist_ok=True)
    flags = config["gcc_flags"]
    run_checked(["gcc", *flags, "-o", C_RUNNER_REL, "c/runner.c", "c/matmul.c"])
    run_checked(["gcc", *flags, "-o", C_TEST_REL, "c/test_matmul.c", "c/matmul.c"])


def build_java(config):
    os.makedirs(os.path.join(ROOT, JAVA_OUT_REL), exist_ok=True)
    sources = ["java/MatMul.java", "java/MatMulRunner.java", "java/MatMulTest.java"]
    run_checked(["javac", *config["javac_flags"], "-d", JAVA_OUT_REL, *sources])


def build_tools(config):
    os.makedirs(os.path.join(ROOT, "tools", "build"), exist_ok=True)
    run_checked(["gcc", *config["gcc_flags"], "-o", TOPOLOGY_EXE_REL, "tools/cpu_topology.c"])


def build_all(config):
    build_c(config)
    build_java(config)
    build_tools(config)


# ------------------------------------------------------------------ executables


def java_home():
    """Resolve the real JDK home (the `java` on PATH may be a launcher stub)."""
    proc = subprocess.run(
        ["java", "-XshowSettings:properties", "-version"],
        capture_output=True,
        text=True,
    )
    for line in proc.stderr.splitlines():
        if line.strip().startswith("java.home ="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("could not determine java.home")


def java_executable():
    return os.path.join(java_home(), "bin", "java.exe" if os.name == "nt" else "java")


def python_executable():
    """Base interpreter: on Windows the venv python.exe is a launcher process."""
    return getattr(sys, "_base_executable", None) or sys.executable


def child_command(language, config, args, java_exe=None):
    """Command line that runs a language runner with the given runner arguments."""
    args = [str(arg) for arg in args]
    if language == "c":
        return [os.path.join(ROOT, C_RUNNER_REL), *args]
    if language == "java":
        java_exe = java_exe or java_executable()
        return [java_exe, *config["java_flags"], "-cp", JAVA_OUT_REL, "MatMulRunner", *args]
    if language == "python":
        # -E/-s: ignore PYTHON* variables and user site-packages; stdlib only.
        return [python_executable(), "-E", "-s", PYTHON_RUNNER_REL, *args]
    raise ValueError(f"unknown language: {language}")


# --------------------------------------------------------------------- topology


def cpu_topology(config=None, refresh=False):
    """Return the logical-processor topology, running tools/cpu_topology.exe if needed."""
    if refresh or not os.path.exists(TOPOLOGY_JSON):
        if not os.path.exists(os.path.join(ROOT, TOPOLOGY_EXE_REL)):
            build_tools(config or load_config())
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


# ------------------------------------------------------------------- child runs

PRIORITY_CLASSES = {
    "normal": getattr(psutil, "NORMAL_PRIORITY_CLASS", None),
    "above_normal": getattr(psutil, "ABOVE_NORMAL_PRIORITY_CLASS", None),
    "high": getattr(psutil, "HIGH_PRIORITY_CLASS", None),
}


def _pump(stream, sink):
    for line in stream:
        sink.put(line)
    sink.put(None)


def run_benchmark_child(cmd, timeout_s, affinity=None, priority=None):
    """Run one benchmark process and sample its peak working set.

    Protocol: the child prints one JSON line, then blocks on stdin. While it is
    still alive the harness reads `memory_info().peak_wset`, then writes a
    newline to release it.

    Returns a dict with keys: status (ok|timeout|oom|error), result (parsed JSON
    or None), peak_wset_bytes (int or None), returncode, stderr.
    """
    proc = psutil.Popen(
        cmd,
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        if affinity:
            proc.cpu_affinity(affinity)
        if priority and PRIORITY_CLASSES.get(priority) is not None:
            proc.nice(PRIORITY_CLASSES[priority])
    except psutil.Error:
        pass

    lines = queue.Queue()
    stderr_lines = queue.Queue()
    threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True).start()
    threading.Thread(target=_pump, args=(proc.stderr, stderr_lines), daemon=True).start()

    deadline = time.monotonic() + timeout_s
    result = None
    timed_out = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            line = lines.get(timeout=remaining)
        except queue.Empty:
            timed_out = True
            break
        if line is None:
            break
        line = line.strip()
        if line.startswith("{"):
            result = json.loads(line)
            break

    peak = None
    if timed_out:
        proc.kill()
        proc.wait()
        status = "timeout"
    elif result is not None and result.get("status") == "ok":
        try:
            peak = proc.memory_info().peak_wset
        except (psutil.Error, AttributeError):
            peak = None
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
    stderr_joined = "".join(stderr_text)
    if status == "error" and "OutOfMemoryError" in stderr_joined:
        status = "oom"
    return {
        "status": status,
        "result": result,
        "peak_wset_bytes": peak,
        "returncode": proc.returncode,
        "stderr": stderr_joined,
    }
