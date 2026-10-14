"""Benchmark harness: builds, generates inputs, runs every (language, n) configuration.

Each configuration runs as ONE child process (warm-ups + measured repetitions),
pinned to the performance-core logical processors with HIGH priority. The
child's peak working set is sampled while it waits on stdin after printing its
results. One CSV row per measured repetition is appended to
<output-dir>/raw_measurements.csv.

If a configuration times out or fails, a row with status=timeout|oom|error
(and no time) is written and larger sizes of that language are recorded as
status=skipped.

Usage:
    .venv\\Scripts\\python benchmark.py                       # full run from bench_config.json
    .venv\\Scripts\\python benchmark.py --sizes 64 128 --output-dir results/smoke
"""

import argparse
import csv
import datetime
import json
import os
import sys

import bench_common

sys.path.insert(0, os.path.join(bench_common.ROOT, "data"))
sys.path.insert(0, os.path.join(bench_common.ROOT, "tools"))
import capture_environment  # noqa: E402
import generate  # noqa: E402

CSV_COLUMNS = [
    "language",
    "n",
    "numeric_type",
    "repetition",
    "time_ns",
    "checksum",
    "peak_wset_bytes",
    "warmups",
    "status",
    "timestamp",
    "note",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the matrix multiplication benchmark.")
    parser.add_argument("--config", default=bench_common.DEFAULT_CONFIG)
    parser.add_argument("--sizes", type=int, nargs="+", help="override sizes from config")
    parser.add_argument("--languages", nargs="+", choices=bench_common.LANGUAGES)
    parser.add_argument("--repetitions", type=int, help="override measured repetitions")
    parser.add_argument("--timeout", type=float, help="per-configuration timeout in seconds")
    parser.add_argument("--output-dir", default="results", help="relative to assignment-1/")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def apply_overrides(config, args):
    if args.sizes:
        config["sizes"] = args.sizes
    if args.languages:
        config["languages"] = args.languages
    if args.repetitions:
        config["repetitions"] = args.repetitions
    if args.timeout:
        config["timeout_seconds"] = args.timeout
    config["sizes"] = sorted(set(config["sizes"]))
    return config


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def rows_for_run(language, n, config, run):
    base = {
        "language": language,
        "n": n,
        "numeric_type": config["numeric_type"],
        "warmups": config["warmups"][language],
        "timestamp": now(),
    }
    if run["status"] == "ok":
        result = run["result"]
        return [
            {
                **base,
                "repetition": index + 1,
                "time_ns": time_ns,
                "checksum": repr(float(checksum)),
                "peak_wset_bytes": run["peak_wset_bytes"],
                "status": "ok",
                "note": "",
            }
            for index, (time_ns, checksum) in enumerate(
                zip(result["times_ns"], result["checksums"])
            )
        ]
    if run["status"] == "timeout":
        note = f"timeout_limit_s={config['timeout_seconds']:g}"
    elif run["status"] == "oom":
        note = "out of memory reported by child process"
    else:
        last = run["stderr"].strip().splitlines()[-1:] or [""]
        note = f"returncode={run['returncode']} {last[0]}".strip()
    return [{**base, "repetition": "", "time_ns": "", "checksum": "",
             "peak_wset_bytes": "", "status": run["status"], "note": note}]


def skipped_row(language, n, config, failed_n, failed_status):
    return {
        "language": language,
        "n": n,
        "numeric_type": config["numeric_type"],
        "repetition": "",
        "time_ns": "",
        "checksum": "",
        "peak_wset_bytes": "",
        "warmups": config["warmups"][language],
        "status": "skipped",
        "timestamp": now(),
        "note": f"skipped after {failed_status} at n={failed_n}",
    }


def append_rows(csv_path, rows):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config = apply_overrides(bench_common.load_config(args.config), args)
    output_dir = os.path.join(bench_common.ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "raw_measurements.csv")

    if not args.skip_build:
        print("Building C, Java and tools ...", flush=True)
        bench_common.build_all(config)

    topology = bench_common.cpu_topology(config)
    affinity = (
        bench_common.performance_cores(topology) if config["pin_to_performance_cores"] else None
    )
    print(f"CPU affinity (P-core logical processors): {affinity}", flush=True)

    print(f"Generating inputs for sizes {config['sizes']} ...", flush=True)
    for n in config["sizes"]:
        generate.ensure_inputs(n)

    environment = capture_environment.capture(config)
    with open(os.path.join(output_dir, "environment.json"), "w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2)
        handle.write("\n")

    java_exe = bench_common.java_executable()
    for language in config["languages"]:
        failure = None
        for n in config["sizes"]:
            if failure:
                append_rows(csv_path, [skipped_row(language, n, config, *failure)])
                print(f"[{language} n={n}] skipped ({failure[1]} at n={failure[0]})", flush=True)
                continue
            cmd = bench_common.child_command(
                language,
                config,
                ["bench", n, bench_common.DATA_DIR_REL, config["warmups"][language],
                 config["repetitions"]],
                java_exe,
            )
            print(f"[{language} n={n}] running ...", flush=True)
            run = bench_common.run_benchmark_child(
                cmd, config["timeout_seconds"], affinity, config["priority_class"]
            )
            append_rows(csv_path, rows_for_run(language, n, config, run))
            if run["status"] == "ok":
                times = run["result"]["times_ns"]
                median_s = sorted(times)[len(times) // 2] / 1e9
                print(f"[{language} n={n}] ok  median={median_s:.6f}s  "
                      f"peak_wset={run['peak_wset_bytes'] / 2**20:.1f} MiB", flush=True)
            else:
                failure = (n, run["status"])
                print(f"[{language} n={n}] {run['status']}: {run['stderr'].strip()[-300:]}",
                      flush=True)
    print(f"Results appended to {csv_path}")


if __name__ == "__main__":
    main()
