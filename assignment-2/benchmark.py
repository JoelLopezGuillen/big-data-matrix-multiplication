"""Benchmark harness for Assignment 2 (dense methods, tile sweep, sparse CSR).

The experiment is selected by the config file:
    bench_config_dense.json   ijk / ikj / blocked on the dense A_n, B_n inputs
    bench_config_tiles.json   blocked with several tile sizes (tile sweep)
    bench_config_sparse.json  own CSR SpGEMM, dense->CSR conversion and dense
                              blocked on matched random sparse inputs

Each configuration (one method/tile/density at one n) runs as ONE child process
(warm-ups + measured repetitions), pinned to the performance-core logical
processors at HIGH priority. Budgets (see bench_common.run_benchmark_child):
    time    every single kernel execution must finish within timeout_seconds;
            otherwise the process is killed and the configuration is recorded
            as status=timeout (no time is recorded).
    memory  before launching, the memory estimate is compared with
            memory_budget_bytes (status=over_memory_budget, not run); while
            running, the working set is polled and the child is killed if it
            exceeds the budget (status=over_memory_budget).
After a non-ok status, every larger n of the same series is recorded as
status=skipped_after_<status>. One CSV row per measured repetition is appended
to <output_dir>/raw_measurements.csv.

Usage:
    .venv\\Scripts\\python benchmark.py --config bench_config_tiles.json
    .venv\\Scripts\\python benchmark.py --config bench_config_dense.json
    .venv\\Scripts\\python benchmark.py --config bench_config_sparse.json
    .venv\\Scripts\\python benchmark.py --config bench_config_dense.json --smoke
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
import generate_sparse  # noqa: E402

CSV_COLUMNS = [
    "experiment",
    "method",
    "representation",
    "phase",
    "n",
    "numeric_type",
    "threads",
    "warmups",
    "repetition",
    "tile",
    "density_percent",
    "sparsity_percent",
    "nnz_a",
    "nnz_b",
    "nnz_c",
    "output_density_percent",
    "status",
    "time_ns",
    "checksum",
    "peak_working_set_bytes",
    "memory_estimate_bytes",
    "timestamp",
    "note",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run an Assignment 2 benchmark.")
    parser.add_argument("--config", default=bench_common.DENSE_CONFIG)
    parser.add_argument("--smoke", action="store_true",
                        help="apply the config's 'smoke' overrides (tiny sizes, 1 repetition)")
    parser.add_argument("--sizes", type=int, nargs="+", help="override sizes (dense/tiles)")
    parser.add_argument("--methods", nargs="+", choices=bench_common.DENSE_METHODS)
    parser.add_argument("--tiles", type=int, nargs="+", help="override blocked tile sizes")
    parser.add_argument("--densities", type=float, nargs="+", help="override densities (%%)")
    parser.add_argument("--repetitions", type=int, help="override measured repetitions")
    parser.add_argument("--timeout", type=float, help="per-execution timeout in seconds")
    parser.add_argument("--memory-budget-gib", type=float, help="override the memory budget")
    parser.add_argument("--output-dir", help="relative to assignment-2/")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def apply_overrides(config, args):
    if args.smoke:
        config.update(config.get("smoke", {}))
    overrides = {
        "sizes": args.sizes,
        "methods": args.methods,
        "blocked_tiles": args.tiles,
        "densities_percent": args.densities,
        "repetitions": args.repetitions,
        "timeout_seconds": args.timeout,
        "output_dir": args.output_dir,
        "memory_budget_bytes": (int(args.memory_budget_gib * 2**30)
                                if args.memory_budget_gib else None),
    }
    for key, value in overrides.items():
        if value:
            config[key] = value
    for key in ("sizes", "dense_compare_sizes", "csr_only_sizes"):
        if key in config:
            config[key] = sorted(set(config[key]))
    return config


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def pct(nnz, n):
    return "" if nnz in ("", None) else f"{100.0 * int(nnz) / (n * n):.6g}"


class Measurement:
    """One configuration to run (or skip): method/representation at one n."""

    def __init__(self, experiment, method, representation, phase, n, series, cmd,
                 estimate, tile="", density="", nnz_a="", nnz_b=""):
        self.experiment = experiment
        self.method = method
        self.representation = representation
        self.phase = phase
        self.n = n
        self.series = series
        self.cmd = cmd
        self.estimate = estimate
        self.tile = tile
        self.density = density
        self.nnz_a = nnz_a
        self.nnz_b = nnz_b

    def label(self):
        parts = [self.method, f"n={self.n}"]
        if self.tile != "":
            parts.append(f"tile={self.tile}")
        if self.density != "":
            parts.append(f"density={generate_sparse.density_tag(self.density)}%")
        return " ".join(parts)

    def base_row(self, config):
        return {
            "experiment": self.experiment,
            "method": self.method,
            "representation": self.representation,
            "phase": self.phase,
            "n": self.n,
            "numeric_type": config["numeric_type"],
            "threads": config["threads"],
            "warmups": config["warmups"],
            "tile": self.tile,
            "density_percent": "" if self.density == "" else f"{self.density:g}",
            "sparsity_percent": "" if self.density == "" else f"{100 - self.density:g}",
            "nnz_a": self.nnz_a,
            "nnz_b": self.nnz_b,
            "nnz_c": "",
            "output_density_percent": "",
            "repetition": "",
            "time_ns": "",
            "checksum": "",
            "peak_working_set_bytes": "",
            "memory_estimate_bytes": self.estimate,
            "timestamp": now(),
            "note": "",
        }


def rows_for_run(measurement, config, run):
    base = measurement.base_row(config)
    setup_note = "; ".join(run.get("setup_warnings", []))
    if run["status"] == "ok":
        result = run["result"]
        nnz_a = result.get("nnz_a", base["nnz_a"])
        nnz_b = result.get("nnz_b", base["nnz_b"])
        nnz_c = result.get("nnz_c", "")
        return [
            {**base, "repetition": index + 1, "time_ns": time_ns,
             "checksum": repr(float(checksum)), "nnz_a": nnz_a, "nnz_b": nnz_b,
             "nnz_c": nnz_c, "output_density_percent": pct(nnz_c, measurement.n),
             "peak_working_set_bytes": run["peak_wset_bytes"], "status": "ok",
             "note": setup_note}
            for index, (time_ns, checksum) in enumerate(
                zip(result["times_ns"], result["checksums"]))
        ]
    if run["status"] == "oom":
        note = "allocation failed in the child process (exit code 3)"
    elif run["status"] in ("timeout", "over_memory_budget"):
        note = run["note"]
    else:
        last = run["stderr"].strip().splitlines()[-1:] or [""]
        note = f"returncode={run['returncode']} {last[0]}".strip()
    note = "; ".join(part for part in (note, setup_note) if part)
    peak = run["peak_wset_bytes"] if run["peak_wset_bytes"] is not None else ""
    return [{**base, "status": run["status"], "peak_working_set_bytes": peak, "note": note}]


def append_rows(csv_path, rows):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def execute(measurement, config, tracker, affinity, csv_path):
    """Run, skip or reject one measurement and append its CSV rows."""
    label = measurement.label()
    skip = tracker.skip_status(measurement.series)
    budget = config["memory_budget_bytes"]
    if skip:
        status, failed_n = skip
        row = {**measurement.base_row(config), "status": status,
               "note": f"{status.removeprefix('skipped_after_')} at n={failed_n} for this series"}
        append_rows(csv_path, [row])
        print(f"[{label}] {status} (n={failed_n})", flush=True)
        return
    if measurement.estimate > budget:
        row = {**measurement.base_row(config), "status": "over_memory_budget",
               "note": f"estimated {measurement.estimate / 2**30:.2f} GiB > budget "
                       f"{budget / 2**30:.2f} GiB; not run"}
        append_rows(csv_path, [row])
        tracker.record(measurement.series, measurement.n, "over_memory_budget")
        print(f"[{label}] over_memory_budget (estimate "
              f"{measurement.estimate / 2**30:.2f} GiB)", flush=True)
        return
    print(f"[{label}] running ...", flush=True)
    run = bench_common.run_benchmark_child(
        measurement.cmd, config["timeout_seconds"], budget,
        config["load_timeout_seconds"], affinity, config["priority_class"])
    append_rows(csv_path, rows_for_run(measurement, config, run))
    tracker.record(measurement.series, measurement.n, run["status"])
    if run["status"] == "ok":
        times = run["result"]["times_ns"]
        median_s = sorted(times)[len(times) // 2] / 1e9
        extra = f" nnz_c={run['result']['nnz_c']}" if "nnz_c" in run["result"] else ""
        print(f"[{label}] ok  median={median_s:.6f}s  "
              f"peak_wset={run['peak_wset_bytes'] / 2**20:.1f} MiB{extra}", flush=True)
    else:
        print(f"[{label}] {run['status']}: {run['note'] or run['stderr'].strip()[-300:]}",
              flush=True)


# ---------------------------------------------------------------- experiments


def dense_measurements(config):
    """Dense and tile-sweep experiments: every (method, tile) series over sizes."""
    for method in config["methods"]:
        tiles = config["blocked_tiles"] if method == "blocked" else [""]
        for tile in tiles:
            for n in config["sizes"]:
                a = bench_common.rel(generate.matrix_path("A", n))
                b = bench_common.rel(generate.matrix_path("B", n))
                cmd = bench_common.runner_command(
                    "bench", method, n, a, b, config["warmups"], config["repetitions"],
                    tile or 0)
                yield Measurement(config["experiment"], method, "dense", "multiply", n,
                                  (method, tile), cmd,
                                  bench_common.dense_memory_estimate_bytes(n), tile=tile)


def sparse_measurements(config):
    """Sparse experiment: per density, sizes ascending; CSR SpGEMM, conversion, dense."""
    data_dir = os.path.join(bench_common.ROOT, config["data_dir"])
    compare = set(config["dense_compare_sizes"])
    sizes = sorted(compare | set(config["csr_only_sizes"]))
    warmups, reps = config["warmups"], config["repetitions"]
    tile = config["dense_tile"]
    for density in config["densities_percent"]:
        for n in sizes:
            nnz = generate_sparse.target_nnz(n, density)
            a_csr = bench_common.rel(generate_sparse.csr_path("A", n, density, data_dir))
            b_csr = bench_common.rel(generate_sparse.csr_path("B", n, density, data_dir))
            common = {"density": density, "nnz_a": nnz, "nnz_b": nnz}
            yield Measurement(
                "sparse", "csr_gustavson", "csr", "multiply", n, ("csr_gustavson", density),
                bench_common.runner_command("sparse-bench", a_csr, b_csr, warmups, reps),
                bench_common.spgemm_memory_estimate_bytes(n, nnz, nnz), **common)
            if n not in compare:
                yield ("not_run", n, density, nnz)
                continue
            a_dense = bench_common.rel(generate_sparse.dense_path("A", n, density, data_dir))
            b_dense = bench_common.rel(generate_sparse.dense_path("B", n, density, data_dir))
            yield Measurement(
                "sparse", "dense_to_csr", "csr", "convert", n, ("dense_to_csr", density),
                bench_common.runner_command("convert-bench", n, a_dense, b_dense, warmups,
                                            reps),
                bench_common.conversion_memory_estimate_bytes(n, nnz, nnz), **common)
            yield Measurement(
                "sparse", config["dense_method"], "dense", "multiply", n,
                (config["dense_method"], density),
                bench_common.runner_command("bench", config["dense_method"], n, a_dense,
                                            b_dense, warmups, reps, tile),
                bench_common.dense_memory_estimate_bytes(n), tile=tile, **common)


def not_run_rows(config, n, density, nnz):
    """Rows documenting that the dense comparison is not run for CSR-only sizes."""
    note = (f"not run by design: n={n} is a CSR-only size (dense comparison sizes: "
            f"{config['dense_compare_sizes']})")
    rows = []
    for method, representation, phase, tile in (
            ("dense_to_csr", "csr", "convert", ""),
            (config["dense_method"], "dense", "multiply", config["dense_tile"])):
        m = Measurement("sparse", method, representation, phase, n, None, None, "",
                        tile=tile, density=density, nnz_a=nnz, nnz_b=nnz)
        rows.append({**m.base_row(config), "status": "not_run", "note": note})
    return rows


def prepare_inputs(config):
    if config["experiment"] in ("dense", "tiles"):
        print(f"Generating dense inputs for sizes {config['sizes']} ...", flush=True)
        for n in config["sizes"]:
            generate.ensure_inputs(n)
        return
    data_dir = os.path.join(bench_common.ROOT, config["data_dir"])
    compare = set(config["dense_compare_sizes"])
    for n in sorted(compare | set(config["csr_only_sizes"])):
        for density in config["densities_percent"]:
            print(f"Generating sparse inputs n={n} density={density:g}% ...", flush=True)
            generate_sparse.ensure_sparse_inputs(n, density, data_dir, with_dense=n in compare)


def main():
    args = parse_args()
    config = apply_overrides(bench_common.load_config(args.config), args)
    output_dir = os.path.join(bench_common.ROOT, config["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "raw_measurements.csv")

    if not args.skip_build:
        print("Building C runner, tests and tools ...", flush=True)
        bench_common.build_all(config["gcc_flags"])

    topology = bench_common.cpu_topology()
    affinity = (bench_common.performance_cores(topology)
                if config["pin_to_performance_cores"] else None)
    print(f"CPU affinity (P-core logical processors): {affinity}", flush=True)

    prepare_inputs(config)
    environment = capture_environment.capture(config)
    with open(os.path.join(output_dir, "environment.json"), "w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2)
        handle.write("\n")

    tracker = bench_common.SeriesTracker()
    if config["experiment"] in ("dense", "tiles"):
        items = dense_measurements(config)
    elif config["experiment"] == "sparse":
        items = sparse_measurements(config)
    else:
        raise ValueError(f"unknown experiment: {config['experiment']}")
    for item in items:
        if isinstance(item, tuple):
            _, n, density, nnz = item
            append_rows(csv_path, not_run_rows(config, n, density, nnz))
            print(f"[dense comparison n={n} density={density:g}%] not_run (CSR-only size)",
                  flush=True)
            continue
        execute(item, config, tracker, affinity, csv_path)
    print(f"Results appended to {csv_path}")


if __name__ == "__main__":
    main()
