"""Summarise raw benchmark measurements and draw the plots of Assignment 2.

For every input raw_measurements.csv (default: results/dense, results/tiles and
results/sparse, whichever exist) the outputs are written next to it, or into
--output-dir:

    summary.csv, summary.md   median, IQR and repetition count per configuration
                              (failed / skipped configurations listed with their
                              status and note, never with a time)
  dense experiment:
    runtime_vs_n.png          log-log runtime vs n per method, n^3 reference
    speedup_vs_n.png          speedup of each method over the ijk baseline
    memory_vs_n.png           peak working set vs n
  tiles experiment:
    tile_sweep.png            runtime vs tile size per n, with the tile sizes
                              whose three-tile working set 3*8*T^2 equals L1d / L2
  sparse experiment:
    sparse_comparison.csv     per (n, density): SpGEMM, conversion, derived
                              end-to-end (= median conversion + median SpGEMM),
                              dense blocked, ratios, nnz and output density
    sparse_time_vs_density.png   time vs input density, one panel per n
    output_density.png           output density vs input density (+ expectation)
    sparse_memory_vs_density.png peak working set vs input density

Usage:
    .venv\\Scripts\\python results\\analyze.py
    .venv\\Scripts\\python results\\analyze.py --input results/smoke/dense/raw_measurements.csv
"""

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(RESULTS_DIR)
CACHE_JSON = os.path.join(ROOT, "tools", "cache_info.json")

KEYS = ["experiment", "method", "representation", "phase", "n", "tile", "density_percent"]
TEXT = "#1f1f1c"
TEXT_MUTED = "#5f5e57"
GRID = "#e4e3dc"
# Fixed categorical palette; every series also gets its own marker so identity
# never relies on colour alone.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#d64a7a", "#b08a1a", "#4a4a45"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
METHOD_LABELS = {
    "ijk": "ijk (baseline)",
    "ikj": "ikj (loop reordering)",
    "blocked": "blocked",
    "csr_gustavson": "CSR SpGEMM (Gustavson)",
    "dense_to_csr": "dense->CSR conversion",
}


def _text_key(series):
    """Normalise a key column to text ('' for missing, '64' rather than '64.0')."""
    def fmt(value):
        if pd.isna(value) or value == "":
            return ""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{number:g}"
    return series.map(fmt)


def _prepare(raw):
    raw = raw.copy()
    for column in KEYS + ["nnz_a", "nnz_b", "nnz_c", "output_density_percent", "note"]:
        if column not in raw:
            raw[column] = ""
    for column in ("tile", "density_percent"):
        raw[column] = _text_key(raw[column])
    for column in ("experiment", "method", "representation", "phase"):
        raw[column] = raw[column].fillna("").astype(str)
    raw["n"] = raw["n"].astype("int64")
    raw["note"] = raw["note"].fillna("").astype(str)
    return raw


def summarise(raw):
    raw = _prepare(raw)
    ok = raw[raw["status"] == "ok"].copy()
    ok["time_s"] = pd.to_numeric(ok["time_ns"]).astype("float64") / 1e9
    for column in ("nnz_a", "nnz_b", "nnz_c", "output_density_percent", "peak_working_set_bytes"):
        ok[column] = pd.to_numeric(ok[column], errors="coerce")
    summary = ok.groupby(KEYS, sort=False).agg(
        repetitions=("time_s", "size"),
        median_s=("time_s", "median"),
        q1_s=("time_s", lambda s: s.quantile(0.25)),
        q3_s=("time_s", lambda s: s.quantile(0.75)),
        min_s=("time_s", "min"),
        max_s=("time_s", "max"),
        peak_wset_mib=("peak_working_set_bytes", lambda s: s.median() / 2**20),
        nnz_a=("nnz_a", "first"),
        nnz_b=("nnz_b", "first"),
        nnz_c=("nnz_c", "first"),
        output_density_percent=("output_density_percent", "first"),
    ).reset_index()
    summary["iqr_s"] = summary["q3_s"] - summary["q1_s"]
    summary["status"] = "ok"
    summary["note"] = ""

    failed = raw[raw["status"] != "ok"].drop_duplicates(KEYS)
    failed = failed[KEYS + ["status", "note", "nnz_a", "nnz_b"]].copy()
    for column in ("nnz_a", "nnz_b"):
        failed[column] = pd.to_numeric(failed[column], errors="coerce")
    summary = pd.concat([summary, failed], ignore_index=True)
    columns = KEYS + ["status", "repetitions", "median_s", "iqr_s", "q1_s", "q3_s", "min_s",
                      "max_s", "peak_wset_mib", "nnz_a", "nnz_b", "nnz_c",
                      "output_density_percent", "note"]
    for column in columns:
        if column not in summary:
            summary[column] = pd.NA
    summary["_tile"] = pd.to_numeric(summary["tile"], errors="coerce")
    summary["_density"] = pd.to_numeric(summary["density_percent"], errors="coerce")
    summary = summary.sort_values(["experiment", "method", "_tile", "_density", "n"])
    return summary[columns].reset_index(drop=True)


def series_label(method, tile):
    label = METHOD_LABELS.get(method, method)
    return f"{label} (tile={tile})" if method == "blocked" and tile else label


def speedups(summary, baseline="ijk"):
    """Speedup of every ok dense configuration over the baseline at the same n."""
    ok = summary[(summary["status"] == "ok") & (summary["representation"] == "dense")]
    base = ok[ok["method"] == baseline].set_index("n")["median_s"]
    table = ok[ok["n"].isin(base.index)].copy()
    table["baseline_median_s"] = table["n"].map(base)
    table["speedup"] = table["baseline_median_s"] / table["median_s"]
    return table[["method", "tile", "n", "median_s", "baseline_median_s", "speedup"]]


def sparse_comparison(summary):
    """Per (n, density): SpGEMM, conversion, derived end-to-end and dense times."""
    sparse = summary[summary["experiment"] == "sparse"]
    rows = []
    for (n, density), group in sparse.groupby(["n", "density_percent"], sort=False):
        def pick(method):
            match = group[group["method"] == method]
            return match.iloc[0] if not match.empty else None

        spgemm, convert = pick("csr_gustavson"), pick("dense_to_csr")
        dense = group[group["representation"] == "dense"]
        dense = dense.iloc[0] if not dense.empty else None

        def median(row):
            return row["median_s"] if row is not None and row["status"] == "ok" else math.nan

        def status(row):
            return row["status"] if row is not None else ""

        spgemm_s, convert_s, dense_s = median(spgemm), median(convert), median(dense)
        end_to_end = spgemm_s + convert_s
        rows.append({
            "n": int(n),
            "density_percent": density,
            "nnz_a": spgemm["nnz_a"] if spgemm is not None else math.nan,
            "nnz_b": spgemm["nnz_b"] if spgemm is not None else math.nan,
            "nnz_c": spgemm["nnz_c"] if spgemm is not None else math.nan,
            "output_density_percent": (spgemm["output_density_percent"]
                                       if spgemm is not None else math.nan),
            "spgemm_s": spgemm_s,
            "spgemm_status": status(spgemm),
            "convert_s": convert_s,
            "convert_status": status(convert),
            "end_to_end_s": end_to_end,
            "dense_s": dense_s,
            "dense_status": status(dense),
            "dense_over_spgemm": dense_s / spgemm_s,
            "dense_over_end_to_end": dense_s / end_to_end,
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["_density"] = pd.to_numeric(table["density_percent"])
        table = table.sort_values(["n", "_density"]).drop(columns="_density")
    return table.reset_index(drop=True)


# ------------------------------------------------------------------- markdown


def _fmt(value, spec):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return format(value, spec) if not isinstance(value, str) else value


def to_markdown(frame, columns):
    """columns: list of (header, column, format spec)."""
    lines = ["| " + " | ".join(header for header, _, _ in columns) + " |",
             "|" + "---|" * len(columns)]
    for row in frame.to_dict("records"):
        lines.append("| " + " | ".join(_fmt(row.get(col), spec) for _, col, spec in columns)
                     + " |")
    return "\n".join(lines) + "\n"


SUMMARY_COLUMNS = [
    ("Method", "method", ""), ("Repr.", "representation", ""), ("Phase", "phase", ""),
    ("n", "n", "d"), ("Tile", "tile", ""), ("Density %", "density_percent", ""),
    ("Status", "status", ""), ("Reps", "repetitions", ".0f"), ("Median (s)", "median_s", ".6g"),
    ("IQR (s)", "iqr_s", ".3g"), ("Q1 (s)", "q1_s", ".6g"), ("Q3 (s)", "q3_s", ".6g"),
    ("Peak WS (MiB)", "peak_wset_mib", ".1f"), ("nnz C", "nnz_c", ".0f"),
    ("Note", "note", ""),
]
SPEEDUP_COLUMNS = [("Method", "method", ""), ("Tile", "tile", ""), ("n", "n", "d"),
                   ("Median (s)", "median_s", ".6g"),
                   ("ijk median (s)", "baseline_median_s", ".6g"),
                   ("Speedup vs ijk", "speedup", ".2f")]
SPARSE_COLUMNS = [
    ("n", "n", "d"), ("Density %", "density_percent", ""), ("nnz A", "nnz_a", ".0f"),
    ("nnz B", "nnz_b", ".0f"), ("nnz C", "nnz_c", ".0f"),
    ("Output density %", "output_density_percent", ".4g"),
    ("SpGEMM (s)", "spgemm_s", ".6g"), ("Convert A+B (s)", "convert_s", ".6g"),
    ("End-to-end (s)", "end_to_end_s", ".6g"), ("Dense blocked (s)", "dense_s", ".6g"),
    ("Dense / SpGEMM", "dense_over_spgemm", ".3g"),
    ("Dense / end-to-end", "dense_over_end_to_end", ".3g"),
    ("SpGEMM status", "spgemm_status", ""), ("Dense status", "dense_status", ""),
]


# ---------------------------------------------------------------------- plots


def style_axes(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED)


def log2_ticks(ax, values):
    values = sorted(set(int(v) for v in values))
    ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.set_xticklabels([str(v) for v in values])
    ax.minorticks_off()


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def dense_series(ok):
    pairs = ok[["method", "tile"]].drop_duplicates().itertuples(index=False)
    order = {"ijk": 0, "ikj": 1, "blocked": 2}
    return sorted(pairs, key=lambda p: (order.get(p.method, 9), pd.to_numeric(p.tile or 0)))


def plot_dense_runtime(summary, path):
    ok = summary[summary["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for index, (method, tile) in enumerate(dense_series(ok)):
        data = ok[(ok["method"] == method) & (ok["tile"] == tile)].sort_values("n")
        yerr = [data["median_s"] - data["q1_s"], data["q3_s"] - data["median_s"]]
        ax.errorbar(data["n"], data["median_s"], yerr=yerr, color=PALETTE[index % 7],
                    marker=MARKERS[index % 7], markersize=5, linewidth=2, capsize=3,
                    label=series_label(method, tile))
    if not ok.empty:
        anchor = ok.sort_values(["n", "median_s"], ascending=[False, True]).iloc[0]
        ns = sorted(ok["n"].unique())
        ax.plot(ns, [anchor["median_s"] * (n / anchor["n"]) ** 3 for n in ns], "--",
                color=TEXT_MUTED, linewidth=1.2, label=r"$\propto n^3$ reference")
        log2_ticks(ax, ns)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (n x n, log2 scale)")
    ax.set_ylabel("Kernel runtime, median (s, log scale)")
    ax.set_title("Dense matrix multiplication runtime vs n (error bars: IQR)")
    style_axes(ax)
    ax.legend(frameon=False)
    save(fig, path)


def plot_speedup(table, path):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for index, (method, tile) in enumerate(dense_series(table)):
        data = table[(table["method"] == method) & (table["tile"] == tile)].sort_values("n")
        ax.plot(data["n"], data["speedup"], color=PALETTE[index % 7],
                marker=MARKERS[index % 7], markersize=5, linewidth=2,
                label=series_label(method, tile))
    if not table.empty:
        log2_ticks(ax, table["n"])
    ax.set_xlabel("Matrix size n (n x n, log2 scale)")
    ax.set_ylabel("Speedup over ijk (median ijk / median method)")
    ax.set_title("Speedup of the optimised dense methods over the ijk baseline")
    style_axes(ax)
    ax.legend(frameon=False)
    save(fig, path)


def plot_dense_memory(summary, path):
    ok = summary[summary["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for index, (method, tile) in enumerate(dense_series(ok)):
        data = ok[(ok["method"] == method) & (ok["tile"] == tile)].sort_values("n")
        ax.plot(data["n"], data["peak_wset_mib"], color=PALETTE[index % 7],
                marker=MARKERS[index % 7], markersize=5, linewidth=2,
                label=series_label(method, tile))
    if not ok.empty:
        ns = sorted(ok["n"].unique())
        ax.plot(ns, [3 * 8 * n * n / 2**20 for n in ns], "--", color=TEXT_MUTED,
                linewidth=1.2, label="3 x 8 n^2 bytes (A, B, C)")
        log2_ticks(ax, ns)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (n x n, log2 scale)")
    ax.set_ylabel("Peak working set of the process (MiB, log scale)")
    ax.set_title("Peak working set vs n")
    style_axes(ax)
    ax.legend(frameon=False)
    save(fig, path)


def cache_sizes():
    """(L1d, L2) bytes of the performance cores from tools/cache_info.json."""
    try:
        with open(CACHE_JSON, encoding="utf-8") as handle:
            entries = json.load(handle)
    except OSError:
        return None, None
    l1 = max((e["size_bytes"] for e in entries if e["level"] == 1 and e["type"] == "data"),
             default=None)
    l2 = min((e["size_bytes"] for e in entries if e["level"] == 2), default=None)
    return l1, l2


def plot_tile_sweep(summary, path):
    ok = summary[(summary["status"] == "ok") & (summary["method"] == "blocked")].copy()
    ok["tile_value"] = pd.to_numeric(ok["tile"])
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for index, n in enumerate(sorted(ok["n"].unique())):
        data = ok[ok["n"] == n].sort_values("tile_value")
        yerr = [data["median_s"] - data["q1_s"], data["q3_s"] - data["median_s"]]
        ax.errorbar(data["tile_value"], data["median_s"], yerr=yerr, color=PALETTE[index % 7],
                    marker=MARKERS[index % 7], markersize=5, linewidth=2, capsize=3,
                    label=f"n={n}")
    l1, l2 = cache_sizes()
    for size, name in ((l1, "L1d"), (l2, "L2")):
        if size:
            tile = math.sqrt(size / (3 * 8))
            ax.axvline(tile, linestyle=":", color=TEXT_MUTED, linewidth=1.2)
            ax.annotate(f"3·8·T² = {name} ({size // 1024} KiB)\nT ≈ {tile:.0f}", (tile, 1),
                        xycoords=("data", "axes fraction"), xytext=(4, -4),
                        textcoords="offset points", va="top", fontsize=8, color=TEXT_MUTED)
    if not ok.empty:
        log2_ticks(ax, ok["tile_value"])
    ax.set_yscale("log")
    ax.set_xlabel("Tile size T (log2 scale)")
    ax.set_ylabel("Kernel runtime, median (s, log scale)")
    ax.set_title("Blocked method: tile-size sweep (error bars: IQR)")
    style_axes(ax)
    ax.legend(frameon=False)
    save(fig, path)


def _sparse_panels(table):
    sizes = sorted(table["n"].unique())
    cols = min(2, len(sizes))
    rows = math.ceil(len(sizes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.2 * rows), dpi=150,
                             squeeze=False)
    for ax in axes.flat[len(sizes):]:
        ax.set_visible(False)
    return fig, list(zip(sizes, axes.flat))


def plot_sparse_time(table, path):
    table = table.copy()
    table["density"] = pd.to_numeric(table["density_percent"])
    fig, panels = _sparse_panels(table)
    series = [("spgemm_s", "CSR SpGEMM"), ("end_to_end_s", "CSR end-to-end (convert + SpGEMM)"),
              ("dense_s", "dense blocked")]
    for n, ax in panels:
        data = table[table["n"] == n].sort_values("density")
        for index, (column, label) in enumerate(series):
            valid = data[data[column].notna()]
            if valid.empty:
                continue
            ax.plot(valid["density"], valid[column], color=PALETTE[index], linewidth=2,
                    marker=MARKERS[index], markersize=5, label=label)
        failed = data[data["spgemm_status"].isin(["timeout", "over_memory_budget"])
                      | data["spgemm_status"].str.startswith("skipped")]
        for row in failed.itertuples():
            ax.annotate(f"SpGEMM {row.spgemm_status}", (row.density, 1),
                        xycoords=("data", "axes fraction"), rotation=90, va="top",
                        fontsize=7, color=TEXT_MUTED)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sorted(data["density"].unique()))
        ax.set_xticklabels([f"{d:g}" for d in sorted(data["density"].unique())])
        ax.minorticks_off()
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Input density (% non-zeros, log scale)")
        ax.set_ylabel("Median time (s, log scale)")
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Sparse (own CSR) vs dense blocked multiplication time vs density")
    save(fig, path)


def plot_output_density(table, path):
    table = table.copy()
    table["density"] = pd.to_numeric(table["density_percent"])
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for index, n in enumerate(sorted(table["n"].unique())):
        data = table[(table["n"] == n) & table["output_density_percent"].notna()]
        data = data.sort_values("density")
        color = PALETTE[index % 7]
        if not data.empty:
            ax.plot(data["density"], data["output_density_percent"], color=color,
                    marker=MARKERS[index % 7], linewidth=2, markersize=5,
                    label=f"n={n} measured")
        grid = sorted(table[table["n"] == n]["density"].unique())
        expected = [100 * (1 - (1 - (d / 100) ** 2) ** n) for d in grid]
        ax.plot(grid, expected, linestyle="--", color=color, linewidth=1,
                label=f"n={n} expected 1-(1-d²)^n")
    ax.set_xscale("log")
    densities = sorted(table["density"].unique())
    ax.set_xticks(densities)
    ax.set_xticklabels([f"{d:g}" for d in densities])
    ax.minorticks_off()
    ax.set_xlabel("Input density of A and B (% non-zeros, log scale)")
    ax.set_ylabel("Output density of C (% non-zeros)")
    ax.set_title("Fill-in: output density vs input density")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    save(fig, path)


def plot_sparse_memory(summary, path):
    sparse = summary[(summary["experiment"] == "sparse") & (summary["status"] == "ok")].copy()
    sparse["density"] = pd.to_numeric(sparse["density_percent"])
    fig, panels = _sparse_panels(sparse)
    for n, ax in panels:
        data = sparse[sparse["n"] == n]
        for index, method in enumerate(["csr_gustavson", "dense_to_csr", "blocked"]):
            rows = data[data["method"] == method].sort_values("density")
            if rows.empty:
                continue
            ax.plot(rows["density"], rows["peak_wset_mib"], color=PALETTE[index],
                    marker=MARKERS[index], linewidth=2, markersize=5,
                    label=METHOD_LABELS.get(method, method))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Input density (% non-zeros, log scale)")
        ax.set_ylabel("Peak working set (MiB, log scale)")
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Peak working set vs input density")
    save(fig, path)


# ----------------------------------------------------------------------- main


def analyze_file(path, output_dir):
    raw = pd.read_csv(path, dtype={"tile": str, "density_percent": str})
    summary = summarise(raw)
    os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    sections = ["# Summary\n", to_markdown(summary, SUMMARY_COLUMNS)]
    written = ["summary.csv", "summary.md"]
    experiments = set(summary["experiment"])
    if "dense" in experiments or "tiles" in experiments:
        dense = summary[summary["experiment"].isin(["dense", "tiles"])]
        if "tiles" in experiments:
            plot_tile_sweep(dense, os.path.join(output_dir, "tile_sweep.png"))
            written.append("tile_sweep.png")
        if "dense" in experiments:
            plot_dense_runtime(dense, os.path.join(output_dir, "runtime_vs_n.png"))
            plot_dense_memory(dense, os.path.join(output_dir, "memory_vs_n.png"))
            table = speedups(dense)
            plot_speedup(table, os.path.join(output_dir, "speedup_vs_n.png"))
            sections += ["\n# Speedup over ijk\n", to_markdown(table, SPEEDUP_COLUMNS)]
            written += ["runtime_vs_n.png", "memory_vs_n.png", "speedup_vs_n.png"]
    if "sparse" in experiments:
        table = sparse_comparison(summary)
        table.to_csv(os.path.join(output_dir, "sparse_comparison.csv"), index=False)
        sections += ["\n# Sparse vs dense\n",
                     "End-to-end = median dense->CSR conversion (A and B) + median SpGEMM "
                     "(a derived value, not a measured run).\n\n",
                     to_markdown(table, SPARSE_COLUMNS)]
        plot_sparse_time(table, os.path.join(output_dir, "sparse_time_vs_density.png"))
        plot_output_density(table, os.path.join(output_dir, "output_density.png"))
        plot_sparse_memory(summary, os.path.join(output_dir, "sparse_memory_vs_density.png"))
        written += ["sparse_comparison.csv", "sparse_time_vs_density.png",
                    "output_density.png", "sparse_memory_vs_density.png"]
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(sections))
    print(f"{path}: wrote {', '.join(written)} to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Summarise benchmark measurements.")
    parser.add_argument("--input", nargs="+", help="raw_measurements.csv file(s)")
    parser.add_argument("--output-dir", help="default: the directory of each input")
    args = parser.parse_args()
    inputs = args.input or [
        os.path.join(RESULTS_DIR, name, "raw_measurements.csv")
        for name in ("tiles", "dense", "sparse")
        if os.path.exists(os.path.join(RESULTS_DIR, name, "raw_measurements.csv"))
    ]
    if not inputs:
        parser.error("no raw_measurements.csv found; pass --input")
    for path in inputs:
        analyze_file(path, args.output_dir or os.path.dirname(os.path.abspath(path)))


if __name__ == "__main__":
    main()
