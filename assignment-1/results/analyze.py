"""Summarise raw benchmark measurements and draw the runtime and memory plots.

Reads raw_measurements.csv and writes, into the output directory:
    summary.csv, summary.md          median, IQR and repetition count per (language, n)
    runtime_vs_n.png                 log-log runtime vs n with an n^3 reference line
    memory_vs_n.png                  peak working set vs n

Usage:
    .venv\\Scripts\\python results\\analyze.py [--input results/raw_measurements.csv]
                                               [--output-dir results]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed categorical order; colour follows the language, plus a distinct marker
# per language so identity never relies on colour alone.
STYLE = {
    "c": {"label": "C", "color": "#2a78d6", "marker": "o"},
    "java": {"label": "Java", "color": "#eb6834", "marker": "s"},
    "python": {"label": "Python", "color": "#1baf7a", "marker": "^"},
}
TEXT_MUTED = "#5f5e57"
GRID = "#e4e3dc"


def summarise(raw):
    ok = raw[raw["status"] == "ok"].copy()
    ok["time_s"] = ok["time_ns"].astype("int64") / 1e9
    grouped = ok.groupby(["language", "n"])
    summary = grouped.agg(
        repetitions=("time_s", "size"),
        median_s=("time_s", "median"),
        q1_s=("time_s", lambda s: s.quantile(0.25)),
        q3_s=("time_s", lambda s: s.quantile(0.75)),
        min_s=("time_s", "min"),
        max_s=("time_s", "max"),
        peak_wset_mib=("peak_wset_bytes", lambda s: s.astype("float64").median() / 2**20),
    ).reset_index()
    summary["iqr_s"] = summary["q3_s"] - summary["q1_s"]
    summary["status"] = "ok"

    failed = raw[raw["status"] != "ok"][["language", "n", "status", "note"]].drop_duplicates(
        ["language", "n"]
    )
    summary = pd.concat([summary, failed], ignore_index=True)
    columns = ["language", "n", "status", "repetitions", "median_s", "q1_s", "q3_s",
               "iqr_s", "min_s", "max_s", "peak_wset_mib", "note"]
    for column in columns:
        if column not in summary:
            summary[column] = pd.NA
    return summary[columns].sort_values(["language", "n"]).reset_index(drop=True)


def to_markdown(summary):
    header = ("| Language | n | Status | Reps | Median (s) | IQR (s) | Q1 (s) | Q3 (s) "
              "| Peak working set (MiB) | Note |")
    lines = [header, "|" + "---|" * 10]

    def fmt(value, spec):
        return "" if pd.isna(value) else format(value, spec)

    for row in summary.itertuples(index=False):
        lines.append(
            f"| {STYLE.get(row.language, {}).get('label', row.language)} | {row.n} "
            f"| {row.status} | {fmt(row.repetitions, '.0f')} | {fmt(row.median_s, '.6g')} "
            f"| {fmt(row.iqr_s, '.3g')} | {fmt(row.q1_s, '.6g')} | {fmt(row.q3_s, '.6g')} "
            f"| {fmt(row.peak_wset_mib, '.1f')} | {'' if pd.isna(row.note) else row.note} |"
        )
    return "\n".join(lines) + "\n"


def size_ticks(ax, sizes):
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.minorticks_off()


def style_axes(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED)


def plot_runtime(summary, path):
    ok = summary[summary["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for language, style in STYLE.items():
        data = ok[ok["language"] == language].sort_values("n")
        if data.empty:
            continue
        yerr = [data["median_s"] - data["q1_s"], data["q3_s"] - data["median_s"]]
        ax.errorbar(data["n"], data["median_s"], yerr=yerr, color=style["color"],
                    marker=style["marker"], markersize=6, linewidth=2, capsize=3,
                    label=style["label"])
        last = data.iloc[-1]
        ax.annotate(style["label"], (last["n"], last["median_s"]), xytext=(6, 0),
                    textcoords="offset points", va="center", color="#1f1f1c", fontsize=9)
    if not ok.empty:
        # n^3 reference anchored at the largest n of the fastest measured series.
        anchor = ok.sort_values(["n", "median_s"], ascending=[False, True]).iloc[0]
        ns = sorted(ok["n"].unique())
        ref = [anchor["median_s"] * (n / anchor["n"]) ** 3 for n in ns]
        ax.plot(ns, ref, linestyle="--", color=TEXT_MUTED, linewidth=1.2,
                label=r"$\propto n^3$ reference")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    size_ticks(ax, sorted(ok["n"].unique()))
    ax.set_xlabel("Matrix size n (n x n, log2 scale)")
    ax.set_ylabel("Kernel runtime, median of repetitions (s, log scale)")
    ax.set_title("Naive matrix multiplication runtime vs n (error bars: IQR)")
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_memory(summary, path):
    ok = summary[summary["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for language, style in STYLE.items():
        data = ok[ok["language"] == language].sort_values("n")
        if data.empty:
            continue
        ax.plot(data["n"], data["peak_wset_mib"], color=style["color"],
                marker=style["marker"], markersize=6, linewidth=2, label=style["label"])
        last = data.iloc[-1]
        ax.annotate(style["label"], (last["n"], last["peak_wset_mib"]), xytext=(6, 0),
                    textcoords="offset points", va="center", color="#1f1f1c", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    size_ticks(ax, sorted(ok["n"].unique()))
    ax.set_xlabel("Matrix size n (n x n, log2 scale)")
    ax.set_ylabel("Peak working set of the process (MiB, log scale)")
    ax.set_title("Peak working set vs n (includes runtime and input matrices)")
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Summarise benchmark measurements.")
    parser.add_argument("--input", default=os.path.join(RESULTS_DIR, "raw_measurements.csv"))
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    summary = summarise(raw)
    os.makedirs(args.output_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    with open(os.path.join(args.output_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write(to_markdown(summary))
    plot_runtime(summary, os.path.join(args.output_dir, "runtime_vs_n.png"))
    plot_memory(summary, os.path.join(args.output_dir, "memory_vs_n.png"))
    print(to_markdown(summary))
    print(f"Wrote summary.csv, summary.md, runtime_vs_n.png, memory_vs_n.png "
          f"to {args.output_dir}")


if __name__ == "__main__":
    main()
