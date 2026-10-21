"""Tests for the summary logic of results/analyze.py."""

import pandas as pd
import pytest

import analyze


def raw_rows(rows):
    defaults = {"experiment": "dense", "representation": "dense", "phase": "multiply",
                "tile": "", "density_percent": "", "nnz_a": "", "nnz_b": "", "nnz_c": "",
                "output_density_percent": "", "peak_working_set_bytes": 2**20, "note": "",
                "status": "ok"}
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    return frame


def test_summary_median_iqr_and_count():
    raw = raw_rows([{"method": "ijk", "n": 64, "time_ns": t, "repetition": i}
                    for i, t in enumerate([1e9, 2e9, 3e9, 4e9, 5e9], start=1)])
    summary = analyze.summarise(raw)
    row = summary.iloc[0]
    assert row["repetitions"] == 5
    assert row["median_s"] == pytest.approx(3.0)
    assert row["q1_s"] == pytest.approx(2.0)
    assert row["q3_s"] == pytest.approx(4.0)
    assert row["iqr_s"] == pytest.approx(2.0)
    assert row["peak_wset_mib"] == pytest.approx(1.0)


def test_summary_keeps_failed_configurations_without_times():
    raw = raw_rows([
        {"method": "ijk", "n": 64, "time_ns": 1e9, "repetition": 1},
        {"method": "ijk", "n": 4096, "time_ns": "", "repetition": "", "status": "timeout",
         "note": "a single kernel execution exceeded 60 s"},
    ])
    summary = analyze.summarise(raw)
    failed = summary[summary["n"] == 4096].iloc[0]
    assert failed["status"] == "timeout"
    assert pd.isna(failed["median_s"])
    assert "60 s" in failed["note"]


def test_summary_separates_tiles():
    raw = raw_rows([
        {"method": "blocked", "n": 1024, "tile": 32, "time_ns": 2e9, "repetition": 1},
        {"method": "blocked", "n": 1024, "tile": 64, "time_ns": 1e9, "repetition": 1},
    ])
    summary = analyze.summarise(raw)
    assert sorted(summary["tile"].tolist()) == ["32", "64"]


def test_speedup_vs_baseline():
    raw = raw_rows([
        {"method": "ijk", "n": 512, "time_ns": 4e9, "repetition": 1},
        {"method": "ikj", "n": 512, "time_ns": 1e9, "repetition": 1},
        {"method": "blocked", "n": 512, "tile": 64, "time_ns": 2e9, "repetition": 1},
    ])
    speed = analyze.speedups(analyze.summarise(raw), baseline="ijk")
    by_method = dict(zip(speed["method"], speed["speedup"]))
    assert by_method["ikj"] == pytest.approx(4.0)
    assert by_method["blocked"] == pytest.approx(2.0)
    assert by_method["ijk"] == pytest.approx(1.0)


def test_sparse_end_to_end_is_conversion_plus_spgemm():
    base = {"experiment": "sparse", "n": 1000, "density_percent": 1, "nnz_a": 10000,
            "nnz_b": 10000, "repetition": 1}
    raw = raw_rows([
        {**base, "method": "csr_gustavson", "representation": "csr", "time_ns": 3e9,
         "nnz_c": 95000, "output_density_percent": 9.5},
        {**base, "method": "dense_to_csr", "representation": "csr", "phase": "convert",
         "time_ns": 1e9},
        {**base, "method": "blocked", "tile": 64, "time_ns": 8e9},
    ])
    table = analyze.sparse_comparison(analyze.summarise(raw))
    row = table.iloc[0]
    assert row["spgemm_s"] == pytest.approx(3.0)
    assert row["convert_s"] == pytest.approx(1.0)
    assert row["end_to_end_s"] == pytest.approx(4.0)
    assert row["dense_s"] == pytest.approx(8.0)
    assert row["dense_over_end_to_end"] == pytest.approx(2.0)
    assert row["nnz_c"] == 95000
