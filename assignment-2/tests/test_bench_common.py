"""Tests for the pure helpers of the benchmark harness (budgets, skip tracking)."""

import math

import pytest

import bench_common as bc


def test_dense_memory_estimate_is_three_matrices():
    assert bc.dense_memory_estimate_bytes(1000) == 3 * 8 * 1000 * 1000


def test_csr_bytes_counts_header_indptr_indices_data():
    assert bc.csr_bytes(10, 7) == 3 * 8 + 11 * 8 + 7 * 4 + 7 * 8


@pytest.mark.parametrize("n, da, db", [(1, 0.5, 0.5), (10, 0.1, 0.2), (1000, 0.01, 0.01)])
def test_expected_output_nnz_formula(n, da, db):
    expected = n * n * (1 - (1 - da * db) ** n)
    assert math.isclose(bc.expected_output_nnz(n, da, db), expected)


def test_expected_output_nnz_full_density_is_dense():
    assert bc.expected_output_nnz(50, 1.0, 1.0) == 50 * 50


def test_spgemm_estimate_includes_inputs_output_and_workspace():
    n, nnz = 100, 500
    est_c = bc.expected_output_nnz(n, nnz / n**2, nnz / n**2)
    expected = 2 * bc.csr_bytes(n, nnz) + bc.csr_bytes(n, math.ceil(est_c)) + n * (4 + 8)
    assert bc.spgemm_memory_estimate_bytes(n, nnz, nnz) == expected


def test_conversion_estimate_includes_dense_inputs_and_csr_outputs():
    n, nnz = 100, 500
    expected = 2 * 8 * n * n + 2 * bc.csr_bytes(n, nnz)
    assert bc.conversion_memory_estimate_bytes(n, nnz, nnz) == expected


def test_series_tracker_skips_larger_sizes_after_timeout():
    tracker = bc.SeriesTracker()
    key = ("ikj", None)
    assert tracker.skip_status(key) is None
    tracker.record(key, 512, "ok")
    assert tracker.skip_status(key) is None
    tracker.record(key, 1024, "timeout")
    assert tracker.skip_status(key) == ("skipped_after_timeout", 1024)
    assert tracker.skip_status(("ikj", 64)) is None


def test_series_tracker_other_failures():
    tracker = bc.SeriesTracker()
    tracker.record("s", 10, "oom")
    assert tracker.skip_status("s") == ("skipped_after_oom", 10)


def test_series_tracker_over_budget_does_not_block_other_series():
    tracker = bc.SeriesTracker()
    tracker.record("a", 10, "over_memory_budget")
    assert tracker.skip_status("a") == ("skipped_after_over_memory_budget", 10)
    assert tracker.skip_status("b") is None
