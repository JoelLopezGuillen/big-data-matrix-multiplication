"""Tests for the sparse input generator (data/generate_sparse.py)."""

import json

import numpy as np
import pytest
import scipy.sparse as sp

import generate_sparse as gs


@pytest.mark.parametrize("percent, tag", [(0.1, "0.1"), (1, "1"), (5.0, "5"), (50, "50")])
def test_density_tag(percent, tag):
    assert gs.density_tag(percent) == tag


@pytest.mark.parametrize("n, percent", [(10, 0.1), (100, 1), (100, 25), (37, 50), (5, 100)])
def test_nnz_is_exact_rounded_density(n, percent):
    assert gs.target_nnz(n, percent) == round(percent / 100 * n * n)
    indptr, indices, data = gs.generate_csr(42, n, percent)
    assert indptr[-1] == len(indices) == len(data) == gs.target_nnz(n, percent)


def test_csr_structure_is_valid_and_values_in_range():
    n = 200
    indptr, indices, data = gs.generate_csr(42, n, 10)
    assert indptr.dtype == np.int64 and indices.dtype == np.int32 and data.dtype == np.float64
    assert indptr[0] == 0 and len(indptr) == n + 1 and np.all(np.diff(indptr) >= 0)
    for i in range(n):
        row = indices[indptr[i]:indptr[i + 1]]
        assert np.all(np.diff(row) > 0), "columns must be strictly increasing"
    assert np.all((indices >= 0) & (indices < n))
    assert np.all(data != 0.0)
    assert np.all((data >= -1.0) & (data < 1.0))


def test_generation_is_deterministic_and_seeds_differ():
    first = gs.generate_csr(42, 50, 5)
    second = gs.generate_csr(42, 50, 5)
    other = gs.generate_csr(43, 50, 5)
    for x, y in zip(first, second):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(first[1], other[1]) or not np.array_equal(first[2], other[2])


def test_csr_file_round_trip(tmp_path):
    n = 30
    indptr, indices, data = gs.generate_csr(43, n, 20)
    path = tmp_path / "m.csr"
    gs.write_csr(path, n, indptr, indices, data)
    header = np.fromfile(path, dtype="<i8", count=3)
    assert list(header) == [n, n, len(data)]
    size = 3 * 8 + (n + 1) * 8 + len(data) * 4 + len(data) * 8
    assert path.stat().st_size == size
    n_read, indptr2, indices2, data2 = gs.read_csr(path)
    assert n_read == n
    np.testing.assert_array_equal(indptr, indptr2)
    np.testing.assert_array_equal(indices, indices2)
    np.testing.assert_array_equal(data, data2)


def test_ensure_writes_matching_dense_and_manifest(tmp_path):
    n, percent = 40, 25
    entry = gs.ensure_sparse_inputs(n, percent, data_dir=tmp_path, with_dense=True)
    for name in ("A", "B"):
        n_read, indptr, indices, data = gs.read_csr(gs.csr_path(name, n, percent, tmp_path))
        dense = np.fromfile(gs.dense_path(name, n, percent, tmp_path), dtype="<f8").reshape(n, n)
        csr = sp.csr_matrix((data, indices, indptr), shape=(n, n))
        np.testing.assert_array_equal(csr.toarray(), dense)
        assert np.count_nonzero(dense) == gs.target_nnz(n, percent)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    key = gs.manifest_key(n, percent)
    assert manifest["entries"][key] == entry
    assert entry["nnz_a"] == entry["nnz_b"] == gs.target_nnz(n, percent)
    assert entry["density_percent"] == percent
    assert entry["sparsity_percent"] == 100 - percent
    assert entry["seeds"] == {"A": 42, "B": 43}
    assert entry["with_dense"] is True


def test_ensure_without_dense_writes_only_csr(tmp_path):
    gs.ensure_sparse_inputs(20, 5, data_dir=tmp_path, with_dense=False)
    assert gs.csr_path("A", 20, 5, tmp_path).exists()
    assert not gs.dense_path("A", 20, 5, tmp_path).exists()
