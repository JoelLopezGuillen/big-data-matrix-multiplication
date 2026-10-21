"""Cross-check the C kernels against library references (references ONLY).

Dense:  every method (ijk, ikj, blocked with several tiles, including tiles that
        do not divide n) runs in `verify` mode on data/A_n{n}.bin, B_n{n}.bin and
        its product is compared with NumPy A @ B.
Sparse: on generated sparse inputs (data/sparse_smoke/),
        * `convert-verify` (own dense->CSR) must reproduce the generated CSR file
          exactly (indptr, indices and data);
        * `sparse-verify` (own Gustavson SpGEMM) is compared with
          scipy.sparse A @ B: values (as dense arrays), the CSR structure
          (sorted, in-range columns), the structural nnz (must equal the nnz of
          the boolean product pattern) and the numerically non-zero entries
          (SciPy drops entries that cancel to exactly 0.0, the C code keeps them
          as structural entries; any disagreement is only allowed on entries
          whose magnitude is <= ATOL).
Values are compared with np.allclose(rtol=1e-9, atol=1e-12), as in Assignment 1.

Usage:
    .venv\\Scripts\\python cross_check.py [--sizes 64 100 130] [--sparse-sizes 50 200]
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import scipy.sparse as sp

import bench_common

sys.path.insert(0, os.path.join(bench_common.ROOT, "data"))
import generate  # noqa: E402
import generate_sparse  # noqa: E402

VERIFY_DIR = os.path.join(bench_common.ROOT, "results", "verify")
SPARSE_DATA_DIR = os.path.join(bench_common.ROOT, "data", "sparse_smoke")
RTOL = 1e-9
ATOL = 1e-12
DENSITIES = [0.1, 1, 5, 10, 25, 50]


def run_runner(*args):
    subprocess.run(bench_common.runner_command(*args), cwd=bench_common.ROOT, check=True)


def error_report(actual, reference):
    abs_err = np.abs(actual - reference)
    rel_err = abs_err / np.maximum(np.abs(reference), np.finfo(float).tiny)
    ok = bool(np.allclose(actual, reference, rtol=RTOL, atol=ATOL))
    return ok, abs_err.max(initial=0.0), rel_err.max(initial=0.0)


def check_dense(sizes, tiles):
    all_ok = True
    for n in sizes:
        generate.ensure_inputs(n)
        a_rel = bench_common.rel(generate.matrix_path("A", n))
        b_rel = bench_common.rel(generate.matrix_path("B", n))
        reference = generate.load_matrix("A", n) @ generate.load_matrix("B", n)
        variants = [("ijk", 0), ("ikj", 0)] + [("blocked", tile) for tile in tiles]
        for method, tile in variants:
            out = os.path.join(VERIFY_DIR, f"{method}_t{tile}_n{n}.bin")
            run_runner("verify", method, n, a_rel, b_rel, bench_common.rel(out), tile)
            product = np.fromfile(out, dtype="<f8").reshape(n, n)
            ok, max_abs, max_rel = error_report(product, reference)
            all_ok &= ok
            label = method if method != "blocked" else f"blocked(tile={tile})"
            print(f"dense  {label:<18} n={n:<5} max_abs_err={max_abs:.3e} "
                  f"max_rel_err={max_rel:.3e} {'OK' if ok else 'MISMATCH'}")
    return all_ok


def csr_from_file(path):
    n, indptr, indices, data = generate_sparse.read_csr(path)
    return n, indptr, indices, data, sp.csr_matrix((data, indices, indptr), shape=(n, n))


def structure_ok(n, indptr, indices):
    if indptr[0] != 0 or np.any(np.diff(indptr) < 0):
        return False
    if len(indices) and (indices.min() < 0 or indices.max() >= n):
        return False
    for i in range(n):
        if np.any(np.diff(indices[indptr[i]:indptr[i + 1]]) <= 0):
            return False
    return True


def check_conversion(n, percent):
    """Own dense->CSR conversion must reproduce the generated CSR exactly."""
    out = os.path.join(VERIFY_DIR, f"convert_A_n{n}_d{generate_sparse.density_tag(percent)}.csr")
    dense_rel = bench_common.rel(generate_sparse.dense_path("A", n, percent, SPARSE_DATA_DIR))
    run_runner("convert-verify", n, dense_rel, bench_common.rel(out))
    _, ip1, ix1, d1 = generate_sparse.read_csr(out)
    _, ip2, ix2, d2 = generate_sparse.read_csr(
        generate_sparse.csr_path("A", n, percent, SPARSE_DATA_DIR))
    return (np.array_equal(ip1, ip2) and np.array_equal(ix1, ix2)
            and np.array_equal(d1, d2))


def check_sparse(sizes):
    all_ok = True
    for n in sizes:
        for percent in DENSITIES:
            generate_sparse.ensure_sparse_inputs(n, percent, SPARSE_DATA_DIR, with_dense=True)
            a_path = generate_sparse.csr_path("A", n, percent, SPARSE_DATA_DIR)
            b_path = generate_sparse.csr_path("B", n, percent, SPARSE_DATA_DIR)
            out = os.path.join(VERIFY_DIR, f"spgemm_n{n}_d{generate_sparse.density_tag(percent)}.csr")
            run_runner("sparse-verify", bench_common.rel(a_path), bench_common.rel(b_path),
                       bench_common.rel(out))
            *_, a = csr_from_file(a_path)
            *_, b = csr_from_file(b_path)
            n_c, indptr, indices, data, mine = csr_from_file(out)

            reference = (a @ b).tocsr()
            reference.eliminate_zeros()
            pattern = (a.astype(bool).astype(np.int64) @ b.astype(bool).astype(np.int64))
            pattern.eliminate_zeros()

            mine_dense = mine.toarray()
            ref_dense = reference.toarray()
            values_ok, max_abs, max_rel = error_report(mine_dense, ref_dense)
            struct_ok = n_c == n and structure_ok(n, indptr, indices)
            nnz_ok = mine.nnz == pattern.nnz
            numeric_nnz = int(np.count_nonzero(data))
            disagree = (mine_dense != 0) ^ (ref_dense != 0)
            zeros_ok = bool(np.all(np.abs(mine_dense[disagree]) <= ATOL)
                            and np.all(np.abs(ref_dense[disagree]) <= ATOL))
            conv_ok = check_conversion(n, percent)
            ok = values_ok and struct_ok and nnz_ok and zeros_ok and conv_ok
            all_ok &= ok
            print(f"sparse n={n:<4} density={generate_sparse.density_tag(percent):>4}% "
                  f"nnz_a={a.nnz:<6} nnz_b={b.nnz:<6} nnz_c={mine.nnz:<6} "
                  f"(pattern {pattern.nnz}, numeric {numeric_nnz}, scipy {reference.nnz}) "
                  f"max_abs_err={max_abs:.3e} max_rel_err={max_rel:.3e} "
                  f"structure={'OK' if struct_ok else 'BAD'} "
                  f"conversion={'OK' if conv_ok else 'BAD'} {'OK' if ok else 'MISMATCH'}")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Cross-check the C kernels against NumPy/SciPy.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 100, 130])
    parser.add_argument("--tiles", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument("--sparse-sizes", type=int, nargs="+", default=[50, 200])
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    if not args.skip_build:
        bench_common.build_c()
    os.makedirs(VERIFY_DIR, exist_ok=True)
    dense_ok = check_dense(args.sizes, args.tiles)
    sparse_ok = check_sparse(args.sparse_sizes)
    ok = dense_ok and sparse_ok
    print("Cross-check PASSED" if ok else "Cross-check FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
