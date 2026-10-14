"""Cross-check the three implementations against NumPy's A @ B.

Each runner is executed in `verify` mode and writes its product to
results/verify/{language}_n{n}.bin; the products are compared with
np.allclose(rtol=1e-9, atol=1e-12). NumPy is used only as the reference.

Usage:
    .venv\\Scripts\\python cross_check.py [--sizes 64 128]
"""

import argparse
import os
import subprocess
import sys

import numpy as np

import bench_common

sys.path.insert(0, os.path.join(bench_common.ROOT, "data"))
import generate  # noqa: E402

VERIFY_DIR_REL = os.path.join("results", "verify")
RTOL = 1e-9
ATOL = 1e-12


def main():
    parser = argparse.ArgumentParser(description="Cross-check implementations against NumPy.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--languages", nargs="+", default=list(bench_common.LANGUAGES))
    args = parser.parse_args()

    config = bench_common.load_config()
    bench_common.build_c(config)
    bench_common.build_java(config)
    java_exe = bench_common.java_executable()
    os.makedirs(os.path.join(bench_common.ROOT, VERIFY_DIR_REL), exist_ok=True)

    all_ok = True
    for n in args.sizes:
        generate.ensure_inputs(n)
        reference = generate.load_matrix("A", n) @ generate.load_matrix("B", n)
        for language in args.languages:
            out_rel = os.path.join(VERIFY_DIR_REL, f"{language}_n{n}.bin")
            cmd = bench_common.child_command(
                language, config, ["verify", n, bench_common.DATA_DIR_REL, out_rel], java_exe
            )
            subprocess.run(cmd, cwd=bench_common.ROOT, check=True)
            product = np.fromfile(os.path.join(bench_common.ROOT, out_rel), dtype="<f8")
            product = product.reshape(n, n)
            abs_err = np.abs(product - reference)
            rel_err = abs_err / np.maximum(np.abs(reference), np.finfo(float).tiny)
            ok = bool(np.allclose(product, reference, rtol=RTOL, atol=ATOL))
            all_ok &= ok
            print(f"{language:>6} n={n:<5} max_abs_err={abs_err.max():.3e} "
                  f"max_rel_err={rel_err.max():.3e} {'OK' if ok else 'MISMATCH'}")
    print("Cross-check PASSED" if all_ok else "Cross-check FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
