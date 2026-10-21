/* Correctness tests for the dense kernels (ijk, ikj, blocked) and the CSR code
 * (dense->CSR conversion, CSR x CSR SpGEMM, binary I/O).
 * Exits with a non-zero status if any check fails. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "csr.h"
#include "matmul.h"

static int failures = 0;

static void fail(const char *name, const char *what) {
    printf("FAIL %s: %s\n", name, what);
    failures++;
}

static void pass(const char *name) { printf("PASS %s\n", name); }

/* Compares two n_rows x n_cols row-major matrices with |a-e| <= atol + rtol|e|. */
static int matrices_close(const char *name, const double *actual,
                          const double *expected, size_t count, double rtol,
                          double atol) {
    if (actual == NULL) {
        fail(name, "result is NULL");
        return 0;
    }
    for (size_t idx = 0; idx < count; idx++) {
        double diff = fabs(actual[idx] - expected[idx]);
        if (diff > atol + rtol * fabs(expected[idx])) {
            printf("FAIL %s: element %zu is %.17g, expected %.17g\n", name, idx,
                   actual[idx], expected[idx]);
            failures++;
            return 0;
        }
    }
    return 1;
}

static void check_matrix(const char *name, const double *actual,
                         const double *expected, size_t n, double tol) {
    if (matrices_close(name, actual, expected, n * n, 0.0, tol)) {
        pass(name);
    }
}

/* Deterministic pseudo-random values in [-1, 1) (LCG, test-only). */
static uint64_t lcg_state = 12345;
static double next_uniform(void) {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(lcg_state >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}

static double *random_matrix(size_t n, double density) {
    double *m = malloc(n * n * sizeof(double));
    for (size_t idx = 0; idx < n * n; idx++) {
        double keep = (next_uniform() + 1.0) / 2.0;
        double value = next_uniform();
        m[idx] = (keep < density && value != 0.0) ? value : 0.0;
    }
    return m;
}

/* ------------------------------------------------------------------ dense */

typedef double *(*dense_kernel)(const double *, const double *, size_t);

static double *blocked64(const double *a, const double *b, size_t n) {
    return matmul_blocked(a, b, n, 64);
}

static double *blocked3(const double *a, const double *b, size_t n) {
    return matmul_blocked(a, b, n, 3);
}

static const struct {
    const char *name;
    dense_kernel kernel;
} KERNELS[] = {
    {"ijk", matmul_ijk},
    {"ikj", matmul_ikj},
    {"blocked(tile=64)", blocked64},
    {"blocked(tile=3)", blocked3},
};
#define N_KERNELS (sizeof(KERNELS) / sizeof(KERNELS[0]))

static void test_identity(void) {
    const size_t n = 5;
    double a[25], id[25];
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            a[i * n + j] = (double)(3 * i + j) - 2.5;
            id[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (size_t k = 0; k < N_KERNELS; k++) {
        char name[96];
        snprintf(name, sizeof(name), "%s identity (A*I == A, I*A == A)",
                 KERNELS[k].name);
        double *c1 = KERNELS[k].kernel(a, id, n);
        double *c2 = KERNELS[k].kernel(id, a, n);
        if (matrices_close(name, c1, a, n * n, 0.0, 0.0) &&
            matrices_close(name, c2, a, n * n, 0.0, 0.0)) {
            pass(name);
        }
        free(c1);
        free(c2);
    }
}

static void test_zero(void) {
    const size_t n = 4;
    double a[16], z[16] = {0};
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            a[i * n + j] = (double)i - (double)j;
        }
    }
    for (size_t k = 0; k < N_KERNELS; k++) {
        char name[96];
        snprintf(name, sizeof(name), "%s zero (A*0 == 0, 0*A == 0)",
                 KERNELS[k].name);
        double *c1 = KERNELS[k].kernel(a, z, n);
        double *c2 = KERNELS[k].kernel(z, a, n);
        if (matrices_close(name, c1, z, n * n, 0.0, 0.0) &&
            matrices_close(name, c2, z, n * n, 0.0, 0.0)) {
            pass(name);
        }
        free(c1);
        free(c2);
    }
}

static void test_known_small(void) {
    const double a2[4] = {1.0, 2.0, 3.0, 4.0};
    const double b2[4] = {5.0, 6.0, 7.0, 8.0};
    const double e2[4] = {19.0, 22.0, 43.0, 50.0};
    const double a3[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0};
    const double b3[9] = {-1.0, 0.5, 2.0, 3.0, -2.0, 1.0, 0.25, 4.0, -3.0};
    const double e3[9] = {5.75, 8.5, -5.0, 12.5, 16.0, -5.0, 19.5, 27.5, -8.0};
    const double one_a = 3.0, one_b = -2.5, one_e = -7.5;
    for (size_t k = 0; k < N_KERNELS; k++) {
        char name[96];
        double *c;
        snprintf(name, sizeof(name), "%s known 1x1", KERNELS[k].name);
        c = KERNELS[k].kernel(&one_a, &one_b, 1);
        check_matrix(name, c, &one_e, 1, 0.0);
        free(c);
        snprintf(name, sizeof(name), "%s known 2x2", KERNELS[k].name);
        c = KERNELS[k].kernel(a2, b2, 2);
        check_matrix(name, c, e2, 2, 0.0);
        free(c);
        snprintf(name, sizeof(name), "%s known 3x3", KERNELS[k].name);
        c = KERNELS[k].kernel(a3, b3, 3);
        check_matrix(name, c, e3, 3, 1e-12);
        free(c);
    }
}

/* All methods agree with the ijk baseline, including sizes that are not a
 * multiple of the tile (partial edge tiles). */
static void test_methods_agree(void) {
    const size_t sizes[] = {1, 7, 63, 64, 65, 130};
    const size_t tiles[] = {16, 64, 128};
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        size_t n = sizes[s];
        double *a = random_matrix(n, 1.0);
        double *b = random_matrix(n, 1.0);
        double *ref = matmul_ijk(a, b, n);
        char name[96];
        snprintf(name, sizeof(name), "ikj == ijk (n=%zu)", n);
        double *c = matmul_ikj(a, b, n);
        if (matrices_close(name, c, ref, n * n, 1e-12, 1e-12)) {
            pass(name);
        }
        free(c);
        for (size_t t = 0; t < sizeof(tiles) / sizeof(tiles[0]); t++) {
            snprintf(name, sizeof(name), "blocked(tile=%zu) == ijk (n=%zu)",
                     tiles[t], n);
            c = matmul_blocked(a, b, n, tiles[t]);
            if (matrices_close(name, c, ref, n * n, 1e-12, 1e-12)) {
                pass(name);
            }
            free(c);
        }
        free(a);
        free(b);
        free(ref);
    }
}

static void test_blocked_rejects_zero_tile(void) {
    const double a[1] = {1.0};
    double *c = matmul_blocked(a, a, 1, 0);
    if (c != NULL) {
        fail("blocked(tile=0) returns NULL", "returned a matrix");
        free(c);
    } else {
        pass("blocked(tile=0) returns NULL");
    }
}

static void test_checksum(void) {
    const double c[4] = {19.0, 22.0, 43.0, 50.0};
    if (checksum(c, 2) != 134.0) {
        fail("checksum", "wrong sum");
    } else {
        pass("checksum");
    }
}

/* -------------------------------------------------------------------- CSR */

static int csr_well_formed(const char *name, const csr_matrix *m) {
    if (m->indptr == NULL || m->indptr[0] != 0 ||
        m->indptr[m->n_rows] != m->nnz) {
        fail(name, "bad indptr bounds");
        return 0;
    }
    for (int64_t i = 0; i < m->n_rows; i++) {
        if (m->indptr[i + 1] < m->indptr[i]) {
            fail(name, "indptr not monotone");
            return 0;
        }
        for (int64_t p = m->indptr[i]; p < m->indptr[i + 1]; p++) {
            if (m->indices[p] < 0 || m->indices[p] >= m->n_cols) {
                fail(name, "column index out of range");
                return 0;
            }
            if (p > m->indptr[i] && m->indices[p] <= m->indices[p - 1]) {
                fail(name, "column indices not strictly increasing");
                return 0;
            }
        }
    }
    return 1;
}

static void test_csr_conversion_known(void) {
    /* [[1 0 2]
     *  [0 0 0]   <- empty row
     *  [0 3 0]] */
    const double dense[9] = {1, 0, 2, 0, 0, 0, 0, 3, 0};
    const int64_t indptr[4] = {0, 2, 2, 3};
    const int32_t indices[3] = {0, 2, 1};
    const double data[3] = {1, 2, 3};
    csr_matrix m;
    const char *name = "csr_from_dense known 3x3 with empty row";
    if (csr_from_dense(dense, 3, 3, &m) != CSR_OK) {
        fail(name, "conversion failed");
        return;
    }
    int ok = csr_well_formed(name, &m) && m.nnz == 3;
    for (int i = 0; ok && i < 4; i++) {
        ok = m.indptr[i] == indptr[i];
    }
    for (int p = 0; ok && p < 3; p++) {
        ok = m.indices[p] == indices[p] && m.data[p] == data[p];
    }
    if (ok) {
        pass(name);
    } else {
        fail(name, "unexpected CSR arrays");
    }
    csr_free(&m);
}

static void test_csr_round_trip(void) {
    const size_t n = 37;
    const double densities[] = {0.0, 0.05, 0.3, 1.0};
    for (size_t d = 0; d < 4; d++) {
        char name[96];
        snprintf(name, sizeof(name), "csr round trip dense->csr->dense (density=%g)",
                 densities[d]);
        double *dense = random_matrix(n, densities[d]);
        csr_matrix m;
        if (csr_from_dense(dense, (int64_t)n, (int64_t)n, &m) != CSR_OK) {
            fail(name, "conversion failed");
            free(dense);
            continue;
        }
        int64_t expected_nnz = 0;
        for (size_t idx = 0; idx < n * n; idx++) {
            expected_nnz += dense[idx] != 0.0;
        }
        double *back = csr_to_dense(&m);
        if (csr_well_formed(name, &m) && m.nnz == expected_nnz &&
            matrices_close(name, back, dense, n * n, 0.0, 0.0)) {
            pass(name);
        } else if (m.nnz != expected_nnz) {
            fail(name, "wrong nnz");
        }
        free(back);
        free(dense);
        csr_free(&m);
    }
}

static void test_csr_all_zero(void) {
    const size_t n = 6;
    double z[36] = {0};
    double a[36];
    for (size_t idx = 0; idx < n * n; idx++) {
        a[idx] = (double)idx - 10.0;
    }
    csr_matrix cz, ca, c1, c2;
    const char *name = "csr all-zero matrix (nnz=0, A*0 == 0, 0*A == 0)";
    if (csr_from_dense(z, 6, 6, &cz) != CSR_OK ||
        csr_from_dense(a, 6, 6, &ca) != CSR_OK ||
        csr_spgemm(&ca, &cz, &c1) != CSR_OK ||
        csr_spgemm(&cz, &ca, &c2) != CSR_OK) {
        fail(name, "operation failed");
        return;
    }
    if (cz.nnz == 0 && c1.nnz == 0 && c2.nnz == 0 && csr_well_formed(name, &c1) &&
        csr_well_formed(name, &c2)) {
        pass(name);
    } else {
        fail(name, "expected empty product");
    }
    csr_free(&cz);
    csr_free(&ca);
    csr_free(&c1);
    csr_free(&c2);
}

static void test_csr_identity(void) {
    const size_t n = 9;
    double *a = random_matrix(n, 0.4);
    double id[81] = {0};
    for (size_t i = 0; i < n; i++) {
        id[i * n + i] = 1.0;
    }
    csr_matrix ca, ci, c1, c2;
    const char *name = "csr identity (A*I == A, I*A == A, same structure)";
    if (csr_from_dense(a, 9, 9, &ca) != CSR_OK ||
        csr_from_dense(id, 9, 9, &ci) != CSR_OK ||
        csr_spgemm(&ca, &ci, &c1) != CSR_OK ||
        csr_spgemm(&ci, &ca, &c2) != CSR_OK) {
        fail(name, "operation failed");
        free(a);
        return;
    }
    double *d1 = csr_to_dense(&c1);
    double *d2 = csr_to_dense(&c2);
    if (ci.nnz == (int64_t)n && c1.nnz == ca.nnz && c2.nnz == ca.nnz &&
        csr_well_formed(name, &c1) && csr_well_formed(name, &c2) &&
        matrices_close(name, d1, a, n * n, 0.0, 0.0) &&
        matrices_close(name, d2, a, n * n, 0.0, 0.0)) {
        pass(name);
    } else if (c1.nnz != ca.nnz || c2.nnz != ca.nnz) {
        fail(name, "wrong nnz");
    }
    free(d1);
    free(d2);
    free(a);
    csr_free(&ca);
    csr_free(&ci);
    csr_free(&c1);
    csr_free(&c2);
}

static void test_csr_known_product(void) {
    /* A = [[1 0 2]    B = [[0 4 0]    A*B = [[0 4 6]
     *      [0 0 0]         [5 0 0]           [0 0 0]
     *      [0 3 0]]        [0 0 3]]          [15 0 0]] */
    const double a[9] = {1, 0, 2, 0, 0, 0, 0, 3, 0};
    const double b[9] = {0, 4, 0, 5, 0, 0, 0, 0, 3};
    const double e[9] = {0, 4, 6, 0, 0, 0, 15, 0, 0};
    csr_matrix ca, cb, cc;
    const char *name = "csr known 3x3 product";
    if (csr_from_dense(a, 3, 3, &ca) != CSR_OK ||
        csr_from_dense(b, 3, 3, &cb) != CSR_OK ||
        csr_spgemm(&ca, &cb, &cc) != CSR_OK) {
        fail(name, "operation failed");
        return;
    }
    double *dc = csr_to_dense(&cc);
    if (cc.nnz == 3 && csr_well_formed(name, &cc) &&
        matrices_close(name, dc, e, 9, 0.0, 0.0)) {
        pass(name);
    } else if (cc.nnz != 3) {
        fail(name, "wrong nnz");
    }
    free(dc);
    csr_free(&ca);
    csr_free(&cb);
    csr_free(&cc);
}

/* Structural entries that cancel to 0.0 are kept: [1 1] x [1 -1]^T = [0]. */
static void test_csr_keeps_cancellation(void) {
    const double a[4] = {1, 1, 0, 0};
    const double b[4] = {1, 0, -1, 0};
    csr_matrix ca, cb, cc;
    const char *name = "csr keeps structural zero from cancellation";
    if (csr_from_dense(a, 2, 2, &ca) != CSR_OK ||
        csr_from_dense(b, 2, 2, &cb) != CSR_OK ||
        csr_spgemm(&ca, &cb, &cc) != CSR_OK) {
        fail(name, "operation failed");
        return;
    }
    if (cc.nnz == 1 && cc.indices[0] == 0 && cc.data[0] == 0.0 &&
        csr_well_formed(name, &cc)) {
        pass(name);
    } else {
        fail(name, "expected one stored 0.0 at (0,0)");
    }
    csr_free(&ca);
    csr_free(&cb);
    csr_free(&cc);
}

static void test_csr_shape_mismatch(void) {
    const double a[6] = {1, 2, 3, 4, 5, 6};
    csr_matrix ca, cc;
    const char *name = "csr_spgemm rejects mismatched shapes";
    csr_from_dense(a, 2, 3, &ca);
    if (csr_spgemm(&ca, &ca, &cc) == CSR_ERR_SHAPE) {
        pass(name);
    } else {
        fail(name, "expected CSR_ERR_SHAPE");
        csr_free(&cc);
    }
    csr_free(&ca);
}

/* SpGEMM agrees with the dense ijk baseline on random sparse matrices, from
 * very sparse to fully dense rows (the latter exercises the sorted-scan path). */
static void test_csr_matches_dense(void) {
    const size_t sizes[] = {1, 8, 50, 120};
    const double densities[] = {0.01, 0.1, 0.5, 1.0};
    for (size_t s = 0; s < 4; s++) {
        for (size_t d = 0; d < 4; d++) {
            size_t n = sizes[s];
            char name[96];
            snprintf(name, sizeof(name), "csr_spgemm == dense ijk (n=%zu, density=%g)",
                     n, densities[d]);
            double *a = random_matrix(n, densities[d]);
            double *b = random_matrix(n, densities[d]);
            double *ref = matmul_ijk(a, b, n);
            csr_matrix ca, cb, cc;
            if (csr_from_dense(a, (int64_t)n, (int64_t)n, &ca) != CSR_OK ||
                csr_from_dense(b, (int64_t)n, (int64_t)n, &cb) != CSR_OK ||
                csr_spgemm(&ca, &cb, &cc) != CSR_OK) {
                fail(name, "operation failed");
            } else {
                double *dc = csr_to_dense(&cc);
                if (csr_well_formed(name, &cc) &&
                    matrices_close(name, dc, ref, n * n, 1e-12, 1e-12)) {
                    pass(name);
                }
                free(dc);
                csr_free(&ca);
                csr_free(&cb);
                csr_free(&cc);
            }
            free(a);
            free(b);
            free(ref);
        }
    }
}

static void test_csr_checksum(void) {
    const double a[4] = {1.5, 0, -2, 4};
    csr_matrix m;
    csr_from_dense(a, 2, 2, &m);
    if (csr_checksum(&m) == 3.5) {
        pass("csr_checksum");
    } else {
        fail("csr_checksum", "wrong sum");
    }
    csr_free(&m);
}

static void test_csr_file_round_trip(const char *tmp_path) {
    const size_t n = 23;
    double *a = random_matrix(n, 0.2);
    csr_matrix m, r;
    const char *name = "csr_write/csr_read round trip";
    csr_from_dense(a, (int64_t)n, (int64_t)n, &m);
    if (csr_write(tmp_path, &m) != CSR_OK || csr_read(tmp_path, &r) != CSR_OK) {
        fail(name, "I/O failed");
        csr_free(&m);
        free(a);
        return;
    }
    int ok = r.n_rows == m.n_rows && r.n_cols == m.n_cols && r.nnz == m.nnz;
    for (int64_t i = 0; ok && i <= m.n_rows; i++) {
        ok = r.indptr[i] == m.indptr[i];
    }
    for (int64_t p = 0; ok && p < m.nnz; p++) {
        ok = r.indices[p] == m.indices[p] && r.data[p] == m.data[p];
    }
    if (ok) {
        pass(name);
    } else {
        fail(name, "data differs after round trip");
    }
    remove(tmp_path);
    csr_free(&m);
    csr_free(&r);
    free(a);
}

int main(int argc, char **argv) {
    test_identity();
    test_zero();
    test_known_small();
    test_methods_agree();
    test_blocked_rejects_zero_tile();
    test_checksum();
    test_csr_conversion_known();
    test_csr_round_trip();
    test_csr_all_zero();
    test_csr_identity();
    test_csr_known_product();
    test_csr_keeps_cancellation();
    test_csr_shape_mismatch();
    test_csr_matches_dense();
    test_csr_checksum();
    test_csr_file_round_trip(argc > 1 ? argv[1] : "test_csr_tmp.csr");
    if (failures > 0) {
        printf("%d test(s) FAILED\n", failures);
        return 1;
    }
    printf("All C tests passed\n");
    return 0;
}
