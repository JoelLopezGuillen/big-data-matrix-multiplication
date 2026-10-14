/* Correctness tests for the naive C matrix multiplication kernel.
 * Exits with a non-zero status if any check fails. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "matmul.h"

static int failures = 0;

static void check_matrix(const char *name, const double *actual,
                         const double *expected, size_t n, double tol) {
    if (actual == NULL) {
        printf("FAIL %s: kernel returned NULL\n", name);
        failures++;
        return;
    }
    for (size_t idx = 0; idx < n * n; idx++) {
        double diff = fabs(actual[idx] - expected[idx]);
        if (diff > tol) {
            printf("FAIL %s: element %zu is %.17g, expected %.17g\n", name,
                   idx, actual[idx], expected[idx]);
            failures++;
            return;
        }
    }
    printf("PASS %s\n", name);
}

static void test_identity(void) {
    const size_t n = 5;
    double a[25], id[25];
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            a[i * n + j] = (double)(3 * i + j) - 2.5;
            id[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    double *c = matmul(a, id, n);
    check_matrix("identity (A*I == A)", c, a, n, 0.0);
    free(c);
}

static void test_zero(void) {
    const size_t n = 4;
    double a[16], z[16] = {0};
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            a[i * n + j] = (double)i - (double)j;
        }
    }
    double *c1 = matmul(a, z, n);
    double *c2 = matmul(z, a, n);
    check_matrix("zero (A*0 == 0)", c1, z, n, 0.0);
    check_matrix("zero (0*A == 0)", c2, z, n, 0.0);
    free(c1);
    free(c2);
}

static void test_known_2x2(void) {
    const double a[4] = {1.0, 2.0, 3.0, 4.0};
    const double b[4] = {5.0, 6.0, 7.0, 8.0};
    const double expected[4] = {19.0, 22.0, 43.0, 50.0};
    double *c = matmul(a, b, 2);
    check_matrix("known 2x2", c, expected, 2, 0.0);
    free(c);
}

static void test_known_3x3(void) {
    const double a[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0};
    const double b[9] = {-1.0, 0.5, 2.0, 3.0, -2.0, 1.0, 0.25, 4.0, -3.0};
    const double expected[9] = {5.75, 8.5, -5.0, 12.5, 16.0,
                                -5.0, 19.5, 27.5, -8.0};
    double *c = matmul(a, b, 3);
    check_matrix("known 3x3 (non-integer values)", c, expected, 3, 1e-12);
    free(c);
}

static void test_checksum(void) {
    const double c[4] = {19.0, 22.0, 43.0, 50.0};
    double s = checksum(c, 2);
    if (s != 134.0) {
        printf("FAIL checksum: got %.17g, expected 134\n", s);
        failures++;
    } else {
        printf("PASS checksum\n");
    }
}

int main(void) {
    test_identity();
    test_zero();
    test_known_2x2();
    test_known_3x3();
    test_checksum();
    if (failures > 0) {
        printf("%d test(s) FAILED\n", failures);
        return 1;
    }
    printf("All C tests passed\n");
    return 0;
}
