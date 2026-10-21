#include "matmul.h"

#include <stdlib.h>

/* Baseline. This is the validated Assignment 1 C kernel (git tag
 * `assignment-1`, assignment-1/c/matmul.c, function `matmul`), copied verbatim
 * and only renamed to matmul_ijk. */
double *matmul_ijk(const double *a, const double *b, size_t n) {
    /* Allocation and zero-initialisation are part of the timed region. */
    double *c = calloc(n * n, sizeof(double));
    if (c == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    return c;
}

double *matmul_ikj(const double *a, const double *b, size_t n) {
    /* Same timing boundary as the baseline: calloc is inside the kernel. */
    double *c = calloc(n * n, sizeof(double));
    if (c == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < n; i++) {
        double *c_row = c + i * n;
        for (size_t k = 0; k < n; k++) {
            const double a_ik = a[i * n + k];
            const double *b_row = b + k * n;
            for (size_t j = 0; j < n; j++) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
    return c;
}

static size_t min_size(size_t x, size_t y) { return x < y ? x : y; }

double *matmul_blocked(const double *a, const double *b, size_t n, size_t tile) {
    if (tile == 0) {
        return NULL;
    }
    /* Same timing boundary as the baseline: calloc is inside the kernel. */
    double *c = calloc(n * n, sizeof(double));
    if (c == NULL) {
        return NULL;
    }
    for (size_t ii = 0; ii < n; ii += tile) {
        const size_t i_end = min_size(ii + tile, n);
        for (size_t kk = 0; kk < n; kk += tile) {
            const size_t k_end = min_size(kk + tile, n);
            for (size_t jj = 0; jj < n; jj += tile) {
                const size_t j_end = min_size(jj + tile, n);
                /* i-k-j micro-kernel on the (ii, kk, jj) tile; edge tiles are
                 * shorter when n is not a multiple of tile. */
                for (size_t i = ii; i < i_end; i++) {
                    double *c_row = c + i * n;
                    for (size_t k = kk; k < k_end; k++) {
                        const double a_ik = a[i * n + k];
                        const double *b_row = b + k * n;
                        for (size_t j = jj; j < j_end; j++) {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
    return c;
}

double checksum(const double *c, size_t n) {
    double total = 0.0;
    for (size_t idx = 0; idx < n * n; idx++) {
        total += c[idx];
    }
    return total;
}
