#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>

/* Dense n x n row-major float64 matrix multiplication kernels.
 *
 * Every kernel returns a newly allocated n*n row-major product C = A * B, or
 * NULL if the allocation fails. The caller owns the returned buffer and must
 * free() it. Allocation and zero-initialisation of C (calloc) happen inside
 * the kernel, so they are part of the timed region for every method. */

/* Baseline: classic i-j-k triple loop (the validated Assignment 1 kernel). */
double *matmul_ijk(const double *a, const double *b, size_t n);

/* Loop reordering i-k-j: A[i][k] is hoisted and the inner loop walks one row
 * of B and one row of C contiguously (unit stride). */
double *matmul_ikj(const double *a, const double *b, size_t n);

/* Cache blocking with square tiles of side `tile` (ii-kk-jj tile loops with an
 * i-k-j micro-kernel inside each tile). n does not need to be a multiple of
 * tile. Returns NULL if tile == 0. */
double *matmul_blocked(const double *a, const double *b, size_t n, size_t tile);

/* Returns the sum of all n*n elements of the row-major matrix c. */
double checksum(const double *c, size_t n);

#endif
