#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>

/* Returns a newly allocated n*n row-major product C = A * B computed with the
 * classic i-j-k triple loop, or NULL if the allocation fails. The caller owns
 * the returned buffer and must free() it. */
double *matmul(const double *a, const double *b, size_t n);

/* Returns the sum of all n*n elements of the row-major matrix c. */
double checksum(const double *c, size_t n);

#endif
