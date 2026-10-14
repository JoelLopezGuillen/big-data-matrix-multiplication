#include "matmul.h"

#include <stdlib.h>

double *matmul(const double *a, const double *b, size_t n) {
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

double checksum(const double *c, size_t n) {
    double total = 0.0;
    for (size_t idx = 0; idx < n * n; idx++) {
        total += c[idx];
    }
    return total;
}
