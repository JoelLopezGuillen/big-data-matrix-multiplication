#ifndef CSR_H
#define CSR_H

#include <stddef.h>
#include <stdint.h>

/* Compressed Sparse Row (CSR) matrix with float64 values.
 *
 * Row i owns the entries indptr[i] .. indptr[i+1]-1 of `indices` (column
 * indices, int32) and `data` (values). Column indices are strictly increasing
 * within each row. Entries are structural: a stored value may be exactly 0.0
 * when it comes from numerical cancellation in a product (see csr_spgemm). */
typedef struct {
    int64_t n_rows;
    int64_t n_cols;
    int64_t nnz;
    int64_t *indptr;  /* n_rows + 1 entries */
    int32_t *indices; /* nnz entries */
    double *data;     /* nnz entries */
} csr_matrix;

/* Status codes returned by the CSR functions. */
#define CSR_OK 0
#define CSR_ERR_ALLOC (-1)
#define CSR_ERR_SHAPE (-2)
#define CSR_ERR_IO (-3)

/* Converts the n_rows x n_cols row-major dense matrix `dense` to CSR, storing
 * every element that is != 0.0. Output arrays are allocated exactly (two
 * passes: count, then fill). On failure *out is left empty. */
int csr_from_dense(const double *dense, int64_t n_rows, int64_t n_cols,
                   csr_matrix *out);

/* Returns a newly allocated row-major dense copy of m (calloc), or NULL. */
double *csr_to_dense(const csr_matrix *m);

/* C = A * B with Gustavson's row-wise algorithm. Two passes over A: a
 * symbolic pass that counts the structural non-zeros of each row of C with a
 * marker array, then exact allocation of C, then a numeric pass that
 * accumulates each row in a dense accumulator of length n_cols(B). Column
 * indices of each output row are emitted in increasing order. All entries that
 * are structurally non-zero are kept, even if they cancel to exactly 0.0. */
int csr_spgemm(const csr_matrix *a, const csr_matrix *b, csr_matrix *out);

/* Sum of all stored values (used to consume results in benchmarks). */
double csr_checksum(const csr_matrix *m);

/* Releases the arrays of m and resets it to an empty matrix. */
void csr_free(csr_matrix *m);

/* Binary CSR file, little-endian:
 *   int64 n_rows, int64 n_cols, int64 nnz,
 *   int64 indptr[n_rows+1], int32 indices[nnz], float64 data[nnz]. */
int csr_read(const char *path, csr_matrix *out);
int csr_write(const char *path, const csr_matrix *m);

#endif
