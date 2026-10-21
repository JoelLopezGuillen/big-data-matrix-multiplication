#include "csr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void csr_reset(csr_matrix *m) { memset(m, 0, sizeof(*m)); }

void csr_free(csr_matrix *m) {
    free(m->indptr);
    free(m->indices);
    free(m->data);
    csr_reset(m);
}

/* Allocates the three arrays for n_rows rows and nnz entries (at least one
 * element each, so that malloc(0) never yields an ambiguous NULL). */
static int csr_alloc(csr_matrix *m, int64_t n_rows, int64_t n_cols, int64_t nnz) {
    csr_reset(m);
    m->n_rows = n_rows;
    m->n_cols = n_cols;
    m->nnz = nnz;
    m->indptr = malloc((size_t)(n_rows + 1) * sizeof(int64_t));
    m->indices = malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int32_t));
    m->data = malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
    if (m->indptr == NULL || m->indices == NULL || m->data == NULL) {
        csr_free(m);
        return CSR_ERR_ALLOC;
    }
    return CSR_OK;
}

int csr_from_dense(const double *dense, int64_t n_rows, int64_t n_cols,
                   csr_matrix *out) {
    csr_reset(out);
    if (n_rows < 0 || n_cols < 0 || n_cols > INT32_MAX) {
        return CSR_ERR_SHAPE;
    }
    /* Pass 1: count the non-zeros to allocate the output exactly. */
    int64_t nnz = 0;
    for (int64_t idx = 0; idx < n_rows * n_cols; idx++) {
        nnz += dense[idx] != 0.0;
    }
    int status = csr_alloc(out, n_rows, n_cols, nnz);
    if (status != CSR_OK) {
        return status;
    }
    /* Pass 2: fill row by row; columns come out in increasing order. */
    int64_t pos = 0;
    out->indptr[0] = 0;
    for (int64_t i = 0; i < n_rows; i++) {
        const double *row = dense + i * n_cols;
        for (int64_t j = 0; j < n_cols; j++) {
            if (row[j] != 0.0) {
                out->indices[pos] = (int32_t)j;
                out->data[pos] = row[j];
                pos++;
            }
        }
        out->indptr[i + 1] = pos;
    }
    return CSR_OK;
}

double *csr_to_dense(const csr_matrix *m) {
    double *dense = calloc((size_t)(m->n_rows * m->n_cols) + 1, sizeof(double));
    if (dense == NULL) {
        return NULL;
    }
    for (int64_t i = 0; i < m->n_rows; i++) {
        for (int64_t p = m->indptr[i]; p < m->indptr[i + 1]; p++) {
            dense[i * m->n_cols + m->indices[p]] = m->data[p];
        }
    }
    return dense;
}

static int compare_int32(const void *x, const void *y) {
    int32_t a = *(const int32_t *)x;
    int32_t b = *(const int32_t *)y;
    return (a > b) - (a < b);
}

static void sort_columns(int32_t *cols, int64_t len) {
    if (len <= 16) {
        for (int64_t s = 1; s < len; s++) {
            int32_t key = cols[s];
            int64_t t = s - 1;
            while (t >= 0 && cols[t] > key) {
                cols[t + 1] = cols[t];
                t--;
            }
            cols[t + 1] = key;
        }
    } else {
        qsort(cols, (size_t)len, sizeof(int32_t), compare_int32);
    }
}

int csr_spgemm(const csr_matrix *a, const csr_matrix *b, csr_matrix *out) {
    csr_reset(out);
    if (a->n_cols != b->n_rows || a->n_rows > INT32_MAX) {
        return CSR_ERR_SHAPE;
    }
    const int64_t n_rows = a->n_rows;
    const int64_t n_cols = b->n_cols;
    /* marker[j] == i  <=>  column j already appeared in output row i. */
    int32_t *marker = malloc((size_t)(n_cols > 0 ? n_cols : 1) * sizeof(int32_t));
    double *acc = malloc((size_t)(n_cols > 0 ? n_cols : 1) * sizeof(double));
    int64_t *indptr = malloc((size_t)(n_rows + 1) * sizeof(int64_t));
    if (marker == NULL || acc == NULL || indptr == NULL) {
        free(marker);
        free(acc);
        free(indptr);
        return CSR_ERR_ALLOC;
    }

    /* Symbolic pass: number of structural non-zeros of each row of C. */
    for (int64_t j = 0; j < n_cols; j++) {
        marker[j] = -1;
    }
    indptr[0] = 0;
    for (int64_t i = 0; i < n_rows; i++) {
        int64_t count = 0;
        for (int64_t p = a->indptr[i]; p < a->indptr[i + 1]; p++) {
            const int32_t k = a->indices[p];
            for (int64_t q = b->indptr[k]; q < b->indptr[k + 1]; q++) {
                const int32_t j = b->indices[q];
                if (marker[j] != (int32_t)i) {
                    marker[j] = (int32_t)i;
                    count++;
                }
            }
        }
        indptr[i + 1] = indptr[i] + count;
    }

    const int64_t nnz = indptr[n_rows];
    int32_t *indices = malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int32_t));
    double *data = malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
    if (indices == NULL || data == NULL) {
        free(indices);
        free(data);
        free(marker);
        free(acc);
        free(indptr);
        return CSR_ERR_ALLOC;
    }

    /* Numeric pass: dense accumulator per row (Gustavson). */
    for (int64_t j = 0; j < n_cols; j++) {
        marker[j] = -1;
    }
    for (int64_t i = 0; i < n_rows; i++) {
        const int64_t start = indptr[i];
        int64_t len = 0;
        for (int64_t p = a->indptr[i]; p < a->indptr[i + 1]; p++) {
            const int32_t k = a->indices[p];
            const double a_ik = a->data[p];
            for (int64_t q = b->indptr[k]; q < b->indptr[k + 1]; q++) {
                const int32_t j = b->indices[q];
                if (marker[j] != (int32_t)i) {
                    marker[j] = (int32_t)i;
                    acc[j] = a_ik * b->data[q];
                    indices[start + len++] = j;
                } else {
                    acc[j] += a_ik * b->data[q];
                }
            }
        }
        /* Emit the row with increasing column indices. For rows that fill a
         * large fraction of the columns, a linear scan of the marker array is
         * cheaper than sorting (heuristic threshold: len >= n_cols / 64). */
        if (len * 64 >= n_cols) {
            int64_t pos = start;
            for (int64_t j = 0; j < n_cols; j++) {
                if (marker[j] == (int32_t)i) {
                    indices[pos] = (int32_t)j;
                    data[pos] = acc[j];
                    pos++;
                }
            }
        } else {
            sort_columns(indices + start, len);
            for (int64_t p = start; p < start + len; p++) {
                data[p] = acc[indices[p]];
            }
        }
    }
    free(marker);
    free(acc);

    out->n_rows = n_rows;
    out->n_cols = n_cols;
    out->nnz = nnz;
    out->indptr = indptr;
    out->indices = indices;
    out->data = data;
    return CSR_OK;
}

double csr_checksum(const csr_matrix *m) {
    double total = 0.0;
    for (int64_t p = 0; p < m->nnz; p++) {
        total += m->data[p];
    }
    return total;
}

/* x86-64 is little-endian, matching the file format. */
int csr_read(const char *path, csr_matrix *out) {
    csr_reset(out);
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return CSR_ERR_IO;
    }
    int64_t header[3];
    if (fread(header, sizeof(int64_t), 3, file) != 3 || header[0] < 0 ||
        header[1] < 0 || header[1] > INT32_MAX || header[2] < 0) {
        fclose(file);
        return CSR_ERR_IO;
    }
    int status = csr_alloc(out, header[0], header[1], header[2]);
    if (status != CSR_OK) {
        fclose(file);
        return status;
    }
    const size_t rows1 = (size_t)(out->n_rows + 1);
    const size_t nnz = (size_t)out->nnz;
    int ok = fread(out->indptr, sizeof(int64_t), rows1, file) == rows1 &&
             fread(out->indices, sizeof(int32_t), nnz, file) == nnz &&
             fread(out->data, sizeof(double), nnz, file) == nnz &&
             out->indptr[0] == 0 && out->indptr[out->n_rows] == out->nnz;
    fclose(file);
    if (!ok) {
        csr_free(out);
        return CSR_ERR_IO;
    }
    return CSR_OK;
}

int csr_write(const char *path, const csr_matrix *m) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        return CSR_ERR_IO;
    }
    const int64_t header[3] = {m->n_rows, m->n_cols, m->nnz};
    const size_t rows1 = (size_t)(m->n_rows + 1);
    const size_t nnz = (size_t)m->nnz;
    int ok = fwrite(header, sizeof(int64_t), 3, file) == 3 &&
             fwrite(m->indptr, sizeof(int64_t), rows1, file) == rows1 &&
             fwrite(m->indices, sizeof(int32_t), nnz, file) == nnz &&
             fwrite(m->data, sizeof(double), nnz, file) == nnz;
    if (fclose(file) != 0) {
        ok = 0;
    }
    return ok ? CSR_OK : CSR_ERR_IO;
}
