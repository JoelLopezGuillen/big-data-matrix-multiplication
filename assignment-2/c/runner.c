/* Benchmark / verification runner for the dense and CSR kernels.
 *
 * Usage:
 *   runner bench         <method> <n> <a.bin> <b.bin> <warmups> <reps> <tile>
 *   runner verify        <method> <n> <a.bin> <b.bin> <out.bin> <tile>
 *   runner sparse-bench  <a.csr> <b.csr> <warmups> <reps>
 *   runner sparse-verify <a.csr> <b.csr> <out.csr>
 *   runner convert-bench <n> <a.bin> <b.bin> <warmups> <reps>
 *   runner convert-verify <n> <a.bin> <out.csr>
 *
 * <method> is ijk, ikj or blocked; <tile> is only used by blocked (pass 0
 * otherwise). Dense files are raw little-endian row-major float64; CSR files
 * use the format documented in csr.h.
 *
 * Bench modes print, one JSON object per line:
 *   {"event": "loaded"}                       after the inputs are in memory
 *   {"event": "done", "phase": ..., ...}      after every kernel execution
 *   {..., "status": "ok", "times_ns": [...], "checksums": [...]}   final result
 * and then block reading one line from stdin so that the harness can sample
 * the peak working set of this still-alive process. Allocation failures print
 * {"status": "oom"} and exit with code 3.
 *
 * Timed regions (QueryPerformanceCounter):
 *   bench          one dense kernel call, including calloc of C
 *   sparse-bench   one csr_spgemm call (symbolic + allocation + numeric)
 *   convert-bench  csr_from_dense(A) + csr_from_dense(B), including allocation
 * File loading, checksums and free() are outside the timed region. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "csr.h"
#include "matmul.h"

#define EXIT_OOM 3

static void exit_oom(void) {
    printf("{\"status\": \"oom\"}\n");
    fflush(stdout);
    exit(EXIT_OOM);
}

static long long ticks_to_ns(LONGLONG ticks, LONGLONG freq) {
    /* Split the conversion to avoid overflowing ticks * 1e9. */
    return (long long)((ticks / freq) * 1000000000LL +
                       ((ticks % freq) * 1000000000LL) / freq);
}

static LONGLONG now_ticks(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}

static LONGLONG tick_frequency(void) {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    return f.QuadPart;
}

static void emit_event(const char *phase, int index) {
    printf("{\"event\": \"done\", \"phase\": \"%s\", \"index\": %d}\n", phase, index);
    fflush(stdout);
}

static void emit_loaded(void) {
    printf("{\"event\": \"loaded\"}\n");
    fflush(stdout);
}

static void wait_for_harness(void) {
    char line[16];
    if (fgets(line, sizeof(line), stdin) == NULL) {
        line[0] = '\0';
    }
}

static void print_samples(const long long *times, const double *sums, int reps,
                          double warmup_sink) {
    printf("\"times_ns\": [");
    for (int r = 0; r < reps; r++) {
        printf("%s%lld", r ? ", " : "", times[r]);
    }
    printf("], \"checksums\": [");
    for (int r = 0; r < reps; r++) {
        printf("%s%.17g", r ? ", " : "", sums[r]);
    }
    printf("], \"warmup_sink\": %.17g}\n", warmup_sink);
    fflush(stdout);
}

static double *load_dense(const char *path, size_t n) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    double *m = malloc(n * n * sizeof(double));
    if (m == NULL) {
        exit_oom();
    }
    /* x86-64 is little-endian, matching the file format. */
    size_t count = fread(m, sizeof(double), n * n, file);
    fclose(file);
    if (count != n * n) {
        fprintf(stderr, "short read from %s\n", path);
        exit(1);
    }
    return m;
}

static void load_csr(const char *path, csr_matrix *m) {
    int status = csr_read(path, m);
    if (status == CSR_ERR_ALLOC) {
        exit_oom();
    }
    if (status != CSR_OK) {
        fprintf(stderr, "cannot read CSR file %s\n", path);
        exit(1);
    }
}

/* ------------------------------------------------------------------ dense */

typedef struct {
    const char *method;
    size_t tile;
} dense_spec;

static int parse_method(const char *method) {
    return strcmp(method, "ijk") == 0 || strcmp(method, "ikj") == 0 ||
           strcmp(method, "blocked") == 0;
}

static double *dense_multiply_or_exit(dense_spec spec, const double *a,
                                      const double *b, size_t n) {
    double *c;
    if (strcmp(spec.method, "ijk") == 0) {
        c = matmul_ijk(a, b, n);
    } else if (strcmp(spec.method, "ikj") == 0) {
        c = matmul_ikj(a, b, n);
    } else {
        c = matmul_blocked(a, b, n, spec.tile);
    }
    if (c == NULL) {
        exit_oom();
    }
    return c;
}

static void alloc_samples(int reps, long long **times, double **sums) {
    *times = malloc((size_t)(reps > 0 ? reps : 1) * sizeof(long long));
    *sums = malloc((size_t)(reps > 0 ? reps : 1) * sizeof(double));
    if (*times == NULL || *sums == NULL) {
        exit_oom();
    }
}

static int dense_bench(dense_spec spec, size_t n, const char *a_path,
                       const char *b_path, int warmups, int reps) {
    double *a = load_dense(a_path, n);
    double *b = load_dense(b_path, n);
    long long *times;
    double *sums;
    alloc_samples(reps, &times, &sums);
    const LONGLONG freq = tick_frequency();
    emit_loaded();

    double warmup_sink = 0.0;
    for (int w = 0; w < warmups; w++) {
        double *c = dense_multiply_or_exit(spec, a, b, n);
        warmup_sink += checksum(c, n);
        free(c);
        emit_event("warmup", w);
    }
    for (int r = 0; r < reps; r++) {
        LONGLONG start = now_ticks();
        double *c = dense_multiply_or_exit(spec, a, b, n);
        LONGLONG end = now_ticks();
        times[r] = ticks_to_ns(end - start, freq);
        sums[r] = checksum(c, n);
        free(c);
        emit_event("repetition", r);
    }
    printf("{\"mode\": \"dense\", \"method\": \"%s\", \"n\": %zu, \"tile\": %zu, "
           "\"status\": \"ok\", \"warmups\": %d, ",
           spec.method, n, spec.tile, warmups);
    print_samples(times, sums, reps, warmup_sink);
    wait_for_harness();
    free(times);
    free(sums);
    free(a);
    free(b);
    return 0;
}

static int dense_verify(dense_spec spec, size_t n, const char *a_path,
                        const char *b_path, const char *out_path) {
    double *a = load_dense(a_path, n);
    double *b = load_dense(b_path, n);
    double *c = dense_multiply_or_exit(spec, a, b, n);
    FILE *out = fopen(out_path, "wb");
    if (out == NULL) {
        fprintf(stderr, "cannot open %s\n", out_path);
        return 1;
    }
    size_t written = fwrite(c, sizeof(double), n * n, out);
    fclose(out);
    free(a);
    free(b);
    free(c);
    return written == n * n ? 0 : 1;
}

/* ----------------------------------------------------------------- sparse */

static void spgemm_or_exit(const csr_matrix *a, const csr_matrix *b, csr_matrix *c) {
    int status = csr_spgemm(a, b, c);
    if (status == CSR_ERR_ALLOC) {
        exit_oom();
    }
    if (status != CSR_OK) {
        fprintf(stderr, "csr_spgemm failed with status %d\n", status);
        exit(1);
    }
}

static int sparse_bench(const char *a_path, const char *b_path, int warmups,
                        int reps) {
    csr_matrix a, b;
    load_csr(a_path, &a);
    load_csr(b_path, &b);
    long long *times;
    double *sums;
    alloc_samples(reps, &times, &sums);
    const LONGLONG freq = tick_frequency();
    emit_loaded();

    double warmup_sink = 0.0;
    int64_t nnz_c = -1;
    for (int w = 0; w < warmups; w++) {
        csr_matrix c;
        spgemm_or_exit(&a, &b, &c);
        warmup_sink += csr_checksum(&c);
        nnz_c = c.nnz;
        csr_free(&c);
        emit_event("warmup", w);
    }
    for (int r = 0; r < reps; r++) {
        csr_matrix c;
        LONGLONG start = now_ticks();
        spgemm_or_exit(&a, &b, &c);
        LONGLONG end = now_ticks();
        times[r] = ticks_to_ns(end - start, freq);
        sums[r] = csr_checksum(&c);
        nnz_c = c.nnz;
        csr_free(&c);
        emit_event("repetition", r);
    }
    printf("{\"mode\": \"sparse\", \"method\": \"csr_gustavson\", \"n\": %lld, "
           "\"status\": \"ok\", \"warmups\": %d, \"nnz_a\": %lld, \"nnz_b\": %lld, "
           "\"nnz_c\": %lld, ",
           (long long)a.n_rows, warmups, (long long)a.nnz, (long long)b.nnz,
           (long long)nnz_c);
    print_samples(times, sums, reps, warmup_sink);
    wait_for_harness();
    free(times);
    free(sums);
    csr_free(&a);
    csr_free(&b);
    return 0;
}

static int sparse_verify(const char *a_path, const char *b_path, const char *out_path) {
    csr_matrix a, b, c;
    load_csr(a_path, &a);
    load_csr(b_path, &b);
    spgemm_or_exit(&a, &b, &c);
    int status = csr_write(out_path, &c);
    csr_free(&a);
    csr_free(&b);
    csr_free(&c);
    return status == CSR_OK ? 0 : 1;
}

static void convert_or_exit(const double *dense, size_t n, csr_matrix *out) {
    int status = csr_from_dense(dense, (int64_t)n, (int64_t)n, out);
    if (status == CSR_ERR_ALLOC) {
        exit_oom();
    }
    if (status != CSR_OK) {
        fprintf(stderr, "csr_from_dense failed with status %d\n", status);
        exit(1);
    }
}

static int convert_bench(size_t n, const char *a_path, const char *b_path,
                         int warmups, int reps) {
    double *a = load_dense(a_path, n);
    double *b = load_dense(b_path, n);
    long long *times;
    double *sums;
    alloc_samples(reps, &times, &sums);
    const LONGLONG freq = tick_frequency();
    emit_loaded();

    double warmup_sink = 0.0;
    int64_t nnz_a = -1, nnz_b = -1;
    for (int w = 0; w < warmups; w++) {
        csr_matrix ca, cb;
        convert_or_exit(a, n, &ca);
        convert_or_exit(b, n, &cb);
        warmup_sink += csr_checksum(&ca) + csr_checksum(&cb);
        nnz_a = ca.nnz;
        nnz_b = cb.nnz;
        csr_free(&ca);
        csr_free(&cb);
        emit_event("warmup", w);
    }
    for (int r = 0; r < reps; r++) {
        csr_matrix ca, cb;
        LONGLONG start = now_ticks();
        convert_or_exit(a, n, &ca);
        convert_or_exit(b, n, &cb);
        LONGLONG end = now_ticks();
        times[r] = ticks_to_ns(end - start, freq);
        sums[r] = csr_checksum(&ca) + csr_checksum(&cb);
        nnz_a = ca.nnz;
        nnz_b = cb.nnz;
        csr_free(&ca);
        csr_free(&cb);
        emit_event("repetition", r);
    }
    printf("{\"mode\": \"convert\", \"method\": \"dense_to_csr\", \"n\": %zu, "
           "\"status\": \"ok\", \"warmups\": %d, \"nnz_a\": %lld, \"nnz_b\": %lld, ",
           n, warmups, (long long)nnz_a, (long long)nnz_b);
    print_samples(times, sums, reps, warmup_sink);
    wait_for_harness();
    free(times);
    free(sums);
    free(a);
    free(b);
    return 0;
}

static int convert_verify(size_t n, const char *a_path, const char *out_path) {
    double *a = load_dense(a_path, n);
    csr_matrix ca;
    convert_or_exit(a, n, &ca);
    int status = csr_write(out_path, &ca);
    csr_free(&ca);
    free(a);
    return status == CSR_OK ? 0 : 1;
}

static size_t parse_size(const char *text) { return (size_t)strtoull(text, NULL, 10); }

static int usage(void) {
    fprintf(stderr,
            "usage: runner bench <ijk|ikj|blocked> <n> <a.bin> <b.bin> <warmups> <reps> <tile>\n"
            "       runner verify <ijk|ikj|blocked> <n> <a.bin> <b.bin> <out.bin> <tile>\n"
            "       runner sparse-bench <a.csr> <b.csr> <warmups> <reps>\n"
            "       runner sparse-verify <a.csr> <b.csr> <out.csr>\n"
            "       runner convert-bench <n> <a.bin> <b.bin> <warmups> <reps>\n"
            "       runner convert-verify <n> <a.bin> <out.csr>\n");
    return 2;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        return usage();
    }
    const char *mode = argv[1];
    if ((strcmp(mode, "bench") == 0 && argc == 9) ||
        (strcmp(mode, "verify") == 0 && argc == 8)) {
        dense_spec spec = {argv[2], parse_size(argv[argc - 1])};
        if (!parse_method(spec.method) ||
            (strcmp(spec.method, "blocked") == 0 && spec.tile == 0)) {
            return usage();
        }
        if (strcmp(spec.method, "blocked") != 0) {
            spec.tile = 0;
        }
        size_t n = parse_size(argv[3]);
        if (mode[0] == 'b') {
            return dense_bench(spec, n, argv[4], argv[5], atoi(argv[6]), atoi(argv[7]));
        }
        return dense_verify(spec, n, argv[4], argv[5], argv[6]);
    }
    if (strcmp(mode, "sparse-bench") == 0 && argc == 6) {
        return sparse_bench(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]));
    }
    if (strcmp(mode, "sparse-verify") == 0 && argc == 5) {
        return sparse_verify(argv[2], argv[3], argv[4]);
    }
    if (strcmp(mode, "convert-bench") == 0 && argc == 7) {
        return convert_bench(parse_size(argv[2]), argv[3], argv[4], atoi(argv[5]),
                             atoi(argv[6]));
    }
    if (strcmp(mode, "convert-verify") == 0 && argc == 5) {
        return convert_verify(parse_size(argv[2]), argv[3], argv[4]);
    }
    return usage();
}
