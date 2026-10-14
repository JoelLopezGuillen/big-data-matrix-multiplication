/* Benchmark / verification runner for the C kernel.
 *
 * Usage:
 *   runner bench  <n> <data_dir> <warmups> <repetitions>
 *   runner verify <n> <data_dir> <output_file>
 *
 * `bench` prints exactly one JSON line with per-repetition kernel times (ns)
 * and checksums, then blocks reading one line from stdin so that the harness
 * can sample the peak working set of this still-alive process. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matmul.h"

#define EXIT_OOM 3

static void exit_oom(void) {
    printf("{\"language\": \"c\", \"status\": \"oom\"}\n");
    fflush(stdout);
    exit(EXIT_OOM);
}

static long long ticks_to_ns(LONGLONG ticks, LONGLONG freq) {
    /* Split the conversion to avoid overflowing ticks * 1e9. */
    return (long long)((ticks / freq) * 1000000000LL +
                       ((ticks % freq) * 1000000000LL) / freq);
}

static double *load_matrix(const char *data_dir, const char *name, size_t n) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s_n%zu.bin", data_dir, name, n);
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

static double *multiply_or_exit(const double *a, const double *b, size_t n) {
    double *c = matmul(a, b, n);
    if (c == NULL) {
        exit_oom();
    }
    return c;
}

static int bench(size_t n, const char *data_dir, int warmups, int reps) {
    double *a = load_matrix(data_dir, "A", n);
    double *b = load_matrix(data_dir, "B", n);
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    double warmup_sink = 0.0;
    for (int w = 0; w < warmups; w++) {
        double *c = multiply_or_exit(a, b, n);
        warmup_sink += checksum(c, n);
        free(c);
    }

    long long *times = malloc((size_t)reps * sizeof(long long));
    double *sums = malloc((size_t)reps * sizeof(double));
    if (times == NULL || sums == NULL) {
        exit_oom();
    }
    for (int r = 0; r < reps; r++) {
        LARGE_INTEGER start, end;
        QueryPerformanceCounter(&start);
        double *c = multiply_or_exit(a, b, n);
        QueryPerformanceCounter(&end);
        times[r] = ticks_to_ns(end.QuadPart - start.QuadPart, freq.QuadPart);
        sums[r] = checksum(c, n);
        free(c);
    }

    printf("{\"language\": \"c\", \"n\": %zu, \"status\": \"ok\", "
           "\"warmups\": %d, \"times_ns\": [",
           n, warmups);
    for (int r = 0; r < reps; r++) {
        printf("%s%lld", r ? ", " : "", times[r]);
    }
    printf("], \"checksums\": [");
    for (int r = 0; r < reps; r++) {
        printf("%s%.17g", r ? ", " : "", sums[r]);
    }
    printf("], \"warmup_sink\": %.17g}\n", warmup_sink);
    fflush(stdout);

    /* Stay alive until the harness has sampled the peak working set. */
    char line[16];
    if (fgets(line, sizeof(line), stdin) == NULL) {
        line[0] = '\0';
    }
    free(times);
    free(sums);
    free(a);
    free(b);
    return 0;
}

static int verify(size_t n, const char *data_dir, const char *output_file) {
    double *a = load_matrix(data_dir, "A", n);
    double *b = load_matrix(data_dir, "B", n);
    double *c = multiply_or_exit(a, b, n);
    FILE *out = fopen(output_file, "wb");
    if (out == NULL) {
        fprintf(stderr, "cannot open %s\n", output_file);
        return 1;
    }
    size_t written = fwrite(c, sizeof(double), n * n, out);
    fclose(out);
    free(a);
    free(b);
    free(c);
    return written == n * n ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc == 6 && strcmp(argv[1], "bench") == 0) {
        return bench((size_t)strtoull(argv[2], NULL, 10), argv[3],
                     atoi(argv[4]), atoi(argv[5]));
    }
    if (argc == 5 && strcmp(argv[1], "verify") == 0) {
        return verify((size_t)strtoull(argv[2], NULL, 10), argv[3], argv[4]);
    }
    fprintf(stderr,
            "usage: runner bench <n> <data_dir> <warmups> <repetitions>\n"
            "       runner verify <n> <data_dir> <output_file>\n");
    return 2;
}
