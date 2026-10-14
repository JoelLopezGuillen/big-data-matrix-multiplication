/** Naive dense matrix multiplication on flat row-major double arrays. */
public final class MatMul {

    private MatMul() {
    }

    /**
     * Returns C = A * B for n x n row-major matrices using the classic i-j-k
     * triple loop. Allocation and zero-initialisation of C (done by the JVM for
     * {@code new double[]}) are part of the timed region.
     */
    public static double[] multiply(double[] a, double[] b, int n) {
        double[] c = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        return c;
    }

    /** Returns the sum of all elements of C (consumes the result). */
    public static double checksum(double[] c) {
        double total = 0.0;
        for (double value : c) {
            total += value;
        }
        return total;
    }
}
