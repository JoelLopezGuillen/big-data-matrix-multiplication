/**
 * Correctness tests for the naive Java matrix multiplication kernel.
 * Uses explicit checks (independent of the -ea flag) and exits with a
 * non-zero status if any check fails.
 */
public final class MatMulTest {

    private static int failures = 0;

    private MatMulTest() {
    }

    private static void checkMatrix(String name, double[] actual, double[] expected, double tol) {
        if (actual == null || actual.length != expected.length) {
            System.out.println("FAIL " + name + ": wrong result length");
            failures++;
            return;
        }
        for (int idx = 0; idx < expected.length; idx++) {
            if (Math.abs(actual[idx] - expected[idx]) > tol) {
                System.out.println("FAIL " + name + ": element " + idx + " is " + actual[idx]
                        + ", expected " + expected[idx]);
                failures++;
                return;
            }
        }
        System.out.println("PASS " + name);
    }

    private static void testIdentity() {
        int n = 5;
        double[] a = new double[n * n];
        double[] id = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = (3 * i + j) - 2.5;
                id[i * n + j] = (i == j) ? 1.0 : 0.0;
            }
        }
        checkMatrix("identity (A*I == A)", MatMul.multiply(a, id, n), a, 0.0);
    }

    private static void testZero() {
        int n = 4;
        double[] a = new double[n * n];
        double[] z = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = i - j;
            }
        }
        checkMatrix("zero (A*0 == 0)", MatMul.multiply(a, z, n), z, 0.0);
        checkMatrix("zero (0*A == 0)", MatMul.multiply(z, a, n), z, 0.0);
    }

    private static void testKnown2x2() {
        double[] a = {1.0, 2.0, 3.0, 4.0};
        double[] b = {5.0, 6.0, 7.0, 8.0};
        double[] expected = {19.0, 22.0, 43.0, 50.0};
        checkMatrix("known 2x2", MatMul.multiply(a, b, 2), expected, 0.0);
    }

    private static void testKnown3x3() {
        double[] a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0};
        double[] b = {-1.0, 0.5, 2.0, 3.0, -2.0, 1.0, 0.25, 4.0, -3.0};
        double[] expected = {5.75, 8.5, -5.0, 12.5, 16.0, -5.0, 19.5, 27.5, -8.0};
        checkMatrix("known 3x3 (non-integer values)", MatMul.multiply(a, b, 3), expected, 1e-12);
    }

    private static void testChecksum() {
        double s = MatMul.checksum(new double[] {19.0, 22.0, 43.0, 50.0});
        if (s != 134.0) {
            System.out.println("FAIL checksum: got " + s + ", expected 134");
            failures++;
        } else {
            System.out.println("PASS checksum");
        }
    }

    public static void main(String[] args) {
        testIdentity();
        testZero();
        testKnown2x2();
        testKnown3x3();
        testChecksum();
        if (failures > 0) {
            System.out.println(failures + " test(s) FAILED");
            System.exit(1);
        }
        System.out.println("All Java tests passed");
    }
}
