import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

/**
 * Benchmark / verification runner for the Java kernel.
 *
 * <pre>
 *   MatMulRunner bench  &lt;n&gt; &lt;data_dir&gt; &lt;warmups&gt; &lt;repetitions&gt;
 *   MatMulRunner verify &lt;n&gt; &lt;data_dir&gt; &lt;output_file&gt;
 * </pre>
 *
 * <p>{@code bench} prints exactly one JSON line with per-repetition kernel
 * times (ns) and checksums, then blocks reading one line from stdin so that
 * the harness can sample the peak working set of this still-alive process.
 */
public final class MatMulRunner {

    private static final int EXIT_OOM = 3;
    private static final int CHUNK_BYTES = 1 << 20;

    private MatMulRunner() {
    }

    /** Loads an n*n little-endian float64 row-major file, reading in chunks. */
    static double[] loadMatrix(Path path, int n) throws IOException {
        double[] m = new double[n * n];
        ByteBuffer buffer = ByteBuffer.allocate(CHUNK_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        int filled = 0;
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            while (filled < m.length) {
                buffer.clear();
                int wanted = Math.min(CHUNK_BYTES, (m.length - filled) * Double.BYTES);
                buffer.limit(wanted);
                while (buffer.hasRemaining()) {
                    if (channel.read(buffer) < 0) {
                        throw new IOException("short read from " + path);
                    }
                }
                buffer.flip();
                int count = wanted / Double.BYTES;
                buffer.asDoubleBuffer().get(m, filled, count);
                filled += count;
            }
        }
        return m;
    }

    /** Writes a matrix as little-endian float64 row-major. */
    static void saveMatrix(Path path, double[] m) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(CHUNK_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE,
                StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            int written = 0;
            while (written < m.length) {
                buffer.clear();
                int count = Math.min(CHUNK_BYTES / Double.BYTES, m.length - written);
                buffer.asDoubleBuffer().put(m, written, count);
                buffer.limit(count * Double.BYTES);
                while (buffer.hasRemaining()) {
                    channel.write(buffer);
                }
                written += count;
            }
        }
    }

    private static void bench(int n, Path dataDir, int warmups, int reps) throws IOException {
        double[] a = loadMatrix(dataDir.resolve("A_n" + n + ".bin"), n);
        double[] b = loadMatrix(dataDir.resolve("B_n" + n + ".bin"), n);
        double warmupSink = 0.0;
        for (int w = 0; w < warmups; w++) {
            warmupSink += MatMul.checksum(MatMul.multiply(a, b, n));
        }
        long[] times = new long[reps];
        double[] sums = new double[reps];
        for (int r = 0; r < reps; r++) {
            long start = System.nanoTime();
            double[] c = MatMul.multiply(a, b, n);
            long end = System.nanoTime();
            times[r] = end - start;
            sums[r] = MatMul.checksum(c);
        }
        StringBuilder json = new StringBuilder();
        json.append("{\"language\": \"java\", \"n\": ").append(n)
                .append(", \"status\": \"ok\", \"warmups\": ").append(warmups)
                .append(", \"times_ns\": [");
        for (int r = 0; r < reps; r++) {
            json.append(r == 0 ? "" : ", ").append(times[r]);
        }
        json.append("], \"checksums\": [");
        for (int r = 0; r < reps; r++) {
            json.append(r == 0 ? "" : ", ").append(sums[r]);
        }
        json.append("], \"warmup_sink\": ").append(warmupSink).append('}');
        System.out.println(json);
        System.out.flush();
        waitForLine(System.in);
    }

    /** Blocks until one line (or end of stream) has been read. */
    private static void waitForLine(InputStream in) throws IOException {
        int ch;
        do {
            ch = in.read();
        } while (ch != -1 && ch != '\n');
    }

    private static void verify(int n, Path dataDir, Path output) throws IOException {
        double[] a = loadMatrix(dataDir.resolve("A_n" + n + ".bin"), n);
        double[] b = loadMatrix(dataDir.resolve("B_n" + n + ".bin"), n);
        saveMatrix(output, MatMul.multiply(a, b, n));
    }

    public static void main(String[] args) throws IOException {
        try {
            if (args.length == 5 && args[0].equals("bench")) {
                bench(Integer.parseInt(args[1]), Paths.get(args[2]),
                        Integer.parseInt(args[3]), Integer.parseInt(args[4]));
            } else if (args.length == 4 && args[0].equals("verify")) {
                verify(Integer.parseInt(args[1]), Paths.get(args[2]), Paths.get(args[3]));
            } else {
                System.err.println("usage: MatMulRunner bench <n> <data_dir> <warmups> <repetitions>");
                System.err.println("       MatMulRunner verify <n> <data_dir> <output_file>");
                System.exit(2);
            }
        } catch (OutOfMemoryError e) {
            System.out.println("{\"language\": \"java\", \"status\": \"oom\"}");
            System.out.flush();
            System.exit(EXIT_OOM);
        }
    }
}
