"""Naive dense matrix multiplication in pure Python (no NumPy in the kernel).

Matrices are represented as lists of lists of Python floats (IEEE-754 float64).
"""


def matmul(a, b):
    """Return C = A * B for square matrices using the classic i-j-k triple loop.

    Allocation and zero-initialisation of the result are part of this function
    and therefore part of the timed region in the benchmark runner.
    """
    n = len(a)
    c = [[0.0] * n for _ in range(n)]
    for i in range(n):
        a_row = a[i]
        c_row = c[i]
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += a_row[k] * b[k][j]
            c_row[j] = total
    return c


def checksum(c):
    """Return the sum of all elements of C (consumes the result)."""
    total = 0.0
    for row in c:
        for value in row:
            total += value
    return total
