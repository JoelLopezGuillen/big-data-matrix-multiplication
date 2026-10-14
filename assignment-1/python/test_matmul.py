"""Correctness tests for the pure-Python naive matrix multiplication kernel."""

import pytest

from matmul import checksum, matmul


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def zeros(n):
    return [[0.0] * n for _ in range(n)]


def assert_matrix_close(actual, expected, tol=1e-12):
    assert len(actual) == len(expected)
    for row_actual, row_expected in zip(actual, expected):
        assert row_actual == pytest.approx(row_expected, abs=tol, rel=tol)


def test_identity_right_multiplication_returns_original_matrix():
    a = [[float(3 * i + j) - 2.5 for j in range(5)] for i in range(5)]
    assert matmul(a, identity(5)) == a


def test_zero_matrix_yields_zero_product():
    a = [[float(i - j) for j in range(4)] for i in range(4)]
    assert matmul(a, zeros(4)) == zeros(4)
    assert matmul(zeros(4), a) == zeros(4)


def test_known_2x2_product():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    assert matmul(a, b) == [[19.0, 22.0], [43.0, 50.0]]


def test_known_3x3_product_with_non_integer_values():
    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]
    b = [[-1.0, 0.5, 2.0], [3.0, -2.0, 1.0], [0.25, 4.0, -3.0]]
    expected = [[5.75, 8.5, -5.0], [12.5, 16.0, -5.0], [19.5, 27.5, -8.0]]
    assert_matrix_close(matmul(a, b), expected)


def test_product_is_not_commutative_for_known_3x3_case():
    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]
    b = [[-1.0, 0.5, 2.0], [3.0, -2.0, 1.0], [0.25, 4.0, -3.0]]
    assert matmul(a, b) != matmul(b, a)


def test_checksum_is_sum_of_all_elements():
    assert checksum([[19.0, 22.0], [43.0, 50.0]]) == 134.0
