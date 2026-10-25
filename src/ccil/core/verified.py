"""Rigorous floating-point helpers for certified bounds (#88).

Values a certificate classifies as certified must be upper bounds on the exact
real quantity, not floating-point estimates of it. Two tools provide that:

- verified_spectral_norm: a float s with ||diag(1/r) M diag(c)||_2 <= s,
  proved by checking that s^2 I - G is positive semidefinite in exact rational
  arithmetic, where G is the exact Gram matrix of the scaled M. The entries of
  M, r, and c are taken as the exact values of their floats.
- forward error bounds (Higham, Accuracy and Stability of Numerical
  Algorithms, 2nd ed., Sec. 3.1): with unit roundoff u and
  gamma_n = n u / (1 - n u), an n-term float64 dot product or sum is within
  gamma_n * sum |terms| of the exact value. upper_norm turns a computed vector
  and a componentwise bound on its error into an upper bound on the exact
  vector's norm.

They assume IEEE 754 binary64 arithmetic with round-to-nearest.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

U64 = 2.0 ** -53
U32 = 2.0 ** -24


def gamma(n: int, u: float = U64) -> float:
    return n * u / (1 - n * u)


def up(x):
    """Round a float or array up by one ulp (toward +inf)."""
    return np.nextafter(np.asarray(x, np.float64), np.inf)


def upper_norm(x: np.ndarray, err: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Row-wise upper bound on ||x_exact / scale||_2, given computed x and |x - x_exact| <= err componentwise."""
    x, err = np.asarray(x, np.float64), np.asarray(err, np.float64)
    d = x.shape[1]
    # ||x_exact / s|| <= ||x / s|| + ||err / s||; each computed norm (division, d squares, sum, sqrt) is within
    # gamma_{d + 3} relative of its exact value.
    g = gamma(d + 3)
    raw = (np.linalg.norm(x / scale, axis=1) + np.linalg.norm(err / scale, axis=1)) * (1 + g) / (1 - g) * (1 + 2 * U64)
    return np.where(raw == 0, 0.0, up(raw))  # a zero vector with no error is exactly zero


def sum_upper(total: np.ndarray, n_ops: int = 8) -> np.ndarray:
    """An upper bound on the exact value of a float64 sum or product chain of nonnegative terms."""
    return up(np.asarray(total, np.float64) * (1 + gamma(n_ops)))


def _exact(M, row_div, col_mul, minus=None):
    """The exact rationals diag(1/row_div) (M - minus) diag(col_mul)."""
    m, n = M.shape
    rd = [Fraction(float(v)) for v in (np.ones(m) if row_div is None else row_div)]
    cm = [Fraction(float(v)) for v in (np.ones(n) if col_mul is None else col_mul)]
    sub = (lambda i, j: Fraction(float(minus[i, j]))) if minus is not None else (lambda i, j: 0)
    return [[(Fraction(float(M[i, j])) - sub(i, j)) * cm[j] / rd[i] for j in range(n)] for i in range(m)]


def _gram(Q):
    m, n = len(Q), len(Q[0])
    if n > m:  # the smaller Gram matrix has the same nonzero spectrum
        Q = [list(col) for col in zip(*Q)]
        m, n = n, m
    return [[sum(Q[k][i] * Q[k][j] for k in range(m)) for j in range(n)] for i in range(n)]


def _is_psd(S) -> bool:
    """Exact positive semidefiniteness of a symmetric rational matrix, by LDL^T with zero-pivot checks."""
    S = [row[:] for row in S]
    n = len(S)
    for k in range(n):
        p = S[k][k]
        if p < 0:
            return False
        if p == 0:
            if any(S[k][j] != 0 for j in range(k + 1, n)):
                return False
            continue
        for i in range(k + 1, n):
            f = S[i][k] / p
            if f:
                for j in range(k + 1, n):
                    S[i][j] -= f * S[k][j]
    return True


def is_spectral_upper_bound(M, s: float, row_div=None, col_mul=None, minus=None) -> bool:
    """Exactly: is ||diag(1/row_div) (M - minus) diag(col_mul)||_2 <= s?"""
    if isinstance(s, bool) or not isinstance(s, (int, float)) or not (np.isfinite(s) and s >= 0):
        return False
    G = _gram(_exact(np.asarray(M, np.float64), row_div, col_mul, minus))
    s2 = Fraction(float(s)) ** 2
    return _is_psd([[(s2 if i == j else 0) - G[i][j] for j in range(len(G))] for i in range(len(G))])


def verified_spectral_norm(M, row_div=None, col_mul=None, minus=None, max_dim: int = 64) -> float:
    """A float proven (in exact arithmetic) to bound ||diag(1/row_div) (M - minus) diag(col_mul)||_2 from above."""
    M = np.asarray(M, np.float64)
    Q = _exact(M, row_div, col_mul, minus)
    if all(v == 0 for row in Q for v in row):
        return 0.0
    if min(M.shape) <= max_dim:
        est = float(np.linalg.norm(np.array([[float(v) for v in row] for row in Q]), 2))
        s = float(up(est * (1 + 2 ** -40)))
        for _ in range(16):
            if is_spectral_upper_bound(M, s, row_div, col_mul, minus):
                return s
            s = float(up(s * (1 + 2 ** -20)))
    # Frobenius norm, an upper bound on the spectral norm, with an exactly verified square root.
    fro2 = sum(v * v for row in Q for v in row)
    s = float(up(np.sqrt(float(fro2))))
    while Fraction(s) ** 2 < fro2:
        s = float(up(s))
    return s
