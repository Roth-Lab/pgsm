from __future__ import division

from scipy.misc import logsumexp as log_sum_exp

import math
import numba
import numpy as np


def exp_normalize(log_p):
    log_p = np.array(log_p)
    p = np.exp(log_p - log_sum_exp(log_p))
    return p / p.sum()


@numba.jit(nopython=True)
def cholesky_update(L, x, alpha=1, inplace=False):
    """ Rank one update of a Cholesky factorized matrix.
    """
    dim = len(x)
    x = x.copy()
    if not inplace:
        L = L.copy()
    for i in range(dim):
        r = np.sqrt(L[i, i] ** 2 + alpha * x[i] ** 2)
        c = r / L[i, i]
        s = x[i] / L[i, i]
        L[i, i] = r
        idx = i + 1
        L[idx:dim, i] = (L[idx:dim, i] + alpha * s * x[idx:dim]) / c
        x[idx:dim] = c * x[idx:dim] - s * L[idx:dim, i]
    return L


def log_factorial(x):
    return log_gamma(x + 1)


def log_binomial_coefficient(n, x):
    return log_factorial(n) - log_factorial(x) - log_factorial(n - x)


@numba.vectorize(["float64(float64)", "int64(float64)"])
def log_gamma(x):
    return math.lgamma(x)


@numba.jit
def outer_product(x, y):
    I = len(x)
    J = len(y)
    result = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            result[i, j] = x[i] * y[j]
    return result
