'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
from __future__ import division

import math
import numba
import numpy as np


def discrete_rvs(p):
    return np.random.multinomial(1, p).argmax()


@numba.jit(cache=True, nopython=True)
def exp_normalize(log_p):
    log_p = np.array(log_p)
    log_norm = log_sum_exp(log_p)
    p = np.exp(log_p - log_norm)
    p = p / p.sum()
    return p, log_norm


@numba.jit(cache=True, nopython=True)
def log_sum_exp(log_X):
    '''
    Given a list of values in log space, log_X. Compute exp(log_X[0] + log_X[1] + ... log_X[n])

    Numerically safer than naive method.
    '''
    max_exp = np.max(log_X)
    if np.isinf(max_exp):
        return max_exp
    total = 0
    for x in log_X:
        total += np.exp(x - max_exp)
    return np.log(total) + max_exp


@numba.jit(cache=True, nopython=True)
def log_normalize(log_p):
    return log_p - log_sum_exp(log_p)


@numba.jit(cache=True, nopython=True)
def cholesky_update(L, x, alpha=1, inplace=True):
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


@numba.jit(cache=True, nopython=True)
def cholesky_log_det(X):
    return 2 * np.sum(np.log(np.diag(X)))


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
