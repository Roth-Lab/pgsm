'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
from __future__ import division

import numba
import numpy as np

from pgsm.math_utils import cholesky_log_det, cholesky_update, log_gamma


class MultivariateNormalPriors(object):

    def __init__(self, dim):
        self.dim = dim

        self.nu = dim + 2

        self.r = 1

        self.u = np.zeros(dim)

        self.S_chol = np.linalg.cholesky(np.eye(dim))

        self.log_det_S = cholesky_log_det(self.S_chol)

    @property
    def S(self):
        return np.dot(self.S_chol, np.conj(self.S_chol.T))


class MultivariateNormalParameters(object):
    __slots__ = ('nu', 'r', 'u', 'S_chol', 'N')

    def __init__(self, nu, r, u, S_chol, N):
        self.nu = nu

        self.r = r

        self.u = u

        self.S_chol = S_chol

        self.N = N

    @property
    def S(self):
        return np.dot(self.S_chol, np.conj(self.S_chol.T))

    def copy(self):
        return MultivariateNormalParameters(self.nu, self.r, self.u.copy(), self.S_chol.copy(), self.N)

    def decrement(self, x):
        self.nu, self.r, self.u, self.S_chol = _decrement_params(x, self.nu, self.r, self.u, self.S_chol, inplace=True)

        self.N -= 1

    def increment(self, x):
        self.nu, self.r, self.u, self.S_chol = _increment_params(x, self.nu, self.r, self.u, self.S_chol, inplace=True)

        self.N += 1


@numba.jit(nopython=True)
def _decrement_params(data_point, nu, r, u, S_chol, inplace=False):
    S_chol = cholesky_update(S_chol, np.sqrt(r / (r - 1)) * (data_point - u), -1, inplace=inplace)

    nu -= 1

    r -= 1

    u = ((r + 1) * u - data_point) / r

    return nu, r, u, S_chol


@numba.jit(nopython=True)
def _increment_params(data_point, nu, r, u, S_chol, inplace=False):
    nu += 1

    r += 1

    u = ((r - 1) * u + data_point) / r

    S_chol = cholesky_update(S_chol, np.sqrt(r / (r - 1)) * (data_point - u), 1, inplace=inplace)

    return nu, r, u, S_chol


class MultivariateNormalDistribution(object):

    def __init__(self, dim, priors=None):
        if priors is None:
            priors = MultivariateNormalPriors(dim)

        self.priors = priors

    def create_params(self):
        return MultivariateNormalParameters(
            self.priors.nu,
            self.priors.r,
            self.priors.u.copy(),
            self.priors.S_chol.copy(),
            0
        )

    def create_params_from_data(self, X):
        N = X.shape[0]

        nu = self.priors.nu + N

        r = self.priors.r + N

        u = (self.priors.r * self.priors.u + np.sum(X, axis=0)) / r

        S = self.priors.S + \
            np.dot(X.T, X) + \
            self.priors.r * np.outer(self.priors.u, self.priors.u) - \
            r * np.outer(u, u)

        S_chol = np.linalg.cholesky(S)

        return MultivariateNormalParameters(nu, r, u, S_chol, N)

    def log_marginal_likelihood(self, params):
        D = self.priors.dim

        N = params.N

        return _log_niw_marginal(
            D, N,
            params.nu, params.r, params.S_chol,
            self.priors.nu, self.priors.r, self.priors.S_chol)

    def log_predictive_likelihood(self, data_point, params):
        '''
        Compute difference between marginal log likelihood with and without data point.

        The implementation is a more efficient equivalent to doing the following.

        ```
        params.increment(data_point)
        diff = self.log_marginal_likelihood(params)
        params.decrement(data_point)
        diff -= self.log_marginal_likelihood(params)
        return diff
        ```

        '''
        return _log_predictive_likelihood(data_point, params.nu, params.r, params.u, params.S_chol)

    def log_predictive_likelihood_bulk(self, data, params):
        return _log_predictive_likelihood_bulk(data, params.nu, params.r, params.u, params.S_chol)

    def log_pairwise_marginals(self, data, params):
        return _log_pairwise_marginals(data, params.nu, params.r, params.u, params.S_chol)


@numba.jit(nopython=True)
def _log_pairwise_marginals(data, nu, r, u, S_chol):
    N = data.shape[0]

    log_p = np.zeros((N, N))

    for i in range(N):
        nu_new, r_new, u_new, S_chol_new = _increment_params(data[i], nu, r, u, S_chol, 1)

        for j in range(i + 1):
            log_p[i, j] = _log_predictive_likelihood(data[j], nu_new, r_new, u_new, S_chol_new)

            log_p[j, i] = log_p[i, j]

    return log_p


@numba.jit(nopython=True)
def _log_predictive_likelihood_bulk(data, nu, r, u, S_chol):
    N = data.shape[0]

    result = np.zeros(N, dtype=np.float64)

    for n in range(N):
        result[n] = _log_predictive_likelihood(data[n], nu, r, u, S_chol)

    return result


@numba.jit(nopython=True)
def _log_predictive_likelihood(data_point, nu, r, u, S_chol):
    D = len(data_point)

    nu_new, r_new, _, S_chol_new = _increment_params(data_point, nu, r, u, S_chol, inplace=False)

    return _log_niw_marginal(D, 1, nu_new, r_new, S_chol_new, nu, r, S_chol)


@numba.jit(nopython=True)
def _log_niw_marginal(D, N, nu_new, r_new, S_chol_new, nu_old, r_old, S_chol_old):
    d = np.arange(1, D + 1)

    return -0.5 * N * D * np.log(np.pi) + \
        0.5 * D * (np.log(r_old) - np.log(r_new)) + \
        0.5 * (nu_old * cholesky_log_det(S_chol_old) - nu_new * cholesky_log_det(S_chol_new)) + \
        np.sum(log_gamma(0.5 * (nu_new + 1 - d)) - log_gamma(0.5 * (nu_old + 1 - d)))
