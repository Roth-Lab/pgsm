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


@numba.jitclass([
    ('nu', numba.float64),
    ('r', numba.float64),
    ('u', numba.float64[:]),
    ('S_chol', numba.float64[:, :]),
    ('N', numba.int64),
])
class MultivariateNormalParameters(object):

    def __init__(self, nu, r, u, S_chol, N):
        self.nu = nu
        self.r = r
        self.u = u
        self.S_chol = S_chol
        self.N = N

    @property
    def log_det_S(self):
        return cholesky_log_det(self.S_chol)

    @property
    def S(self):
        return np.dot(self.S_chol, np.conj(self.S_chol.T))

    def copy(self):
        return MultivariateNormalParameters(self.nu, self.r, self.u.copy(), self.S_chol.copy(), self.N)

    def decrement(self, x):
        self.S_chol = cholesky_update(self.S_chol, np.sqrt(self.r / (self.r - 1)) * (x - self.u), -1)
        self.nu -= 1
        self.r -= 1
        self.u = ((self.r + 1) * self.u - x) / self.r
        self.N -= 1

    def increment(self, x):
        self.nu += 1
        self.r += 1
        self.u = ((self.r - 1) * self.u + x) / self.r
        self.N += 1
        self.S_chol = cholesky_update(self.S_chol, np.sqrt(self.r / (self.r - 1)) * (x - self.u), 1)


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
            0)

    def log_marginal_likelihood(self, params):
        D = self.priors.dim
        N = params.N
        d = np.arange(1, D + 1)
        return -0.5 * N * D * np.log(np.pi) + \
            0.5 * D * (np.log(self.priors.r) - np.log(params.r)) + \
            0.5 * (self.priors.nu * self.priors.log_det_S - params.nu * params.log_det_S) + \
            np.sum(log_gamma(0.5 * (params.nu + 1 - d)) - log_gamma(0.5 * (self.priors.nu + 1 - d)))

    def log_marginal_likelihood_diff(self, data_point, params):
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
        D = self.priors.dim
        u = (params.r * params.u + data_point) / (params.r + 1)
        diff = np.sqrt((params.r + 1) / params.r) * (data_point - u)
        S_chol = cholesky_update(params.S_chol, diff, 1, inplace=False)
        return -0.5 * D * np.log(np.pi) + \
            0.5 * D * (np.log(params.r) - np.log(params.r + 1)) + \
            0.5 * (params.nu * cholesky_log_det(params.S_chol) - (params.nu + 1) * cholesky_log_det(S_chol)) + \
            log_gamma(0.5 * (params.nu + 1)) - log_gamma(0.5 * (params.nu + 1 - D))
