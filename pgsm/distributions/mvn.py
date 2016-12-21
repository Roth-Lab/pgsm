from __future__ import division

import numpy as np

from pgsm.math_utils import cholesky_log_det, cholesky_update, log_gamma, outer_product


class MultivariateNormalSufficientStatistics(object):

    def __init__(self, X):
        X = np.atleast_2d(X)
        self.N = X.shape[0]
        self.X = np.sum(X, axis=0)
        self.S = np.dot(X.T, X)

    def copy(self):
        copy = MultivariateNormalSufficientStatistics.__new__(MultivariateNormalSufficientStatistics)
        copy.N = self.N
        copy.X = self.X.copy()
        copy.S = self.S.copy()
        return copy

    def decrement(self, x):
        self.N -= 1
        self.X -= x
        self.S -= outer_product(x, x)

    def increment(self, x):
        self.N += 1
        self.X += x
        self.S += outer_product(x, x)


class MultivariateNormalPriors(object):

    def __init__(self, dim):
        self.dim = dim

        self.nu = dim + 2
        self.r = 1
        self.S = np.eye(dim)
        self.u = np.zeros(dim)
        self.log_det_S = np.linalg.slogdet(self.S)[1]


class MultivariateNormalParameters(object):

    def __init__(self, priors, suff_stats):
        self.priors = priors
        self.suff_stats = suff_stats
        self._update_nu()
        self._update_r()
        self._update_u()
        self._update_S_chol()

    @property
    def log_det_S(self):
        return cholesky_log_det(self.S_chol)

    @property
    def S(self):
        return np.dot(self.S_chol, np.conj(self.S_chol.T))

    def copy(self):
        copy = MultivariateNormalParameters.__new__(MultivariateNormalParameters)
        copy.priors = self.priors
        copy.suff_stats = self.suff_stats.copy()
        copy.nu = self.nu
        copy.r = self.r
        copy.u = self.u.copy()
        copy.S_chol = self.S_chol.copy()
        return copy

    def decrement(self, x):
        self.suff_stats.decrement(x)
        self.S_chol = cholesky_update(self.S_chol, np.sqrt(self.r / (self.r - 1)) * (x - self.u), -1)
        self._update_nu()
        self._update_r()
        self._update_u()

    def increment(self, x):
        self.suff_stats.increment(x)
        self._update_nu()
        self._update_r()
        self._update_u()
        self.S_chol = cholesky_update(self.S_chol, np.sqrt(self.r / (self.r - 1)) * (x - self.u), 1)

    def _update_nu(self):
        self.nu = self.priors.nu + self.suff_stats.N

    def _update_r(self):
        self.r = self.priors.r + self.suff_stats.N

    def _update_u(self):
        self.u = ((self.priors.r * self.priors.u) + self.suff_stats.X) / self.r

    def _update_S_chol(self):
        S = self.priors.S + self.suff_stats.S + \
            self.priors.r * outer_product(self.priors.u, self.priors.u) - \
            self.r * outer_product(self.u, self.u)
        self.S_chol = np.linalg.cholesky(S)


class MultivariateNormal(object):

    def __init__(self, priors):
        self.priors = priors

    def create_params(self, x):
        suff_stats = MultivariateNormalSufficientStatistics(x)
        return MultivariateNormalParameters(self.priors, suff_stats)

    def log_marginal_likelihood(self, params):
        D = self.priors.dim
        N = params.suff_stats.N
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
