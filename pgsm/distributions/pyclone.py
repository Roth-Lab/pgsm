'''
Created on 19 Jan 2017

@author: Andrew Roth
'''
from __future__ import division

import numpy as np

from pgsm.math_utils import log_sum_exp, log_normalize


class PycloneParameters(object):

    def __init__(self, psi, N):
        self._psi = psi

        self.N = N

    @property
    def psi(self):
        return log_normalize(self._psi)

    def copy(self):
        return PycloneParameters(self.psi.copy(), self.N)

    def decrement(self, x):
        self._psi -= x

        self.N -= 1

    def increment(self, x):
        self._psi += x

        self.N += 1


class PyCloneDistribution(object):

    def __init__(self, grid_size):
        self.grid_size = grid_size

    def create_params(self):
        uniform_log_prior = -np.log(self.grid_size) * np.ones(self.grid_size)

        return PycloneParameters(uniform_log_prior, 0)

    def create_params_from_data(self, X):
        X = np.atleast_2d(X)

        return PycloneParameters(np.sum(X, axis=0), X.shape[0])

    def log_marginal_likelihood(self, params):
        return log_sum_exp(params._psi)

    def log_predictive_likelihood(self, data_point, params):
        return log_sum_exp(data_point + params.psi)

    def log_predictive_likelihood_bulk(self, data, params):
        log_p = np.zeros(len(data))

        for i, data_point in enumerate(data):
            log_p[i] = self.log_predictive_likelihood(data_point, params)

        return log_p

    def log_pairwise_marginals(self, data, params):
        num_data_points = len(data)

        log_p = np.zeros((num_data_points, num_data_points))

        for i in range(num_data_points):
            for j in range(num_data_points):
                if i == j:
                    continue

                params = self.create_params_from_data(data[[i, j]])

                log_p[i, j] = self.log_marginal_likelihood(params)

        return log_p
