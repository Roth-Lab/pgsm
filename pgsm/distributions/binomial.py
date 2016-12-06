from __future__ import division

from scipy.special import betaln as log_beta

import numpy as np

from pgsm.math_utils import log_binomial_coefficient


class BinomialSufficientStatistics(object):

    def __init__(self, X):
        X = np.atleast_2d(X)
        self.N = X.shape[0]
        self.a = np.sum(X[:, 0])
        self.b = np.sum(X[:, 1])

    def copy(self):
        copy = BinomialSufficientStatistics.__new__(BinomialSufficientStatistics)
        copy.N = self.N
        copy.a = self.a
        copy.b = self.b
        return copy

    def decrement(self, x):
        self.N -= 1
        self.a -= x[0]
        self.b -= x[1]

    def increment(self, x):
        self.N += 1
        self.a += x[0]
        self.b += x[1]


class BinomialPriors(object):

    def __init__(self, a, b):
        self.a = a
        self.b = b


class BinomialParameters(object):

    def __init__(self, priors, ss):
        self.priors = priors
        self.ss = ss
        self.a = priors.a + ss.a
        self.b = priors.b + ss.b

    def copy(self):
        copy = BinomialParameters.__new__(BinomialParameters)
        copy.priors = self.priors
        copy.ss = self.ss.copy()
        copy.a = self.a
        copy.b = self.b
        return copy

    def decrement(self, x):
        self.ss.decrement(x)
        self._update()

    def increment(self, x):
        self.ss.increment(x)
        self._update()

    def _update(self):
        self.a = self.priors.a + self.ss.a
        self.b = self.priors.b + self.ss.b


class Binomial(object):

    def __init__(self, priors):
        self.priors = priors

    def create_params(self, x):
        ss = BinomialSufficientStatistics(x)
        return BinomialParameters(self.priors, ss)

    def log_marginal_likelihood(self, params):
        return 0 + log_beta(params.a, params.b) - log_beta(self.priors.a, self.priors.b)
    
    def incremental_log_marginal_likelihood(self, data_point, params):
        params.increment(data_point)
        ll = self.log_marginal_likelihood(params)
        params.decrement(data_point)
        ll -= self.log_marginal_likelihood(params)
        return ll
