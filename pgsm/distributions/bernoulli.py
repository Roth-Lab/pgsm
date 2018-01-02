'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
from scipy.special import betaln as log_beta

import numpy as np


class BetaPriors(object):

    def __init__(self, a, b):
        self.a = a

        self.b = b


class BernoulliParameters(object):

    def __init__(self, a, b, N):
        self.a = a

        self.b = b

        self.N = N

    def copy(self):
        return BernoulliParameters(self.a.copy(), self.b.copy(), self.N)

    def decrement(self, x):
        self.a -= x

        self.b -= (1 - x)

        self.N -= 1

    def increment(self, x):
        self.a += x

        self.b += (1 - x)

        self.N += 1


class BernoulliDistribution(object):

    def __init__(self, priors):
        self.priors = priors

    def create_params(self):
        return BernoulliParameters(self.priors.a.copy(), self.priors.b.copy(), 0)

    def create_params_from_data(self, data):
        params = self.create_params()

        for data_point in data:
            params.increment(data_point)

        return params

    def log_marginal_likelihood(self, params):
        return np.sum(log_beta(params.a, params.b) - log_beta(self.priors.a, self.priors.b))

    def log_predictive_likelihood(self, data_point, params):
        ll = log_beta(params.a + data_point, params.b + (1 - data_point))

        ll -= log_beta(params.a, params.b)

        return np.sum(ll)

    def log_predictive_likelihood_bulk(self, data, params):
        result = []

        for data_point in data:
            result.append(self.log_predictive_likelihood(data_point, params))

        return np.array(result)
