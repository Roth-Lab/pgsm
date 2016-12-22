from __future__ import division

from scipy.special import betaln as log_beta


class BinomialPriors(object):

    def __init__(self, a, b):
        self.a = a
        self.b = b


class BinomialParameters(object):

    def __init__(self, a, b, N):
        self.a = a
        self.b = b
        self.N = N

    def copy(self):
        return BinomialParameters(self.a, self.b, self.N)

    def decrement(self, x):
        self.a -= x[0]
        self.b -= x[1]
        self.N -= 1

    def increment(self, x):
        self.a += x[0]
        self.b += x[1]
        self.N += 1


class Binomial(object):

    def __init__(self, priors):
        self.priors = priors

    def create_params(self):
        return BinomialParameters(self.priors.a, self.priors.b, 0)

    def log_marginal_likelihood(self, params):
        return log_beta(params.a, params.b) - log_beta(self.priors.a, self.priors.b)

    def log_marginal_likelihood_diff(self, data_point, params):
        return log_beta(params.a + data_point[0], params.b + data_point[1]) - log_beta(params.a, params.b)
