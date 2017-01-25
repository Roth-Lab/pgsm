'''
Created on 25 Jan 2017

@author: Andrew Roth
'''
import numpy as np


class MockParams(object):

    def __init__(self, N):
        self.N = N

    def copy(self):
        return MockParams(self.N)

    def decrement(self, x):
        self.N -= 1

    def increment(self, x):
        self.N += 1


class MockDistribution(object):

    def create_params(self):
        return MockParams(0)

    def create_params_from_data(self, X):
        X = np.atleast_2d(X)
        return MockParams(len(X))

    def log_marginal_likelihood(self, params):
        return 0

    def log_predictive_likelihood(self, data_point, params):
        return 0

    def log_predictive_likelihood_bulk(self, data, params):
        return 0


class MockPartitionPrior(object):

    def log_tau_1(self, x):
        return 0

    def log_tau_2(self, x):
        return 0

    def log_tau_1_diff(self, x):
        return 0

    def log_tau_2_diff(self, x):
        return 0
