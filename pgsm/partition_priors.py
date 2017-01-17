'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
from __future__ import division

import math

from pgsm.math_utils import log_factorial, log_gamma


class PartitionPrior(object):

    def log_likelihood(self, block_sizes):
        return self.log_tau_1(len(block_sizes)) + sum([self.log_tau_2(x) for x in block_sizes])

    def log_tau_1(self, x):
        raise NotImplementedError()

    def log_tau_2(self, x):
        raise NotImplementedError()


class DirichletProcessPartitionPrior(PartitionPrior):

    def __init__(self, alpha):
        self.alpha = alpha

    def log_tau_1(self, x):
        return x * math.log(self.alpha)

    def log_tau_2(self, x):
        return log_factorial(x - 1)

    def log_tau_1_diff(self, x):
        return math.log(self.alpha)

    def log_tau_2_diff(self, x):
        if x == 0:
            return 0

        else:
            return math.log(x)


class FiniteDirichletPartitionPrior(PartitionPrior):

    def __init__(self, alpha, dim):
        self.alpha = alpha

        self.dim = dim

    def log_tau_1(self, x):
        if x <= self.dim:
            return 0

        else:
            return float('-inf')

    def log_tau_2(self, x):
        return log_gamma(x + self.alpha)

    def log_tau_1_diff(self, x):
        if x <= (self.dim - 1):
            return 0

        else:
            return float('-inf')

    def log_tau_2_diff(self, x):
        return math.log(x + self.alpha)
