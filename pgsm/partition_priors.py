from __future__ import division

import math

from pgsm.math_utils import log_factorial


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
