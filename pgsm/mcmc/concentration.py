'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
from __future__ import division

import numpy as np
import scipy.stats as stats


class GammaPriorConcentrationSampler(object):
    '''
    Gibbs update assuming a gamma prior on the concentration parameter.
    '''

    def __init__(self, a, b):
        '''
        Args :
            a : (float) Shape parameter of the gamma prior.
            b : (float) Rate parameter of the gamma prior.
        '''
        self.a = a
        self.b = b

    def sample(self, old_value, num_clusters, num_data_points):
        a = self.a
        b = self.b

        k = num_clusters
        n = num_data_points

        eta = stats.beta.rvs(old_value + 1, n)

        x = (a + k - 1) / (n * (b - np.log(eta)))

        pi = x / (1 + x)

        label = stats.bernoulli.rvs(pi)

        scale = b - np.log(eta)

        if label == 0:
            new_value = stats.gamma.rvs(a + k, scale)
        else:
            new_value = stats.gamma.rvs(a + k - 1, scale)

        return new_value
