'''
Created on 2 Jan 2017

@author: Andrew Roth
'''
import numpy as np

from pgsm.mcmc.concentration import GammaPriorConcentrationSampler


class DirichletProcessSampler(object):

    def __init__(self, partition_sampler, init_alpha=1.0, prior_a=1.0, prior_b=1.0):
        self.partition_sampler = partition_sampler

        self.concentration_sampler = GammaPriorConcentrationSampler(prior_a, prior_b)

    @property
    def alpha(self):
        return self.partition_prior.alpha

    @property
    def dist(self):
        return self.partition_sampler.dist

    @property
    def partition_prior(self):
        return self.partition_sampler.partition_prior

    def sample(self, clustering, data, num_iters=1):
        for _ in range(num_iters):
            clustering = self.partition_sampler.sample(clustering, data, num_iters=1)
            self.partition_sampler.partition_prior.alpha = self.concentration_sampler.sample(
                self.partition_sampler.partition_prior.alpha,
                len(np.unique(clustering)),
                len(clustering),
            )
        return clustering
