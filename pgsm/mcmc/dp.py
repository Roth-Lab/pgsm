'''
Created on 2 Jan 2017

@author: Andrew Roth
'''
import copy
import numpy as np

from pgsm.mcmc.concentration import GammaPriorConcentrationSampler


class DirichletProcessSampler(object):

    def __init__(self, partition_sampler, prior_a=1.0, prior_b=1.0):
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


class CoupledDirichletProcessSplitMerge(object):

    def __init__(self, sampler, init_aux_clustering=None, prior_a=1.0, prior_b=1.0):

        self.sampler_1 = DirichletProcessSampler(sampler, prior_a=prior_a, prior_b=prior_b)

        self.sampler_2 = copy.deepcopy(self.sampler_1)

        self.sampler_1.partition_sampler.split_merge_setup_kernel.num_adaptation_iters = 0

        self.sampler_2.partition_sampler.split_merge_setup_kernel.num_adaptation_iters = 0

        self.aux_clustering = init_aux_clustering

    @property
    def alpha(self):
        return self.sampler_1.partition_prior.alpha

    @property
    def dist(self):
        return self.sampler_1.partition_sampler.dist

    @property
    def partition_prior(self):
        return self.sampler_1.partition_sampler.partition_prior

    def sample(self, clustering, data, num_iters=1):
        for _ in range(num_iters):
            clustering = self._sample(clustering, data)

        return clustering

    def _sample(self, clustering, data):
        if self.aux_clustering is None:
            self.aux_clustering = clustering.copy()

        self.sampler_1.partition_sampler.split_merge_setup_kernel.update(self.aux_clustering)

        self.sampler_2.partition_sampler.split_merge_setup_kernel.update(clustering)

        clustering = self.sampler_1.sample(clustering, data)

        self.aux_clustering = self.sampler_2.sample(self.aux_clustering, data)

        return clustering
