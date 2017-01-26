'''
Created on 2 Jan 2017

@author: Andrew Roth
'''
from pgsm.mcmc.collapsed_gibbs import CollapsedGibbsSampler


class MixedSampler(object):

    def __init__(self, dist, partition_prior, split_merge_sampler, gibbs_per_split_merge=1):
        self.dist = dist

        self.partition_prior = partition_prior

        self.gibbs_per_split_merge = gibbs_per_split_merge

        self.gibbs_sampler = CollapsedGibbsSampler(dist, partition_prior)

        self.split_merge_sampler = split_merge_sampler

    @property
    def split_merge_setup_kernel(self):
        return self.split_merge_sampler.split_merge_setup_kernel

    def sample(self, clustering, data, num_iters=1):
        for _ in range(num_iters):
            clustering = self.gibbs_sampler.sample(clustering, data, num_iters=self.gibbs_per_split_merge)

            clustering = self.split_merge_sampler.sample(clustering, data, num_iters=1)

        return clustering
