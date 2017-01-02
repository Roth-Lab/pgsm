'''
Created on 2 Jan 2017

@author: Andrew Roth
'''
from pgsm.mcmc.collapsed_gibbs import CollapsedGibbsSampler
from pgsm.mcmc.particle_gibbs_split_merge import ParticleGibbsSplitMergeSampler
from pgsm.mcmc.sams import SequentiallyAllocatedMergeSplitSampler


class MixedSampler(object):

    def __init__(self, dist, partition_prior, gibbs_per_split_merge=1, split_merge_sampler='pgsm', **kwargs):
        self.dist = dist
        self.partition_prior = partition_prior

        self.gibbs_per_split_merge = gibbs_per_split_merge

        self.gibbs_sampler = CollapsedGibbsSampler(dist, partition_prior)
        if split_merge_sampler == 'pgsm':
            self.split_merge_sampler = ParticleGibbsSplitMergeSampler.create_from_dist(dist, partition_prior, **kwargs)
        elif split_merge_sampler == 'sams':
            self.split_merge_sampler = SequentiallyAllocatedMergeSplitSampler(dist, partition_prior)
        else:
            raise Exception('Unknown split merge sampler')

    def sample(self, clustering, data, num_iters=1):
        for _ in range(num_iters):
            clustering = self.gibbs_sampler.sample(clustering, data, num_iters=self.gibbs_per_split_merge)
            clustering = self.split_merge_sampler.sample(clustering, data, num_iters=1)
        return clustering
