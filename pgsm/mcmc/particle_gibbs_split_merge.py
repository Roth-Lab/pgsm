'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
from __future__ import division

from pgsm.math_utils import discrete_rvs
from pgsm.particle_utils import get_cluster_labels
from pgsm.utils import relabel_clustering

import pgsm


class ParticleGibbsSplitMergeSampler(object):

    @classmethod
    def create_from_dist(
            cls,
            dist,
            partition_prior,
            num_anchors=2,
            num_particles=20,
            resample_threshold=0.5,
            setup_kernel=None,
            use_annealed=True):

        if use_annealed:
            smc_kernel = pgsm.smc.kernels.AnnealedSplitMergeKernel(dist, partition_prior)

        else:
            smc_kernel = pgsm.smc.kernels.FullyAdaptedSplitMergeKernel(dist, partition_prior)

        if setup_kernel is None:
            setup_kernel = pgsm.mcmc.split_merge_setup.UniformSplitMergeSetupKernel()

        smc_sampler = pgsm.smc.samplers.ImplicitParticleGibbsSampler(
            num_particles,
            resample_threshold=resample_threshold,
        )

        return cls(smc_kernel, smc_sampler, setup_kernel, num_anchors=num_anchors)

    def __init__(self, smc_kernel, smc_sampler, split_merge_setup_kernel, num_anchors=None):
        self.smc_kernel = smc_kernel

        self.smc_sampler = smc_sampler

        self.num_anchors = num_anchors

        self.split_merge_setup_kernel = split_merge_setup_kernel

    @property
    def dist(self):
        return self.smc_kernel.dist

    @property
    def partition_prior(self):
        return self.smc_kernel.partition_prior

    def sample(self, clustering, data, num_iters=1):
        for _ in range(num_iters):
            anchors, sigma = self._setup_split_merge(clustering)

            self.smc_kernel.setup(anchors, clustering, data, sigma)

            particles_weights = self.smc_sampler.sample(data[sigma], self.smc_kernel)

            sampled_particle = self._sample_particle(particles_weights)

            self._get_updated_clustering(clustering, sampled_particle, sigma)

            clustering = relabel_clustering(clustering)

        return clustering

    def setup(self, clustering, data):
        self.anchor_proposal.update(clustering, data, self.smc_kernel.dist, self.smc_kernel.partition_prior)

    def _get_updated_clustering(self, clustering, particle, sigma):
        restricted_clustering = get_cluster_labels(particle)

        max_idx = clustering.max()

        clustering[sigma] = restricted_clustering + max_idx + 1

        return relabel_clustering(clustering)

    def _sample_particle(self, particles_weights):
        particles = particles_weights.keys()

        weights = particles_weights.values()

        particle_idx = discrete_rvs(weights)

        return particles[particle_idx]

    def _setup_split_merge(self, clustering):
        if self.num_anchors is None:
            p = [0.6, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.0125]

            num_anchors = discrete_rvs(p) + 2

        else:
            num_anchors = self.num_anchors

        return self.split_merge_setup_kernel.setup_split_merge(clustering, num_anchors)
