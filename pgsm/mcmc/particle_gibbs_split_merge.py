from __future__ import division

import numpy as np

from pgsm.math_utils import discrete_rvs
from pgsm.particle_utils import get_cluster_labels
from pgsm.utils import setup_split_merge, relabel_clustering

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
            use_annealed=True):
        if use_annealed:
            kernel = pgsm.smc.kernels.AnnealedSplitMergeKernel(dist, partition_prior)
        else:
            kernel = pgsm.smc.kernels.FullyAdaptedSplitMergeKernel(dist, partition_prior)
        smc_sampler = pgsm.smc.samplers.ImplicitParticleGibbsSampler(
            num_particles,
            resample_threshold=resample_threshold,
        )
        return cls(kernel, smc_sampler, num_anchors=num_anchors)

    def __init__(self, kernel, smc_sampler, num_anchors=None):
        self.kernel = kernel
        self.smc_sampler = smc_sampler

        self.num_anchors = num_anchors

    @property
    def dist(self):
        return self.kernel.dist

    @property
    def partition_prior(self):
        return self.kernel.partition_prior

    def sample(self, clustering, data, num_iters=1):
        for _ in range(num_iters):
            anchors, sigma = self._setup_split_merge(clustering)
            self.kernel.setup(anchors, clustering, data, sigma)
            particles_weights = self.smc_sampler.sample(data[sigma], self.kernel)
            sampled_particle = self._sample_particle(particles_weights)
            self._get_updated_clustering(clustering, sampled_particle, sigma)
            clustering = relabel_clustering(clustering)
        return clustering

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
            num_anchors = np.random.poisson(0.8) + 2
            num_anchors = min(num_anchors, 6)
        else:
            num_anchors = self.num_anchors
        return setup_split_merge(clustering, num_anchors)
