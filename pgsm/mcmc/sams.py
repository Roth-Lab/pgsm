'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
from __future__ import division

import numpy as np

from pgsm.particle_utils import get_log_normalisation, get_cluster_labels
from pgsm.smc.kernels import FullyAdaptedSplitMergeKernel
from pgsm.utils import relabel_clustering


class SequentiallyAllocatedMergeSplitSampler(object):

    def __init__(self, dist, partition_prior, split_merge_setup_kernel):
        self.dist = dist

        self.partition_prior = partition_prior

        self.split_merge_setup_kernel = split_merge_setup_kernel

        self.kernel = FullyAdaptedSplitMergeKernel(self.dist, self.partition_prior)

    def sample(self, clustering, data, num_iters=1):
        for _ in range(num_iters):
            clustering = self._sample(clustering, data)

        return clustering

    def _sample(self, clustering, data):
        anchors, sigma = self.split_merge_setup_kernel.setup_split_merge(clustering, 2)

        self.kernel.setup(anchors, clustering, data, sigma)

        clustering_sigma = clustering[sigma]

        data_sigma = data[sigma]

        propose_merge = (clustering_sigma[0] != clustering_sigma[1])

        if propose_merge:
            merge_clustering, merge_mh_factor = self._merge(data_sigma)

            split_clustering, split_mh_factor = self._split(data_sigma, constrained_clustering=clustering_sigma)

            forward_factor = merge_mh_factor

            reverse_factor = split_mh_factor

            restricted_clustering = merge_clustering

        else:
            merge_clustering, merge_mh_factor = self._merge(data_sigma)

            split_clustering, split_mh_factor = self._split(data_sigma)

            forward_factor = split_mh_factor

            reverse_factor = merge_mh_factor

            restricted_clustering = split_clustering

        log_ratio = forward_factor - reverse_factor

#         print split_mh_factor, merge_mh_factor, log_ratio

        u = np.random.random()

        if log_ratio >= np.log(u):
            max_idx = clustering.max()

            clustering[sigma] = restricted_clustering + max_idx + 1

            clustering = relabel_clustering(clustering)

        return clustering

    def _merge(self, data):
        particle = self.kernel.create_particle(
            0,
            data[0],
            None,
            log_q={0: 0}
        )

        particle = self.kernel.create_particle(
            0,
            data[1],
            particle,
            log_q={0: self.kernel.log_target_density([self.dist.create_params_from_data(data[:2]), ])}
        )

        for data_point in data[2:]:
            particle = self.kernel.propose(data_point, particle)

        clustering = get_cluster_labels(particle)

        log_mh_factor = get_log_normalisation(particle)

        return clustering, log_mh_factor

    def _split(self, data, constrained_clustering=None):
        particle = self.kernel.create_particle(
            0,
            data[0],
            None,
            log_q={0: self.dist.log_marginal_likelihood(self.dist.create_params_from_data(data[0]))}
        )

        particle = self.kernel.create_particle(
            1,
            data[1],
            particle,
            log_q={1: self.dist.log_marginal_likelihood(self.dist.create_params_from_data(data[1]))}
        )

        if constrained_clustering is None:
            for data_point in data[2:]:
                particle = self.kernel.propose(data_point, particle)

        else:
            constrained_clustering = relabel_clustering(constrained_clustering)

            for block_idx, data_point in zip(constrained_clustering[2:], data[2:]):
                particle = self.kernel.create_particle(block_idx, data_point, particle)

        clustering = get_cluster_labels(particle)

        log_mh_factor = get_log_normalisation(particle)

        return clustering, log_mh_factor
