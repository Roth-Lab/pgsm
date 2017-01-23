'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
from __future__ import division

import numpy as np

from pgsm.particle_utils import get_constrained_path, get_log_normalisation, get_cluster_labels
from pgsm.smc.kernels import FullyAdaptedSplitMergeKernel
from pgsm.utils import relabel_clustering

from pgsm.smc.kernels import SplitMergeParticle


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

        u = np.random.random()

        if log_ratio >= np.log(u):
            max_idx = clustering.max()

            clustering[sigma] = restricted_clustering + max_idx + 1

            clustering = relabel_clustering(clustering)

        return clustering

    def _merge(self, data):
        clustering = np.zeros(len(data), dtype=np.int)

        particle = get_constrained_path(clustering, data, self.kernel)[-1]

        return clustering, get_log_normalisation(particle)

    def _split(self, data, constrained_clustering=None):
        if constrained_clustering is None:
            params = []

            params.append(self.dist.create_params_from_data(data[0]))

            particle = SplitMergeParticle(0, params, 1, 0, None)

            params.append(self.dist.create_params_from_data(data[1]))

            particle = SplitMergeParticle(1, params, 2, self.kernel.log_target_density(params), particle)

            for data_point in data[2:]:
                particle = self.kernel.propose(data_point, particle)

        else:
            particle = get_constrained_path(constrained_clustering, data, self.kernel)[-1]

        clustering = get_cluster_labels(particle)

        return clustering, get_log_normalisation(particle)
