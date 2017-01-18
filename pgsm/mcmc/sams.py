'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
from __future__ import division

import numpy as np

from pgsm.math_utils import discrete_rvs, log_normalize
# from pgsm.smc.kernels import SplitMergeParticle
# from pgsm.smc.samplers import ParticleSwarm
from pgsm.utils import relabel_clustering
# from pgsm.particle_utils import get_constrained_path, get_log_normalisation, get_clustering, get_cluster_labels


class SequentiallyAllocatedMergeSplitSampler(object):

    def __init__(self, dist, partition_prior, split_merge_setup_kernel):
        self.dist = dist

        self.partition_prior = partition_prior

        self.split_merge_setup_kernel = split_merge_setup_kernel

    def sample(self, clustering, data, num_iters=1):
        for _ in range(num_iters):
            clustering = self._sample(clustering, data)

        return clustering

    def _sample(self, clustering, data):
        anchors, sigma = self.split_merge_setup_kernel.setup_split_merge(clustering, 2)

        num_anchor_blocks = len(np.unique([clustering[a] for a in anchors]))

        num_global_blocks = len(np.unique(clustering))

        num_outside_blocks = num_global_blocks - num_anchor_blocks

        clustering_sigma = clustering[sigma]

        data_sigma = data[sigma]

        propose_merge = (clustering_sigma[0] != clustering_sigma[1])

        if propose_merge:
            merge_clustering, merge_mh_factor = self._merge(data_sigma, num_outside_blocks)

            split_clustering, split_mh_factor = self._split(
                data_sigma,
                num_outside_blocks,
                constrained_path=clustering_sigma
            )

            forward_factor = merge_mh_factor

            reverse_factor = split_mh_factor

            restricted_clustering = np.array(merge_clustering, dtype=int)

        else:
            merge_clustering, merge_mh_factor = self._merge(data_sigma, num_outside_blocks)

            split_clustering, split_mh_factor = self._split(data_sigma, num_outside_blocks)

            forward_factor = split_mh_factor

            reverse_factor = merge_mh_factor

            restricted_clustering = np.array(split_clustering, dtype=int)

        log_ratio = forward_factor - reverse_factor

        u = np.random.random()

        if log_ratio >= np.log(u):
            max_idx = clustering.max()

            clustering[sigma] = restricted_clustering + max_idx + 1

            clustering = relabel_clustering(clustering)

        return clustering

    def _merge(self, data, num_outside_blocks):
        clustering = np.zeros(len(data), dtype=np.int)

        log_q = 0

        params = [self.dist.create_params_from_data(data), ]

        log_p = self._log_target(num_outside_blocks, params)

        mh_factor = log_p - log_q

        return clustering, mh_factor

    def _split(self, data, num_outside_blocks, constrained_path=None):
        num_blocks = 2

        if constrained_path is not None:
            constrained_path = relabel_clustering(constrained_path)

        clustering = [0, 1]

        params = [
            self.dist.create_params_from_data(data[0]),
            self.dist.create_params_from_data(data[1]),
        ]

        log_q = 0

        for i, data_point in enumerate(data[num_blocks:], num_blocks):
            log_block_probs = np.zeros(num_blocks, dtype=float)

            for block_idx, cluster_params in enumerate(params):
                log_block_probs[block_idx] = self.partition_prior.log_tau_2_diff(cluster_params.N)

                log_block_probs[block_idx] += self.dist.log_predictive_likelihood(data_point, cluster_params)

            log_block_probs = log_normalize(log_block_probs)

            if constrained_path is None:
                block_probs = np.exp(log_block_probs)

                block_probs = block_probs / block_probs.sum()

                c = discrete_rvs(block_probs)

            else:
                c = constrained_path[i]

            clustering.append(c)

            log_q += log_block_probs[c]

            params[c].increment(data_point)

        log_p = self._log_target(num_outside_blocks, params)

        mh_factor = log_p - log_q

        return clustering, mh_factor

    def _log_target(self, num_outside_blocks, params):
        num_anchors = len(params)

        log_p = self.partition_prior.log_tau_1(num_anchors + num_outside_blocks)

        for block_params in params:
            log_p += self.partition_prior.log_tau_2(block_params.N)

            log_p += self.dist.log_marginal_likelihood(block_params)

        return log_p
