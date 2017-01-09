'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
from __future__ import division

import numpy as np

from pgsm.math_utils import discrete_rvs, log_normalize
from pgsm.utils import relabel_clustering, setup_split_merge


class SequentiallyAllocatedMergeSplitSampler(object):

    def __init__(self, dist, partition_prior):
        self.dist = dist

        self.partition_prior = partition_prior

    def sample(self, clustering, data, num_iters=1):
        for _ in range(num_iters):
            clustering = self._sample(clustering, data)

        return relabel_clustering(clustering)

    def _sample(self, clustering, data):
        anchors, sigma = setup_split_merge(clustering, 2)

        num_anchor_blocks = len(np.unique([clustering[a] for a in anchors]))

        num_global_blocks = len(np.unique(clustering))

        num_outside_blocks = num_global_blocks - num_anchor_blocks

        clustering_sigma = clustering[sigma]

        data_sigma = data[sigma]

        merge_clustering, merge_mh_factor = self._merge(data_sigma, num_outside_blocks)

        if clustering[anchors[0]] == clustering[anchors[1]]:
            split_clustering, split_mh_factor = self._split(data_sigma, num_outside_blocks)

            forward_factor = split_mh_factor

            reverse_factor = merge_mh_factor

            restricted_clustering = np.array(split_clustering, dtype=int)

        else:
            split_clustering, split_mh_factor = self._split(
                data_sigma,
                num_outside_blocks,
                constrained_path=clustering_sigma
            )

            forward_factor = merge_mh_factor

            reverse_factor = split_mh_factor

            restricted_clustering = np.array(merge_clustering, dtype=int)

        log_ratio = forward_factor - reverse_factor

        u = np.random.random()

        if log_ratio >= np.log(u):
            max_idx = clustering.max()

            clustering[sigma] = restricted_clustering + max_idx + 1

            clustering = relabel_clustering(clustering)

        return clustering

    def _merge(self, data, num_outside_blocks):
        num_blocks = 1

        clustering = np.zeros(len(data), dtype=np.int)

        log_q = 0

        params = self.dist.create_params_from_data(data)

        log_p = self.partition_prior.log_tau_1(num_outside_blocks + num_blocks) + \
            self.partition_prior.log_tau_2(params.N) + self.dist.log_marginal_likelihood(params)

        mh_factor = log_p - log_q

        return clustering, mh_factor

    def _split(self, data, num_outside_blocks, constrained_path=None):
        num_blocks = 2

        if constrained_path is not None:
            constrained_path = relabel_clustering(constrained_path)

        clustering = range(num_blocks)

        params = [self.dist.create_params(), self.dist.create_params()]

        for i in range(num_blocks):
            params[i].increment(data[i])

        log_q = 0

        log_block_probs = np.zeros(num_blocks)

        for i, data_point in enumerate(data[num_blocks:]):
            for block_idx, cluster_params in enumerate(params):
                log_block_probs[block_idx] = self.partition_prior.log_tau_2_diff(cluster_params.N) + \
                    self.dist.log_predictive_likelihood(data_point, cluster_params)

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

        log_p = self.partition_prior.log_tau_1(num_outside_blocks + num_blocks) + \
            sum([self.partition_prior.log_tau_2(x.N) + self.dist.log_marginal_likelihood(x) for x in params])

        mh_factor = log_p - log_q

        return clustering, mh_factor
