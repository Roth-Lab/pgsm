'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
import numpy as np

from pgsm.math_utils import discrete_rvs, log_normalize
from pgsm.utils import relabel_clustering, setup_split_merge


class SequentiallyAllocatedMergeSplitSampler(object):

    def __init__(self, dist, partition_prior):
        self.dist = dist
        self.partition_prior = partition_prior

    def sample(self, clustering, data, num_iters=1000):
        for _ in range(num_iters):
            clustering = self._sample(clustering, data)
        return clustering

    def _sample(self, clustering, data):
        anchors, sigma = setup_split_merge(clustering, 2)
        _, unchanged_block_sizes = np.unique([x for x in clustering if x not in sigma], return_counts=True)
        merge_clustering, merge_mh_factor = self._merge(data, sigma, unchanged_block_sizes)
        split_clustering, split_mh_factor = self._split(data, sigma, unchanged_block_sizes)
        if clustering[anchors[0]] == clustering[anchors[1]]:
            forward_factor = split_mh_factor
            reverse_factor = merge_mh_factor
            restricted_clustering = np.array(split_clustering, dtype=int)
        else:
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

    def _merge(self, data, sigma, unchanged_block_sizes):
        clustering = [0 for _ in range(len(sigma))]
        log_q = 0
        params = [self.dist.create_params(), ]
        for x in data[sigma]:
            params[0].increment(x)
        block_sizes = list(unchanged_block_sizes) + [x.N for x in params]
        log_p = self._log_p(block_sizes, params)
        mh_factor = log_p - log_q
        return clustering, mh_factor

    def _split(self, data, sigma, unchanged_block_sizes):
        clustering = [0, 1]
        log_q = 0
        params = [self.dist.create_params(), self.dist.create_params()]
        for i in range(2):
            params[i].increment(data[sigma[i]])
        for idx in sigma[2:]:
            data_point = data[idx]
            log_block_probs = []
            for cluster_params in params:
                log_block_probs.append(
                    np.log(cluster_params.N + 1) + self.dist.log_marginal_likelihood_diff(data_point, cluster_params)
                )
            log_block_probs = np.array(log_block_probs)
            log_block_probs = log_normalize(log_block_probs)
            block_probs = np.exp(log_block_probs)
            block_probs = block_probs / block_probs.sum()
            c = discrete_rvs(block_probs)
            clustering.append(c)
            log_q += log_block_probs[c]
            params[c].increment(data_point)
        block_sizes = list(unchanged_block_sizes) + [x.N for x in params]
        log_p = self._log_p(block_sizes, params)
        mh_factor = log_p - log_q
        return clustering, mh_factor

    def _log_p(self, block_sizes, params):
        return self.partition_prior.log_likelihood(block_sizes) + \
            sum([self.dist.log_marginal_likelihood(x) for x in params])
