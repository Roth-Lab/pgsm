'''
Created on 9 Jan 2017

@author: Andrew Roth
'''
from __future__ import division

import numpy as np

from pgsm.math_utils import discrete_rvs, exp_normalize, log_sum_exp
from pgsm.utils import relabel_clustering


class SplitMergeSetupKernel(object):
    '''
    Base class for kernel to setup a split merge move.
    '''

    def __init__(self, data, dist, partition_prior, num_adaptation_iters=float('inf')):
        self.data = data

        self.dist = dist

        self.partition_prior = partition_prior

        self.num_adaptation_iters = num_adaptation_iters

        self.iter = 0

        self.num_data_points = len(data)

    def setup_split_merge(self, clustering, num_anchors):
        self.iter += 1

        clustering = np.array(clustering, dtype=np.int)

        if self.iter < self.num_adaptation_iters:
            self._update(clustering)

        num_data_points = len(clustering)

        num_anchors = min(num_anchors, num_data_points)

        anchors = self._propose_anchors(num_anchors)

        anchor_clusters = set([clustering[a] for a in anchors])

        sigma = set()

        for c in anchor_clusters:
            sigma.update(np.argwhere(clustering == c).flatten())

        sigma = list(sigma)

        for x in anchors:
            sigma.remove(x)

        np.random.shuffle(sigma)

        return anchors, list(anchors) + sigma

    def _propose_anchors(self, num_anchors):
        raise NotImplementedError()

    def _update(self, clustering):
        pass


class UniformSplitMergeSetupKernel(SplitMergeSetupKernel):
    '''
    Setup a split merge move by selecting anchors uniformly.
    '''

    def _propose_anchors(self, num_anchors):
        return np.random.choice(np.arange(self.num_data_points), replace=False, size=num_anchors)


class GibbsSplitMergeSetupKernel(SplitMergeSetupKernel):

    def __init__(self, data, dist, partition_prior, num_adaptation_iters=float('inf'), threshold=0.01):
        SplitMergeSetupKernel.__init__(self, data, dist, partition_prior, num_adaptation_iters=num_adaptation_iters)

        self.threshold = threshold

        self.max_clusters_seen = 0

    def _propose_anchors(self, num_achors):
        anchor_1 = np.random.randint(0, self.num_data_points)

        if anchor_1 not in self.data_to_clusters:
            self._set_data_to_clusters(anchor_1)

        if len(self.data_to_clusters[anchor_1]) == 0:
            return np.random.choice(np.arange(self.num_data_points), replace=False, size=2)

        cluster = np.random.choice(self.data_to_clusters[anchor_1])

        cluster_members = set(self.clusters_to_data[cluster])

        cluster_members.discard(anchor_1)

        if len(cluster_members) == 0:
            return np.random.choice(np.arange(len(self.data_to_clusters)), replace=False, size=2)

        anchor_2 = np.random.choice(list(cluster_members), replace=False, size=1)

        return int(anchor_1), int(anchor_2)

    def _update(self, clustering):
        num_clusters = len(np.unique(clustering))

        if num_clusters <= self.max_clusters_seen:
            return

        else:
            self.max_clusters_seen = num_clusters

        print 'Updating anchor proposal distribution'

        self.cluster_params = {}

        self.clusters_to_data = {}

        self.data_to_clusters = {}

        self.clustering = relabel_clustering(clustering)

        for c in np.unique(clustering):
            cluster_data = self.data[clustering == c]

            self.cluster_params[c] = self.dist.create_params_from_data(cluster_data)

            self.clusters_to_data[c] = np.where(clustering == c)[0].flatten()

    def _set_data_to_clusters(self, data_idx):
        data_point = self.data[data_idx]

        log_p = np.zeros(len(self.cluster_params))

        for c, block_params in self.cluster_params.items():
            if self.clustering[data_idx] == c:
                block_params.decrement(data_point)

            if block_params.N == 0:
                log_p[c] = float('-inf')

            else:
                log_p[c] = self.partition_prior.log_tau_2(block_params.N)

                log_p[c] += self.dist.log_predictive_likelihood(data_point, block_params)

            if self.clustering[data_idx] == c:
                block_params.increment(data_point)

        p, _ = exp_normalize(log_p)

        self.data_to_clusters[data_idx] = []

        for c in self.cluster_params:
            cluster_idx = self.cluster_params.keys().index(c)

            if p[cluster_idx] >= self.threshold:
                self.data_to_clusters[data_idx].append(c)


class ClusterInformedSplitMergeSetupKernel(SplitMergeSetupKernel):

    def __init__(self, data, dist, partition_prior, num_adaptation_iters=float('inf'), use_prior_weight=False):
        SplitMergeSetupKernel.__init__(self, data, dist, partition_prior, num_adaptation_iters=num_adaptation_iters)

        self.use_prior_weight = use_prior_weight

        self.max_clusters_seen = 0

    def _propose_anchors(self, num_achors):
        anchor_1 = np.random.randint(0, self.num_data_points)

        cluster_1 = self.data_to_clusters[anchor_1]

        cluster_2 = discrete_rvs(self.cluster_probs[cluster_1])

        cluster_members = set(self.clusters_to_data[cluster_2])

        cluster_members.discard(anchor_1)

        if len(cluster_members) == 0:
            anchor_1, anchor_2 = np.random.choice(np.arange(self.num_data_points), replace=False, size=2)

        else:
            anchor_2 = np.random.choice(list(cluster_members))

        return anchor_1, anchor_2

    def _update(self, clustering):
        num_clusters = len(np.unique(clustering))

        if num_clusters <= self.max_clusters_seen:
            return

        else:
            self.max_clusters_seen = num_clusters

        print 'Updating proposal'

        clustering = relabel_clustering(clustering)

        clusters = np.unique(clustering)

        self.cluster_probs = np.zeros((num_clusters, num_clusters))

        self.clusters_to_data = {}

        self.data_to_clusters = {}

        margs = {}

        for c in clusters:
            cluster_data = self.data[clustering == c]

            cluster_params = self.dist.create_params_from_data(cluster_data)

            margs[c] = self.dist.log_marginal_likelihood(cluster_params)

            if self.use_prior_weight:
                margs[c] += self.partition_prior.log_tau_2(cluster_params.N)

            self.clusters_to_data[c] = np.where(clustering == c)[0].flatten()

            for i in self.clusters_to_data[c]:
                self.data_to_clusters[i] = c

        for c_i in clusters:
            log_p = np.ones(num_clusters) * float('-inf')

            for c_j in clusters:
                if c_i == c_j:
                    continue

                merged_data = self.data[(clustering == c_i) | (clustering == c_j)]

                merged_params = self.dist.create_params_from_data(merged_data)

                merge_marg = self.dist.log_marginal_likelihood(merged_params)

                if self.use_prior_weight:
                    merge_marg += self.partition_prior.log_tau_2(merged_params.N)

                log_p[c_j] = merge_marg - (margs[c_i] + margs[c_j])

            log_p[c_i] = log_sum_exp(log_p)

            self.cluster_probs[c_i], _ = exp_normalize(log_p)


class PointInformedSplitMergeSetupKernel(SplitMergeSetupKernel):

    def __init__(self, data, dist, partition_prior, num_adaptation_iters=float('inf')):
        SplitMergeSetupKernel.__init__(self, data, dist, partition_prior, num_adaptation_iters=num_adaptation_iters)

        self.data_to_clusters = {}

        params = self.dist.create_params()

        self.log_seperate_margs = self.dist.log_predictive_likelihood_bulk(self.data, params)

    def _propose_anchors(self, num_anchors):
        anchor_1 = np.random.randint(0, self.num_data_points)

        if anchor_1 not in self.data_to_clusters:
            self._set_data_to_clusters(anchor_1)

        log_p_anchor = self.data_to_clusters[anchor_1].copy()

        do_merge = (np.random.random() <= 0.5)

        alpha = np.random.beta(1, 9) * 100

        if do_merge:
            x = np.percentile(log_p_anchor, alpha)

            log_p_anchor[log_p_anchor > x] = float('-inf')

            log_p_anchor[log_p_anchor <= x] = 0

        else:
            x = np.percentile(log_p_anchor, 100 - alpha)

            log_p_anchor[log_p_anchor > x] = 0

            log_p_anchor[log_p_anchor <= x] = float('-inf')

        log_p_anchor[anchor_1] = float('-inf')

        if np.isneginf(np.max(log_p_anchor)):
            idx = np.arange(self.num_data_points)

            idx = list(idx)

            idx.remove(anchor_1)

            anchor_2 = np.random.choice(idx)

        else:
            idx = np.where(~np.isneginf(log_p_anchor))[0].flatten()

            anchor_2 = np.random.choice(idx)

        return anchor_1, anchor_2

    def _update(self, clustering):
        pass

    def _set_data_to_clusters(self, i):
        self.data_to_clusters[i] = np.zeros(self.num_data_points)

        params = self.dist.create_params_from_data(self.data[i])

        log_pairwise_margs = self.dist.log_predictive_likelihood_bulk(self.data, params)

        for j in range(self.num_data_points):
            self.data_to_clusters[i][j] = log_pairwise_margs[j] - \
                (self.log_seperate_margs[i] + self.log_seperate_margs[j])
