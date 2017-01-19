'''
Created on 9 Jan 2017

@author: Andrew Roth
'''
from __future__ import division

import numpy as np

from pgsm.math_utils import discrete_rvs, exp_normalize
from pgsm.utils import relabel_clustering


class SplitMergeSetupKernel(object):
    '''
    Base class for kernel to setup a split merge move.
    '''

    def __init__(self, data, dist, partition_prior):
        self.data = data

        self.dist = dist

        self.partition_prior = partition_prior

        self.num_data_points = len(data)

    def setup_split_merge(self, clustering, num_anchors):
        clustering = np.array(clustering, dtype=np.int)

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

    def __init__(self, data, dist, partition_prior, threshold=0.01):
        SplitMergeSetupKernel.__init__(self, data, dist, partition_prior)

        self.threshold = threshold

        self.max_clusters_seen = 0

    def _propose_anchors(self, num_achors):
        anchor_1 = np.random.randint(0, len(self.data_to_clusters))

        if len(self.data_to_clusters[anchor_1]) == 0:
            return np.random.choice(np.arange(len(self.data_to_clusters)), replace=False, size=2)

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

        self.clusters_to_data = {}

        self.data_to_clusters = {}

        clustering = relabel_clustering(clustering)

        clusters = np.unique(clustering)

        idx = np.arange(len(self.data))

        np.random.shuffle(idx)

        params = []

        for c in clusters:
            cluster_data = self.data[clustering == c]

            cluster_params = self.dist.create_params_from_data(cluster_data)

            params.append(cluster_params)

            self.clusters_to_data[c] = np.where(clustering == c)[0].flatten()

        log_p = np.zeros(num_clusters)

        for i in idx:
            data_point = self.data[i]

            self.data_to_clusters[i] = []

            for c, block_params in enumerate(params):
                if clustering[i] == c:
                    block_params.decrement(data_point)

                if params[c].N == 0:
                    log_p[c] = float('-inf')

                else:
                    log_p[c] = self.partition_prior.log_tau_2(params[c].N)

                    log_p[c] += self.dist.log_predictive_likelihood(data_point, block_params)

                if clustering[i] == c:
                    params[c].increment(data_point)

            p, _ = exp_normalize(log_p)

            for c, _ in enumerate(params):
                if p[c] >= self.threshold:
                    self.data_to_clusters[i].append(c)


class ClusterInformedSplitMergeSetupKernel(SplitMergeSetupKernel):

    def __init__(self, data, dist, partition_prior, threshold=0.01):
        SplitMergeSetupKernel.__init__(self, data, dist, partition_prior)

        self.clusters_to_data = {}

        self.data_to_clusters = {}

        self.max_clusters_seen = 0

    def _propose_anchors(self, num_achors):
        anchor_1 = np.random.randint(0, self.num_data_points)

        cluster_1 = self.data_to_clusters[anchor_1]

        log_p = self.log_p[cluster_1]

        p, _ = exp_normalize(log_p)

        cluster_2 = discrete_rvs(p)

        cluster_members = set(self.clusters_to_data[cluster_2])

        cluster_members.discard(anchor_1)

        if len(cluster_members) == 0:
            anchor_1, anchor_2 = np.random.choice(np.arange(self.num_data_points), replace=False, size=2)

        else:
            anchor_2 = np.random.choice(list(cluster_members))

        return int(anchor_1), int(anchor_2)

    def _update(self, clustering):
        num_clusters = len(np.unique(clustering))

        if num_clusters <= self.max_clusters_seen:
            return

        else:
            self.max_clusters_seen = num_clusters

        print 'Updating proposal'

        clustering = relabel_clustering(clustering)

        clusters = np.unique(clustering)

        self.clusters_to_data = {}

        self.data_to_clusters = {}

        self.log_p = np.zeros((num_clusters, num_clusters))

        margs = {}

        for c in clusters:
            cluster_data = self.data[clustering == c]

            cluster_params = self.dist.create_params_from_data(cluster_data)

            margs[c] = self.dist.log_marginal_likelihood(cluster_params)

            self.clusters_to_data[c] = np.where(clustering == c)[0].flatten()

            for i in self.clusters_to_data[c]:
                self.data_to_clusters[i] = c

        for c_i in clusters:
            for c_j in clusters:
                if c_i == c_j:
                    continue

                merged_data = self.data[(clustering == c_i) | (clustering == c_j)]

                merged_params = self.dist.create_params_from_data(merged_data)

                self.log_p[c_i, c_j] = self.dist.log_marginal_likelihood(merged_params) - (margs[c_i] + margs[c_j])


class PointInformedSplitMergeSetupKernel(SplitMergeSetupKernel):

    def __init__(self, data, dist, partition_prior):
        SplitMergeSetupKernel.__init__(self, data, dist, partition_prior)

        self.log_p = None

    def _propose_anchors(self, num_anchors):
        anchor_1 = np.random.randint(0, self.num_data_points)

        log_p_anchor = self.log_p[anchor_1].copy()

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
        if self.log_p is not None:
            return

        self.log_p = np.zeros((self.num_data_points, self.num_data_points))

        params = self.dist.create_params()

        log_seperate_margs = self.dist.log_predictive_likelihood_bulk(self.data, params)

        log_pairwise_margs = self.dist.log_pairwise_marginals(self.data, params)

        for i in range(self.num_data_points):
            for j in range(i):
                self.log_p[i, j] = log_pairwise_margs[i, j] - (log_seperate_margs[i] + log_seperate_margs[j])

                self.log_p[j, i] = self.log_p[i, j]


class AdaptiveInformedSplitMergeSetupKernel(SplitMergeSetupKernel):

    def __init__(self, data, dist, partition_prior):
        SplitMergeSetupKernel.__init__(self, data, dist, partition_prior)

        self.max_clusters_seen = 0

    def _propose_anchors(self, num_anchors):
        if num_anchors != 2:
            raise Exception('Informed anchor proposal only works for 2 anchors.')

        cluster_1 = np.random.randint(0, len(self.clusters_to_data))

        is_singleton = (len(self.clusters_to_data[cluster_1]) == 1)

        not_single_cluster = (len(self.clusters_to_data) > 1)

        do_merge = (np.random.random() < 0.5)

        # Merge
        if is_singleton or (not_single_cluster and do_merge):
            anchor_1 = np.random.choice(self.clusters_to_data[cluster_1])

            p, _ = exp_normalize(self.log_p_merge[cluster_1])

            cluster_2 = discrete_rvs(p)

            anchor_2 = np.random.choice(self.clusters_to_data[cluster_2])

        # Split
        else:
            anchor_1, anchor_2 = np.random.choice(self.clusters_to_data[cluster_1], replace=False, size=2)

        return anchor_1, anchor_2

    def _update(self, clustering):
        num_clusters = len(np.unique(clustering))

        if num_clusters <= self.max_clusters_seen:
            return

        else:
            self.max_clusters_seen = num_clusters

        clustering = relabel_clustering(clustering)

        clusters = np.unique(clustering)

        print 'Updating proposal'

        self.clusters_to_data = {}

        self.data_to_clusters = {}

        for c in clusters:
            self.clusters_to_data[c] = np.where(clustering == c)[0].flatten()

            for i in self.clusters_to_data[c]:
                self.data_to_clusters[i] = c

        self.log_p_merge = np.zeros((num_clusters, num_clusters))

        for cluster_1 in clusters:
            cluster_1_params = self.dist.create_params_from_data(self.data[clustering == cluster_1])

            cluster_1_log_marg = self.partition_prior.log_tau_2(cluster_1_params.N) + \
                self.dist.log_marginal_likelihood(cluster_1_params)

            for cluster_2 in clusters:
                if cluster_1 == cluster_2:
                    self.log_p_merge[cluster_1, cluster_2] = float('-inf')

                else:
                    cluster_2_params = self.dist.create_params_from_data(self.data[clustering == cluster_2])

                    cluster_2_log_marg = self.partition_prior.log_tau_2(cluster_2_params.N) + \
                        self.dist.log_marginal_likelihood(cluster_2_params)

                    merged_data = self.data[(clustering == cluster_1) | (clustering == cluster_2)]

                    merged_params = self.dist.create_params_from_data(merged_data)

                    merged_log_marg = self.partition_prior.log_tau_2(merged_params.N) + \
                        self.dist.log_marginal_likelihood(merged_params)

                    self.log_p_merge[cluster_1, cluster_2] = merged_log_marg - \
                        (cluster_1_log_marg + cluster_2_log_marg)
