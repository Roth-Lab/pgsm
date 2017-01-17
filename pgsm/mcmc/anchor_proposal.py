'''
Created on 9 Jan 2017

@author: Andrew Roth
'''
from __future__ import division

import numpy as np

from pgsm.math_utils import discrete_rvs, exp_normalize


class AnchorProposal(object):

    def propose_anchors(self, num_anchors):
        raise NotImplementedError()

    def update(self, clustering, data, dist, partition_prior):
        raise NotImplementedError()


class OriginalInformedProposal(object):

    def __init__(self, threshold=0.01):
        self.max_clusters_seen = 0

        self.threshold = threshold

        self.clusters_to_data = {}

        self.data_to_clusters = {}

    def propose_anchors(self, num_achors):
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

    def update(self, clustering, data, dist, partition_prior):
        clusters = np.unique(clustering)

        num_clusters = len(clusters)

        if num_clusters <= self.max_clusters_seen:
            return

        else:
            self.max_clusters_seen = num_clusters

        print 'Updating anchor proposal distribution'

        idx = np.arange(len(data))

        np.random.shuffle(idx)

        params = []

        for c in clusters:
            params.append(dist.create_params_from_data(data[clustering == c]))

            self.clusters_to_data[c] = np.where(clustering == c)[0].flatten()

        log_p = np.zeros(num_clusters)

        for i in idx:
            self.data_to_clusters[i] = []

            for c, block_params in enumerate(params):
                if clustering[i] == c:
                    block_params.decrement(data[i])

                if params[c].N == 0:
                    log_p[c] = float('-inf')

                else:
                    log_p[c] = partition_prior.log_tau_2(params[c].N) + \
                        dist.log_predictive_likelihood(data[i], block_params)

                if clustering[i] == c:
                    params[c].increment(data[i])

            p, _ = exp_normalize(log_p)

            for c, _ in enumerate(params):
                if p[c] >= self.threshold:
                    self.data_to_clusters[i].append(c)


class ClusterInformed(AnchorProposal):

    def update(self, clustering, data, dist, partition_prior):
        clusters = np.unique(clustering)

        num_clusters = len(clusters)

        if num_clusters <= self.max_clusters_seen:
            return

        else:
            self.max_clusters_seen = num_clusters

        params = []

        for c in clusters:
            params.append(dist.create_params_from_data(data[clustering == c]))

        for c_i in clusters:
            for c_j in clusters:
                if c_i == c_j:
                    continue

                merged_params = dist.create_params_from_data(data[(clustering == c_i) | (clustering == c_j)])

        self.num_data_points = data.shape[0]

        self.log_p = np.zeros((self.num_data_points, self.num_data_points))

        params = dist.create_params()

        log_seperate_margs = dist.log_predictive_likelihood_bulk(data, params)

        log_pairwise_margs = dist.log_pairwise_marginals(data, params)

        for i in range(self.num_data_points):
            for j in range(i):
                self.log_p[i, j] = log_pairwise_margs[i, j] - (log_seperate_margs[i] + log_seperate_margs[j])

                self.log_p[j, i] = self.log_p[i, j]


class UniformAnchorProposal(AnchorProposal):

    def propose_anchors(self, num_anchors):
        return np.random.choice(np.arange(self.num_data_points), replace=False, size=num_anchors)

    def update(self, clustering, data, dist, partition_prior):
        self.num_data_points = len(data)


class InformedAnchorProposal(AnchorProposal):

    def propose_anchors(self, num_anchors):
        anchor_1 = np.random.randint(0, self.num_data_points)

        log_p_anchor = self.log_p[anchor_1].copy()

#         x = np.percentile(log_p_anchor, 25)

#         log_p_anchor[log_p_anchor > x] = float('-inf')
        alpha = np.random.beta(2, 8) * 100

        if np.random.random() <= 0.5:
            #             log_p_anchor = -log_p_anchor
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
            p, _ = exp_normalize(log_p_anchor)

            anchor_2 = discrete_rvs(p)

        return anchor_1, anchor_2

    def setup(self, data, dist):
        self.num_data_points = data.shape[0]

        self.log_p = np.zeros((self.num_data_points, self.num_data_points))

        params = dist.create_params()

        log_seperate_margs = dist.log_predictive_likelihood_bulk(data, params)

        log_pairwise_margs = dist.log_pairwise_marginals(data, params)

        for i in range(self.num_data_points):
            for j in range(i):
                self.log_p[i, j] = log_pairwise_margs[i, j] - (log_seperate_margs[i] + log_seperate_margs[j])

                self.log_p[j, i] = self.log_p[i, j]
