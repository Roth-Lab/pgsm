'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
from __future__ import division

from scipy.misc import logsumexp as log_sum_exp

import numpy as np


def cluster_entropy(clustering):
    _, size = np.unique(clustering, return_counts=True)

    frac = size / size.sum()

    entropy = -1 * np.sum(frac * np.log(frac))

    return entropy


def held_out_log_predicitive(clustering, dist, partition_prior, test_data, train_data):
    clustering = relabel_clustering(clustering)

    block_params = {}

    log_cluster_prior = {}

    block_ids = np.unique(clustering)

    num_blocks = len(block_ids)

    for z in block_ids:
        block_params[z] = dist.create_params()

        N_z = np.sum(clustering == z)

        log_cluster_prior[z] = partition_prior.log_tau_2_diff(N_z)

    block_params[num_blocks] = dist.create_params()

    log_cluster_prior[num_blocks] = partition_prior.log_tau_1_diff(num_blocks)

    weight_norm = log_sum_exp(log_cluster_prior.values())

    for z in log_cluster_prior:
        log_cluster_prior[z] -= weight_norm

    for x, z in zip(train_data, clustering):
        block_params[z].increment(x)

    log_p = np.zeros((test_data.shape[0], len(log_cluster_prior)))

    for z, w in log_cluster_prior.items():
        log_p[:, z] = w + dist.log_predictive_likelihood_bulk(test_data, block_params[z])

    return np.sum(log_sum_exp(log_p, axis=1))


def relabel_clustering(clustering):
    relabeled = []

    orig_block_ids, first_index = np.unique(clustering, return_index=True)

    blocks = list(orig_block_ids[np.argsort(first_index)])

    for c in clustering:
        relabeled.append(blocks.index(c))

    return np.array(relabeled, dtype=np.int)


def setup_split_merge(anchor_proposal, clustering, num_anchors):
    clustering = np.array(clustering, dtype=np.int)

    num_data_points = len(clustering)

    num_anchors = min(num_anchors, num_data_points)

    anchors = anchor_proposal.propose_anchors(num_anchors)

    anchor_clusters = set([clustering[a] for a in anchors])

    sigma = set()

    for c in anchor_clusters:
        sigma.update(np.argwhere(clustering == c).flatten())

    sigma = list(sigma)

    for x in anchors:
        sigma.remove(x)

    np.random.shuffle(sigma)

    return anchors, list(anchors) + sigma
