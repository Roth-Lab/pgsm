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


def mean_log_prediction(clustering, cluster_prior, data, dist, held_out):
    clustering = relabel_clustering(clustering)
    block_params = {}
    log_cluster_prior = {}
    block_ids = np.unique(clustering)
    num_blocks = len(block_ids)
    for z in block_ids:
        block_params[z] = dist.create_params()
        N_z = np.sum(clustering == z)
        log_cluster_prior[z] = cluster_prior.log_tau_2_diff(N_z)
    block_params[num_blocks] = dist.create_params()
    log_cluster_prior[num_blocks] = cluster_prior.log_tau_1_diff(num_blocks)
    weight_norm = log_sum_exp(log_cluster_prior.values())
    for z in log_cluster_prior:
        log_cluster_prior[z] -= weight_norm
    for x, z in zip(data, clustering):
        block_params[z].increment(x)
    log_p = np.zeros((held_out.shape[0], len(log_cluster_prior)))
    for z, w in log_cluster_prior.items():
        log_p[:, z] = w + dist.log_predictive_likelihood_bulk(held_out, block_params[z])
    return np.sum(log_sum_exp(log_p, axis=1)) / len(held_out)


def relabel_clustering(clustering):
    relabeled = []
    orig_block_ids, first_index = np.unique(clustering, return_index=True)
    blocks = list(orig_block_ids[np.argsort(first_index)])
    for c in clustering:
        relabeled.append(blocks.index(c))
    return np.array(relabeled, dtype=np.int)


def setup_split_merge(clustering, num_anchors):
    clustering = np.array(clustering, dtype=np.int)
    num_data_points = len(clustering)
    num_anchors = min(num_anchors, num_data_points)
    anchors = np.random.choice(np.arange(num_data_points), replace=False, size=num_anchors)
    anchor_clusters = set([clustering[a] for a in anchors])
    sigma = set()
    for c in anchor_clusters:
        sigma.update(np.argwhere(clustering == c).flatten())
    sigma = list(sigma)
    for x in anchors:
        sigma.remove(x)
    np.random.shuffle(sigma)
    return anchors, list(anchors) + sigma
