'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
from __future__ import division

from scipy.misc import logsumexp as log_sum_exp

import numpy as np


def cluster_entropy(clustering):
    entropy = 0
    for z in np.unique(clustering):
        size = np.sum(clustering == z)
        frac = size / len(clustering)
        entropy -= frac * np.log(frac)
    return entropy


def mean_log_prediction(clustering, cluster_prior, data, dist, held_out):
    block_params = {}
    weights = {}
    for z in np.unique(clustering):
        block_params[z] = dist.create_params()
        N_z = np.sum(clustering == z)
        weights[z] = cluster_prior.log_tau_2_diff(N_z)
    block_params['new'] = dist.create_params()
    weights['new'] = cluster_prior.log_tau_1_diff(len(np.unique(clustering)))
    weight_norm = log_sum_exp(weights.values())
    for z in weights:
        weights[z] -= weight_norm
    for x, z in zip(data, clustering):
        block_params[z].increment(x)
    log_p = 0
    for x in held_out:
        log_p_x = []
        for z, w in weights.items():
            log_p_x.append(w + dist.log_marginal_likelihood_diff(x, block_params[z]))
        log_p += log_sum_exp(log_p_x)
    return log_p


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
