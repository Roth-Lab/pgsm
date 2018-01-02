from scipy.misc import logsumexp as log_sum_exp

import numpy as np

from pgsm.math_utils import log_normalize


def cluster_entropy(clustering):
    _, size = np.unique(clustering, return_counts=True)

    frac = size / size.sum()

    entropy = -1 * np.sum(frac * np.log(frac))

    return entropy


def held_out_log_predicitive(clustering, dist, partition_prior, test_data, train_data, per_point=False):
    clustering = relabel_clustering(clustering)

    block_params = []

    log_cluster_prior = []

    block_ids = sorted(np.unique(clustering))

    for z in block_ids:
        params = dist.create_params_from_data(train_data[clustering == z])

        block_params.append(params)

        log_cluster_prior.append(partition_prior.log_tau_2_diff(params.N))

    num_blocks = len(block_ids)

    block_params.append(dist.create_params())

    log_cluster_prior.append(partition_prior.log_tau_1_diff(num_blocks))

    log_cluster_prior = np.array(log_cluster_prior)

    log_cluster_prior = log_normalize(log_cluster_prior)

    log_p = np.zeros((test_data.shape[0], len(log_cluster_prior)))

    for z, (w, params) in enumerate(zip(log_cluster_prior, block_params)):
        log_p[:, z] = w + dist.log_predictive_likelihood_bulk(test_data, params)

    if per_point:
        return log_sum_exp(log_p, axis=1)

    else:
        return np.sum(log_sum_exp(log_p, axis=1))


def log_joint_probability(clustering, data, dist, partition_prior):
    ll = partition_prior.log_tau_1(len(np.unique(clustering)))

    for z in np.unique(clustering):
        params = dist.create_params_from_data(data[clustering == z])

        ll += partition_prior.log_tau_2(params.N)

        ll += dist.log_marginal_likelihood(params)

    return ll


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
