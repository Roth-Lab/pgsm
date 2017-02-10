'''
Created on 9 Feb 2017

@author: Andrew Roth
'''
import numpy as np

from pgsm.math_utils import exp_normalize
from pgsm.utils import log_joint_probability, relabel_clustering


def get_all_clusterings(size):
    for partition in get_all_partitions(range(size)):
        clustering = np.zeros(size, dtype=int)

        for block_idx, block in enumerate(partition):
            for data_idx in block:
                clustering[data_idx] = block_idx

        yield clustering


def get_all_partitions(collection):
    if len(collection) == 1:
        yield [collection]

        return

    first = collection[0]

    for smaller in get_all_partitions(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]

        yield [[first]] + smaller


def get_exact_posterior(data, dist, partition_prior):
    '''
    Compute the exact posterior of the clustering model.

    Returns a dictionary mapping clusterings to posterior probability.
    '''
    log_p = []

    clusterings = []

    for c in get_all_clusterings(data.shape[0]):
        clusterings.append(tuple(relabel_clustering(c).astype(int)))

        log_p.append(log_joint_probability(c, data, dist, partition_prior))

    p, _ = exp_normalize(np.array(log_p))

    return dict(zip(clusterings, p))
