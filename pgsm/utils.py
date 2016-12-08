'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
import numpy as np


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