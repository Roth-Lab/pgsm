from __future__ import division

import numpy as np

from pgsm.geneaolgy import get_cluster_labels
from pgsm.kernels import NaiveSplitMergeKernel
from pgsm.math_utils import exp_normalize


class ParticleGibbsSplitMerge(object):

    def __init__(self, dist, partition_prior, pgs, kernel_cls=NaiveSplitMergeKernel, num_anchors=2):
        self.dist = dist
        self.partition_prior = partition_prior
        self.pgs = pgs
        
        self.kernel_cls = kernel_cls
        self.num_anchors = num_anchors

    def sample(self, data, init_clustering=None, num_iters=1000):
        clustering = self._init_clustering(data, init_clustering)
        for _ in range(num_iters):
            self.num_anchors = np.random.randint(2, 10)
            sigma = self._setup_split_merge(clustering)
            num_cluster_diff = len(np.unique(clustering)) - len(np.unique(clustering[sigma[:self.num_anchors]]))
            kernel = self.kernel_cls(
                self.dist,
                num_cluster_diff, 
                self.partition_prior,
                num_anchors=self.num_anchors)
            constrained_path = self._get_constrained_path(clustering[sigma], data[sigma], kernel)
            particles = self.pgs.sample(constrained_path, data[sigma], kernel)
            probs = exp_normalize([x.log_w for x in particles])
            particle_idx = np.random.multinomial(1, probs).argmax()
            sampled_particle = particles[particle_idx]
            restricted_clustering = get_cluster_labels(sampled_particle)
            max_idx = clustering.max()
            # clustering[sigma] = restricted_clustering + max_idx
            for sigma_idx, z in enumerate(restricted_clustering):
                clustering[sigma[sigma_idx]] = z + max_idx + 1
            _, clustering = np.unique(clustering, return_inverse=True)
        return clustering

    def _get_constrained_path(self, clustering, data, kernel):
        constrained_path = [None,]
        anchors = []
        for a in clustering[:self.num_anchors]:
            if a not in anchors:
                anchors.append(a)
        for i in range(data.shape[0]):
            for block_idx, a in enumerate(anchors):
                if clustering[i] == a:
                    constrained_path.append(kernel.create_particle(block_idx, data[i], constrained_path[-1]))
        return constrained_path

    def _init_clustering(self, data, init_clustering):
        if init_clustering is None:
            clustering = np.zeros(data.shape[0])
        elif isinstance(init_clustering, np.ndarray):
            clustering = init_clustering
        elif init_clustering == 'separate':
            clustering = np.arange(data.shape[0])
        return clustering

    def _setup_split_merge(self, clustering):
        anchors = np.random.choice(np.arange(len(clustering)), replace=False, size=self.num_anchors)
        anchor_clusters = set([clustering[a] for a in anchors])
        sigma = set()
        for a in anchor_clusters:
            sigma.update(np.argwhere(clustering == a).flatten())
        sigma = list(sigma)
        for x in anchors:
            sigma.remove(x)
        np.random.shuffle(sigma)
        return list(anchors) + sigma
