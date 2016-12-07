from __future__ import division

import numpy as np

from pgsm.geneaolgy import get_cluster_labels, relabel_clustering
from pgsm.math_utils import exp_normalize


class ParticleGibbsSplitMerge(object):

    def __init__(self, kernel, pgs, num_anchors=2):
        self.kernel = kernel
        self.pgs = pgs
        
        self.num_anchors = num_anchors

    def sample(self, clustering, data, num_iters=1000):
        for _ in range(num_iters):
            anchors, sigma = self._setup_split_merge(clustering)
            self.kernel.setup(anchors, clustering, data, sigma)
            particles = self.pgs.sample(data[sigma], self.kernel)
            sampled_particle = self._sample_particle(particles)
            self._get_updated_clustering(clustering, sampled_particle, sigma)
        return clustering
    
    def _get_updated_clustering(self, clustering, particle, sigma):
        restricted_clustering = get_cluster_labels(particle)
        max_idx = clustering.max()
        clustering[sigma] = restricted_clustering + max_idx + 1
        return relabel_clustering(clustering)
    
    def _sample_particle(self, particles):
        probs = exp_normalize([x.log_w for x in particles])
        particle_idx = np.random.multinomial(1, probs).argmax()
        return particles[particle_idx]
    
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
        return anchors, list(anchors) + sigma
