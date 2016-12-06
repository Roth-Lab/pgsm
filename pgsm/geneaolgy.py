from __future__ import division

from collections import defaultdict

import numpy as np


def iter_particles(particle):
    while particle is not None:
        yield particle
        particle = particle.parent_particle


def get_genealogy_length(particle):
    l = 0
    for _ in iter_particles(particle):
        l += 1
    return l


def get_clustering(particle):
    clusters = defaultdict(list)
    particles = reversed(list(iter_particles(particle)))
    for idx, p in enumerate(particles):
        clusters[p.value.block_idx].append(idx)
    return clusters


def get_cluster_labels(particle):
    N = get_genealogy_length(particle)
    clustering = get_clustering(particle)
    Z = np.zeros(N)
    for z, block in clustering.items():
        for idx in block:
            Z[idx] = z
    return Z
