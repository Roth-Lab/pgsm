'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
from __future__ import division

from collections import defaultdict

import numpy as np

from pgsm.utils import relabel_clustering


def iter_particles(particle):
    while particle is not None:
        yield particle

        particle = particle.parent_particle


def get_log_normalisation(particle):
    log_Z = 0

    for particle in iter_particles(particle):
        log_Z += particle.log_w

    return log_Z


def get_genealogy_length(particle):
    l = 0

    for _ in iter_particles(particle):
        l += 1

    return l


def get_clustering(particle):
    clusters = defaultdict(list)

    particles = reversed(list(iter_particles(particle)))

    for idx, p in enumerate(particles):
        clusters[p.block_idx].append(idx)

    return clusters


def get_cluster_labels(particle):
    N = get_genealogy_length(particle)

    clustering = get_clustering(particle)

    Z = np.zeros(N)

    for z, block in clustering.items():
        for idx in block:
            Z[idx] = z

    return Z


def get_constrained_path(clustering, data, kernel):
    constrained_path = []

    clustering = relabel_clustering(clustering)

    particle = None

    for c, x in zip(clustering, data):
        particle = kernel.create_particle(c, x, particle)

        constrained_path.append(particle)

    return constrained_path
