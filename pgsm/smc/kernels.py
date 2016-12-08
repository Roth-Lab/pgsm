from __future__ import division

from scipy.misc import logsumexp as log_sum_exp

import numpy as np
import random

from pgsm.math_utils import exp_normalize
from pgsm.particle_utils import get_constrained_path


class SplitMergeParticle(object):

    def __init__(self, block_idx, log_g, log_w, parent_particle, posterior_params):
        self.block_idx = block_idx
        self.log_g = log_g
        self.log_w = log_w
        self.parent_particle = parent_particle
        self._posterior_params = posterior_params

    @property
    def posterior_params(self):
        return [x.copy() for x in self._posterior_params]

    def copy(self):
        return SplitMergeParticle(self.block_idx, self.log_g, self.log_w, self.parent_particle, self.posterior_params)


class SMCKernel(object):

    def __init__(self, dist, partition_prior):
        self.dist = dist
        self.partition_prior = partition_prior

    def create_initial_particle(self, data_point):
        return self.create_particle(0, data_point, None)

    def create_particle(self, block_idx, data_point, parent_particle, log_q=None):
        posterior_params = self._get_posterior_params(block_idx, data_point, parent_particle)
        log_g = self._compute_log_intermediate_target(posterior_params)
        if log_q is None:
            log_q = self._get_log_q(data_point, parent_particle)
        log_q_norm = log_sum_exp(log_q.values())
        if parent_particle is None:
            log_w = 0
        else:
            log_w = log_q_norm - parent_particle.log_g
        return SplitMergeParticle(
            block_idx=block_idx,
            log_g=log_g,
            log_w=log_w,
            parent_particle=parent_particle,
            posterior_params=posterior_params
        )

    def propose(self, data_point, parent_particle, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return self._propose(data_point, parent_particle)

    def setup(self, anchors, clustering, data, sigma):
        self.num_anchors = len(anchors)

        c_size = len(np.unique(clustering))
        c_bar_size = len(np.unique(anchors))
        self.num_cluster_diff = c_size - c_bar_size

        self.constrained_path = get_constrained_path(clustering[sigma], data[sigma], self)

    def _can_add_block(self, posterior_params):
        num_data_points = sum([x.ss.N for x in posterior_params])
        return (num_data_points < self.num_anchors) and (len(posterior_params) < self.num_anchors)

    def _compute_log_intermediate_target(self, posterior_params):
        block_sizes = [x.ss.N for x in posterior_params]
        log_g = self.partition_prior.log_tau_1(self.num_cluster_diff + len(block_sizes)) + \
            sum([self.partition_prior.log_tau_2(x) for x in block_sizes]) + \
            sum([self.dist.log_marginal_likelihood(x) for x in posterior_params])
        return log_g

    def _get_posterior_params(self, block_idx, data_point, parent_particle):
        if parent_particle is None:
            posterior_params = []
        else:
            posterior_params = parent_particle.posterior_params

        if block_idx > (len(posterior_params) - 1):
            posterior_params.append(self.dist.create_params(data_point))
        else:
            posterior_params[block_idx].increment(data_point)

        return posterior_params

    def _propose(self, data_point, parent_particle):
        log_q = self._get_log_q(data_point, parent_particle)
        block_probs = exp_normalize(log_q.values())
        block_idx = np.random.choice(log_q.keys(), p=block_probs)
        return self.create_particle(block_idx, data_point, parent_particle, log_q=log_q)


class NaiveSplitMergeKernel(SMCKernel):

    def _get_log_q(self, data_point, parent_particle):
        if parent_particle is None:
            posterior_params = []
        else:
            posterior_params = parent_particle.posterior_params

        log_q = {}

        for block_idx, _ in enumerate(posterior_params):
            log_q[block_idx] = 0

        if self._can_add_block(posterior_params):
            block_idx = len(posterior_params)
            log_q[block_idx] = 0

        return log_q


class FullyAdaptedSplitMergeKernel(SMCKernel):

    def _get_log_q(self, data_point, parent_particle):
        if parent_particle is None:
            posterior_params = []
        else:
            posterior_params = parent_particle.posterior_params

        log_q = {}

        for block_idx, params in enumerate(posterior_params):
            params.increment(data_point)
            log_q[block_idx] = self._compute_log_intermediate_target(posterior_params)
            params.decrement(data_point)

        if self._can_add_block(posterior_params):
            block_idx = len(posterior_params)
            posterior_params.append(self.dist.create_params(data_point))
            log_q[block_idx] = self._compute_log_intermediate_target(posterior_params)

        return log_q
