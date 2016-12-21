from __future__ import division

from scipy.misc import logsumexp as log_sum_exp

import numpy as np
import random

from pgsm.math_utils import exp_normalize
from pgsm.particle_utils import get_constrained_path


class SplitMergeParticle(object):

    def __init__(self, block_idx, log_w, parent_particle, posterior_params):
        self.block_idx = block_idx
        self.log_w = log_w
        self.parent_particle = parent_particle
        self._posterior_params = posterior_params

    @property
    def posterior_params(self):
        return [x.copy() for x in self._posterior_params]

    def copy(self):
        return SplitMergeParticle(self.block_idx, self.log_w, self.parent_particle, self.posterior_params)


# class AnnealedSplitMergeParticle(SplitMergeParticle):
#
#     def __init__(
#             self,
#             block_idx,
#             generation,
#             log_g_anchor,
#             log_g,
#             log_w,
#             parent_particle,
#             posterior_params):
#
#         SplitMergeParticle.__init__(self, block_idx, log_g, log_w, parent_particle, posterior_params)
#         self.generation = generation
#         self.log_g_anchor = log_g_anchor
#
#     def copy(self):
#         return AnnealedSplitMergeParticle(
#             self.block_idx,
#             self.generation,
#             self.log_g_anchor,
#             self.log_g,
#             self.log_w,
#             self.parent_particle,
#             self.posterior_params)


class SMCKernel(object):

    def __init__(self, dist, partition_prior):
        self.dist = dist
        self.partition_prior = partition_prior

    def create_initial_particle(self, data_point):
        return self.create_particle(0, data_point, None)

    def create_particle(self, block_idx, data_point, parent_particle, log_q=None, log_q_norm=None):
        posterior_params = self._get_posterior_params(block_idx, data_point, parent_particle)
        if log_q is None:
            log_q = self._get_log_q(data_point, parent_particle)
        if log_q_norm is None:
            log_q_norm = log_sum_exp(log_q.values())
        if parent_particle is None:
            log_w = 0
        else:
            log_w = log_q_norm
        return SplitMergeParticle(
            block_idx=block_idx,
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
        self.num_generations = len(sigma)

        c_size = len(np.unique(clustering))
        c_bar_size = len(np.unique(anchors))
        self.num_cluster_diff = c_size - c_bar_size

        self.constrained_path = get_constrained_path(clustering[sigma], data[sigma], self)

    def _can_add_block(self, posterior_params):
        num_data_points = sum([x.suff_stats.N for x in posterior_params])
        return (num_data_points < self.num_anchors) and (len(posterior_params) < self.num_anchors)

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
        block_probs, log_q_norm = exp_normalize(log_q.values())
        block_idx = np.random.choice(log_q.keys(), p=block_probs)
        return self.create_particle(block_idx, data_point, parent_particle, log_q=log_q, log_q_norm=log_q_norm)

#
# class NaiveSplitMergeKernel(SMCKernel):
#
#     def _get_log_q(self, data_point, parent_particle):
#         if parent_particle is None:
#             posterior_params = []
#         else:
#             posterior_params = parent_particle.posterior_params
#
#         log_q = {}
#
#         for block_idx, _ in enumerate(posterior_params):
#             log_q[block_idx] = 0
#
#         if self._can_add_block(posterior_params):
#             block_idx = len(posterior_params)
#             log_q[block_idx] = 0
#
#         return log_q


class FullyAdaptedSplitMergeKernel(SMCKernel):

    def _get_log_q(self, data_point, parent_particle):
        if parent_particle is None:
            posterior_params = []
        else:
            posterior_params = parent_particle.posterior_params

        log_q = {}

        for block_idx, params in enumerate(posterior_params):
            log_q[block_idx] = self.partition_prior.log_tau_2_diff(params.suff_stats.N) + \
                self.dist.log_marginal_likelihood_diff(data_point, params)

        if self._can_add_block(posterior_params):
            block_idx = len(posterior_params)
            params = self.dist.create_params(data_point)
            # TODO: Check this missing global cluster diff
            log_q[block_idx] = self.partition_prior.log_tau_1_diff(len(posterior_params)) + \
                self.partition_prior.log_tau_2_diff(1) + self.dist.log_marginal_likelihood(params)

        return log_q


# class AnnealedSplitMergeKernel(SMCKernel):
#
#     def create_particle(self, block_idx, data_point, parent_particle, log_q=None):
#         if parent_particle is None:
#             generation = 1
#         else:
#             generation = parent_particle.generation + 1
#
#         posterior_params = self._get_posterior_params(block_idx, data_point, parent_particle)
#
#         if log_q is None:
#             log_q = self._get_log_q(data_point, parent_particle)
#         log_q_norm = log_sum_exp(log_q.values())
#
#         if generation <= self.num_anchors:
#             log_g = -np.log(len(log_q))
#         else:
#             log_g = self._compute_log_intermediate_target(posterior_params)
#
#         if parent_particle is None:
#             log_w = 0
#         else:
#             log_w = log_q_norm - parent_particle.log_g
#
#         if generation < self.num_anchors:
#             log_g_anchor = None
#         elif generation == self.num_anchors:
#             log_g_anchor = log_g
#         else:
#             log_g_anchor = parent_particle.log_g_anchor
#
#         return AnnealedSplitMergeParticle(
#             block_idx=block_idx,
#             generation=generation,
#             log_g_anchor=log_g_anchor,
#             log_g=log_g,
#             log_w=log_w,
#             parent_particle=parent_particle,
#             posterior_params=posterior_params
#         )
#
#     def _get_log_q(self, data_point, parent_particle):
#         if parent_particle is None:
#             generation = 1
#             posterior_params = []
#         else:
#             generation = parent_particle.generation + 1
#             posterior_params = parent_particle.posterior_params
#
#         log_q = {}
#
#         if generation <= self.num_anchors:
#             for block_idx, _ in enumerate(posterior_params):
#                 log_q[block_idx] = 0
#
#             if self._can_add_block(posterior_params):
#                 block_idx = len(posterior_params)
#                 log_q[block_idx] = 0
#
#         else:
#             log_annealing_correction = self._get_log_annealing_correction(parent_particle)
#             for block_idx, params in enumerate(posterior_params):
#                 params.increment(data_point)
#                 log_q[block_idx] = log_annealing_correction + self._compute_log_intermediate_target(posterior_params)
#                 params.decrement(data_point)
#
#         return log_q
#
#     def _get_log_annealing_correction(self, parent_particle):
#         t = parent_particle.generation + 1
#         n = self.num_generations
#         s = self.num_anchors
#         return ((t - s) / (n - s) - 1) * parent_particle.log_g_anchor
