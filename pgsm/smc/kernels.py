from __future__ import division

from scipy.misc import logsumexp as log_sum_exp

import numpy as np
import random

from pgsm.math_utils import exp_normalize
from pgsm.particle_utils import get_constrained_path


class SplitMergeParticle(object):

    def __init__(self, block_idx, block_params, log_w, parent_particle):
        self.block_idx = block_idx
        self.block_params_ = block_params  # Only access this if you won't modify
        self.log_w = log_w
        self.parent_particle = parent_particle
        if len(self.block_params_) == 0:
            self.generation = 0
        else:
            self.generation = sum([x.N for x in self.block_params_])

    @property
    def block_params(self):
        return [x.copy() for x in self.block_params_]

    def copy(self):
        return SplitMergeParticle(self.block_idx, self.block_params, self.log_w, self.parent_particle)


class AnnealedSplitMergeParticle(SplitMergeParticle):

    def __init__(
            self,
            block_idx,
            block_params,
            log_annealing_correction,
            log_w,
            parent_particle):

        SplitMergeParticle.__init__(self, block_idx, block_params, log_w, parent_particle)
        self.log_annealing_correction = log_annealing_correction

    def copy(self):
        return AnnealedSplitMergeParticle(
            self.block_idx,
            self.block_params,
            self.log_annealing_correction,
            self.log_w,
            self.parent_particle)


class AbstractSplitMergKernel(object):

    def __init__(self, dist, partition_prior):
        self.dist = dist
        self.partition_prior = partition_prior

    def can_add_block(self, parent_particle):
        '''
        Check if a descendant particle can add a new block.
        '''
        if parent_particle is None:
            return True
        else:
            return (parent_particle.generation < self.num_anchors)

    def create_initial_particle(self, data_point):
        return self.create_particle(0, data_point, None)

    def create_particle(self, block_idx, data_point, parent_particle, log_q=None, log_q_norm=None):
        '''
        Create a descenedant particle from a parent particle by adding data point to a block.
        '''
        block_params = self._get_block_params(block_idx, data_point, parent_particle)
        if log_q is None:
            log_q = self.get_log_q(data_point, parent_particle)
        if log_q_norm is None:
            log_q_norm = log_sum_exp(log_q.values())
        return self._create_particle(block_idx, block_params, data_point, log_q, log_q_norm, parent_particle)

    def log_target_density(self, params):
        log_g = self.partition_prior.log_tau_1(len(params) + self.num_cluster_diff)
        for block_params in params:
            log_g += self.partition_prior.log_tau_2(block_params.N)
            log_g += self.dist.log_marginal_likelihood(block_params)
        return log_g

    def propose(self, data_point, parent_particle, seed=None):
        '''
        Propose a particle for t given a particle from t - 1 and a data point.
        '''
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        log_q = self.get_log_q(data_point, parent_particle)
        block_probs, log_q_norm = exp_normalize(log_q.values())
        block_idx = np.random.multinomial(1, block_probs).argmax()
        block_idx = log_q.keys()[block_idx]
        return self.create_particle(block_idx, data_point, parent_particle, log_q=log_q, log_q_norm=log_q_norm)

    def setup(self, anchors, clustering, data, sigma):
        '''
        Setup kernel for a split merge run based on anchors, current clustering, data and permutation of indices.
        '''
        self.num_anchors = len(anchors)
        self.num_generations = len(sigma)

        c_size = len(np.unique(clustering))
        c_bar_size = len(np.unique(anchors))
        self.num_cluster_diff = c_size - c_bar_size

        self.constrained_path = get_constrained_path(clustering[sigma], data[sigma], self)

    def _get_block_params(self, block_idx, data_point, parent_particle):
        '''
        Get posterior parameters from parent particle updated by adding data_point to a block.
        '''
        if parent_particle is None:
            block_params = []
        else:
            block_params = parent_particle.block_params

        if block_idx > (len(block_params) - 1):
            params = self.dist.create_params()
            params.increment(data_point)
            block_params.append(params)
        else:
            block_params[block_idx].increment(data_point)

        return block_params

    def get_log_q(self, data_point, parent_particle):
        '''
        Get the unnormalized proposal
        '''
        raise NotImplementedError

    def _create_particle(self, block_idx, block_params, data_point, log_q, log_q_norm, parent_particle):
        raise NotImplementedError


class UniformSplitMergeKernel(AbstractSplitMergKernel):
    '''
    Propose next state uniformly from available states.
    '''

    def get_log_q(self, data_point, parent_particle):
        if parent_particle is None:
            block_params = []
        else:
            block_params = parent_particle.block_params_

        log_q = {}

        for block_idx, _ in enumerate(block_params):
            log_q[block_idx] = 0

        if self.can_add_block(parent_particle):
            block_idx = len(block_params)
            log_q[block_idx] = 0

        return log_q

    def _create_particle(self, block_idx, block_params, data_point, log_q, log_q_norm, parent_particle):
        if parent_particle is None:
            log_w = 0
        else:
            log_w = parent_particle.log_w + self.log_target_density(block_params) - \
                self.log_target_density(parent_particle.block_params_) - log_q_norm
        return SplitMergeParticle(
            block_idx=block_idx,
            block_params=block_params,
            log_w=log_w,
            parent_particle=parent_particle)


class FullyAdaptedSplitMergeKernel(AbstractSplitMergKernel):
    '''
    Propose next with probability proportional to target density.
    '''

    def get_log_q(self, data_point, parent_particle):
        if parent_particle is None:
            block_params = []
        else:
            block_params = parent_particle.block_params_

        log_q = {}

        for block_idx, params in enumerate(block_params):
            log_q[block_idx] = self.partition_prior.log_tau_2_diff(params.N) + \
                self.dist.log_marginal_likelihood_diff(data_point, params)

        if self.can_add_block(parent_particle):
            block_idx = len(block_params)
            params = self.dist.create_params()
            params.increment(data_point)
            # TODO: Check this missing global cluster diff
            log_q[block_idx] = \
                self.partition_prior.log_tau_1_diff(self.num_cluster_diff + len(block_params) + 1) + \
                self.partition_prior.log_tau_2_diff(1) + self.dist.log_marginal_likelihood(params)

        return log_q

    def _create_particle(self, block_idx, block_params, data_point, log_q, log_q_norm, parent_particle):
        if parent_particle is None:
            log_w = 0
        else:
            log_w = parent_particle.log_w + log_q_norm
        return SplitMergeParticle(
            block_idx=block_idx,
            block_params=block_params,
            log_w=log_w,
            parent_particle=parent_particle)


class AnnealedSplitMergeKernel(AbstractSplitMergKernel):
    '''
    Propose new states uniformly until all anchors are added then use fully adapted proposal.
    '''

    def get_log_q(self, data_point, parent_particle):
        if parent_particle is None:
            block_params = []
        else:
            block_params = parent_particle.block_params_

        log_q = {}

        if self.can_add_block(parent_particle):
            for block_idx, _ in enumerate(block_params):
                log_q[block_idx] = 0

            block_idx = len(block_params)
            log_q[block_idx] = 0

        else:
            for block_idx, params in enumerate(block_params):
                log_q[block_idx] = parent_particle.log_annealing_correction + \
                    self.partition_prior.log_tau_2_diff(params.N) + \
                    self.dist.log_marginal_likelihood_diff(data_point, params)

        return log_q

    def _create_particle(self, block_idx, block_params, data_point, log_q, log_q_norm, parent_particle):
        if parent_particle is None:
            generation = 1
        else:
            generation = parent_particle.generation + 1

        if parent_particle is None:
            log_w = 0
        else:
            log_w = parent_particle.log_w + log_q_norm

        if generation < self.num_anchors:
            log_annealing_correction = None
        elif generation == self.num_anchors:
            n = self.num_generations
            s = self.num_anchors
            log_annealing_correction = (1 / (n - s)) * self.log_target_density(block_params)
        else:
            log_annealing_correction = parent_particle.log_annealing_correction

        return AnnealedSplitMergeParticle(
            block_idx=block_idx,
            block_params=block_params,
            log_annealing_correction=log_annealing_correction,
            log_w=log_w,
            parent_particle=parent_particle)
