from collections import namedtuple

import numpy as np
import random

from pgsm.math_utils import exp_normalize, log_sum_exp
from pgsm.particle_utils import get_constrained_path

SplitMergeParticle = namedtuple(
    'SplitMergeParticle',
    ('block_idx', 'block_params', 'generation', 'log_w', 'parent_particle'),
)

AnnealedSplitMergeParticle = namedtuple(
    'AnnealedSplitMergeParticle',
    ('block_idx', 'block_params', 'generation', 'log_annealing_correction', 'log_w', 'parent_particle'),
)


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

    def copy_particle(self, particle):
        return SplitMergeParticle(
            particle.block_idx,
            tuple([x.copy() for x in particle.block_params]),
            particle.generation,
            particle.log_w,
            particle.parent_particle
        )

    def create_initial_particle(self, data_point):
        return self.create_particle(0, data_point, None)

    def create_particle(self, block_idx, data_point, parent_particle, log_q=None, log_q_norm=None):
        '''
        Create a descendant particle from a parent particle by adding data point to a block.
        '''
        block_params = self._get_block_params(block_idx, data_point, parent_particle)

        if log_q is None:
            log_q = self.get_log_q(data_point, parent_particle)

        if log_q_norm is None:
            log_q_norm = log_sum_exp(np.array(list(log_q.values())))

        return self._create_particle(block_idx, block_params, data_point, log_q, log_q_norm, parent_particle)

    def log_target_density(self, block_params):
        num_blocks = len(block_params)

        log_g = self.partition_prior.log_tau_1(num_blocks + self.num_outside_blocks)

        for params in block_params:
            log_g += self.partition_prior.log_tau_2(params.N)

            log_g += self.dist.log_marginal_likelihood(params)

        return log_g

    def propose(self, data_point, parent_particle, seed=None):
        '''
        Propose a particle for t given a particle from t - 1 and a data point.
        '''
        if seed is not None:
            random.seed(seed)

            np.random.seed(seed)

        log_q = self.get_log_q(data_point, parent_particle)

        block_probs, log_q_norm = exp_normalize(np.array(list(log_q.values())))

        block_idx = np.random.multinomial(1, block_probs).argmax()

        block_idx = list(log_q.keys())[block_idx]

        return self.create_particle(block_idx, data_point, parent_particle, log_q=log_q, log_q_norm=log_q_norm)

    def setup(self, anchors, clustering, data, sigma, set_constrained_path=True):
        '''
        Setup kernel for a split merge run based on anchors, current clustering, data and permutation of indices.
        '''
        self.num_anchors = len(anchors)

        self.num_generations = len(sigma)

        num_anchor_blocks = len(np.unique([clustering[a] for a in anchors]))

        num_global_blocks = len(np.unique(clustering))

        self.num_outside_blocks = num_global_blocks - num_anchor_blocks

        if set_constrained_path:
            self.constrained_path = get_constrained_path(clustering[sigma], data[sigma], self)

    def _get_block_params(self, block_idx, data_point, parent_particle):
        '''
        Get posterior parameters from parent particle updated by adding data_point to a block.
        '''
        if parent_particle is None:
            block_params = []

        else:
            block_params = [x.copy() for x in parent_particle.block_params]

        if block_idx > (len(block_params) - 1):
            params = self.dist.create_params()

            params.increment(data_point)

            block_params.append(params)

        else:
            block_params[block_idx].increment(data_point)

        return block_params

    def _get_generation(self, parent_particle):
        if parent_particle is None:
            generation = 1

        else:
            generation = parent_particle.generation + 1

        return generation

    def get_log_q(self, data_point, parent_particle):
        '''
        Get the unnormalized proposal
        '''
        raise NotImplementedError

    def _create_particle(self, block_idx, block_params, data_point, log_q, log_q_norm, parent_particle):
        raise NotImplementedError


class UniformSplitMergeKernel(AbstractSplitMergKernel):
    '''
    Propose next state uniformly from available states. This is for pedagogical purposes. The implementation is slow.
    '''

    def get_log_q(self, data_point, parent_particle):
        if parent_particle is None:
            block_params = []

        else:
            block_params = parent_particle.block_params

        log_q = {}

        for block_idx, _ in enumerate(block_params):
            log_q[block_idx] = 0

        if self.can_add_block(parent_particle):
            block_idx = len(block_params)

            log_q[block_idx] = 0

        return log_q

    def _create_particle(self, block_idx, block_params, data_point, log_q, log_q_norm, parent_particle):
        # Initial particle
        if parent_particle is None:
            log_w = self.log_target_density(block_params) - log_q_norm

        else:
            # Ratio of target densities
            log_w = self.log_target_density(block_params) - self.log_target_density(parent_particle.block_params)

            # Proposal contribution
            log_w -= log_q_norm

        return SplitMergeParticle(
            block_idx=block_idx,
            block_params=tuple(block_params),
            generation=self._get_generation(parent_particle),
            log_w=log_w,
            parent_particle=parent_particle
        )


class FullyAdaptedSplitMergeKernel(AbstractSplitMergKernel):
    '''
    Propose next state with probability proportional to target density.
    '''

    def get_log_q(self, data_point, parent_particle):
        if parent_particle is None:
            block_params = []

        else:
            block_params = parent_particle.block_params

        log_q = {}

        for block_idx, params in enumerate(block_params):
            log_q[block_idx] = self.partition_prior.log_tau_2_diff(params.N)

            log_q[block_idx] += self.dist.log_predictive_likelihood(data_point, params)

        if self.can_add_block(parent_particle):
            block_idx = len(block_params)

            params = self.dist.create_params()

            num_blocks = len(block_params)

            log_q[block_idx] = self.partition_prior.log_tau_1_diff(self.num_outside_blocks + num_blocks)

            log_q[block_idx] += self.partition_prior.log_tau_2_diff(params.N)

            log_q[block_idx] += self.dist.log_predictive_likelihood(data_point, params)

        return log_q

    def _create_particle(self, block_idx, block_params, data_point, log_q, log_q_norm, parent_particle):
        return SplitMergeParticle(
            block_idx=block_idx,
            block_params=tuple(block_params),
            generation=self._get_generation(parent_particle),
            log_w=log_q_norm,
            parent_particle=parent_particle
        )


class AnnealedSplitMergeKernel(AbstractSplitMergKernel):
    '''
    Propose next state uniformly until all anchors are added then use fully adapted proposal.
    '''

    def copy_particle(self, particle):
        return AnnealedSplitMergeParticle(
            particle.block_idx,
            tuple([x.copy() for x in particle.block_params]),
            particle.generation,
            particle.log_annealing_correction,
            particle.log_w,
            particle.parent_particle
        )

    def get_log_q(self, data_point, parent_particle):
        if parent_particle is None:
            block_params = []

        else:
            block_params = parent_particle.block_params

        log_q = {}

        # Sample uniformly from possible states if we are still adding anchor points
        if self.can_add_block(parent_particle):
            for block_idx, _ in enumerate(block_params):
                log_q[block_idx] = 0

            block_idx = len(block_params)

            log_q[block_idx] = 0

        # Otherwise do the normally fully adapted proposal plus the annealing correction
        else:
            for block_idx, params in enumerate(block_params):
                log_q[block_idx] = parent_particle.log_annealing_correction

                log_q[block_idx] += self.partition_prior.log_tau_2_diff(params.N)

                log_q[block_idx] += self.dist.log_predictive_likelihood(data_point, params)

        return log_q

    def _create_particle(self, block_idx, block_params, data_point, log_q, log_q_norm, parent_particle):
        generation = self._get_generation(parent_particle)

        if generation < self.num_anchors:
            log_annealing_correction = None

        elif generation == self.num_anchors:
            n = self.num_generations

            s = self.num_anchors

            if n == s:
                log_annealing_correction = None

                log_q_norm = self.log_target_density(block_params)

            else:
                log_annealing_correction = (1 / (n - s)) * self.log_target_density(block_params)

        else:
            log_annealing_correction = parent_particle.log_annealing_correction

        return AnnealedSplitMergeParticle(
            block_idx=block_idx,
            block_params=tuple(block_params),
            generation=generation,
            log_annealing_correction=log_annealing_correction,
            log_w=log_q_norm,
            parent_particle=parent_particle
        )
