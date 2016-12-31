from __future__ import division

import numpy as np
import random

from pgsm.math_utils import exp_normalize, log_sum_exp
from pgsm.particle_utils import get_constrained_path
from pgsm.smc.samplers import ParticleSwarm


class SplitMergeParticle(object):

    def __init__(self, block_idx, block_params, log_w, parent_particle):
        self.block_idx = block_idx
        self.block_params_ = block_params  # Only access this if you won't modify
        self.log_w = log_w
        self.parent_particle = parent_particle

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
            log_q_norm = log_sum_exp(np.array(log_q.values()))
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
        block_probs, log_q_norm = exp_normalize(log_q.values())
        block_idx = np.random.multinomial(1, block_probs).argmax()
        block_idx = log_q.keys()[block_idx]
        return self.create_particle(block_idx, data_point, parent_particle, log_q=log_q, log_q_norm=log_q_norm)

    def setup(self, anchors, clustering, data, sigma):
        '''
        Setup kernel for a split merge run based on anchors, current clustering, data and permutation of indices.
        '''
        self.num_anchors = len(anchors)
        num_anchor_blocks = len(np.unique([clustering[a] for a in anchors]))
        num_global_blocks = len(np.unique(clustering))
        self.num_outside_blocks = num_global_blocks - num_anchor_blocks

        permuted_clustering = clustering[sigma]
        permuted_data = data[sigma]
        self.data = permuted_data[self.num_anchors:]
        constrained_path = get_constrained_path(permuted_clustering, permuted_data, self)
        self.constrained_path = constrained_path[self.num_anchors:]

        anchor_constrained_path = constrained_path[:self.num_anchors]
        anchor_data = permuted_data[:self.num_anchors]
        unweighted_swarm = self._create_unweighted_init_swarm(anchor_constrained_path, anchor_data)
        self.init_swarm = self._create_weighted_init_swarm(unweighted_swarm)

    def _create_unweighted_init_swarm(self, anchor_constrained_path, anchor_data):
        '''
        Create a swarm which contains all possible particles that could be created from the anchors.
        '''
        swarm = ParticleSwarm()
        swarm.add_particle(0, anchor_constrained_path[0])
        for constrained_particle, data_point in zip(anchor_constrained_path[1:], anchor_data[1:]):
            new_swarm = ParticleSwarm()
            for parent_particle in swarm.particles:
                is_constrained_parent = (parent_particle == constrained_particle.parent_particle)
                num_blocks = len(parent_particle.block_params_)
                for block_idx in range(num_blocks + 1):
                    if is_constrained_parent and (block_idx == constrained_particle.block_idx):
                        particle = constrained_particle
                    else:
                        particle = self.create_particle(block_idx, data_point, parent_particle, log_q=0, log_q_norm=0)
                    new_swarm.add_particle(0, particle)
            swarm = new_swarm
        return swarm

    def _create_weighted_init_swarm(self, unweighted_swarm):
        swarm = ParticleSwarm()
        for particle in unweighted_swarm.particles:
            particle.log_w = self.log_target_density(particle.block_params_)
            # Prune particles with 0 probability
            if np.isneginf(particle.log_w):
                continue
            swarm.add_particle(particle.log_w, particle)
        return swarm

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
    Propose next state uniformly from available states. This is for pedagogical purposes. The implementation is slow.
    '''

    def get_log_q(self, data_point, parent_particle):
        log_q = {}
        for block_idx, _ in enumerate(parent_particle.block_params_):
            log_q[block_idx] = 0
        return log_q

    def _create_particle(self, block_idx, block_params, data_point, log_q, log_q_norm, parent_particle):
        # Ratio of target densities
        log_w = self.log_target_density(block_params) - self.log_target_density(parent_particle.block_params_)
        # Proposal contribution
        log_w -= log_q[block_idx] - log_q_norm
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
        log_q = {}
        for block_idx, params in enumerate(parent_particle.block_params_):
            log_q[block_idx] = self.partition_prior.log_tau_2_diff(params.N)
            log_q[block_idx] += self.dist.log_marginal_likelihood_diff(data_point, params)
        return log_q

    def _create_particle(self, block_idx, block_params, data_point, log_q, log_q_norm, parent_particle):
        return SplitMergeParticle(
            block_idx=block_idx,
            block_params=block_params,
            log_w=log_q_norm,
            parent_particle=parent_particle)


class AnnealedSplitMergeKernel(AbstractSplitMergKernel):
    '''
    Propose new states uniformly until all anchors are added then use fully adapted proposal.
    '''

    def get_log_q(self, data_point, parent_particle):
        log_q = {}
        for block_idx, params in enumerate(parent_particle.block_params_):
            log_q[block_idx] = parent_particle.log_annealing_correction
            log_q[block_idx] += self.partition_prior.log_tau_2_diff(params.N)
            log_q[block_idx] += self.dist.log_marginal_likelihood_diff(data_point, params)
        return log_q

    def _create_particle(self, block_idx, block_params, data_point, log_q, log_q_norm, parent_particle):
        generation = sum([x.N for x in block_params])
        if generation < self.num_anchors:
            log_annealing_correction = 0
        elif generation == self.num_anchors:
            log_annealing_correction = (1 / len(self.data)) * self.log_target_density(block_params)
        else:
            log_annealing_correction = parent_particle.log_annealing_correction

        return AnnealedSplitMergeParticle(
            block_idx=block_idx,
            block_params=block_params,
            log_annealing_correction=log_annealing_correction,
            log_w=log_q_norm,
            parent_particle=parent_particle)
