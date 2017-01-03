'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
from __future__ import division

import numpy as np

from pgsm.math_utils import exp_normalize, log_sum_exp


class ParticleSwarm(object):

    def __init__(self):
        self.particles = []

        self._log_norm_const = None
        self._unnormalized_log_weights = []

    @property
    def ess(self):
        return 1 / np.sum(np.square(self.weights))

    @property
    def log_norm_const(self):
        if self._log_norm_const is None:
            self._log_norm_const = log_sum_exp(self.unnormalized_log_weights)
        return self._log_norm_const

    @property
    def log_weights(self):
        return self.unnormalized_log_weights - self.log_norm_const

    @property
    def num_particles(self):
        return len(self.particles)

    @property
    def relative_ess(self):
        return self.ess / self.num_particles

    @property
    def unnormalized_log_weights(self):
        return np.array(self._unnormalized_log_weights)

    @property
    def weights(self):
        weights = np.exp(self.log_weights)
        weights = weights / weights.sum()
        return weights

    def add_particle(self, log_weight, particle):
        '''
        Args:
            log_weight: Unnormalized log weight of particle
            particle: Particle
        '''
        self.particles.append(particle)
        self._unnormalized_log_weights.append(log_weight)
        self._log_norm_const = None

    def to_dict(self):
        return dict(zip(self.particles, self.weights))


class ImplicitParticleSwarm(ParticleSwarm):

    def __init__(self):
        ParticleSwarm.__init__(self)
        self.multiplicities = []

    def __iter__(self):
        for x in zip(self.log_weights, self.multiplicities, self.particles):
            yield x

    @property
    def aggregate_log_weights(self):
        return np.log(self.multiplicities) + self.log_weights

    @property
    def aggregate_weights(self):
        result = np.exp(self.aggregate_log_weights)
        return result / result.sum()

    @property
    def ess(self):
        return 1 / np.sum(self.multiplicities * np.square(self.weights))

    @property
    def log_norm_const(self):
        if self._log_norm_const is None:
            self._log_norm_const = log_sum_exp(np.log(self.multiplicities) + self.unnormalized_log_weights)
        return self._log_norm_const

    @property
    def num_particles(self):
        return sum(self.multiplicities)

    @property
    def weights(self):
        weights = np.exp(self.log_weights)
        return weights

    def add_particle(self, log_weight, particle, multiplicity=1):
        '''
        Args:
            log_weight: Unnormalized log weight of particle
            particle: Particle
        '''
        self.multiplicities.append(multiplicity)
        self.particles.append(particle)
        self._unnormalized_log_weights.append(log_weight)
        self._log_norm_const = None

    def to_dict(self):
        return dict(zip(self.particles, self.aggregate_weights))


class SMCSampler(object):

    def __init__(self, num_particles, resample_threshold=0.5, verbose=False):
        self.num_particles = num_particles
        self.resample_threshold = resample_threshold
        self.verbose = verbose

    def resample_if_necessary(self, swarm, conditional_particle=None):
        if swarm.relative_ess <= self.resample_threshold:
            new_swarm = ParticleSwarm()
            if self.verbose:
                print 'Resampling', swarm.relative_ess
            log_uniform_weight = -np.log(self.num_particles)
            if conditional_particle is None:
                multiplicities = np.random.multinomial(self.num_particles, swarm.weights)
            else:
                multiplicities = np.random.multinomial(self.num_particles - 1, swarm.weights)
                new_swarm.add_particle(log_uniform_weight, conditional_particle)
            for particle, multiplicity in zip(swarm.particles, multiplicities):
                for _ in range(multiplicity):
                    new_swarm.add_particle(log_uniform_weight, particle.copy())
        else:
            new_swarm = swarm
        return new_swarm

    def sample(self, data, kernel):
        raise NotImplemented()

    def _check_collapse(self, particles):
        collapse = True
        for p in particles:
            if len(p.block_params_) > 1:
                collapse = False
                break
        if collapse and self.verbose:
            print 'Particle swarm has collapsed to merge'
        return collapse


class IndependentSMCSampler(SMCSampler):

    def sample(self, data, kernel):
        swarm = kernel.init_swarm
        for data_point in kernel.data:
            new_swarm = ParticleSwarm()
            for parent_log_W, parent_particle in zip(swarm.log_weights, swarm.particles):
                particle = kernel.propose(data_point, parent_particle)
                new_swarm.add_particle(parent_log_W + particle.log_w, particle)
            swarm = self.resample_if_necessary(swarm)
            if self._check_collapse(swarm.particles):
                return {kernel.constrained_path[-1]: 0}
        return swarm.to_dict()


class ParticleGibbsSampler(SMCSampler):

    def sample(self, data, kernel):
        swarm = kernel.init_swarm
        for constrained_particle, data_point in zip(kernel.constrained_path, kernel.data):
            new_swarm = ParticleSwarm()
            for parent_log_W, parent_particle in zip(swarm.log_weights, swarm.particles):
                if parent_particle == constrained_particle.parent_particle:
                    particle = constrained_particle
                else:
                    particle = kernel.propose(data_point, parent_particle)
                new_swarm.add_particle(parent_log_W + particle.log_w, particle)
            swarm = self.resample_if_necessary(new_swarm, conditional_particle=constrained_particle)
            if self._check_collapse(swarm.particles):
                return {kernel.constrained_path[-1]: 0}
        return swarm.to_dict()


class ImplicitParticleGibbsSampler(SMCSampler):

    def resample_if_necessary(self, swarm, conditional_particle=None):
        if swarm.relative_ess <= self.resample_threshold:
            new_swarm = ImplicitParticleSwarm()
            if self.verbose:
                print 'Resampling', swarm.relative_ess
            if conditional_particle is None:
                multiplicities = np.random.multinomial(self.num_particles, swarm.aggregate_weights)
            else:
                multiplicities = np.random.multinomial(self.num_particles - 1, swarm.aggregate_weights)
                multiplicities[swarm.particles.index(conditional_particle)] += 1
            log_uniform_weight = -np.log(self.num_particles)
            for multiplicity, particle in zip(multiplicities, swarm.particles):
                if multiplicity == 0:
                    continue
                new_swarm.add_particle(log_uniform_weight, particle, multiplicity=multiplicity)
        else:
            new_swarm = swarm
        return new_swarm

    def sample(self, data, kernel):
        swarm = ImplicitParticleSwarm()
        for log_weight, particle in zip(kernel.init_swarm.log_weights, kernel.init_swarm.particles):
            swarm.add_particle(log_weight, particle, 1)
        for constrained_particle, data_point in zip(kernel.constrained_path, kernel.data):
            new_swarm = ImplicitParticleSwarm()
            for parent_log_W, parent_multiplicity, parent_particle in swarm:
                is_constrained_parent = (parent_particle == constrained_particle.parent_particle)
                log_q = kernel.get_log_q(data_point, parent_particle)
                block_probs, log_q_norm = exp_normalize(np.array(log_q.values()))
                if is_constrained_parent:
                    multiplicities = np.random.multinomial(parent_multiplicity - 1, block_probs)
                    multiplicities[log_q.keys().index(constrained_particle.block_idx)] += 1
                else:
                    multiplicities = np.random.multinomial(parent_multiplicity, block_probs)
                for block_idx, multiplicity in zip(log_q.keys(), multiplicities):
                    if multiplicity == 0:
                        continue
                    if is_constrained_parent and (block_idx == constrained_particle.block_idx):
                        particle = constrained_particle
                    else:
                        particle = kernel.create_particle(
                            block_idx,
                            data_point,
                            parent_particle,
                            log_q=log_q,
                            log_q_norm=log_q_norm)
                    new_swarm.add_particle(parent_log_W + particle.log_w, particle, multiplicity=multiplicity)
            swarm = self.resample_if_necessary(new_swarm, conditional_particle=constrained_particle)
            if self._check_collapse(swarm.particles):
                return {kernel.constrained_path[-1]: 0}
            assert constrained_particle in swarm.particles
        return swarm.to_dict()
