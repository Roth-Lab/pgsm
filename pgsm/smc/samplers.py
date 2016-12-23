'''
Created on 2014-02-25

@author: Andrew Roth
'''
from __future__ import division

import numpy as np

from pgsm.math_utils import exp_normalize


class SMCSampler(object):

    def __init__(self, num_particles, resample_threshold=0.5, verbose=False):
        self.num_particles = num_particles
        self.resample_threshold = resample_threshold
        self.verbose = verbose

    def resample_if_necessary(self, new_particles):
        particle_probs, _ = exp_normalize([float(p.log_w) for p in new_particles])
        ess = 1 / np.sum(np.square(particle_probs))
        if (ess / self.num_particles) <= self.resample_threshold:
            particles = []
            if self.verbose:
                print 'Resampling', ess
            multiplicities = np.random.multinomial(self.num_particles, particle_probs)
            for particle, multiplicity in zip(new_particles, multiplicities):
                particles.extend([particle.copy() for _ in range(multiplicity)])
        else:
            particles = new_particles
        return particles

    def sample(self, data, kernel):
        raise NotImplemented()


class SimpleSMCSampler(SMCSampler):

    def sample(self, data, kernel):
        particles = [kernel.create_initial_particle(data[0]) for _ in range(self.num_particles)]
        for data_point in data[1:]:
            new_particles = []
            for p in particles:
                new_particles.append(kernel.propose(data_point, p))
            particles = self.resample_if_necessary(new_particles)
        return particles


class ParticleGibbsSampler(SMCSampler):

    def sample(self, data, kernel):
        constrained_path = kernel.constrained_path
        particles = [kernel.create_initial_particle(data[0]) for _ in range(self.num_particles - 1)]
        particles.append(constrained_path[0])
        for i in range(1, data.shape[0]):
            new_particles = []
            for p_idx in range(self.num_particles - 1):
                new_particles.append(kernel.propose(data[i], particles[p_idx]))
            new_particles.append(constrained_path[i])
            particles = self.resample_if_necessary(new_particles)
            # Overwrite first particle with constrained path
            particles[-1] = constrained_path[i].copy()
            if self._check_collapse(kernel, particles):
                print 'Collapse'
                return [constrained_path[-1], ]
        return particles

    def _check_collapse(self, kernel, particles):
        if kernel.can_add_block(particles[0]):
            collapse = False
        else:
            collapse = True
            for p in particles:
                if len(p.block_params_) > 1:
                    collapse = False
                    break
        return collapse


class ImplicitParticleGibbsSampler(SMCSampler):

    def resample_if_necessary(self, constrained_particle, new_particles):
        particle_probs, _ = exp_normalize([float(np.log(m) + p.log_w) for p, m in new_particles.items()])
        ess = 1 / np.sum(np.square(particle_probs))
        if (ess / self.num_particles) <= self.resample_threshold:
            particles = {}
            if self.verbose:
                print 'Resampling', ess
            multiplicities = np.random.multinomial(self.num_particles - 1, particle_probs)
            for p, m in zip(new_particles, multiplicities):
                if p == constrained_particle:
                    m += 1
                particles[p] = m
        else:
            particles = new_particles
        return particles

    def sample(self, data, kernel):
        constrained_path = kernel.constrained_path
        particles = {
            kernel.create_initial_particle(data[0]): self.num_particles - 1,
            constrained_path[0]: 1
        }
        for i in range(1, data.shape[0]):
            new_particles = {}
            constrained_particle = constrained_path[i]
            for parent_particle, parent_multiplicity in particles.items():
                is_constrained_parent = (parent_particle == constrained_path[i - 1])
                log_q = kernel.get_log_q(data[i], parent_particle)
                block_probs, log_q_norm = exp_normalize(log_q.values())
                if is_constrained_parent:
                    multiplicities = np.random.multinomial(parent_multiplicity - 1, block_probs)
                    multiplicities[log_q.keys().index(constrained_path[i].block_idx)] += 1
                else:
                    multiplicities = np.random.multinomial(parent_multiplicity, block_probs)
                for block_idx, particle_multiplicity in zip(log_q.keys(), multiplicities):
                    if particle_multiplicity == 0:
                        continue
                    if is_constrained_parent and (constrained_particle.block_idx == block_idx):
                        particle = constrained_path[i]
                    else:
                        particle = kernel.create_particle(
                            block_idx,
                            data[i],
                            parent_particle,
                            log_q=log_q,
                            log_q_norm=log_q_norm)
                    new_particles[particle] = particle_multiplicity
            particles = self.resample_if_necessary(constrained_particle, new_particles)
            if self._check_collapse(kernel, particles):
                print 'Collapse'
                return [constrained_path[-1], ]
            assert new_particles[constrained_path[i]] > 0
            assert sum(new_particles.values()) == self.num_particles
        final_particles = []
        for particle, multiplicity in particles.items():
            for _ in range(multiplicity):
                final_particles.append(particle)
        return final_particles

    def _check_collapse(self, kernel, particles):
        if kernel.can_add_block(particles.keys()[0]):
            collapse = False
        else:
            collapse = True
            for p in particles.keys():
                if len(p.block_params_) > 1:
                    collapse = False
                    break
        return collapse
