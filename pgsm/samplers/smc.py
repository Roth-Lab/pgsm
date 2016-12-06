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
        particle_probs = exp_normalize([p.log_w for p in new_particles])
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


class SimpleSMCSampler(SMCSampler):
    """ Simple SMC sampler. Does conditional resampling when the relative ESS falls below threshold.
    """

    def sample(self, data, kernel):
        particles = [kernel.create_initial_particle() for _ in range(self.num_particles)]
        for data_point in data:
            new_particles = []
            for p in particles:
                new_particles.append(kernel.propose(data_point, p))
            particles = self.resample_if_necessary(new_particles)
        return particles


class ParticleGibbsSampler(SMCSampler):

    def sample(self, constrained_path, data, kernel):
        particles = [kernel.create_initial_particle(data[0]) for _ in range(self.num_particles)]
        for i in range(1, data.shape[0]):
            new_particles = []
            for p_idx in range(self.num_particles - 1):
                new_particles.append(kernel.propose(data[i], particles[p_idx]))
            new_particles.append(constrained_path[i + 1])
            particles = self.resample_if_necessary(new_particles)
            # Overwrite first particle with constrained path
            particles[-1] = constrained_path[i + 1].copy()
        return particles
