'''
Created on 8 Dec 2016

@author: Andrew Roth
'''
from __future__ import division

import numpy as np

from pgsm.math_utils import discrete_rvs, log_normalize
# from pgsm.smc.kernels import SplitMergeParticle
# from pgsm.smc.samplers import ParticleSwarm
from pgsm.utils import relabel_clustering, setup_split_merge
# from pgsm.particle_utils import get_constrained_path, get_log_normalisation, get_clustering, get_cluster_labels


class SequentiallyAllocatedMergeSplitSampler(object):

    def __init__(self, anchor_proposal, dist, partition_prior):
        self.anchor_proposal = anchor_proposal

        self.dist = dist

        self.partition_prior = partition_prior

    def sample(self, clustering, data, num_iters=1):
        for _ in range(num_iters):
            clustering = self._sample(clustering, data)

        return clustering

    def setup(self, clustering, data):
        pass
#         self.anchor_proposal.setup(data, self.dist)

    def _sample(self, clustering, data):
        self.anchor_proposal.update(clustering, data, self.dist, self.partition_prior)

        anchors, sigma = setup_split_merge(self.anchor_proposal, clustering, 2)

        num_anchor_blocks = len(np.unique([clustering[a] for a in anchors]))

        num_global_blocks = len(np.unique(clustering))

        num_outside_blocks = num_global_blocks - num_anchor_blocks

        clustering_sigma = clustering[sigma]

        data_sigma = data[sigma]

        propose_merge = (clustering_sigma[0] != clustering_sigma[1])

        if propose_merge:
            merge_clustering, merge_mh_factor = self._merge(data_sigma, num_outside_blocks)

            split_clustering, split_mh_factor = self._split(
                data_sigma,
                num_outside_blocks,
                constrained_path=clustering_sigma
            )

            forward_factor = merge_mh_factor

            reverse_factor = split_mh_factor

            restricted_clustering = np.array(merge_clustering, dtype=int)

#             assert np.all(relabel_clustering(clustering_sigma) == relabel_clustering(split_clustering))

        else:
            merge_clustering, merge_mh_factor = self._merge(data_sigma, num_outside_blocks)

            split_clustering, split_mh_factor = self._split(data_sigma, num_outside_blocks)

            forward_factor = split_mh_factor

            reverse_factor = merge_mh_factor

            restricted_clustering = np.array(split_clustering, dtype=int)

        log_ratio = forward_factor - reverse_factor

        u = np.random.random()

#         assert len(np.unique(merge_clustering)) == 1
#         assert len(np.unique(split_clustering)) == 2
#         assert len(restricted_clustering) == len(sigma)

        if log_ratio >= np.log(u):
            max_idx = clustering.max()

            clustering[sigma] = restricted_clustering + max_idx + 1

            clustering = relabel_clustering(clustering)

        return clustering

    def _merge(self, data, num_outside_blocks):
        clustering = np.ones(len(data), dtype=np.int)

        log_q = 0

        params = [self.dist.create_params_from_data(data), ]

        log_p = self._log_target(num_outside_blocks, params)

        mh_factor = log_p - log_q

#         assert len(clustering) == sum([x.N for x in params])

        return clustering, mh_factor

    def _split(self, data, num_outside_blocks, constrained_path=None):
        num_blocks = 2

        if constrained_path is not None:
            constrained_path = relabel_clustering(constrained_path)

#             assert constrained_path[0] == 0 and constrained_path[1] == 1

        clustering = range(num_blocks)

        params = [self.dist.create_params(), self.dist.create_params()]

        for i in range(num_blocks):
            params[i].increment(data[i])

        log_q = 0
#
#         T = len(data[num_blocks:]) - num_blocks

        for i, data_point in enumerate(data[num_blocks:], num_blocks):
            log_block_probs = np.zeros(num_blocks, dtype=float)

            for block_idx, cluster_params in enumerate(params):
                log_block_probs[block_idx] = self.partition_prior.log_tau_2_diff(cluster_params.N)

                log_block_probs[block_idx] += self.dist.log_predictive_likelihood(data_point, cluster_params)

#                 log_block_probs[block_idx] *= T

            log_block_probs = log_normalize(log_block_probs)

            if constrained_path is None:
                block_probs = np.exp(log_block_probs)

                block_probs = block_probs / block_probs.sum()

                c = discrete_rvs(block_probs)

            else:
                c = constrained_path[i]

            clustering.append(c)

            log_q += log_block_probs[c]

            params[c].increment(data_point)

#         assert len(clustering) == sum([x.N for x in params])

        log_p = self._log_target(num_outside_blocks, params)

        mh_factor = log_p - log_q

        return clustering, mh_factor

    def _log_target(self, num_outside_blocks, params):
        num_anchors = len(params)

        log_p = self.partition_prior.log_tau_1(num_anchors + num_outside_blocks)

        for block_params in params:
            log_p += self.partition_prior.log_tau_2(block_params.N)

            log_p += self.dist.log_marginal_likelihood(block_params)

        return log_p

# class SequentiallyAllocatedMergeSplitSampler(object):
#
#     def __init__(self, anchor_proposal, kernel):
#         self.anchor_proposal = anchor_proposal
#
#         self.kernel = kernel
#
#     @property
#     def dist(self):
#         return self.kernel.dist
#
#     @property
#     def partition_prior(self):
#         return self.kernel.partition_prior
#
#     def sample(self, clustering, data, num_iters=1):
#         for _ in range(num_iters):
#             clustering = self._sample(clustering, data)
#
#         return clustering
#
#     def setup(self, clustering, data):
#         self.anchor_proposal.update(clustering, data, self.kernel.dist, self.kernel.partition_prior)
#
#     def _sample(self, clustering, data):
#         self.anchor_proposal.update(clustering, data, self.kernel.dist, self.kernel.partition_prior)
#
#         anchors, sigma = setup_split_merge(self.anchor_proposal, clustering, 2)
#
#         self.kernel.setup(anchors, clustering, data, sigma)
#
#         propose_merge = clustering[anchors[0]] != clustering[anchors[1]]
#
#         merge_particle = get_constrained_path(np.zeros(len(sigma)), data[sigma], self.kernel)[-1]
#
#         merge_sampler = MergeSampler()
#
#         merge_particle = merge_sampler.sample(data, self.kernel)
#
#         if propose_merge:
#             split_particle = get_constrained_path(clustering[sigma], data[sigma], self.kernel)[-1]
#
#             restricted_clustering = np.zeros(len(sigma))
#
#         else:
#             sampler = SplitSampler()
#
#             split_particle = sampler.sample(data[sigma], self.kernel)
#
#             restricted_clustering = get_cluster_labels(split_particle)
#
#         merge_mh_factor = self.kernel.log_target_density(merge_particle.block_params)  # get_log_normalisation(merge_particle)
#
#         split_mh_factor = get_log_normalisation(split_particle)
#
# #         print self.kernel.log_target_density(merge_particle.block_params), self.kernel.log_target_density(split_particle.block_params)
# #         print merge_mh_factor, split_mh_factor
# #         print '*' * 100
#
#         if propose_merge:
#             log_mh_ratio = merge_mh_factor - split_mh_factor
#
#         else:
#             log_mh_ratio = split_mh_factor - merge_mh_factor
#
#         u = np.random.random()
#
#         if log_mh_ratio >= np.log(u):
#             if propose_merge:
#                 print 'Merge accepted', log_mh_ratio
#                 print '#' * 100
#             max_idx = clustering.max()
#
#             clustering[sigma] = restricted_clustering + max_idx + 1
#
#             clustering = relabel_clustering(clustering)
#
#         return clustering
#
#
# class MergeSampler(object):
#
#     def sample(self, data, kernel):
#         particle = self._create_init_particle(kernel, data[:2])
#
#         for data_point in data[2:]:
#             particle = kernel.propose(data_point, particle)
#
#         return particle
#
#     def _create_init_particle(self, kernel, anchor_data):
#         params = kernel.dist.create_params()
#
#         params.increment(anchor_data[0])
#
#         init_particle = kernel.create_particle(
#             0,
#             anchor_data[0],
#             None,
#             log_q={0: 0},
#         )
#
#         params.increment(anchor_data[1])
#
#         init_particle = kernel.create_particle(
#             0,
#             anchor_data[1],
#             init_particle,
#             log_q={0: kernel.log_target_density([params, ])},
#         )
#
#         return init_particle
#
#
# class SplitSampler(object):
#
#     def sample(self, data, kernel):
#         particle = self._create_init_particle(kernel, data[:2])
#
#         for data_point in data[2:]:
#             particle = kernel.propose(data_point, particle)
#
#         return particle
#
#     def _create_init_particle(self, kernel, anchor_data):
#         params = [
#             kernel.dist.create_params_from_data(anchor_data[0]),
#             kernel.dist.create_params_from_data(anchor_data[1])
#         ]
#
#         init_particle = kernel.create_particle(
#             0,
#             anchor_data[0],
#             None,
#             log_q={0: 0},
#         )
#
#         init_particle = kernel.create_particle(
#             1,
#             anchor_data[1],
#             init_particle,
#             log_q={0: float('-inf'), 1: kernel.log_target_density(params)}
#         )
#
#         return init_particle
