'''
Created on 25 Jan 2017

@author: andrew
'''
import unittest

import numpy as np

import pgsm.distributions.mvn as mvn
import pgsm.tests.mocks as mocks

from pgsm.mcmc.sams import SequentiallyAllocatedMergeSplitSampler
from pgsm.mcmc.split_merge_setup import UniformSplitMergeSetupKernel
from pgsm.math_utils import log_normalize
from pgsm.partition_priors import DirichletProcessPartitionPrior


class Test(unittest.TestCase):

    def test_empty(self):
        N = 100

        clustering = np.random.randint(0, 10, size=N)

        data = np.random.normal(size=100)

        dist = mocks.MockDistribution()

        partition_prior = mocks.MockPartitionPrior()

        split_merge_setup_kernel = UniformSplitMergeSetupKernel(data, dist, partition_prior)

        sampler = SequentiallyAllocatedMergeSplitSampler(dist, partition_prior, split_merge_setup_kernel)

        anchors, sigma = split_merge_setup_kernel.setup_split_merge(clustering, 2)

        sampler.kernel.setup(anchors, clustering, data, sigma)

        clustering, mh_ratio = sampler._merge(data)

        self.assertAlmostEqual(mh_ratio, 0)

        clustering, mh_ratio = sampler._split(data)

        self.assertAlmostEqual(mh_ratio, np.log(2) * (N - 2))

    def test_likelihood(self):
        dim = 10

        N = 100

        clustering = np.random.randint(0, 10, size=N)

        data = np.random.multivariate_normal(np.random.random(size=dim) * 100, np.eye(dim), size=N)

        dist = mvn.MultivariateNormalDistribution(dim)

        partition_prior = mocks.MockPartitionPrior()

        split_merge_setup_kernel = UniformSplitMergeSetupKernel(data, dist, partition_prior)

        sampler = SequentiallyAllocatedMergeSplitSampler(dist, partition_prior, split_merge_setup_kernel)

        anchors, sigma = split_merge_setup_kernel.setup_split_merge(clustering, 2)

        sampler.kernel.setup(anchors, clustering, data, sigma)

        clustering, mh_ratio = sampler._merge(data)

        log_p = dist.log_marginal_likelihood(dist.create_params_from_data(data))

        self.assertAlmostEqual(mh_ratio, log_p)

        clustering, mh_ratio = sampler._split(data)

        clustering = clustering.astype(int)

        log_q = 0

        params = [dist.create_params_from_data(data[0]), dist.create_params_from_data(data[1])]

        for c, x in zip(clustering[2:], data[2:]):
            block_probs = np.zeros(2)

            for i in range(2):
                block_probs[i] = dist.log_predictive_likelihood(x, params[i])

            block_probs = log_normalize(block_probs)

            log_q += block_probs[c]

            params[c].increment(x)

        log_p = sum([dist.log_marginal_likelihood(x) for x in params])

        self.assertAlmostEqual(mh_ratio, log_p - log_q)

    def test_partition_prior(self):
        dim = 10

        N = 3

        clustering = np.random.randint(0, 10, size=N)

        data = np.random.multivariate_normal(np.random.random(size=dim) * 100, np.eye(dim), size=N)

        dist = mocks.MockDistribution()

        partition_prior = DirichletProcessPartitionPrior(0.1234)

        split_merge_setup_kernel = UniformSplitMergeSetupKernel(data, dist, partition_prior)

        sampler = SequentiallyAllocatedMergeSplitSampler(dist, partition_prior, split_merge_setup_kernel)

        anchors, sigma = split_merge_setup_kernel.setup_split_merge(clustering, 2)

        sampler.kernel.setup(anchors, clustering, data, sigma)

        clustering, mh_ratio = sampler._merge(data)

        log_p = partition_prior.log_likelihood([N, ])

        self.assertAlmostEqual(mh_ratio, log_p)

        clustering, mh_ratio = sampler._split(data)

        clustering = clustering.astype(int)

        log_q = 0

        params = [dist.create_params_from_data(data[0]), dist.create_params_from_data(data[1])]

        for c, x in zip(clustering[2:], data[2:]):
            block_probs = np.zeros(2)

            for i in range(2):
                block_probs[i] = partition_prior.log_tau_2_diff(params[i].N)

            block_probs = log_normalize(block_probs)

            log_q += block_probs[c]

            params[c].increment(x)

        log_p = sum([dist.log_marginal_likelihood(x) for x in params])

        self.assertAlmostEqual(mh_ratio, log_p - log_q)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_empty']
    unittest.main()
