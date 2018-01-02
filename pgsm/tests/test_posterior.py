import unittest

from collections import defaultdict, Counter

import numpy as np

from pgsm.distributions.mvn import MultivariateNormalDistribution
from pgsm.mcmc.collapsed_gibbs import CollapsedGibbsSampler
from pgsm.mcmc.particle_gibbs_split_merge import ParticleGibbsSplitMergeSampler
from pgsm.mcmc.sams import SequentiallyAllocatedMergeSplitSampler
from pgsm.mcmc.split_merge_setup import UniformSplitMergeSetupKernel
from pgsm.partition_priors import DirichletProcessPartitionPrior
from pgsm.utils import relabel_clustering

from . import simulate_test_data

from .exact_posterior import get_exact_posterior
from .mocks import MockDistribution, MockPartitionPrior


class Test(unittest.TestCase):

    def test_gibbs(self):
        def sampler_factory(data, dist, partition_prior):
            return CollapsedGibbsSampler(dist, partition_prior)

        self._run_single_mvn_test(sampler_factory, dim=2)

    def test_pgsm(self):
        def sampler_factory(data, dist, partition_prior):
            setup_kernel = UniformSplitMergeSetupKernel(data, dist, partition_prior)

            return ParticleGibbsSplitMergeSampler.create_from_dist(
                dist,
                partition_prior,
                setup_kernel,
                num_particles=20,
                resample_threshold=0.5,
                use_annealed=True
            )

        self._run_single_mvn_test(sampler_factory, dim=2)

    def test_sams(self):
        def sampler_factory(data, dist, partition_prior):
            setup_kernel = UniformSplitMergeSetupKernel(data, dist, partition_prior)

            return SequentiallyAllocatedMergeSplitSampler(dist, partition_prior, setup_kernel)

        self._run_single_mvn_test(sampler_factory, dim=2)

    def _get_normal_model(self, dim):
        return MultivariateNormalDistribution(dim), DirichletProcessPartitionPrior(1.1234)

    def _run_single_mvn_test(
            self,
            sampler_factory,
            dim=2,
            mock_dist=False,
            mock_partition=False,
            num_data_points=None,
            num_iters=int(1e4),
            seeds=None):

        if num_data_points is None:
            num_data_points = list(range(2, 9))

        if seeds is None:
            seeds = [np.random.RandomState(x).randint(0, int(1e6)) for x in range(5)]

        for n in num_data_points:
            for seed in seeds:
                data = simulate_test_data.single_mvn_cluster(n, dim=dim, seed=seed)

                dist, partition_prior = self._get_normal_model(dim)

                if mock_dist:
                    dist = MockDistribution()

                if mock_partition:
                    partition_prior = MockPartitionPrior()

                partition_prior.alpha = np.random.RandomState(seed).gamma(0.5, scale=2)

                print(n, seed, partition_prior.alpha)

                sampler = sampler_factory(data, dist, partition_prior)

                pred_probs = self._run_sampler_posterior(data, sampler, num_iters=num_iters)

                true_probs = get_exact_posterior(data, dist, partition_prior)

                self._test_posterior(pred_probs, true_probs)

    def _run_sampler_posterior(self, data, sampler, burnin=int(1e2), num_iters=int(1e4)):
        clustering = np.zeros(data.shape[0], dtype=int)

        test_counts = Counter()

        for i in range(num_iters):
            clustering = sampler.sample(clustering, data)

            if i >= burnin:
                test_counts[tuple(relabel_clustering(clustering))] += 1

        posterior_probs = defaultdict(float)

        norm_const = sum(test_counts.values())

        for key in test_counts:
            posterior_probs[key] = test_counts[key] / norm_const

        return posterior_probs

    def _test_posterior(self, pred_probs, true_probs):
        print(sorted(pred_probs.items()))
        print()
        print(sorted(true_probs.items()))
        for key in true_probs:
            self.assertAlmostEqual(pred_probs[key], true_probs[key], delta=0.02)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_gibbs']
    unittest.main()
