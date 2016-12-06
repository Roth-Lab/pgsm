from __future__ import division

from sklearn.metrics import homogeneity_completeness_v_measure

import numpy as np

from pgsm.distributions.binomial import Binomial, BinomialPriors
from pgsm.distributions.mvn import MultivariateNormal, MultivariateNormalPriors
from pgsm.geneaolgy import get_clustering, get_cluster_labels
from pgsm.kernels import NaiveSplitMergeKernel, FullyAdaptedSplitMergeKernel
from pgsm.partition_priors import DirichletProcessPartitionPrior
from pgsm.samplers.collapsed_gibbs import CollapsedGibbsSampler
from pgsm.samplers.particle_gibbs_split_merge import ParticleGibbsSplitMerge
from pgsm.samplers.smc import ParticleGibbsSampler

from simulate import simulate_binomial_data, simulate_mvn_data

model = 'binomial'
if model == 'binomial':
    N = 100
    D = 1000
    data = simulate_binomial_data(D, mix_weights=np.ones(3) * 1 / 3, p=[0.1, 0.5, 0.9], size=N)
    priors = BinomialPriors(10, 10)
    dist = Binomial(priors)

elif model == 'mvn':
    N = 100
    mu = np.array([[-10, -10], [10, 10], [10, -10]])
    D = mu.shape[1]
    sigma = np.eye(D)
    data = simulate_mvn_data(mu=mu, sigma=sigma, size=N)
#     D = 10
#     K = 10
#     data = simulate_mvn_data(dim=D, num_clusters=K, size=N)
    priors = MultivariateNormalPriors(dim=D)
    dist = MultivariateNormal(priors)

# print data['Z'], data['X']

partition_prior = DirichletProcessPartitionPrior(1.0)
# kernel = NaiveSplitMergeKernel(dist, partition_prior)
pgs = ParticleGibbsSampler(25)
sampler = ParticleGibbsSplitMerge(dist, partition_prior, pgs, kernel_cls=FullyAdaptedSplitMergeKernel, num_anchors=4)
# sampler = ParticleGibbsSplitMerge(dist, partition_prior, pgs, kernel_cls=NaiveSplitMergeKernel, num_anchors=4)
gibbs_sampler = CollapsedGibbsSampler(dist, partition_prior)
X = data['X']
# Z = np.arange(X.shape[0])  # sampler.sample(X, num_iters=1, init_clustering='separate')
Z = np.zeros(X.shape[0], dtype=np.int)
# Z = data['Z'].copy()
# Z = np.random.randint(0, 10, size=X.shape[0])
for i in range(100):
    print homogeneity_completeness_v_measure(data['Z'], Z), len(np.unique(Z))
    Z = sampler.sample(X, init_clustering=Z, num_iters=10)
#     print homogeneity_completeness_v_measure(data['Z'], Z), len(np.unique(Z))
#     gibbs_sampler.sample(X, init_clustering=Z, num_iters=4)
