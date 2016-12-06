from __future__ import division

from sklearn.metrics import v_measure_score

import numpy as np
import scipy.stats as stats

from pgsm.distributions.binomial import Binomial, BinomialPriors
from pgsm.distributions.mvn import MultivariateNormal, MultivariateNormalPriors
from pgsm.geneaolgy import get_clustering, get_cluster_labels
from pgsm.kernels import NaiveSplitMergeKernel
from pgsm.math_utils import exp_normalize
from pgsm.partition_priors import DirichletProcessPartitionPrior
from pgsm.samplers.mcmc import CollapsedGibbsSampler

from simulate import simulate_binomial_data, simulate_mvn_data

model = 'mvn'
if model == 'binomial':
    N = 100
    D = 1000
    data = simulate_binomial_data(D, mix_weights=[0.5, 0.5], p=[0.1, 0.9], size=N)
    priors = BinomialPriors(1, 1)
    dist = Binomial(priors)

elif model == 'mvn':
    N = 100
    D = 10
    K = 4
    mu = [[-10, -10, 0], [10, 10, 0]]
    sigma = np.eye(D)
    data = simulate_mvn_data(dim=D, num_clusters=K, size=N)
    priors = MultivariateNormalPriors(dim=D)
    dist = MultivariateNormal(priors)
    # print data['Z'], data['mu']

dp_prior = DirichletProcessPartitionPrior(1.0)
sampler = CollapsedGibbsSampler(dist, dp_prior)
X = data['X']
Z = sampler.sample(X, num_iters=1)
for i in range(1000):
    print v_measure_score(data['Z'], Z), len(np.unique(Z))
    Z = sampler.sample(X, init_clustering=Z, num_iters=1)
# Z = np.random.multinomial(1, [0.5, 0.5], size=N).argmax(axis=1)
# sigma = np.arange(N)
# for i in range(100):
#     np.random.shuffle(sigma)
#     assert np.all(Z == Z[sigma][np.argsort(sigma)])
#     constrained_path = get_constrained_path(kernel, X[sigma], Z[sigma])
#     particles = smc.sample(constrained_path, X[sigma])
#     probs = exp_normalize([x.log_w for x in particles])
#     p_idx = np.random.multinomial(1, probs).argmax()
#     p = particles[p_idx]
#     Z = get_cluster_labels(N, p)[np.argsort(sigma)]
#     print '#' * 100

#     print '#' * 100

# print p.log_w, len(get_clustering(X.shape[0], p))
# print get_clustering(X.shape[0], p)
# x[0][0].log_w
# get_clustering(X.shape[0], x[0][0])
