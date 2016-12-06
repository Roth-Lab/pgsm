from distributions.mvn import MultivariateNormal, MultivariateNormalPriors
from kernels import NaiveSplitMergeKernel
from partition_priors import DirichletProcessPartitionPrior
from sampler import NaiveSMC
from geneaolgy import get_clustering
from math_utils import exp_normalize

import numpy as np

D = 1000
N = 10
X_1 = np.random.multivariate_normal(100 * np.ones(D), np.eye(D), size=N)
X_2 = np.random.multivariate_normal(-100 * np.ones(D), np.eye(D), size=N)
X = np.vstack([X_1, X_2])
np.random.shuffle(X)
# print X[:2]

priors = MultivariateNormalPriors(D)
dist = MultivariateNormal(priors)
partition_prior = DirichletProcessPartitionPrior(1.0)
kernel = NaiveSplitMergeKernel(dist, partition_prior)
smc = NaiveSMC(kernel, 10)
particles = smc.sample(X)
probs = exp_normalize([x.log_w for x in particles])
p_idx = np.random.multinomial(1, probs).argmax()
p = particles[p_idx]
print p.log_w, len(get_clustering(X.shape[0], p))
print get_clustering(X.shape[0], p)
