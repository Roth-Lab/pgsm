from distributions.mvn import MultivariateNormal, MultivariateNormalParameters, MultivariateNormalPriors, MultivariateNormalSufficientStatistics
from kernels import BasicSplitMergeKernel
from partition_priors import DirichletProcessPartitionPrior

import numpy as np
import random

N = 1000
D = 100
X = np.random.multivariate_normal(np.zeros(D), np.eye(D), size=N)
# ss = MultivariateNormalSufficientStatistics(X[0])
# priors = MultivariateNormalPriors(D)
# params = MultivariateNormalParameters(priors, ss)
# params_1 = params.copy()
# params_1.increment(X[1])
# params.u
# dist = MultivariateNormal(priors)
# dist.create_params(X[0]).u
# priors = MultivariateNormalPriors(D)
# mvn = MultivariateNormal(priors)
# mvn.create_params(X[0])
# print mvn.log_marginal_likelihood()
# mvn.increment(X[1])
# print mvn.log_marginal_likelihood()
# mvn.decrement(X[1])
# print mvn.log_marginal_likelihood()
# dp_prior = DirichletProcessPartitionPrior(1.0)
# # print dp_prior.log_marginal_likelihood([3, 4])
# dist_factory = lambda x: MultivariateNormal(priors, MultivariateNormalSufficientStatistics(x))


dp_prior = DirichletProcessPartitionPrior(1.0)
mvn_prior = MultivariateNormalPriors(D)
dist = MultivariateNormal(mvn_prior)
kernel = BasicSplitMergeKernel(dist, dp_prior)
for _ in range(10):
    p = None
    for i in range(N):
        p = kernel.propose(X[i], p, random.random())
    print len(p.value.posterior_params), p.value.posterior_params[0].ss.N,
kernel.dist.log_marginal_likelihood(p.value.posterior_params[0])
