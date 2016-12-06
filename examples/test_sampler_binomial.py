from __future__ import division

import numpy as np
import scipy.stats as stats

from pgsm.distributions.binomial import Binomial, BinomialPriors
from pgsm.geneaolgy import get_clustering, get_cluster_labels
from pgsm.kernels import NaiveSplitMergeKernel, FullyAdaptedSplitMergeKernel
from pgsm.math_utils import exp_normalize
from pgsm.partition_priors import DirichletProcessPartitionPrior
from pgsm.samplers.smc import ParticleGibbsSampler

from sklearn.metrics import v_measure_score


def simulate_data(n, mix_weights=None, num_clusters=None, p=None, size=1):
    if p is None:
        if mix_weights is not None:
            num_clusters = len(mix_weights)
        p = np.random.beta(1, 1, size=num_clusters)
    else:
        p = np.array(p)

    if mix_weights is None:
        mix_weights = np.random.dirichlet(np.ones(num_clusters))
    elif num_clusters is None:
        num_clusters = len(mix_weights)

    Z = np.random.multinomial(1, mix_weights, size=size).argmax(axis=1)
    a = stats.binom.rvs(n, p[Z])  # , size=size)
    return {'X': np.column_stack([a, n - a]), 'Z': Z, 'p': p, 'mix_weights': mix_weights}


def get_constrained_path(kernel, X, Z):
    constrained_path = [kernel.create_initial_particle()]
    constrained_path.append(kernel.create_particle(X[0], constrained_path[-1], 1))

    if Z[0] == Z[1]:
        state = 2
    else:
        state = 4
    constrained_path.append(kernel.create_particle(X[1], constrained_path[-1], state))

    for i in range(2, X.shape[0]):
        if constrained_path[-1].value.state == 2:
            state = 2
        elif Z[i] == Z[0]:
            state = 3
        elif Z[i] == Z[1]:
            state = 4
        else:
            raise Exception()
        constrained_path.append(kernel.create_particle(X[i], constrained_path[-1], state))
    return constrained_path

N = 100
D = 1000
data = simulate_data(D, mix_weights=[0.5, 0.5], p=[0.1, 0.9], size=N)
print data['p'], data['mix_weights']

priors = BinomialPriors(1, 1)
dist = Binomial(priors)
dp_prior = DirichletProcessPartitionPrior(2.0)
smc = ParticleGibbsSampler(100)
X = data['X']
Z = np.random.multinomial(1, [0.5, 0.5], size=N).argmax(axis=1)
sigma = np.arange(N)
for i in range(100):
    np.random.shuffle(sigma)
    assert np.all(Z == Z[sigma][np.argsort(sigma)])
    kernel = FullyAdaptedSplitMergeKernel(dist, 0, dp_prior)
    constrained_path = get_constrained_path(kernel, X[sigma], Z[sigma])
    particles = smc.sample(constrained_path, X[sigma], kernel)
    probs = exp_normalize([x.log_w for x in particles])
    p_idx = np.random.multinomial(1, probs).argmax()
    p = particles[p_idx]
    Z = get_cluster_labels(p)[np.argsort(sigma)]
    print '#' * 100
    print v_measure_score(data['Z'], Z), len(np.unique(Z))
    print '#' * 100

# print p.log_w, len(get_clustering(X.shape[0], p))
# print get_clustering(X.shape[0], p)
# x[0][0].log_w
# get_clustering(X.shape[0], x[0][0])
