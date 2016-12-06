from __future__ import division

import numpy as np
import scipy.stats as stats


def simulate_binomial_data(n, mix_weights=None, num_clusters=None, p=None, size=1):
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


def simulate_mvn_data(dim=2, mix_weights=None, mu=None, num_clusters=1, sigma=None, size=1):
    if mu is None:
        mu = stats.multivariate_normal.rvs(np.zeros(dim), 100 * np.eye(dim), size=num_clusters)
    else:
        mu = np.array(mu)
        num_clusters = mu.shape[0]
        dim = mu.shape[1]

    if sigma is None:
        sigma = stats.wishart.rvs(df=(dim + 2), scale=np.eye(dim), size=num_clusters)
    else:
        sigma = np.stack([sigma for _ in range(num_clusters)], axis=0)

    mix_weights = np.random.dirichlet(np.ones(num_clusters))

    Z = np.random.multinomial(1, mix_weights, size=size).argmax(axis=1)
    if num_clusters == 1:
        X = stats.multivariate_normal.rvs(mu, sigma, size=size)
    else:
        X = []
        for z in Z:
            X.append(stats.multivariate_normal.rvs(mu[z], sigma[z]))
        X = np.array(X)

    return {'X': X, 'Z': Z, 'mu': mu, 'sigma': sigma, 'mix_weights': mix_weights}

simulate_mvn_data(dim=1, num_clusters=2, size=10)
