'''
Created on 19 Jan 2017

@author: Andrew Roth
'''
from __future__ import division

import numba
import numpy as np
import pandas as pd

from pgsm.math_utils import log_binomial_coefficient, log_normalize, log_sum_exp


class PycloneParameters(object):

    def __init__(self, log_pdf_grid, N):
        self.log_pdf_grid = log_pdf_grid

        self.N = N

    @property
    def normalized_log_pdf_grid(self):
        return log_normalize(self.log_pdf_grid)

    def copy(self):
        return PycloneParameters(self.log_pdf_grid.copy(), self.N)

    def decrement(self, x):
        self.log_pdf_grid -= x

        self.N -= 1

    def increment(self, x):
        self.log_pdf_grid += x

        self.N += 1


class PyCloneDistribution(object):

    def __init__(self, grid_size):
        self.grid_size = grid_size

    def create_params(self):
        uniform_log_prior = -np.log(self.grid_size) * np.ones(self.grid_size)

        return PycloneParameters(uniform_log_prior, 0)

    def create_params_from_data(self, X):
        X = np.atleast_2d(X)

        return PycloneParameters(np.sum(X, axis=0), X.shape[0])

    def log_marginal_likelihood(self, params):
        return log_sum_exp(params.log_pdf_grid)

    def log_predictive_likelihood(self, data_point, params):
        return log_sum_exp(data_point + params.normalized_log_pdf_grid)

    def log_predictive_likelihood_bulk(self, data, params):
        log_p = np.zeros(len(data))

        for i, data_point in enumerate(data):
            log_p[i] = self.log_predictive_likelihood(data_point, params)

        return log_p

    def log_pairwise_marginals(self, data, params):
        num_data_points = len(data)

        log_p = np.zeros((num_data_points, num_data_points))

        for i in range(num_data_points):
            for j in range(num_data_points):
                if i == j:
                    continue

                params = self.create_params_from_data(data[[i, j]])

                log_p[i, j] = self.log_marginal_likelihood(params)

        return log_p


def load_data_from_file(file_name, error_rate=1e-3, grid_size=1000, perfect_prior=False, tumour_content=1.0):
    '''
    Given a PyClone input tsv formatted file, load the discretized grid of likelihoods.

    See https://bitbucket.org/aroth85/pyclone/wiki/Usage for information about the input file format. For debugging
    purposes, this file can also include information about the mutational genotype for use with the perfrect_prior
    argument.
    '''
    data = []

    df = pd.read_csv(file_name, sep='\t')

    for _, row in df.iterrows():
        a = row['ref_counts']

        b = row['var_counts']

        cn_n = row['normal_cn']

        major_cn = row['major_cn']

        total_cn = row['major_cn'] + row['minor_cn']

        # Use the true mutational genotype information
        if perfect_prior:
            cn = [
                [len(row['g_n']), len(row['g_r']), len(row['g_v'])]
            ]

            mu = [
                [error_rate, error_rate, min(1 - error_rate, row['g_v'].count('B') / len(row['g_v']))]
            ]

            log_pi = [0, ]

        # Elicit mutational genotype prior based on major and minor copy number
        else:
            cn = []

            mu = []

            log_pi = []

            # Consider all possible mutational genotypes consistent with mutation before CN change
            for x in range(1, major_cn + 1):
                cn.append((cn_n, cn_n, total_cn))

                mu.append((error_rate, error_rate, min(1 - error_rate, x / total_cn)))

                log_pi.append(0)

            # Consider mutational genotype of mutation before CN change if not already added
            mutation_after_cn = (cn_n, total_cn, total_cn)

            if mutation_after_cn not in cn:
                cn.append(mutation_after_cn)

                mu.append((error_rate, error_rate, min(1 - error_rate, 1 / total_cn)))

                log_pi.append(0)

                assert len(set(cn)) == 2

        cn = np.array(cn, dtype=int)

        mu = np.array(mu, dtype=float)

        log_pi = log_normalize(np.array(log_pi, dtype=float))

        data.append(DataPoint(a, b, cn, mu, log_pi))

    return convert_data_to_discrete_grid(data, grid_size=grid_size, tumour_content=tumour_content)


def convert_data_to_discrete_grid(data, grid_size=1000, tumour_content=1.0):
    log_ll = np.zeros((len(data), grid_size))

    for i, data_point in enumerate(data):
        for j, cellular_prevalence in enumerate(np.linspace(0, 1, grid_size)):
            log_ll[i, j] = compute_log_p_sample(data_point, cellular_prevalence, tumour_content)

    return log_ll


@numba.jitclass([
    ('a', numba.int64),
    ('b', numba.int64),
    ('cn', numba.int64[:, :]),
    ('mu', numba.float64[:, :]),
    ('log_pi', numba.float64[:]),
])
class DataPoint(object):

    def __init__(self, a, b, cn, mu, log_pi):
        self.a = a
        self.b = b
        self.cn = cn
        self.mu = mu
        self.log_pi = log_pi


@numba.jit(nopython=True)
def compute_log_p_sample(data, f, t):
    C = len(data.cn)

    population_prior = np.zeros(3)
    population_prior[0] = (1 - t)
    population_prior[1] = t * (1 - f)
    population_prior[2] = t * f

    ll = np.ones(C, dtype=np.float64) * np.inf * -1

    for c in range(C):
        e_vaf = 0

        norm_const = 0

        for i in range(3):
            e_cn = population_prior[i] * data.cn[c, i]

            e_vaf += e_cn * data.mu[c, i]

            norm_const += e_cn

        e_vaf /= norm_const

        ll[c] = data.log_pi[c] + binomial_log_pdf(data.a + data.b, data.b, e_vaf)

    return log_sum_exp(ll)


@numba.jit(nopython=True)
def binomial_log_likelihood(n, x, p):
    return x * np.log(p) + (n - x) * np.log(1 - p)


@numba.jit(nopython=True)
def binomial_log_pdf(n, x, p):
    return log_binomial_coefficient(n, x) + binomial_log_likelihood(n, x, p)
