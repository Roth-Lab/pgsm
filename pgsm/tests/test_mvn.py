'''
Created on 24 Jan 2017

@author: andrew
'''
from __future__ import division

import unittest

import numpy as np
import scipy.stats as stats

from pgsm.math_utils import cholesky_log_det, cholesky_update

import pgsm.distributions.mvn as mvn


class Test(unittest.TestCase):

    def test_cholesky_det(self):
        dim = 10

        X = stats.invwishart.rvs(dim + 1, np.eye(dim))

        X_chol = np.linalg.cholesky(X)

        np.testing.assert_allclose(cholesky_log_det(X_chol), np.linalg.slogdet(X)[1])

    def test_cholesky_update(self):
        dim = 10

        L = np.eye(dim)

        X = np.random.multivariate_normal(np.zeros(dim), 10 * np.eye(dim), size=100)

        for x in X:
            L = cholesky_update(L, x)

        np.testing.assert_allclose(np.dot(L, L.T), np.eye(dim) + np.dot(X.T, X))

    def test_predictive(self):
        dim = 10

        train_data = np.random.multivariate_normal(np.random.random(size=dim) * 100, np.eye(dim), size=100)

        test_data = np.random.multivariate_normal(np.random.random(size=dim) * 100, np.eye(dim), size=100)

        dist = mvn.MultivariateNormalDistribution(dim)

        params = dist.create_params_from_data(train_data)

        pred_1 = []

        pred_2 = []

        for data_point in test_data:
            pred_1.append(dist.log_predictive_likelihood(data_point, params))

            params.increment(data_point)

            diff = dist.log_marginal_likelihood(params)

            params.decrement(data_point)

            diff -= dist.log_marginal_likelihood(params)

            pred_2.append(diff)

        self._assert_params_equal(dist.create_params_from_data(train_data), params)

        np.testing.assert_allclose(pred_1, pred_2)

    def test_marginal(self):
        '''
        Test the marginal is equivalent to the Chib estimator approach.
        '''
        dim = 10

        data = np.random.multivariate_normal(np.random.random(size=dim) * 100, np.eye(dim), size=100)

        dist = mvn.MultivariateNormalDistribution(dim)

        prior_params = dist.create_params()

        posterior_params = dist.create_params_from_data(data)

        test_mu = np.zeros(dim)

        test_cov = np.eye(dim)

        chib = np.sum(stats.multivariate_normal.logpdf(data, test_mu, test_cov))

        chib += self._niw_log_pdf(test_mu, test_cov, prior_params)

        chib -= self._niw_log_pdf(test_mu, test_cov, posterior_params)

        marg = dist.log_marginal_likelihood(posterior_params)

        self.assertAlmostEqual(chib, marg)

    def test_increment(self):
        dim = 10

        data = np.random.multivariate_normal(np.random.random(size=dim) * 100, np.eye(dim), size=100)

        dist = mvn.MultivariateNormalDistribution(dim)

        params_1 = dist.create_params_from_data(data)

        params_2 = dist.create_params()

        for data_point in data:
            params_2.increment(data_point)

        self._assert_params_equal(params_1, params_2)

    def test_decrement(self):
        dim = 10

        data = np.random.multivariate_normal(np.random.random(size=dim) * 100, np.eye(dim), size=100)

        dist = mvn.MultivariateNormalDistribution(dim)

        params_1 = dist.create_params_from_data(data)

        params_2 = dist.create_params()

        for data_point in data:
            params_1.decrement(data_point)

        self._assert_params_equal(params_1, params_2)

    def _assert_params_equal(self, params_1, params_2):
        np.testing.assert_allclose(params_1.nu, params_2.nu)

        np.testing.assert_allclose(params_1.r, params_2.r)

        np.testing.assert_allclose(params_1.u, params_2.u, atol=1e-8)

        np.testing.assert_allclose(params_1.S_chol, params_2.S_chol, atol=1e-8)

        np.testing.assert_allclose(params_1.N, params_2.N)

    def _niw_log_pdf(self, mean, cov, params):
        log_p = stats.multivariate_normal.logpdf(mean, params.u, (1 / params.r) * cov)

        log_p += stats.invwishart.logpdf(cov, params.nu, params.S)

        return log_p

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_cholesky']
    unittest.main()
