from __future__ import division

import numpy as np

from pgsm.math_utils import exp_normalize


class Table(object):

    def __init__(self, customers, dish):
        self.customers = customers
        self.dish = dish


class CollapsedGibbsSampler(object):

    def __init__(self, dist, partition_prior):
        self.dist = dist
        self.partition_prior = partition_prior

    def sample(self, clustering, data, num_iters=1000):
        tables = self._get_tables(clustering, data)
        for _ in range(num_iters):
            clustering, tables = self._resample_params(clustering, data, tables)
        return clustering

    def _get_tables(self, clustering, data):
        tables = []
        for z in np.unique(clustering):
            idx = np.argwhere(clustering == z).flatten()
            tables.append(Table(list(idx), self.dist.create_params(data[idx])))
        return tables

    def _resample_params(self, clustering, data, tables):
        num_data_points = data.shape[0]
        idx = np.arange(num_data_points)
        np.random.shuffle(idx)
        for i in idx:
            clustering, tables = self._prune(clustering, tables)
            x = data[i]
            z = clustering[i]
            tables[z].customers.remove(i)
            tables[z].dish.decrement(x)

            log_p = []
            new_params = self.dist.create_params(x)
            log_p.append(self.partition_prior.log_tau_1(1) + self.partition_prior.log_tau_2(1) +
                         self.dist.log_marginal_likelihood(new_params))
            for table in tables:
                if table.dish.suff_stats.N == 0:
                    continue
                table.dish.increment(x)
                log_p_c = self.partition_prior.log_tau_2(table.dish.suff_stats.N) + \
                    self.dist.log_marginal_likelihood(table.dish)
                table.dish.decrement(x)
                log_p_c -= self.partition_prior.log_tau_2(table.dish.suff_stats.N) + \
                    self.dist.log_marginal_likelihood(table.dish)
                log_p.append(log_p_c)

            p = exp_normalize(log_p)
            z = np.random.multinomial(1, p).argmax() - 1
            if z == -1:
                tables.append(Table([i, ], new_params))
            else:
                tables[z].customers.append(i)
                tables[z].dish.increment(x)

        return clustering, tables

    def _prune(self, clustering, tables):
        tables = [x for x in tables if x.dish.suff_stats.N > 0]
        for z, table in enumerate(tables):
            for i in table.customers:
                clustering[i] = z
        return clustering, tables
