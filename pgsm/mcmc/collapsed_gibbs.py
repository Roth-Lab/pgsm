import numpy as np

from pgsm.math_utils import exp_normalize


class Table(object):

    def __init__(self, dish):
        self.dish = dish

        self.customers = set()

    def __contains__(self, idx):
        return idx in self.customers

    def add_customer(self, idx, value):
        self.customers.add(idx)

        self.dish.increment(value)

    def remove_customer(self, idx, value):
        self.customers.remove(idx)

        self.dish.decrement(value)


class CollapsedGibbsSampler(object):

    def __init__(self, dist, partition_prior):
        self.dist = dist

        self.partition_prior = partition_prior

    def sample(self, clustering, data, num_iters=1):
        tables = self._get_tables(clustering, data)

        for _ in range(num_iters):
            clustering, tables = self._resample_partition(clustering, data, tables)

        return clustering

    def _get_tables(self, clustering, data):
        tables = []

        for z in np.unique(clustering):
            idx = np.argwhere(clustering == z).flatten()

            dish = self.dist.create_params_from_data(data[idx])

            table = Table(dish)

            table.customers.update(idx)

            tables.append(table)

        return tables

    def _resample_partition(self, clustering, data, tables):
        num_data_points = data.shape[0]

        idx = np.arange(num_data_points)

        np.random.shuffle(idx)

        for customer_idx in idx:
            customer_data_point = data[customer_idx]

            table_idx = self._find_table(customer_idx, tables)

            tables[table_idx].remove_customer(customer_idx, customer_data_point)

            tables = [t for t in tables if t.dish.N > 0]

            new_table_idx = self._resample_customer_table_idx(customer_data_point, tables)

            is_new_table = (new_table_idx == len(tables))

            if is_new_table:
                dish = self.dist.create_params()

                tables.append(Table(dish))

            tables[new_table_idx].add_customer(customer_idx, customer_data_point)

        return self._prune(clustering, tables)

    def _resample_customer_table_idx(self, customer_data_point, tables):
        log_p = np.zeros(len(tables) + 1, dtype=np.float64)

        log_p[-1] = self._log_prob_new_table(customer_data_point, len(tables))

        for c, table in enumerate(tables):
            log_p[c] = self._log_prob_join_table(customer_data_point, table.dish)

        p, _ = exp_normalize(log_p)

        new_table_idx = np.random.multinomial(1, p).argmax()

        return new_table_idx

    def _log_prob_new_table(self, data_point, num_tables):
        new_params = self.dist.create_params()

        new_params.increment(data_point)

        log_p = self.partition_prior.log_tau_1_diff(num_tables)

        log_p += self.partition_prior.log_tau_2(1)

        log_p += self.dist.log_marginal_likelihood(new_params)

        return log_p

    def _log_prob_join_table(self, data_point, dish):
        log_p = self.partition_prior.log_tau_2_diff(dish.N)

        log_p += self.dist.log_predictive_likelihood(data_point, dish)

        return log_p

    def _prune(self, clustering, tables):
        tables = [x for x in tables if x.dish.N > 0]

        for z, table in enumerate(tables):
            clustering[np.array(list(table.customers))] = z

        return clustering, tables

    def _find_table(self, customer, tables):
        for i, t in enumerate(tables):
            if customer in t:
                return i
