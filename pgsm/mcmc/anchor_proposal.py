'''
Created on 9 Jan 2017

@author: Andrew Roth
'''
from __future__ import division

import numpy as np
from pgsm.math_utils import discrete_rvs, exp_normalize


class AnchorProposal(object):

    def propose_anchors(self, num_anchors):
        raise NotImplementedError()

    def setup(self, data, dist):
        raise NotImplementedError()


class UniformAnchorProposal(AnchorProposal):

    def propose_anchors(self, num_anchors):
        return np.random.choice(np.arange(self.num_data_points), replace=False, size=num_anchors)

    def setup(self, data, dist):
        self.num_data_points = len(self.data)


class InformedAnchorProposal(AnchorProposal):

    def propose_anchors(self, num_anchors):
        anchor_1 = np.random.randint(0, self.num_data_points)

        log_p_anchor = self.log_p[anchor_1].copy()

        if np.random.random() <= 0.5:
            log_p_anchor = -log_p_anchor

        log_p_anchor[anchor_1] = float('-inf')

        p, _ = exp_normalize(log_p_anchor)

        anchor_2 = discrete_rvs(p)

        return anchor_1, anchor_2

    def setup(self, data, dist):
        self.num_data_points = data.shape[0]

        self.log_p = np.zeros((self.num_data_points, self.num_data_points))

        params = dist.create_params()

        log_seperate_margs = dist.log_predictive_likelihood_bulk(data, params)

        for i in range(self.num_data_points):
            params = dist.create_params_from_data(np.atleast_2d(data[i]))

            log_merged_margs = dist.log_predictive_likelihood_bulk(data, params)

            for j in range(i):
                self.log_p[i, j] = 0.5 * log_merged_margs[j] - (log_seperate_margs[i] + log_seperate_margs[j])

                self.log_p[j, i] = self.log_p[i, j]
