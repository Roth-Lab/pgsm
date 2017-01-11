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
        self.num_data_points = len(data)


class InformedAnchorProposal(AnchorProposal):

    def propose_anchors(self, num_anchors):
        anchor_1 = np.random.randint(0, self.num_data_points)

        log_p_anchor = self.log_p[anchor_1].copy()

#         x = np.percentile(log_p_anchor, 25)

#         log_p_anchor[log_p_anchor > x] = float('-inf')
        alpha = np.random.beta(2, 8) * 100

        if np.random.random() <= 0.5:
            #             log_p_anchor = -log_p_anchor
            x = np.percentile(log_p_anchor, alpha)
            log_p_anchor[log_p_anchor > x] = float('-inf')
            log_p_anchor[log_p_anchor <= x] = 0
        else:
            x = np.percentile(log_p_anchor, 100 - alpha)
            log_p_anchor[log_p_anchor > x] = 0
            log_p_anchor[log_p_anchor <= x] = float('-inf')

        log_p_anchor[anchor_1] = float('-inf')

        if np.isneginf(np.max(log_p_anchor)):
            idx = np.arange(self.num_data_points)

            idx = list(idx)

            idx.remove(anchor_1)

            anchor_2 = np.random.choice(idx)

        else:
            p, _ = exp_normalize(log_p_anchor)

            anchor_2 = discrete_rvs(p)

        return anchor_1, anchor_2

    def setup(self, data, dist):
        self.num_data_points = data.shape[0]

        self.log_p = np.zeros((self.num_data_points, self.num_data_points))

        params = dist.create_params()

        log_seperate_margs = dist.log_predictive_likelihood_bulk(data, params)

        log_pairwise_margs = dist.log_pairwise_marginals(data, params)

        for i in range(self.num_data_points):
            for j in range(i):
                self.log_p[i, j] = log_pairwise_margs[i, j] - (log_seperate_margs[i] + log_seperate_margs[j])

                self.log_p[j, i] = self.log_p[i, j]
