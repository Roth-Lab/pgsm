'''
Created on 9 Feb 2017

@author: Andrew Roth
'''
import numpy as np


def single_mvn_cluster(num_data_points, dim=2, seed=None):
    random_state = np.random.RandomState(seed=seed)

    return random_state.multivariate_normal(np.zeros(dim), np.eye(dim), size=num_data_points)
