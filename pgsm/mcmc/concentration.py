import numpy as np
import scipy.stats as stats


class GammaPriorConcentrationSampler(object):
    '''
    Gibbs update assuming a gamma prior on the concentration parameter.
    '''

    def __init__(self, a, b):
        '''
        Args :
            a : (float) Shape parameter of the gamma prior.
            b : (float) Rate parameter of the gamma prior.
        '''
        self.a = a

        self.b = b

    def sample(self, old_value, num_clusters, num_data_points):
        a = self.a

        b = self.b

        k = num_clusters

        n = num_data_points

        eta = stats.beta.rvs(a=old_value + 1, b=n)

        shape = (a + k - 1)

        rate = b - np.log(eta)

        x = shape / (n * rate)

        pi = x / (1 + x)

        shape += stats.bernoulli.rvs(pi)

        new_value = stats.gamma.rvs(shape, scale=(1 / rate))

        new_value = max(new_value, 1e-10)  # Catch numerical error

        return new_value
