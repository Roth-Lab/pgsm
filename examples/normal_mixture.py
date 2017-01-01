'''
Simple example of 5 2D multivariate normals.
'''

from sklearn.metrics import homogeneity_completeness_v_measure

import matplotlib.pyplot as pp
import numpy as np
import pandas as pd
import seaborn as sb

from pgsm.distributions.mvn import MultivariateNormalDistribution
from pgsm.mcmc.collapsed_gibbs import CollapsedGibbsSampler
from pgsm.mcmc.concentration import GammaPriorConcentrationSampler
from pgsm.mcmc.particle_gibbs_split_merge import ParticleGibbsSplitMergeSampler
from pgsm.partition_priors import DirichletProcessPartitionPrior

from pgsm.misc import Timer


def plot_clustering(clustering, data, title):
    plot_df = pd.DataFrame(data, columns=['0', '1'])
    plot_df['cluster'] = clustering
    g = sb.lmplot(x='0', y='1', data=plot_df, hue='cluster', fit_reg=False)
    g.ax.set_title(title)
    pp.draw()


def print_info(pred_clustering, true_clustering, iteration, time):
    print 'Iteration: {0}, Elapsed time: {1}'.format(i, t.elapsed)
    print 'Number of cluster: {}'.format(len(np.unique(pred_clustering)))
    print 'Homogeneity: {0}, Completeness: {1}, V-measure: {2}'.format(
        *homogeneity_completeness_v_measure(pred_clustering, true_clustering)
    )


def simulate_data():
    mu = [[10, 10], [10, -10], [-10, 10], [-10, -10], [0, 0]]
    cov = np.eye(2)
    X = []
    Z = []
    for z, m in enumerate(mu):
        X.append(np.random.multivariate_normal(m, cov, size=100))
        Z.append(z * np.ones(100))
    X = np.vstack(X)
    Z = np.concatenate(Z)
    return X, Z


data, true_clustering = simulate_data()
dist = MultivariateNormalDistribution(2)
partition_prior = DirichletProcessPartitionPrior(1)
conc_sampler = GammaPriorConcentrationSampler(1, 1)
pgsm_sampler = ParticleGibbsSplitMergeSampler.create_from_dist(dist, partition_prior)
gibbs_sampler = CollapsedGibbsSampler(dist, partition_prior)

num_data_points = data.shape[0]

with Timer() as t:
    pred_clustering = np.zeros(data.shape[0])
    for i in range(1000):
        if i % 10 == 0:
            t.stop()
            print_info(pred_clustering, true_clustering, i, t.elapsed)
            t.start()
        pred_clustering = pgsm_sampler.sample(pred_clustering, data)
        pred_clustering = gibbs_sampler.sample(pred_clustering, data)
        num_clusters = len(np.unique(pred_clustering))
        partition_prior.alpha = conc_sampler.sample(partition_prior.alpha, num_clusters, num_data_points)

plot_clustering(pred_clustering, data, 'Predicted clustering')
plot_clustering(true_clustering, data, 'True clustering')
pp.show()
