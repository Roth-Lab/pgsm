'''
Simple example of 4 2D multivariate normals.
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
from pgsm.mcmc.split_merge_setup import UniformSplitMergeSetupKernel


def plot_clustering(clustering, data, title):
    plot_df = pd.DataFrame(data, columns=['0', '1'])
    plot_df['cluster'] = clustering
    g = sb.lmplot(x='0', y='1', data=plot_df, hue='cluster', fit_reg=False)
    g.ax.set_title(title)
    pp.draw()


def print_info(pred_clustering, true_clustering, iteration):
    print 'Iteration: {0}'.format(i)
    print 'Number of cluster: {}'.format(len(np.unique(pred_clustering)))
    print 'Homogeneity: {0}, Completeness: {1}, V-measure: {2}'.format(
        *homogeneity_completeness_v_measure(pred_clustering, true_clustering)
    )


def simulate_data(nun_data_points_per_cluster=100):
    mu = [[10, 10], [10, -10], [-10, 10], [-10, -10]]
    cov = np.eye(2)
    X = []
    Z = []
    for z, m in enumerate(mu):
        X.append(np.random.multivariate_normal(m, cov, size=nun_data_points_per_cluster))
        Z.append(z * np.ones(nun_data_points_per_cluster))
    X = np.vstack(X)
    Z = np.concatenate(Z).astype(int)
    return X, Z

np.random.seed(0)
data, true_clustering = simulate_data(100)

dist = MultivariateNormalDistribution(2)
partition_prior = DirichletProcessPartitionPrior(1)

gibbs_sampler = CollapsedGibbsSampler(dist, partition_prior)

setup_kernel = UniformSplitMergeSetupKernel(data, dist, partition_prior)
pgsm_sampler = ParticleGibbsSplitMergeSampler.create_from_dist(dist, partition_prior, setup_kernel, num_anchors=2)

conc_sampler = GammaPriorConcentrationSampler(1, 1)

num_data_points = data.shape[0]

pred_clustering = np.zeros(num_data_points)
for i in range(100):
    if i % 10 == 0:
        print_info(pred_clustering, true_clustering, i)
    pred_clustering = pgsm_sampler.sample(pred_clustering, data)
    pred_clustering = gibbs_sampler.sample(pred_clustering, data)
    num_clusters = len(np.unique(pred_clustering))
    partition_prior.alpha = conc_sampler.sample(partition_prior.alpha, num_clusters, num_data_points)

plot_clustering(pred_clustering, data, 'Predicted clustering')
plot_clustering(true_clustering, data, 'True clustering')
pp.show()
