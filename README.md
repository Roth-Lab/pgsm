# PGSM

Python implementation of samplers from "Particle Gibbs Split-Merge Sampling for Bayesian Inference in Mixture Models".
Version 0.1.1 of the software was used for all experiments in the paper and the code can be found [here](https://bitbucket.org/aroth85/pgsm/get/0.1.1.zip).

Please report any issues using the issue tracker.
Questions about the software or PGSM method can be asked on the [user group](https://groups.google.com/forum/#!forum/pgsm-user-group).

## Installation

The following dependencies need to be installed

- python 2.7.x
- numba
- numpy
- pandas
- scipy

> We suggest using [MiniConda](https://conda.io/miniconda.html) to get Python and then use the `conda` command to install the dependencies.

Once the required dependencies are installed the software can be installed by running `setup.py install` from the repository root directory.

> If you wish to make modifications to the code you can use `python setup.py develop` to install the package in development mode.
This will allow your modifications to take effect immediately.

## Running

For an example of how to run the code see the `examples/normal_mixture.py`.
Below we give details on how the library is structured, and how samplers can be instantiated and used for sampling. 

### Samplers

There are three basic samplers implemented.

1. `pgsm.mcmc.collapsed_gibbs.CollapsedGibbsSampler` - Standard collapsed gibbs sampler.

2. `pgsm.mcmc.particle_gibbs_split_merge.ParticleGibbsSplitMergeSampler` - Particle Gibbs Split Merge (PGSM) sampler.

3. `pgsm.mcmc.sams.SequentiallyAllocatedMergeSplitSampler` - Sequentially Allocated Split Merge (SAMS) sampler.

In addition there are two convenience samplers that wrap the basic samplers.

1. `pgsm.mcmc.mixed.MixedSampler` - Mixed sampler which is designed to interleave Gibbs and split merge kernels.

2. `pgsm.mcmc.dp.DirichletProcessSampler` - Dirichlet sampler which wraps any sampler and uses a Dirichlet process partition prior with a Gamma prior for the concentration parameter.

To instantiate the basic samplers you will need to define a distribution and a partition prior.
Supported distributions are 

- Normal likelihood with a Normal inverse Wishart prior
- Multivariate Bernoulli with Beta a prior
- The domain specific PyClone model with a Uniform prior

and can be found in `pgsm.distributions`.
Supported partition priors are

- Dirichlet process 
- Finite symmetric Dirichlet

and can be found in `pgsm.partition_priors`.  

Both the SAMS and PGSM samplers also require a SplitMergeSetupKernel. 
The main function of this kernel is to propose pairs of anchor points.
A number of kernels are provided in `pgsm.mcmc.split_merge_setup`.
Only the UniformSplitMergeSetupKernel can currently be used with the PGSM when more than two anchors are used.

### Setting up a sampler

In the following example we will setup a model with a 2D Normal distribution and Dirichlet process partition prior.

First we setup the distribution.

```python
from pgsm.distributions.mvn import MultivariateNormalDistribution

dim = 2
dist = MultivariateNormalDistribution(dim)
```

Next we setup the Dirichlet process partition prior with a concentration of 1.0.

```python
from pgsm.partition_priors import DirichletProcessPartitionPrior

init_concentration = 1.0
partition_prior = DirichletProcessPartitionPrior(init_concentration)
```

Now we can create a collapsed Gibbs sampler.

```python
from pgsm.mcmc.collapsed_gibbs import CollapsedGibbsSampler

gibbs_sampler = CollapsedGibbsSampler(dist, partition_prior)
```

For the SAMS and PGSM sampler we first create a setup kernel.
For simplicity we will use the uniform setup kernel which proposes anchor points uniformly at random.

```
from pgsm.mcmc.split_merge_setup import UniformSplitMergeSetupKernel

setup_kernel = UniformSplitMergeSetupKernel(data, dist, partition_prior)
```

Now we can create a SAMS sampler.

```python
from pgsm.mcmc.sams import SequentiallyAllocatedMergeSplitSampler

sams_sampler = SequentiallyAllocatedMergeSplitSampler(dist, partition_prior, setup_kernel)
```

The PGSM sampler requires two additional component before we can create it.
First, the sequential Monte Carlo (SMC) kernel.
There are three options which are provided in `pgsm.smc.kernels`.

- `pgsm.smc.kernels.UniformSplitMergeKernel` - Uniformly propose an new anchor block.
- `pgsm.smc.kernels.FullyAdaptedSplitMergeKernel` - Propose a new anchor block with probability proportional to importance sampling weight.
- `pgsm.smc.kernels.AnnealedSplitMergeKernel` - The annealed proposals described in the PGSM paper.

We will use the annealed kernel.

```python
from pgsm.smc.kernels import AnnealedSplitMergeKernel

smc_kernel = AnnealedSplitMergeKernel(dist, partition_prior)
```

Second, we need and SMC sampler.
There are three options.

- `pgsm.smc.samplers.IndependentSMCSampler` - Standard SMC with no conditional path. 
This must be combined with a Metropolis Hastings accept reject step to create the particle independent Metropolis Hastings (PIMH) sampler.
This is provided for pedagogical reasons, but not used in the PGSM paper.
- `pgsm.smc.samplers.ParticleGibbsSampler` - Standard particle Gibbs (PG) sampler.
This is provided for pedagogical purposes, but the `ImplicitParticleGibbsSampler` should be used in practice.
- `pgsm.smc.samplers.ImplicitParticleGibbsSampler` - More computationally efficient implementation of the PG sampler. 

We will use the `ImplicitParticleGibbsSampler` which was used for the PGSM paper.
We will use 20 particles and a relative ESS resample threshold of 0.5. 

```python
from pgsm.smc.samplers import ImplicitParticleGibbsSampler

smc_sampler = ImplicitParticleGibbsSampler(20, resample_threshold=0.5)
```

Finally we can create the PGSM sampler.

```python
from pgsm.mcmc.particle_gibbs_split_merge import ParticleGibbsSplitMergeSampler

pgsm_sampler = ParticleGibbsSplitMergeSampler(smc_kernel, smc_sampler, split_merge_setup_kernel)
```

### Using a sampler

Once a sampler has been created the process for running the sampler is the same.
All samplers implement the `sample` method which takes the current clustering and observed data as arguments and returns an updated clustering.

Assume we a have `numpy` array, `data`, with our data, where columns are dimensions, and row are data points.
To use the PGSM sampler create above (other samplers are the same) for 100 iterations starting from the fully connected initialization we dow the following.

```python
import numpy as np

clustering = np.zeros(data.shape[0) # Creates the initial clustering with all data points together.

for i in range(100):
    clustering = pgsm_sampler.sample(clustering, data)
```
