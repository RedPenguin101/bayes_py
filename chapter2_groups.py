import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import scipy.stats as stats

tips = pd.read_csv('data/tips.csv')
tips.tail()

type(tips['tip']) # pandas.core.series.Series
type(tips['tip'].values) # ndarray

tip = tips['tip'].values

categories = np.array(['Thur', 'Fri', 'Sat', 'Sun'])
type(pd.Categorical(tips['day'], categories=categories)) # pandas.core.arrays.categorical.Categorical

idx = pd.Categorical(tips['day'], categories=categories).codes
type(idx) # ndarray

# with pm.Model() as model:
    # μ = pm.Normal('μ', mu=0, sigma=10, shape=4)
    # σ = pm.HalfNormal('σ', sigma=10, shape=4)
    # y = pm.Normal('y', mu=μ[idx], sigma=σ[idx], observed=tip)

coords = {'days': categories, 'days_flat': categories[idx]}

categories[idx]

with pm.Model(coords=coords) as model:
    μ = pm.HalfNormal('μ', sigma=5, dims='days')
    σ = pm.HalfNormal('σ', sigma=1, dims='days')
    y = pm.Gamma('y', mu=μ[idx], sigma=σ[idx], observed=tip, dims='days_flat')
    idata = pm.sample()
    idata.extend(pm.sample_posterior_predictive(idata))

_, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.5, bottom=0.15)
az.plot_ppc(idata, num_pp_samples=100, coords={'days_flat':[categories]},flatten=[], ax=axes)
plt.savefig('c2_groups_ppc')

posterior = az.extract(idata)
comparisons = [(categories[i], categories[j]) for i in range(4) for j in range(i+1, 4)]

plt.clf()
_, axes = plt.subplots(3,2,figsize=(13,9),sharex=True)
plt.subplots_adjust(hspace=0.5, bottom=0.15)
dist = stats.Normal(mu=0,sigma=1)

for (i,j), ax in zip(comparisons, axes.ravel()):
    means_diff = posterior['μ'].sel(days=i) - posterior['μ'].sel(days=j)
    d_cohen = (means_diff /
               np.sqrt((posterior['σ'].sel(days=i)**2 +
                        posterior['σ'].sel(days=j)**2) / 2)
               ).mean().item()
    ps = dist.cdf(d_cohen/(2**0.5))
    az.plot_posterior(means_diff.values, ref_val=0, ax=ax)
    ax.set_title(f'{i} - {j}')
    ax.plot(0, label=f"Cohen's d = {d_cohen:.2f}\nProb sup = {ps:.2f}", alpha=0)
    ax.legend(loc=1)

plt.savefig('c2_groups_posterior')
