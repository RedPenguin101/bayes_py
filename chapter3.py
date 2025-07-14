import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

data = pd.read_csv('data/chemical_shifts_theo_exp.csv')

data.tail()

diff = data.theo - data.exp

categories = pd.Categorical(data.aa).categories
idx = pd.Categorical(data.aa).codes
coords = {'aa':categories}

with pm.Model(coords=coords) as model_nh:
    μ = pm.Normal('μ', mu=0, sigma=10, dims='aa')
    σ = pm.HalfNormal('σ', sigma=10, dims='aa')
    y = pm.Normal('y', mu=μ[idx], sigma=σ[idx], observed=diff)
    idata_nh = pm.sample()

with pm.Model(coords=coords) as model_h:
    μ_mu = pm.Normal('μ_mu', mu=0, sigma=10)
    μ_sd = pm.HalfNormal('μ_sd', sigma=10)

    μ = pm.Normal('μ', mu=μ_mu, sigma=μ_sd, dims='aa')
    σ = pm.HalfNormal('σ', sigma=10, dims='aa')

    y = pm.Normal('y', mu=μ[idx], sigma=σ[idx], observed=diff)
    idata_h = pm.sample()

axes = az.plot_forest([idata_nh, idata_h], model_names=['non_hierarchical', 'hierarchical'],
                      var_names='μ', combined=True, r_hat=False, ess=False, figsize=(10, 7),
                      colors='cycle')
y_lims = axes[0].get_ylim()
axes[0].vlines(idata_h.posterior['μ_mu'].mean(), *y_lims, color="k", ls=":");
plt.savefig("c3_shifts_forest.png")

