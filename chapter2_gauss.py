import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import halfnorm

data = np.loadtxt('data/chemical_shifts.csv')
data.max()

plt.clf()
plt.boxplot(data, vert=False)
plt.savefig('c2_gauss_box.png', dpi=100)
plt.hist(data)
plt.savefig('c2_gauss_hist.png')

# half normal illustration

plt.clf()
plt.plot(np.linspace(0, 20, 100),
         halfnorm.pdf(np.linspace(0, 20, 100), scale=5))
plt.savefig('c2_gauss_halfnorm.png', dpi=100)

# model

with pm.Model() as model:
    μ = pm.Uniform('μ', lower=40, upper=70)
    σ = pm.HalfNormal('σ', sigma=5)
    Y = pm.Normal('Y', mu=μ, sigma=σ, observed=data)
    idata = pm.sample()

plt.clf()
az.plot_trace(idata)
plt.savefig('c2_gauss_trace')

az.plot_pair(idata, kind='kde', marginals=True)
plt.savefig('c2_gauss_pair')

az.summary(idata, kind='stats').round(2)

pm.sample_posterior_predictive(idata, model, extend_inferencedata=True)

az.plot_ppc(idata, num_pp_samples=100)
plt.savefig('c2_gauss_ppc')

with pm.Model() as model_t:
    ν = pm.Exponential('ν', 1/30)
    μ = pm.Uniform('μ', lower=40, upper=75)
    σ = pm.HalfNormal('σ', sigma=10)
    Y = pm.StudentT('Y', mu=μ, sigma=σ, nu=ν, observed=data)
    idata_t = pm.sample()

plt.clf()
az.plot_trace(idata_t)
plt.savefig('c2_gauss_studentt_trace')

az.summary(idata_t, kind='stats').round(2)

pm.sample_posterior_predictive(idata_t, model_t, extend_inferencedata=True)

ax = az.plot_ppc(idata_t, num_pp_samples=100)
ax.set_xlim(40,70)
plt.savefig('c2_gauss_studentt_ppc')
