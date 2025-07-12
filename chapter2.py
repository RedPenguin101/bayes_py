import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import beta

np.random.seed(123)

trials = 4
theta_real = 0.35
data = np.random.binomial(n=1, p=theta_real, size=trials)
alpha_prior = 1
beta_prior = 1

# Coin flip using analytics

data
success = np.count_nonzero(data == 1)
alpha_post = alpha_prior+success
beta_post = beta_prior+trials-success

plt.clf()
plt.plot(np.linspace(0, 1, 100),
         beta.pdf(np.linspace(0, 1, 100), alpha_post, beta_post))
plt.savefig('c2_coin_analytic.png', dpi=100)

# Coin flip using numerical

with pm.Model() as model:
    θ = pm.Beta('θ', alpha=1, beta=1)
    y = pm.Bernoulli('y', p=θ, observed=data)
    idata = pm.sample(1000)


az.plot_posterior(idata)
plt.savefig('c2_coin_posterior.png', dpi=100)
az.plot_trace(idata)
plt.savefig('c2_coin_trace.png', dpi=100)
plt.show()

az.summary(idata, kind='stats').round(2)

az.plot_trace(idata, combined=True)

az.plot_bf(idata, var_name='θ', prior=np.random.uniform(0,1,10000), ref_val=0.5)
plt.savefig('c2_coin_savage_dickey', dpi=100)

az.plot_posterior(idata, rope=[0.45, 0.55])
plt.savefig('c2_coin_rope.png', dpi=100)

plt.show()
