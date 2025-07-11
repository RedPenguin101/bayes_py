import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

trials = 4
theta_real = 0.35

data = np.random.binomial(n=1, p=theta_real, size=trials)

with pm.Model() as model:
    θ = pm.Beta('θ', alpha=1, beta=1)
    y = pm.Bernoulli('y', p=θ, observed=data)
    idata = pm.sample(1000)

az.plot_trace(idata)
plt.savefig('c2_coin_trace.png')
