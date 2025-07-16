from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import arviz as az

np.random.seed(123)

x = np.linspace(0, 1, 1000)

b11 = beta.pdf(x, 1, 1)
b2020 = beta.pdf(x, 20, 20)
b14 = beta.pdf(x, 1, 4)

plt.plot(x, b11, label="B(1,1)")
plt.plot(x, b2020, label="B(20,20)")
plt.plot(x, b14, label="B(1,4)")
plt.legend()
plt.yticks([])
plt.savefig("exercises_c2_betas.png")

plt.clf()

data = np.random.binomial(n=1, p=0.35, size=4)

with pm.Model() as model_11:
    θ = pm.Beta('θ_11', 1, 1)
    y = pm.Bernoulli('y', p=θ, observed=data)
    idata_11 = pm.sample()


with pm.Model() as model_20:
    θ = pm.Beta('θ_20', 20, 20)
    y = pm.Bernoulli('y', p=θ, observed=data)
    idata_20 = pm.sample()


with pm.Model() as model_14:
    θ = pm.Beta('θ_14', 1, 4)
    y = pm.Bernoulli('y', p=θ, observed=data)
    idata_14 = pm.sample()

plt.clf()
ax = idata_11.posterior.θ_11[0].to_pandas().plot.kde(color='b')
ax.plot(x, beta.pdf(x, 1+1, 1+3), linestyle='dashed', label="Anθ_11", color='b', alpha=0.5)

idata_20.posterior.θ_20[0].to_pandas().plot.kde(ax=ax, color='orange')
ax.plot(x, beta.pdf(x, 20+1, 20+3), linestyle='dashed', label="Anθ_20", color='orange', alpha=0.5)

idata_14.posterior.θ_14[0].to_pandas().plot.kde(ax=ax, color='g')
ax.plot(x, beta.pdf(x, 1+1, 4+3), linestyle='dashed', label="Anθ_14", color='g', alpha=0.5)

plt.yticks([])
plt.legend()
plt.savefig('exercises_2_1_beta_comparison')

plt.show()

with pm.Model() as model_u1:
    θ = pm.Uniform('θ_u1', 0, 1)
    y = pm.Bernoulli('y', p=θ, observed=data)
    idata_u1 = pm.sample()

plt.clf()
ax = idata_11.posterior.θ_11[0].to_pandas().plot.kde(color='b')
idata_u1.posterior.θ_u1[0].to_pandas().plot.kde(ax=ax, color='orange')
plt.legend()
plt.savefig('exercises_2_1_beta_uniform')
plt.show()

with pm.Model() as model_u2:
    θ = pm.Uniform('θ_u2', -1, 2)
    y = pm.Bernoulli('y', p=θ, observed=data)
    idata_u2 = pm.sample()

az.plot_trace(idata_u2)
plt.savefig('exercises_2_1_bad_uniform')
plt.show()
