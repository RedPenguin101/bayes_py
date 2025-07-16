import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

heads = 3
tails = 10
n = 10

x = np.linspace(0, 1, n)
prior = np.repeat(1/n, n)
likelihood = stats.binom.pmf(heads, heads+tails, x)
posterior = likelihood * prior

plt.plot(x, posterior)
plt.yticks([])
plt.savefig("c10_grid_method")
plt.show()
