# 1.7 coin flipping / beta binomial

prior_params = [(1,1),(20,20),(1,4)]
data = [(0,0),(1,1),(2,1),(3,1),(4,1),(8,4),(16,6),(32,9),(50,13),(150,48)]

for N, y in data:
    print(f"{N} throws, {y} heads")
    for alpha, beta in prior_params:
        print(f"\t with Prior {alpha}, {beta}: {alpha+y}, {beta+N-y}")

# 1.9 Summarizing the Posterior

import numpy as np
import arviz as az
import matplotlib.pyplot as plt

az.plot_posterior({'θ': np.random.beta(4, 12, 1000)})
plt.savefig('summarize_posterior.png', dpi=100)

prior_alpha = 
