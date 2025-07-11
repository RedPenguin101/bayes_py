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

# Ex1.2

sample_space = ['strawberry', 'strawberry', 'blueberry', 'cinnamon']


def P(S, A):
    if set(A).issubset(set(S)):
        return len(A)/len(S)
    else:
        return 0


P(sample_space, ['strawberry', 'strawberry'])  # => 0.5
P(sample_space, ['strawberry', 'strawberry', 'blueberry'])  # =>0.75

sample_space2 = {'strawberry', 'blueberry', 'cinnamon'}


def P2(S, A, probs):
    if set(A).issubset(set(S)):
        return sum([probs[e] for e in A])
    else:
        return 0


P2(sample_space2, {'strawberry'}, {'strawberry': 0.5, 'blueberry': 0.25,
                                   'cinnamon': 0.25})
P2(sample_space2, {'blueberry', 'strawberry'},
   {'strawberry': 0.5, 'blueberry': 0.25, 'cinnamon': 0.25})
