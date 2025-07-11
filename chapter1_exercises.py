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
