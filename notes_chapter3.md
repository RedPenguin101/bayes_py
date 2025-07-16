# Hierarchical Models

In the previous chapter we modelled tips using a model that assigned
_independent_ priors to each category. That assumes that knowing the
tip for one day gives us no information about tips on another day.

This chapter looks at models which share information between groups:
hierarchical models using hyperpriors.

```
        pooled    unpooled    hierarchical
hyper                              Φ
                                 / | \
prior     θ       θ1 θ2 θ3      θ1 θ2 θ3
        / | \     |  |  |       |  |   |
group  y1 y2 y3   y1 y2 y3      y1 y2 ym
```

Unpooled is what we used for the tips dataset.

To see the impact of this we'll do unpooled and hierarchical models
for the differences between theoretical and experimental chemical
shifts (it still doesn't matter what these are)

