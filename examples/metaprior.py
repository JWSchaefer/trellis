"""Example: Metaprior (hierarchical Bayes)."""

import jax

from trellis import Spec
from trellis.prior import HalfNormal, LogNormalLearnable, Normal, NormalLearnable


class MetaPriorSpec(Spec):
    ell: LogNormalLearnable[
        float,
        NormalLearnable[float, Normal[float], HalfNormal[float]],
        HalfNormal[float],
    ]


meta_spec = MetaPriorSpec(
    ell=LogNormalLearnable(
        value=1.0,
        mu=NormalLearnable(
            value=0.0,
            loc=Normal(value=0.0, loc=0.0, scale=100.0),
            scale=HalfNormal(value=1.0, scale=10.0),
        ),
        sigma=HalfNormal(value=1.0, scale=1.0),
    )
)
meta_params = meta_spec.init_params()

print(f'Params.ell.value: {meta_params.ell.value}')
print(f'Params.ell.mu.value: {meta_params.ell.mu.value}')
print(f'Params.ell.mu.loc.value: {meta_params.ell.mu.loc.value}')
print(f'Params.ell.mu.scale.value: {meta_params.ell.mu.scale.value}')
print(f'Params.ell.sigma.value: {meta_params.ell.sigma.value}')

# Tree leaves - should contain ALL learnable values in the hierarchy
leaves = jax.tree.leaves(meta_params)
print(f'\ntree_leaves(params): {leaves}')
print(
    f'Number of leaves: {len(leaves)} (expected 5: value, mu.value, mu.loc.value, mu.scale.value, sigma.value)'
)
