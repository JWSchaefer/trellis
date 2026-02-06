import jax
import jax.numpy as jnp
import jax.random as jr

from trellis import Spec
from trellis.prior import HalfNormal, LogNormalLearnable, Normal


class LearnablePriorSpec(Spec):
    ell: LogNormalLearnable[float, Normal[float], HalfNormal[float]]


learnable_spec = LearnablePriorSpec(
    ell=LogNormalLearnable(
        value=1.0,
        mu=Normal(value=0.0, loc=0.0, scale=10.0),
        sigma=HalfNormal(value=1.0, scale=1.0),
    )
)
learnable_params = learnable_spec.init_params()

print(f'Spec: {learnable_spec}')
print(f'Params: {learnable_params}')
print(f'Params.ell.value: {learnable_params.ell.value}')
print(f'Params.ell.mu.value: {learnable_params.ell.mu.value}')
print(f'Params.ell.sigma.value: {learnable_params.ell.sigma.value}\n')

leaves = jax.tree.leaves(learnable_params)
print(f'tree_leaves(params): {leaves}')
print(f'Number of leaves: {len(leaves)}\n')

# Evaluate log prob
x = jnp.array(1.5)
ell_state = learnable_spec.ell.init_state()
lp = learnable_spec.ell.log_prob(x, learnable_params.ell, ell_state)
print(f'log_prob({x}) = {lp}')

# Sample
key = jr.PRNGKey(0)
sample = learnable_spec.ell.sample(
    key, learnable_params.ell, ell_state, shape=(3,)
)
print(f'sample(shape=(3,)) = {sample}')
