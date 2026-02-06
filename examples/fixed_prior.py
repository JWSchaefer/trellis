import jax
import jax.numpy as jnp
import jax.random as jr

from trellis import Spec
from trellis.prior import NoPrior


class FixedPriorSpec(Spec):
    ell: NoPrior[float]


fixed_spec = FixedPriorSpec(ell=NoPrior(value=1.0))
fixed_params = fixed_spec.init_params()
fixed_state = fixed_spec.init_state()

print(f'Spec: {fixed_spec}')
print(f'Params: {fixed_params}')
print(f'Params.ell.value: {fixed_params.ell.value}')
print(f'State: {fixed_state}')

# Test the prior directly
ell_prior = fixed_spec.ell
prior_params = ell_prior.init_params()
prior_state = ell_prior.init_state()

print(f'\nPrior: {ell_prior}')
print(f'Prior.Params: {prior_params}')
print(f'Prior.Params.value: {prior_params.value}')

# Evaluate log prob
x = jnp.array(1.5)
lp = ell_prior.log_prob(x, prior_params, prior_state)
print(f'\nlog_prob({x}) = {lp}')

# Sample
key = jr.PRNGKey(0)
sample = ell_prior.sample(key, prior_params, prior_state, shape=(3,))
print(f'sample(shape=(3,)) = {sample}')

# Tree leaves
print(f'\ntree_leaves(params): {jax.tree.leaves(fixed_params)}')

# Transforms
transforms = fixed_spec.get_transforms()
print(f'Transforms: {transforms}')
