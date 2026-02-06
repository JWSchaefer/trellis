"""Example: NoPrior (parameter without distribution)."""

import jax
import jax.numpy as jnp

from trellis import Spec
from trellis.prior import NoPrior


class NoPriorSpec(Spec):
    x: NoPrior[float]


no_prior_spec = NoPriorSpec(x=NoPrior(value=42.0))
no_prior_params = no_prior_spec.init_params()

print(f'Params.x.value: {no_prior_params.x.value}')
print(f'tree_leaves: {jax.tree.leaves(no_prior_params)}')
print(
    f'log_prob: {no_prior_spec.x.log_prob(jnp.asarray(no_prior_params.x.value), no_prior_params.x, no_prior_spec.x.init_state())}'
)
