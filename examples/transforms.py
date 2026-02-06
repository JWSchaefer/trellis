"""Example: Constrained/unconstrained transforms round-trip."""

import jax.numpy as jnp

from trellis import Model, Spec
from trellis.prior import LogNormal


class PositiveParams(Spec):
    """A spec with positive-constrained parameters."""
    ell: LogNormal[float]      # Log transform (positive)
    sigma: LogNormal[float]    # Log transform (positive)


spec = PositiveParams(
    ell=LogNormal(value=2.0, mu=0.0, sigma=1.0),
    sigma=LogNormal(value=0.5, mu=0.0, sigma=1.0),
)

model = Model.from_spec(spec)

# Constrained params (what the model uses)
print('Constrained params:')
print(f'  ell = {model.params.ell.value}')
print(f'  sigma = {model.params.sigma.value}')

# Transform to unconstrained space (for optimization)
unconstrained = model.to_unconstrained()
print('\nUnconstrained params:')
print(f'  ell = {unconstrained.ell.value}')      # log(2.0) ~ 0.693
print(f'  sigma = {unconstrained.sigma.value}')  # log(0.5) ~ -0.693

# Round-trip back to constrained
model2 = model.from_unconstrained(unconstrained)
print('\nRound-trip constrained params:')
print(f'  ell = {model2.params.ell.value}')
print(f'  sigma = {model2.params.sigma.value}')

# Verify round-trip
print(f'\nRound-trip matches: {jnp.allclose(model.params.ell.value, model2.params.ell.value)}')

# Get the transforms dataclass
transforms = spec.get_transforms()
print(f'\nTransforms: {transforms}')
print(f'  ell transform: {transforms.ell.value}')
print(f'  sigma transform: {transforms.sigma.value}')
