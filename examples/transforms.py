"""Example: Constrained/unconstrained transforms round-trip."""

import jax.numpy as jnp

from trellis import Model, Spec
from trellis.prior import LogNormal


class PositiveParams(Spec):
    """A spec with positive-constrained parameters."""

    ell: LogNormal[float]      # Log transform (positive)
    scale: LogNormal[float]    # Log transform (positive)


spec = PositiveParams(
    ell=LogNormal(value=2.0, loc=0.0, scale=1.0),
    scale=LogNormal(value=0.5, loc=0.0, scale=1.0),
)

model = Model.from_spec(spec)

# Constrained params (what the model uses)
print('Constrained params:')
print(f'  ell = {model.params.ell.value}')
print(f'  scale = {model.params.scale.value}')

# Transform to unconstrained space (for optimization)
unconstrained = model.to_unconstrained()
print('\nUnconstrained params:')
print(f'  ell = {unconstrained.ell.value}')      # log(2.0) ~ 0.693
print(f'  scale = {unconstrained.scale.value}')  # log(0.5) ~ -0.693

# Round-trip back to constrained
model2 = model.from_unconstrained(unconstrained)
print('\nRound-trip constrained params:')
print(f'  ell = {model2.params.ell.value}')
print(f'  scale = {model2.params.scale.value}')

# Verify round-trip
print(
    f'\nRound-trip matches: {jnp.allclose(model.params.ell.value, model2.params.ell.value)}'
)

# Get the transforms dataclass
transforms = spec.get_transforms()
print(f'\nTransforms: {transforms}')
print(f'  ell transform: {transforms.ell.value}')
print(f'  scale transform: {transforms.scale.value}')
