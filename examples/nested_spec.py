"""Example: Nested Spec structure."""

import jax

from trellis import Model, Spec
from trellis.prior import LogNormal


# A spec with nested priors on its parameters
class MyModel(Spec):
    alpha: LogNormal[float]
    beta: LogNormal[float]


# Instantiate with concrete prior values
model_spec = MyModel(
    alpha=LogNormal(value=1.0, loc=0.0, scale=1.0),
    beta=LogNormal(value=2.0, loc=0.0, scale=0.5),
)

# The params tree mirrors the spec structure
params = model_spec.init_params()
print('Params structure:')
print(f'  params.alpha.value = {params.alpha.value}')
print(f'  params.beta.value = {params.beta.value}')

# Tree leaves are all the learnable values
leaves = jax.tree.leaves(params)
print(f'\nTree leaves: {leaves}')
print(f'Number of leaves: {len(leaves)}')

# Create a Model for the full workflow
model = Model.from_spec(model_spec)
print(f'\nModel.params: {model.params}')
