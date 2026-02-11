import jax

from trellis import Model, Spec
from trellis.prior import LogNormal


class MyModel(Spec):
    alpha: LogNormal[float]
    beta: LogNormal[float]


model_spec = MyModel(
    alpha=LogNormal(value=1.0, loc=0.0, scale=1.0),
    beta=LogNormal(value=2.0, loc=0.0, scale=1.0),
)

model = Model.from_spec(model_spec)
params = model.params

# Tree leaves - all learnable values
leaves = jax.tree.leaves(params)
print(f'Leaves: {leaves}')

# Tree structure
structure = jax.tree.structure(params)
print(f'Structure: {structure}\n')

# Tree map - apply function to all leaves
doubled = jax.tree.map(lambda x: x * 2, params)
print('Doubled params:')
print(f'  alpha.value: {doubled.alpha.value}')
print(f'  beta.value: {doubled.beta.value}\n')

# Flatten/unflatten for optimizers
flat, treedef = jax.tree.flatten(params)
print(f'Flattened: {flat}')
print(f'Treedef: {treedef}')

# Reconstruct from flat
reconstructed = jax.tree.unflatten(treedef, flat)
print(f'\nReconstructed: {reconstructed}\n')

# Model's flatten_params for 1D array (optimizer-friendly)
flat_array, structure = model.flatten_params()
print('Model.flatten_params:')
print(f'  flat_array: {flat_array}')
print(f'  structure: {structure}')

# Unflatten back to model
model2 = model.unflatten_params(flat_array, structure)
print(f'\nUnflattened model params: {model2.params}')
