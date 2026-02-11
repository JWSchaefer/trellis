"""Example demonstrating list[Spec] support in trellis.

This example shows how to define a Spec with a list of nested Specs,
and how the params/state/transforms are correctly generated as tuples.
"""

import jax

from trellis import Model, Spec
from trellis.prior import LogNormal


class KernelSpec(Spec):
    """A simple kernel specification with a lengthscale parameter."""

    lengthscale: LogNormal[float]


class MultiKernelModel(Spec):
    """A model with multiple kernels specified as a list."""

    kernels: list[KernelSpec]


def main():
    model = MultiKernelModel(
        kernels=[
            KernelSpec(lengthscale=LogNormal(value=1.0, loc=0.0, scale=1.0)),
            KernelSpec(lengthscale=LogNormal(value=2.0, loc=0.0, scale=1.0)),
            KernelSpec(lengthscale=LogNormal(value=0.5, loc=0.0, scale=1.0)),
        ]
    )

    params = model.init_params()
    print('=== Params ===')
    print(f'Type of params.kernels: {type(params.kernels)}')
    print(f'Number of kernels: {len(params.kernels)}')
    for i, kernel_params in enumerate(params.kernels):
        print(
            f'  Kernel {i} lengthscale value: {kernel_params.lengthscale.value}'
        )

    # Test pytree operations
    print('\n=== Pytree Operations ===')
    flat, treedef = jax.tree.flatten(params)
    print(f'Flattened leaves: {flat}')
    reconstructed = jax.tree.unflatten(treedef, flat)
    print(f'Reconstructed equals original: {params == reconstructed}')

    # Test transforms
    print('\n=== Transforms ===')
    transforms = model.get_transforms()
    print(f'Type of transforms.kernels: {type(transforms.kernels)}')
    print(f'Number of transform tuples: {len(transforms.kernels)}')

    # Test Model workflow
    print('\n=== Model Workflow ===')
    full_model = Model.from_spec(model)
    unconstrained = full_model.to_unconstrained()
    print(f'Unconstrained params type: {type(unconstrained.kernels)}')

    # Test log_prior
    log_p = full_model.log_prior(unconstrained)
    print(f'Log prior: {log_p}')

    # Test round-trip
    constrained_model = full_model.from_unconstrained(unconstrained)
    print(
        f'Round-trip successful: {constrained_model.params == full_model.params}'
    )

    # Test empty list
    print('\n=== Empty List ===')
    empty_model = MultiKernelModel(kernels=[])
    empty_params = empty_model.init_params()
    print(f'Empty kernels: {empty_params.kernels}')
    print(f'Empty kernels type: {type(empty_params.kernels)}')

    print('\n=== All tests passed! ===')


if __name__ == '__main__':
    main()
