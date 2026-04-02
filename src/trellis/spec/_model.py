from __future__ import annotations

from dataclasses import dataclass, fields, replace

import jax
import jax.numpy as jnp
from beartype.typing import Any, Generic

from .._types import Scalar
from ..transform._transform import Transform
from ._params_structure import LearnableParamsStructure, ParamsStructure
from ._spec import Spec
from ._types import CS, P, S, Tr


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Model(Generic[P, S, CS, Tr]):
    """
    Immutable container binding a Spec with typed params and state.

    Model provides parameter infrastructure:
    - flatten_params / unflatten_params: For external optimizers
    - to_unconstrained / from_unconstrained: Transform to/from optimization space
    - log_prior: Evaluate prior densities with Jacobian correction
    - replace_params / replace_state: Immutable updates

    Spec behavior (fit, __call__, domain methods) should be called explicitly:
        state = model.spec.fit(model.params, model.state, x, y)
        model = model.replace_state(state)
        preds = model.spec(model.params, model.state, x_test)
    """

    spec: Tr
    params: P
    state: S

    @classmethod
    def from_spec(cls, spec: Spec[P, S, CS, Tr]) -> Model[P, S, CS, Tr]:
        """Create Model from an instantiated Spec."""
        return cls(
            spec=spec,
            params=spec.init_params(),
            state=spec.init_state(),
        )

    def replace_params(self, params: P) -> Model[P, S, CS, Tr]:
        """Return new Model with updated params."""
        return replace(self, params=params)

    def replace_state(self, state: S) -> Model[P, S, CS, Tr]:
        """Return new Model with updated state."""
        return replace(self, state=state)

    def flatten_params(self) -> tuple[jnp.ndarray, ParamsStructure]:
        """Flatten params to a 1D array for use with external optimizers.

        Returns:
            flat: 1D array containing all parameter values.
            structure: Structure information needed for unflattening.

        """
        leaves, treedef = jax.tree_util.tree_flatten(self.params)

        arrays = []
        shapes = []
        dtypes = []

        for leaf in leaves:
            arr = jnp.atleast_1d(jnp.asarray(leaf))
            arrays.append(arr.ravel())
            shapes.append(arr.shape)
            dtypes.append(arr.dtype)

        flat = jnp.concatenate(arrays) if arrays else jnp.array([])

        structure = ParamsStructure(
            treedef=treedef,
            shapes=tuple(shapes),
            dtypes=tuple(dtypes),
        )

        return flat, structure

    def unflatten_params(
        self, flat: jnp.ndarray, structure: ParamsStructure
    ) -> Model[P, S, CS, Tr]:
        """Reconstruct Model with params from a 1D array.

        Args:
            flat: 1D array of parameter values.
            structure: Structure information from flatten_params.

        Returns:
            New Model with reconstructed params.
        """
        leaves = []
        offset = 0

        for shape, dtype in zip(structure.shapes, structure.dtypes):
            size = structure._prod(shape)
            leaf_flat = flat[offset : offset + size]
            leaf = leaf_flat.reshape(shape).astype(dtype)

            if shape == (1,):
                leaf = leaf[0]

            leaves.append(leaf)
            offset += size

        params = jax.tree_util.tree_unflatten(structure.treedef, leaves)
        return self.replace_params(params)

    def flatten_learnable(self) -> tuple[jnp.ndarray, LearnableParamsStructure]:
        """Flatten learnable params only, excluding fixed (NoPrior) params.

        Fixed params are stored in the structure for restoration during
        unflatten_learnable(). Use this for MCMC/optimization where fixed
        params should not receive gradients.

        Returns:
            flat: 1D array containing only learnable parameter values.
            structure: LearnableParamsStructure with fixed values for restoration.
        """
        from ..prior._noprior import NoPrior

        noprior_params_name = NoPrior.Params.__name__

        def is_fixed_params(x: Any) -> bool:
            return type(x).__name__ == noprior_params_name

        # Flatten with NoPriorParams treated as opaque leaves
        leaves, treedef = jax.tree_util.tree_flatten(
            self.params, is_leaf=is_fixed_params
        )

        learnable_arrays = []
        learnable_shapes = []
        learnable_dtypes = []
        fixed_indices = []
        fixed_values = []

        for i, leaf in enumerate(leaves):
            if is_fixed_params(leaf):
                fixed_indices.append(i)
                fixed_values.append(leaf)
            else:
                arr = jnp.atleast_1d(jnp.asarray(leaf))
                learnable_arrays.append(arr.ravel())
                learnable_shapes.append(arr.shape)
                learnable_dtypes.append(arr.dtype)

        flat = (
            jnp.concatenate(learnable_arrays)
            if learnable_arrays
            else jnp.array([])
        )

        structure = LearnableParamsStructure(
            treedef=treedef,
            shapes=tuple(learnable_shapes),
            dtypes=tuple(learnable_dtypes),
            fixed_indices=tuple(fixed_indices),
            fixed_values=tuple(fixed_values),
        )

        return flat, structure

    def unflatten_learnable(
        self, flat: jnp.ndarray, structure: LearnableParamsStructure
    ) -> 'Model[P, S, CS, Tr]':
        """Reconstruct Model with params from learnable-only flat array.

        Fixed values are restored from the structure.

        Args:
            flat: 1D array of learnable parameter values.
            structure: LearnableParamsStructure from flatten_learnable.

        Returns:
            New Model with reconstructed params (learnable + fixed).
        """
        # Reconstruct learnable leaves
        learnable_leaves = []
        offset = 0

        for shape, dtype in zip(structure.shapes, structure.dtypes):
            size = structure._prod(shape)
            leaf_flat = flat[offset : offset + size]
            leaf = leaf_flat.reshape(shape).astype(dtype)

            if shape == (1,):
                leaf = leaf[0]

            learnable_leaves.append(leaf)
            offset += size

        # Merge learnable and fixed leaves in original order
        n_total = len(learnable_leaves) + len(structure.fixed_indices)
        all_leaves = [None] * n_total

        # Place fixed values at their original positions
        for idx, val in zip(structure.fixed_indices, structure.fixed_values):
            all_leaves[idx] = val

        # Fill remaining positions with learnable values
        learnable_iter = iter(learnable_leaves)
        for i in range(n_total):
            if all_leaves[i] is None:
                all_leaves[i] = next(learnable_iter)

        params = jax.tree_util.tree_unflatten(structure.treedef, all_leaves)
        return self.replace_params(params)

    def get_learnable_transforms(self) -> list:
        """Get flat list of transforms for learnable parameters only.

        Excludes transforms for fixed (NoPrior) parameters, matching the
        structure returned by flatten_learnable().

        Returns:
            List of Transform instances for learnable parameters.
        """
        from ..prior._noprior import NoPrior

        noprior_params_name = NoPrior.Params.__name__

        def is_fixed_params(x: Any) -> bool:
            return type(x).__name__ == noprior_params_name

        # Flatten params and transforms with same is_leaf to ensure alignment
        flat_params, _ = jax.tree_util.tree_flatten(
            self.params, is_leaf=is_fixed_params
        )
        transforms_tree = self.spec.get_transforms()
        flat_transforms, _ = jax.tree_util.tree_flatten(
            transforms_tree, is_leaf=is_fixed_params
        )

        # Return only transforms where the corresponding param is not fixed
        return [
            t for t, p in zip(flat_transforms, flat_params)
            if not is_fixed_params(p)
        ]

    def to_unconstrained(self) -> P:
        """Transform current params to unconstrained space."""
        transforms = self.spec.get_transforms()
        return self._apply_transforms(self.params, transforms, inverse=True)

    def from_unconstrained(self, raw_params: P) -> Model[P, S, CS, Tr]:
        """Return new Model with params transformed from unconstrained space."""
        transforms = self.spec.get_transforms()
        constrained = self._apply_transforms(
            raw_params, transforms, inverse=False
        )
        return self.replace_params(constrained)

    def _apply_transforms(
        self, params: Any, transforms: Any, inverse: bool
    ) -> Any:
        """Apply transforms to params.

        Args:
            params: Params dataclass instance
            transforms: Transforms dataclass instance
            inverse: If True, apply inverse transform (constrained -> unconstrained)

        Returns:
            New Params dataclass with transformed values
        """
        result = {}

        for f in fields(params):
            name = f.name
            value = getattr(params, name)
            transform = getattr(transforms, name)

            if isinstance(transform, Transform):
                if inverse:
                    result[name] = transform.inverse(value)
                else:
                    result[name] = transform.forward(value)
            elif isinstance(value, tuple) and isinstance(transform, tuple):
                # Handle list[Spec] - parallel iteration over tuples
                result[name] = tuple(
                    self._apply_transforms(v, t, inverse)
                    for v, t in zip(value, transform)
                )
            else:
                result[name] = self._apply_transforms(
                    value, transform, inverse
                )

        return type(params)(**result)

    def log_prior(self, unconstrained_params: P) -> Scalar:
        """
        Evaluate log prior with Jacobian correction.

        Traverses the Spec tree to find all Prior instances and
        evaluates their log_prob on constrained parameter values.
        """
        transforms = self.spec.get_transforms()
        constrained_params = self._apply_transforms(
            unconstrained_params, transforms, inverse=False
        )

        prior_lp = self.spec.eval_priors(constrained_params)

        jacobian = self._log_det_jacobian(unconstrained_params, transforms)

        return prior_lp + jacobian

    def _log_det_jacobian(
        self, unconstrained_params: Any, transforms: Any
    ) -> Scalar:
        """Sum of log|det J| for all transforms.

        Args:
            unconstrained_params: Params dataclass of unconstrained values
            transforms: Transforms dataclass instance

        Returns:
            Sum of log determinant of Jacobians
        """
        total = jnp.zeros(())

        for f in fields(transforms):
            name = f.name
            value = getattr(unconstrained_params, name)
            transform = getattr(transforms, name)

            if isinstance(transform, Transform):
                total = total + transform.log_det_jacobian(value)
            elif isinstance(value, tuple) and isinstance(transform, tuple):
                # Handle list[Spec]
                for v, t in zip(value, transform):
                    total = total + self._log_det_jacobian(v, t)
            else:
                total = total + self._log_det_jacobian(value, transform)

        return total

    def tree_flatten(self):
        return self.params, self.state

    @classmethod
    def tree_unflatten(cls, spec, children):
        params, state = children
        return cls(spec=spec, params=params, state=state)
