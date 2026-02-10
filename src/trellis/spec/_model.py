from __future__ import annotations

from dataclasses import dataclass, fields, replace

import jax
import jax.numpy as jnp
from beartype.typing import Any, Generic

from .._types import Scalar
from ..prior._prior import Prior
from ..transform._transform import Transform
from ._params_structure import ParamsStructure
from ._spec import Spec
from ._types import CS, P, S, Tr


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Model(Generic[P, S, CS, Tr]):
    """
    Immutable container binding a Spec with typed params and state.
    """

    spec: Spec[P, S, CS, Tr]
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

        prior_lp = self._eval_priors(self.spec, constrained_params)

        jacobian = self._log_det_jacobian(unconstrained_params, transforms)

        return prior_lp + jacobian

    def _eval_priors(self, spec: 'Spec', params: Any) -> Scalar:
        """Recursively evaluate all priors in the Spec tree.

        Walks the Spec tree and typed params in parallel. When a field
        is a Prior (which is a Spec), evaluates its log_prob and then
        recurses into its own fields to evaluate nested priors (hyperpriors).

        Args:
            spec: Current Spec node in the tree
            params: Corresponding typed Params dataclass

        Returns:
            Sum of all log prior probabilities
        """
        total = jnp.zeros(())

        for f in fields(params):
            name = f.name
            nested_spec = getattr(spec, name, None)
            nested_params = getattr(params, name, None)

            if nested_params is None:
                continue

            # Handle list[Spec] - both spec and params are tuples
            if isinstance(nested_spec, tuple) and isinstance(
                nested_params, tuple
            ):
                for spec_item, params_item in zip(nested_spec, nested_params):
                    if isinstance(spec_item, Prior):
                        value = jnp.asarray(params_item.value)
                        state = spec_item.init_state()
                        total = total + spec_item.log_prob(
                            value, params_item, state
                        )
                        total = total + self._eval_priors(
                            spec_item, params_item
                        )
                    elif isinstance(spec_item, Spec):
                        total = total + self._eval_priors(
                            spec_item, params_item
                        )
                continue

            if isinstance(nested_spec, Prior):
                # This is a Prior - evaluate its log_prob
                value = jnp.asarray(nested_params.value)
                state = nested_spec.init_state()
                total = total + nested_spec.log_prob(
                    value, nested_params, state
                )
                # Recurse into the Prior's fields for nested priors (hyperpriors)
                total = total + self._eval_priors(nested_spec, nested_params)

            elif isinstance(nested_spec, Spec):
                # Non-Prior Spec (e.g., Kernel) - just recurse
                total = total + self._eval_priors(nested_spec, nested_params)

        return total

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

    def __call__(self, *args, **kwargs) -> Any:
        if not callable(self.spec):
            raise TypeError(f'{type(self.spec).__name__} is not callable')
        return self.spec(self.params, self.state, *args, **kwargs)

    def tree_flatten(self):
        return self.params, self.state

    @classmethod
    def tree_unflatten(cls, spec, children):
        params, state = children
        return cls(spec=spec, params=params, state=state)
