import jax.numpy as jnp
from jax import Array

from .._types import Scalar, T
from ._transform import Transform


class Identity(Transform[T]):
    """Identity transform for unconstrained parameters."""

    def forward(self, unconstrained: T) -> Array:
        return jnp.asarray(unconstrained)

    def inverse(self, constrained: T) -> Array:
        return jnp.asarray(constrained)

    def log_det_jacobian(self, unconstrained: T) -> Scalar:
        return jnp.zeros(())
