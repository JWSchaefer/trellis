import jax.numpy as jnp
from beartype.typing import Generic
from jax import Array

from .._types import Scalar, T
from ._transform import Transform


class Log(Transform[T], Generic[T]):
    """
    forward: x -> exp(x)
    inverse: y -> log(y)
    """

    def forward(self, unconstrained: T) -> Array:
        return jnp.exp(unconstrained)

    def inverse(self, constrained: T) -> Array:
        return jnp.log(constrained)

    def log_det_jacobian(self, unconstrained: T) -> Scalar:
        # d/dx(exp(x)) = exp(x), so log|J| = log(exp(x)) = x
        return jnp.sum(unconstrained)
