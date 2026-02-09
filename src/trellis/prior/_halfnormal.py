import jax.numpy as jnp
import jax.random as jr
from beartype import beartype
from beartype.typing import ClassVar, Generic, Optional, Tuple, Type
from jax import Array

from .._types import Scalar, T
from ..spec._parameter import Parameter
from ..transform._log import Log
from ..transform._transform import Transform
from ..typing._typecheck import typecheck
from ._prior import Prior
from ._types import ScalePrior


@beartype
class HalfNormal(Prior[T]):
    """Half-normal prior with fixed hyperparameters: X ~ HalfNormal(scale).

    X = |Z| where Z ~ Normal(0, scale).

    For learnable hyperparameters, use HalfNormalLearnable.

    Example:
        prior = HalfNormal(value=1.0, scale=1.0)
        params = prior.init_params()  # params.value = 1.0
        lp = prior.log_prob(params.value, params, prior.init_state())
    """

    value: Parameter[T]
    scale: float
    transform: ClassVar[Type[Transform]] = Log

    @typecheck
    def log_prob(
        self,
        value: Array,
        params: 'HalfNormal.Params',
        state: 'HalfNormal.State',
    ) -> Scalar:
        """Log probability density at value (must be positive)."""
        element_log_prob = (
            0.5 * jnp.log(2 / jnp.pi)
            - jnp.log(self.scale)
            - 0.5 * (value / self.scale) ** 2
        )
        return jnp.sum(element_log_prob)

    @typecheck
    def sample(
        self,
        rng_key: Array,
        params: 'HalfNormal.Params',
        state: 'HalfNormal.State',
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Array:
        """Sample from prior (returns positive values)."""
        z = jr.normal(rng_key, shape=self._sample_shape(params, shape))
        return self.scale * jnp.abs(z)


@beartype
class HalfNormalLearnable(Prior[T], Generic[T, ScalePrior]):
    """Half-normal prior with learnable hyperparameters: X ~ HalfNormal(scale).

    X = |Z| where Z ~ Normal(0, scale).

    Generic over T (output type) and ScalePrior (prior on scale).

    ScalePrior must be a Prior subtype (nested Prior Spec).

    Example:
        prior = HalfNormalLearnable[float, LogNormal[float]](
            value=1.0,
            scale=LogNormal(value=1.0, mu=0.0, sigma=1.0),
        )
    """

    value: Parameter[T]
    scale: ScalePrior
    transform: ClassVar[Type[Transform]] = Log

    @typecheck
    def log_prob(
        self,
        value: Array,
        params: 'HalfNormalLearnable.Params',
        state: 'HalfNormalLearnable.State',
    ) -> Scalar:
        """Log probability density at value (must be positive)."""
        scale_val = params.scale.value

        element_log_prob = (
            0.5 * jnp.log(2 / jnp.pi)
            - jnp.log(scale_val)
            - 0.5 * (value / scale_val) ** 2
        )
        return jnp.sum(element_log_prob)

    @typecheck
    def sample(
        self,
        rng_key: Array,
        params: 'HalfNormalLearnable.Params',
        state: 'HalfNormalLearnable.State',
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Array:
        """Sample from prior (returns positive values)."""
        z = jr.normal(rng_key, shape=self._sample_shape(params, shape))
        return params.scale.value * jnp.abs(z)
