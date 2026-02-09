import jax.numpy as jnp
import jax.random as jr
from beartype import beartype
from beartype.typing import ClassVar, Generic, Optional, Tuple, Type
from jax import Array

from .._types import Scalar, T
from ..spec._parameter import Parameter
from ..transform._identity import Identity
from ..transform._transform import Transform
from ..typing._typecheck import typecheck
from ._prior import Prior
from ._types import LocPrior, ScalePrior


@beartype
class Normal(Prior[T]):
    """Normal prior with fixed hyperparameters: X ~ Normal(loc, scale).

    For learnable hyperparameters, use NormalLearnable.

    Example:
        prior = Normal(value=0.0, loc=0.0, scale=1.0)
        params = prior.init_params()  # params.value = 0.0
        lp = prior.log_prob(params.value, params, prior.init_state())
    """

    value: Parameter[T]
    loc: float
    scale: float
    transform: ClassVar[Type[Transform]] = Identity

    @typecheck
    def log_prob(
        self,
        value: Array,
        params: 'Normal.Params',
        state: 'Normal.State',
    ) -> Scalar:
        """Log probability density at value."""
        element_log_prob = (
            -0.5 * jnp.log(2 * jnp.pi)
            - jnp.log(self.scale)
            - 0.5 * ((value - self.loc) / self.scale) ** 2
        )
        return jnp.sum(element_log_prob)

    @typecheck
    def sample(
        self,
        rng_key: Array,
        params: 'Normal.Params',
        state: 'Normal.State',
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Array:
        """Sample from prior."""
        z = jr.normal(rng_key, shape=self._sample_shape(params, shape))
        return self.loc + self.scale * z


@beartype
class NormalLearnable(Prior[T], Generic[T, LocPrior, ScalePrior]):
    """Normal prior with learnable hyperparameters: X ~ Normal(loc, scale).

    Generic over T (output type), LocPrior (prior on loc), ScalePrior (prior on scale).

    LocPrior and ScalePrior must be Prior subtypes (nested Prior Specs).

    Example:
        prior = NormalLearnable[float, Normal[float], HalfNormal[float]](
            value=0.0,
            loc=Normal(value=0.0, loc=0.0, scale=100.0),
            scale=HalfNormal(value=1.0, scale=10.0),
        )
    """

    value: Parameter[T]
    loc: LocPrior
    scale: ScalePrior
    transform: ClassVar[Type[Transform]] = Identity

    @typecheck
    def log_prob(
        self,
        value: Array,
        params: 'NormalLearnable.Params',
        state: 'NormalLearnable.State',
    ) -> Scalar:
        """Log probability density at value."""
        loc_val = params.loc.value
        scale_val = params.scale.value

        element_log_prob = (
            -0.5 * jnp.log(2 * jnp.pi)
            - jnp.log(scale_val)
            - 0.5 * ((value - loc_val) / scale_val) ** 2
        )
        return jnp.sum(element_log_prob)

    @typecheck
    def sample(
        self,
        rng_key: Array,
        params: 'NormalLearnable.Params',
        state: 'NormalLearnable.State',
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Array:
        """Sample from prior."""
        z = jr.normal(rng_key, shape=self._sample_shape(params, shape))
        return params.loc.value + params.scale.value * z
