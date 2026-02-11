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
from ._types import MuPrior, SigmaPrior


@beartype
class LogNormal(Prior[T]):
    """Log-normal prior with fixed hyperparameters: log(X) ~ Normal(mu, sigma).

    For learnable hyperparameters, use LogNormalLearnable.

    Example:
        prior = LogNormal(value=1.0, mu=0.0, sigma=1.0)
        params = prior.init_params()  # params.value = 1.0
        lp = prior.log_prob(params.value, params, prior.init_state())
    """

    value: Parameter[T]
    loc: float
    scale: float
    transform: ClassVar[Type[Transform]] = Log

    @typecheck
    def log_prob(
        self,
        value: Array,
        params: 'LogNormal.Params',
        state: 'LogNormal.State',
    ) -> Scalar:
        """Log probability density at value in constrained space."""
        log_x = jnp.log(value)
        element_log_prob = (
            -log_x
            - jnp.log(self.scale)
            - 0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * ((log_x - self.loc) / self.scale) ** 2
        )
        return jnp.sum(element_log_prob)

    @typecheck
    def sample(
        self,
        rng_key: Array,
        params: 'LogNormal.Params',
        state: 'LogNormal.State',
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Array:
        """Sample from prior (returns constrained value)."""
        z = jr.normal(rng_key, shape=self._sample_shape(params, shape))
        return jnp.exp(self.loc + self.scale * z)


@beartype
class LogNormalLearnable(Prior[T], Generic[T, MuPrior, SigmaPrior]):
    """Log-normal prior with learnable hyperparameters: log(X) ~ Normal(mu, sigma).

    Generic over T (output type), MuPrior (prior on mu), SigmaPrior (prior on sigma).

    MuPrior and SigmaPrior must be Prior subtypes (nested Prior Specs).

    Example:
        prior = LogNormalLearnable[float, Normal[float], HalfNormal[float]](
            value=1.0,
            mu=Normal(value=0.0, loc=0.0, scale=10.0),
            sigma=HalfNormal(value=1.0, scale=1.0),
        )
        params = prior.init_params()
        # params.value = 1.0
        # params.mu.value = 0.0
        # params.sigma.value = 1.0
    """

    value: Parameter[T]
    loc: MuPrior
    scale: SigmaPrior
    transform: ClassVar[Type[Transform]] = Log

    @typecheck
    def log_prob(
        self,
        value: Array,
        params: 'LogNormalLearnable.Params',
        state: 'LogNormalLearnable.State',
    ) -> Scalar:
        """Log probability density at value in constrained space."""
        loc_val = params.loc.value
        scale_val = params.scale.value

        log_x = jnp.log(value)
        element_log_prob = (
            -log_x
            - jnp.log(scale_val)
            - 0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * ((log_x - loc_val) / scale_val) ** 2
        )
        return jnp.sum(element_log_prob)

    @typecheck
    def sample(
        self,
        rng_key: Array,
        params: 'LogNormalLearnable.Params',
        state: 'LogNormalLearnable.State',
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Array:
        """Sample from prior (returns constrained value)."""
        z = jr.normal(rng_key, shape=self._sample_shape(params, shape))
        return jnp.exp(params.loc.value + params.scale.value * z)
