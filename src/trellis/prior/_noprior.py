import jax.numpy as jnp
from beartype import beartype
from beartype.typing import ClassVar, Optional, Tuple, Type
from jax import Array

from .._types import Scalar, T
from ..spec._parameter import Parameter
from ..transform._identity import Identity
from ..transform._transform import Transform
from ..typing._typecheck import typecheck
from ._prior import Prior


@beartype
class NoPrior(Prior[T]):
    """Wrapper for fixed parameters without probabilistic priors.

    Use for parameters that should remain constant during inference.
    NoPrior parameters are:
    - Excluded from flatten_learnable() (not sampled in MCMC)
    - log_prob() returns 0 (no prior contribution)
    - sample() returns the current value unchanged

    Example:
        prior = NoPrior(value=1.0)
        params = prior.init_params()  # params.value = 1.0
        lp = prior.log_prob(params.value, params, prior.init_state())  # 0.0
    """

    value: Parameter[T]
    transform: ClassVar[Type[Transform]] = Identity
    is_fixed: ClassVar[bool] = True

    @typecheck
    def log_prob(
        self,
        value: Array,
        params: 'NoPrior.Params',
        state: 'NoPrior.State',
    ) -> Scalar:
        """Returns zero - no prior contribution."""
        return jnp.zeros(())

    @typecheck
    def sample(
        self,
        rng_key: Array,
        params: 'NoPrior.Params',
        state: 'NoPrior.State',
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Array:
        """Returns the current parameter value broadcast to the requested shape.

        NoPrior has no distribution, so sampling just tiles the value.
        """
        sample_shape = self._sample_shape(params, shape)
        value = jnp.asarray(params.value)
        return jnp.broadcast_to(value, sample_shape)
