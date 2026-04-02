from abc import ABC, abstractmethod
from dataclasses import fields as dataclass_fields

from beartype.typing import Any, ClassVar, Generic, Optional, Tuple, Type
from jax import Array

from .._types import Scalar, T
from ..spec._parameter import Parameter
from ..spec._spec import Spec
from ..transform._identity import Identity
from ..transform._transform import Transform


class Prior(Spec, ABC, Generic[T]):
    """Prior distribution that wraps a parameter value.

    Every Prior contains:
    - value: Parameter[T] - the actual parameter value (terminal leaf)
    - transform: ClassVar - the transform to apply (e.g., Log for positive values)
    - is_fixed: ClassVar[bool] - if True, excluded from flatten_learnable()

    Prior extends Spec, so hyperparameters can be:
    - Fixed (plain types like float) - NOT in params tree, accessed via self.x
    - Learnable (nested Prior Specs) - IN params tree, accessed via params.x.value

    Patterns:

    1. Fixed hyperparameters:
        class LogNormal(Prior[T]):
            value: Parameter[T]  # The wrapped parameter
            mu: float            # Fixed, not in tree
            sigma: float         # Fixed, not in tree
            transform: ClassVar[Type[Transform]] = Log

    2. Learnable hyperparameters:
        class LogNormalLearnable(Prior[T], Generic[T, MuPrior, SigmaPrior]):
            value: Parameter[T]  # The wrapped parameter
            mu: MuPrior          # Nested Prior, in tree
            sigma: SigmaPrior    # Nested Prior, in tree

    Access pattern:
        params.lengthscale.value  # The parameter value
        params.lengthscale.mu.value  # Nested hyperparameter (if learnable)
    """

    # All Priors must have a value field - subclasses redeclare it
    value: Parameter[T]

    # Default transform - subclasses override for constrained values
    transform: ClassVar[Type[Transform]] = Identity

    # If True, excluded from flatten_learnable() (default: learnable)
    is_fixed: ClassVar[bool] = False

    def __init__(self, *, transform: Optional[Type[Transform]] = None, **kwargs):
        """Initialize Prior with optional transform override.

        Args:
            transform: Optional transform to use instead of class default.
            **kwargs: Arguments for the Prior fields (value, hyperparameters).
        """
        super().__init__(**kwargs)
        self._transform_override = transform

    @abstractmethod
    def log_prob(
        self,
        value: Array,
        params: 'Prior.Params',
        state: 'Prior.State',
    ) -> Scalar:
        """Log probability density at value (in constrained space).

        Args:
            value: The value to evaluate (typically params.value)
            params: Prior's parameters (contains value and any learnable hyperparams)
            state: Prior's state (derived values)

        Returns:
            Log probability (sum of element-wise log probs for arrays)
        """
        ...

    @staticmethod
    def _sample_shape(
        params: 'Prior.Params',
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[int, ...]:
        """Compute sample shape from params and optional batch shape.

        Args:
            params: Prior's parameters (uses params.value.shape)
            shape: Optional batch dimensions to prepend

        Returns:
            Combined shape: (batch_dims..., param_dims...)
        """
        import jax.numpy as jnp

        param_shape = jnp.asarray(params.value).shape
        return (shape or ()) + param_shape

    @abstractmethod
    def sample(
        self,
        rng_key: Array,
        params: 'Prior.Params',
        state: 'Prior.State',
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Array:
        """Sample from prior (returns constrained value).

        Args:
            rng_key: JAX random key
            params: Prior's parameters (learnable hyperparameters)
            state: Prior's state (derived hyperparameters)
            shape: Output shape (default: scalar)

        Returns:
            Sample from the prior distribution
        """
        ...

    def _build_transforms(self) -> Any:
        """Override to use Prior's transform for the value field.

        Uses instance-level transform override if provided, otherwise
        falls back to the class-level transform ClassVar.
        Other fields (nested Priors) get their own transforms recursively.
        """
        transforms_cls = getattr(self.__class__, 'Transforms')
        values = {}

        for field in dataclass_fields(transforms_cls):
            name = field.name

            if name == 'value':
                # Use instance override if provided, otherwise class default
                transform_cls = self._transform_override or self.__class__.transform
                if isinstance(transform_cls, type):
                    values[name] = transform_cls()
                else:
                    values[name] = transform_cls
            else:
                # For other fields, check if they're nested Specs or list[Spec]
                nested = getattr(self, name, None)
                if isinstance(nested, tuple):
                    # list[Spec] stored as tuple
                    values[name] = tuple(
                        item._build_transforms() if isinstance(item, Spec) else Identity()
                        for item in nested
                    )
                elif isinstance(nested, Spec):
                    values[name] = nested._build_transforms()
                else:
                    values[name] = Identity()

        return transforms_cls(**values)
