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
        """Override to use Prior's class-level transform for the value field.

        The transform ClassVar on the Prior class applies to the `value` field.
        Other fields (nested Priors) get their own transforms recursively.
        """
        transforms_cls = getattr(self.__class__, 'Transforms')
        values = {}

        for field in dataclass_fields(transforms_cls):
            name = field.name

            if name == 'value':
                # Use the Prior's class-level transform for value
                transform_cls = self.__class__.transform
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
