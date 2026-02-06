from abc import ABC, abstractmethod

from beartype.typing import Generic
from jax import Array

from .._types import Scalar, T


class Transform(ABC, Generic[T]):
    """Bijective mapping between constrained and unconstrained spaces.

    - forward: unconstrained (R) -> constrained (e.g., R+)
    - inverse: constrained -> unconstrained
    - log_det_jacobian: for MCMC correction when sampling in unconstrained space
    """

    @abstractmethod
    def forward(self, unconstrained: T) -> Array:
        """Map from unconstrained to constrained space."""
        ...

    @abstractmethod
    def inverse(self, constrained: T) -> Array:
        """Map from constrained to unconstrained space."""
        ...

    @abstractmethod
    def log_det_jacobian(self, unconstrained: T) -> Scalar:
        """Log determinant of Jacobian |d(forward)/d(unconstrained)|."""
        ...
