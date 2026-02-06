from __future__ import annotations

from beartype.typing import Generic

from .._types import T


class Parameter(Generic[T]):
    """Pure type marker for terminal leaf parameters.

    Parameters are the terminal leaves of a Spec tree - they hold
    the actual learnable values. Parameters should not contain
    hidden structure (priors, transforms, etc.).

    Use in type hints to declare leaf parameters:
        class MyPrior(Prior[float]):
            value: Parameter[float]  # Terminal leaf

    Priors and transforms are now specified at the Prior level,
    not on Parameters.
    """

    pass
