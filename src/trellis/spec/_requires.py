"""State validation utilities."""

from dataclasses import fields as dataclass_fields
from functools import wraps

from beartype.typing import Callable, TypeVar

R = TypeVar('R')


class StateValidationError(ValueError):
    """Raised when required state fields are None."""

    def __init__(self, spec_name: str, missing_fields: list[str]):
        self.spec_name = spec_name
        self.missing_fields = missing_fields
        fields_str = ', '.join(f"'{f}'" for f in missing_fields)
        super().__init__(
            f"{spec_name}.State validation failed: "
            f"required fields {fields_str} are None"
        )


def _validate_and_convert(
    spec_cls: type,
    state: object,
    required_fields: tuple[str, ...],
) -> object:
    """Validate state fields and convert to CheckedState."""
    checked_cls = getattr(spec_cls, 'CheckedState')

    missing = [f for f in required_fields if getattr(state, f, None) is None]
    if missing:
        raise StateValidationError(spec_cls.__name__, missing)

    checked_fields = {f.name for f in dataclass_fields(checked_cls)}
    kwargs = {
        f.name: getattr(state, f.name)
        for f in dataclass_fields(type(state))
        if f.name in checked_fields
    }

    return checked_cls(**kwargs)


def requires(
    spec_cls: type, *field_names: str
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Decorator that validates required state fields before function execution.

    Args:
        spec_cls: The Spec class whose State is being validated
        *field_names: Names of state fields that must not be None

    Returns:
        Decorator that validates state and converts State -> CheckedState

    Example:
        @requires(GPModel, 'cholesky', 'alpha')
        def predict(state: GPModel.CheckedState, params: GPModel.Params, x: Array):
            return state.cholesky @ state.alpha

    Raises:
        StateValidationError: If any required field is None
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> R:
            state = kwargs.get('state', args[0] if args else None)
            if state is None:
                raise ValueError("No state argument found")

            checked_state = _validate_and_convert(spec_cls, state, field_names)

            if 'state' in kwargs:
                kwargs['state'] = checked_state
                return func(*args, **kwargs)
            else:
                return func(checked_state, *args[1:], **kwargs)

        return wrapper

    return decorator
