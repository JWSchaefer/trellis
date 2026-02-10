"""State validation utilities."""

import inspect
from dataclasses import fields as dataclass_fields
from functools import wraps

from beartype.roar import BeartypeCallHintViolation
from beartype.typing import Callable, TypeVar, Union
from jaxtyping import TypeCheckError

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

    try:
        return checked_cls(**kwargs)
    except BeartypeCallHintViolation as e:
        raise TypeCheckError(str(e)) from e


def _resolve_spec_cls(
    spec_cls: Union[type, str],
    func: Callable,
    args: tuple,
) -> type:
    """Resolve spec_cls to an actual type, handling forward references."""
    if isinstance(spec_cls, type):
        return spec_cls

    # String forward reference - resolve it
    class_name = spec_cls

    # For methods, check if self's class matches the name
    if args:
        self_obj = args[0]
        if hasattr(self_obj, '__class__'):
            # Check self's class and its MRO
            for cls in type(self_obj).__mro__:
                if cls.__name__ == class_name:
                    return cls

    # Try to resolve from the function's globals
    func_globals = getattr(func, '__globals__', None)
    if func_globals is not None and class_name in func_globals:
        return func_globals[class_name]

    raise ValueError(
        f"Could not resolve class '{class_name}'. "
        f"Ensure the class is defined or use the actual class instead of a string."
    )


def _find_state_arg(
    func: Callable,
    args: tuple,
    kwargs: dict,
) -> tuple[object, int | None, str | None]:
    """Find the state argument in args/kwargs.

    Returns:
        (state_value, arg_index, kwarg_name)
        - arg_index is set if state was found in positional args
        - kwarg_name is set if state was found in kwargs
    """
    # Check kwargs first
    if 'state' in kwargs:
        return kwargs['state'], None, 'state'

    # Get parameter names from function signature
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    # Find 'state' parameter position
    if 'state' in param_names:
        state_idx = param_names.index('state')
        if state_idx < len(args):
            return args[state_idx], state_idx, None

    # Fallback: assume first arg (for standalone functions)
    if args:
        return args[0], 0, None

    raise ValueError("No state argument found")


def requires(
    spec_cls: Union[type, str], *field_names: str
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Decorator that validates required state fields before function execution.

    Args:
        spec_cls: The Spec class whose State is being validated.
            Can be the class itself or a string name (for forward references
            when decorating methods inside the class definition).
        *field_names: Names of state fields that must not be None

    Returns:
        Decorator that validates state and converts State -> CheckedState

    Example:
        # With class reference (standalone functions or after class is defined)
        @requires(GPModel, 'cholesky', 'alpha')
        def predict(state: GPModel.CheckedState, params: GPModel.Params, x: Array):
            return state.cholesky @ state.alpha

        # With string forward reference (methods inside class definition)
        class MyModel(Spec):
            cached: State[Array]

            @requires('MyModel', 'cached')
            def compute(self, params, state: 'MyModel.CheckedState'):
                return state.cached * 2

    Raises:
        StateValidationError: If any required field is None
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> R:
            resolved_cls = _resolve_spec_cls(spec_cls, func, args)
            state, arg_idx, kwarg_name = _find_state_arg(func, args, kwargs)

            if state is None:
                raise ValueError("State argument is None")

            checked_state = _validate_and_convert(resolved_cls, state, field_names)

            if kwarg_name is not None:
                kwargs[kwarg_name] = checked_state
                return func(*args, **kwargs)
            elif arg_idx is not None:
                new_args = list(args)
                new_args[arg_idx] = checked_state
                return func(*new_args, **kwargs)
            else:
                raise ValueError("Could not determine how to pass checked state")

        return wrapper

    return decorator
