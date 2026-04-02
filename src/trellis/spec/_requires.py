"""State validation utilities."""

import inspect
from dataclasses import Field
from dataclasses import fields as dataclass_fields
from functools import wraps
from typing import ClassVar, NamedTuple, Protocol, runtime_checkable

from beartype import beartype
from beartype.roar import BeartypeCallHintViolation
from beartype.typing import Callable, TypeVar, Union
from jaxtyping import TypeCheckError, jaxtyped

from .._config import config

R = TypeVar('R')


@runtime_checkable
class DataclassInstance(Protocol):
    """Protocol for dataclass instances."""

    __dataclass_fields__: ClassVar[dict[str, Field]]


class StateRequirement(NamedTuple):
    """Specification for validating a single state parameter."""

    param_name: str
    spec_cls: Union[type, str]
    field_names: tuple[str, ...]


class StateValidationError(ValueError):
    """Raised when required state fields are None."""

    def __init__(
        self,
        spec_name: str,
        missing_fields: list[str],
        param_name: str = 'state',
    ):
        self.spec_name = spec_name
        self.missing_fields = missing_fields
        self.param_name = param_name
        fields_str = ', '.join(f"'{f}'" for f in missing_fields)
        super().__init__(
            f"{spec_name}.State validation failed for parameter '{param_name}': "
            f'required fields {fields_str} are None'
        )


@jaxtyped(typechecker=beartype)
def _create_checked_state(checked_cls: type, **kwargs):
    """Create CheckedState in isolated jaxtyping context.

    This runs in its own dimension memo, so state dimensions
    don't conflict with input dimensions from the outer function.
    """
    return checked_cls(**kwargs)


def _validate_and_convert(
    spec_cls: type,
    state: DataclassInstance,
    required_fields: tuple[str, ...],
    param_name: str = 'state',
) -> DataclassInstance:
    """Validate state fields and convert to CheckedState."""
    checked_cls = getattr(spec_cls, 'CheckedState')
    checked_fields = {f.name for f in dataclass_fields(checked_cls)}

    # Verify required fields exist on the CheckedState class
    invalid_fields = [f for f in required_fields if f not in checked_fields]
    if invalid_fields:

        fields_str = ', '.join(f"'{f}'" for f in invalid_fields)
        raise ValueError(
            f"@requires references fields {fields_str} that don't exist on "
            f"{spec_cls.__name__}.CheckedState. Check that you're using the "
            f'correct Spec class (got {spec_cls.__name__}, input state is '
            f'{type(state).__name__}).'
        )

    missing = [f for f in required_fields if getattr(state, f, None) is None]
    if missing:
        raise StateValidationError(spec_cls.__name__, missing, param_name)

    kwargs = {
        f.name: getattr(state, f.name)
        for f in dataclass_fields(type(state))
        if f.name in checked_fields
    }

    try:
        return _create_checked_state(checked_cls, **kwargs)
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

    class_name = spec_cls

    # For methods, check if self's class matches the name
    if args:
        self_obj = args[0]
        if hasattr(self_obj, '__class__'):
            for cls in type(self_obj).__mro__:
                if cls.__name__ == class_name:
                    return cls

    # Try to resolve from the function's globals
    func_globals = getattr(func, '__globals__', None)
    if func_globals is not None and class_name in func_globals:
        return func_globals[class_name]

    raise ValueError(
        f"Could not resolve class '{class_name}'. "
        f'Ensure the class is defined or use the actual class instead.'
    )


def _find_param_arg(
    func: Callable,
    args: tuple,
    kwargs: dict,
    param_name: str,
) -> tuple[object, int | None, str | None]:
    """Find a named parameter in args/kwargs.

    Returns:
        (value, arg_index, kwarg_name)
    """
    if param_name in kwargs:
        return kwargs[param_name], None, param_name

    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    if param_name in param_names:
        param_idx = param_names.index(param_name)
        if param_idx < len(args):
            return args[param_idx], param_idx, None

    raise ValueError(
        f"Parameter '{param_name}' not found in function arguments"
    )


def requires(
    **state_requirements: tuple,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Decorator that validates required state fields before function execution.

    Args:
        **state_requirements: Mapping from parameter name to
            (SpecClass, 'field1', 'field2', ...) tuples. SpecClass can be
            the class itself or a string name (for forward references).

    Returns:
        Decorator that validates state(s) and converts State -> CheckedState

    Example:
        # Single state parameter
        @requires(state=(GPModel, 'cholesky', 'alpha'))
        def predict(state: GPModel.CheckedState, params: GPModel.Params, x: Array):
            return state.cholesky @ state.alpha

        # String forward reference (methods inside class definition)
        class MyModel(Spec):
            cached: State[Array]

            @requires(state=('MyModel', 'cached'))
            def compute(self, params, state: 'MyModel.CheckedState'):
                return state.cached * 2

        # Multiple state parameters
        @requires(
            state=('MFKernel',),
            coupling_state=('LinearARCoupling', 'w_m', 'w_k'),
        )
        def __call__(self, params, state, coupling_params, coupling_state, x, ell):
            ...

    Raises:
        StateValidationError: If any required field is None
    """
    if not state_requirements:
        raise ValueError(
            'requires() needs at least one state requirement.\n'
            "Example: @requires(state=('SpecClass', 'field1', 'field2'))"
        )

    # Parse requirements at decoration time
    requirements = []
    for param_name, spec_tuple in state_requirements.items():
        if not isinstance(spec_tuple, tuple) or len(spec_tuple) < 1:
            raise ValueError(
                f"Invalid requirement for '{param_name}': "
                f"expected tuple of (SpecClass, 'field1', ...), got {spec_tuple}"
            )
        spec_cls = spec_tuple[0]
        field_names = spec_tuple[1:]
        requirements.append(
            StateRequirement(param_name, spec_cls, field_names)
        )

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> R:
            if not config.get('runtime_state_validation', True):
                return func(*args, **kwargs)

            new_args = list(args)
            new_kwargs = dict(kwargs)

            for req in requirements:
                resolved_cls = _resolve_spec_cls(req.spec_cls, func, args)
                state, arg_idx, kwarg_name = _find_param_arg(
                    func, tuple(new_args), new_kwargs, req.param_name
                )

                if state is None:
                    raise ValueError(f"Parameter '{req.param_name}' is None")

                assert isinstance(state, DataclassInstance)
                checked_state = _validate_and_convert(
                    resolved_cls, state, req.field_names, req.param_name
                )

                if kwarg_name is not None:
                    new_kwargs[kwarg_name] = checked_state
                elif arg_idx is not None:
                    new_args[arg_idx] = checked_state
                else:
                    raise ValueError(
                        f'Could not determine how to pass checked state '
                        f"for parameter '{req.param_name}'"
                    )

            return func(*new_args, **new_kwargs)

        return wrapper

    return decorator
