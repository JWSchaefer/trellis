from abc import ABC, ABCMeta
from dataclasses import fields as dataclass_fields
from dataclasses import make_dataclass

import jax.tree_util
from beartype.typing import (
    Any,
    ClassVar,
    Generic,
    Optional,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)
from jax import Array

from ..transform import Identity
from ..transform._transform import Transform
from ._parameter import Parameter
from ._state import State
from ._types import CS, P, S, Tr


def _is_spec_bound(tv: TypeVar) -> bool:
    """Check if a TypeVar is bounded to Spec (including Prior)."""
    bound = getattr(tv, '__bound__', None)
    if bound is None:
        return False
    bound_origin = get_origin(bound) or bound
    return isinstance(bound_origin, type) and issubclass(bound_origin, Spec)


def _is_parameter_bound(tv: TypeVar) -> bool:
    """Check if a TypeVar is bounded to Parameter."""
    bound = getattr(tv, '__bound__', None)
    if bound is None:
        return False
    bound_origin = get_origin(bound) or bound
    return isinstance(bound_origin, type) and issubclass(
        bound_origin, Parameter
    )


def _is_list_of_specs(ann) -> tuple[bool, type | None]:
    """Check if annotation is list[SomeSpec]. Returns (is_list, inner_spec_type)."""
    origin = get_origin(ann)
    if origin is not list:
        return False, None
    args = get_args(ann)
    if not args:
        return False, None
    inner_type = args[0]
    inner_origin = get_origin(inner_type) or inner_type
    if isinstance(inner_origin, type) and issubclass(inner_origin, Spec):
        return True, inner_origin
    return False, None


class SpecMeta(ABCMeta):
    """Metaclass that generates Params, State, and Transforms classes for Specs."""

    ATTRS = ('Params', 'State', 'CheckedState', 'Transforms')

    _params_cache: dict[type, type] = {}
    _state_cache: dict[type, type] = {}
    _checked_state_cache: dict[type, type] = {}
    _transforms_cache: dict[type, type] = {}

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip base classes (Spec, Prior)
        if name in ('Spec', 'Prior'):
            return cls

        # Generate and attach Params/State/CheckedState/Transforms classes
        setattr(cls, 'Params', mcs._generate_params_class(cls))
        setattr(cls, 'State', mcs._generate_state_class(cls))
        setattr(cls, 'CheckedState', mcs._generate_checked_state_class(cls))
        setattr(cls, 'Transforms', mcs._generate_transforms_class(cls))

        return cls

    @classmethod
    def _generate_params_class(mcs, spec_cls: type) -> type:
        """Generate frozen dataclass for a Spec's parameters."""
        if spec_cls in mcs._params_cache:
            return mcs._params_cache[spec_cls]

        hints = get_type_hints(spec_cls)
        fields = []

        for name, ann in hints.items():
            if name in mcs.ATTRS:
                continue

            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, Parameter):
                inner_type = get_args(ann)[0] if get_args(ann) else Any
                fields.append((name, inner_type))

            elif isinstance(origin, TypeVar):
                if _is_parameter_bound(origin) or _is_spec_bound(origin):
                    fields.append((name, Any))

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_params = mcs._generate_params_class(origin)
                fields.append((name, nested_params))

            elif origin is list:
                is_list, inner_spec = _is_list_of_specs(ann)
                if is_list and inner_spec is not None:
                    mcs._generate_params_class(inner_spec)  # Ensure nested class exists
                    fields.append((name, tuple))

        class_name = f'{spec_cls.__name__}Params'
        ParamsClass = make_dataclass(class_name, fields, frozen=True)

        jax.tree_util.register_dataclass(
            ParamsClass,
            data_fields=[f[0] for f in fields],
            meta_fields=[],
        )

        mcs._params_cache[spec_cls] = ParamsClass
        return ParamsClass

    @classmethod
    def _generate_state_class(mcs, spec_cls: type) -> type:
        """Generate frozen dataclass for a Spec's state."""
        if spec_cls in mcs._state_cache:
            return mcs._state_cache[spec_cls]

        hints = get_type_hints(spec_cls)
        fields = []

        for name, ann in hints.items():
            if name in mcs.ATTRS:
                continue

            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, State):
                inner_type = get_args(ann)[0] if get_args(ann) else Any
                fields.append((name, Optional[inner_type], None))

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_state = mcs._generate_state_class(origin)
                if dataclass_fields(nested_state):
                    fields.append((name, nested_state))

            elif origin is list:
                is_list, inner_spec = _is_list_of_specs(ann)
                if is_list and inner_spec is not None:
                    nested_state = mcs._generate_state_class(inner_spec)
                    if dataclass_fields(nested_state):
                        fields.append((name, tuple))

        class_name = f'{spec_cls.__name__}State'
        StateClass = make_dataclass(class_name, fields, frozen=True)

        jax.tree_util.register_dataclass(
            StateClass,
            data_fields=[f[0] for f in fields],
            meta_fields=[],
        )

        mcs._state_cache[spec_cls] = StateClass
        return StateClass

    @classmethod
    def _generate_checked_state_class(mcs, spec_cls: type) -> type:
        """Generate frozen dataclass for validated state (non-Optional fields)."""
        if spec_cls in mcs._checked_state_cache:
            return mcs._checked_state_cache[spec_cls]

        hints = get_type_hints(spec_cls)
        fields = []

        for name, ann in hints.items():
            if name in mcs.ATTRS:
                continue

            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, State):
                inner_type = get_args(ann)[0] if get_args(ann) else Any
                fields.append((name, inner_type))  # Non-Optional

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_checked = mcs._generate_checked_state_class(origin)
                if dataclass_fields(nested_checked):
                    fields.append((name, nested_checked))

            elif origin is list:
                is_list, inner_spec = _is_list_of_specs(ann)
                if is_list and inner_spec is not None:
                    nested_checked = mcs._generate_checked_state_class(inner_spec)
                    if dataclass_fields(nested_checked):
                        fields.append((name, tuple))

        class_name = f'{spec_cls.__name__}CheckedState'
        CheckedStateClass = make_dataclass(class_name, fields, frozen=True)

        jax.tree_util.register_dataclass(
            CheckedStateClass,
            data_fields=[f[0] for f in fields],
            meta_fields=[],
        )

        mcs._checked_state_cache[spec_cls] = CheckedStateClass
        return CheckedStateClass

    @classmethod
    def _generate_transforms_class(mcs, spec_cls: type) -> type:
        """Generate frozen dataclass for a Spec's transforms."""
        if spec_cls in mcs._transforms_cache:
            return mcs._transforms_cache[spec_cls]

        hints = get_type_hints(spec_cls)
        fields = []

        for name, ann in hints.items():
            if name in mcs.ATTRS:
                continue

            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, Parameter):
                fields.append((name, Transform))

            elif isinstance(origin, TypeVar):
                if _is_parameter_bound(origin) or _is_spec_bound(origin):
                    fields.append((name, Any))

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_transforms = mcs._generate_transforms_class(origin)
                fields.append((name, nested_transforms))

            elif origin is list:
                is_list, inner_spec = _is_list_of_specs(ann)
                if is_list and inner_spec is not None:
                    mcs._generate_transforms_class(inner_spec)  # Ensure nested class exists
                    fields.append((name, tuple))

        class_name = f'{spec_cls.__name__}Transforms'
        TransformsClass = make_dataclass(class_name, fields, frozen=True)

        mcs._transforms_cache[spec_cls] = TransformsClass
        return TransformsClass


class Spec(ABC, Generic[P, S, CS, Tr], metaclass=SpecMeta):
    """Base class for declarative model definitions.

    Subclasses define model structure via type hints:

    Fields can be:
    - Parameter[T]: Terminal leaf values (in Params tree)
    - State[T]: Cached/derived values (in State tree)
    - Nested Spec (including Prior): Branches with their own structure

    All learnable values are terminal Parameter[T] leaves.
    All branching/composite structures are Specs.

    Example:
        class MyModel(Spec):
            kernel: Matern52[LogNormal[float]]
            noise: LogNormal[float]

        model = MyModel(
            kernel=Matern52(
                lengthscale=LogNormal(value=0.5, mu=0.0, sigma=1.0),
                sigma=LogNormal(value=1.0, mu=0.0, sigma=1.0),
            ),
            noise=LogNormal(value=0.01, mu=-4.0, sigma=0.5),
        )

        params = model.init_params()
        params.kernel.lengthscale.value  # 0.5
        params.noise.value               # 0.01
    """

    Params: type[P]  # Set by metaclass
    State: type[S]  # Set by metaclass
    CheckedState: type[CS]  # Set by metaclass
    Transforms: type[Tr]  # Set by metaclass

    def __init__(self, **kwargs):
        """Initialize with values for Parameters and nested Specs.

        Args:
            **kwargs: Values for each Parameter and Spec field defined in type hints.

        Raises:
            TypeError: If required arguments are missing or unexpected arguments provided.
        """
        hints = get_type_hints(self.__class__)
        expected = {
            n
            for n, ann in hints.items()
            if n not in SpecMeta.ATTRS and get_origin(ann) is not ClassVar
        }

        # Check for unexpected arguments
        unexpected = set(kwargs) - expected
        if unexpected:
            raise TypeError(
                f"Unexpected argument(s): {', '.join(sorted(unexpected))}"
            )

        for name, ann in hints.items():
            # Skip class attributes set by metaclass
            if name in SpecMeta.ATTRS:
                continue

            # Skip ClassVar annotations (class-level, not instance)
            if get_origin(ann) is ClassVar:
                continue

            origin = get_origin(ann) or ann

            if name in kwargs:
                value = kwargs[name]
                # Convert lists to tuples for immutability
                if origin is list and isinstance(value, list):
                    value = tuple(value)
                setattr(self, name, value)
            elif isinstance(origin, type) and issubclass(origin, State):
                setattr(self, name, None)
            else:
                raise TypeError(f'Missing required argument: {name}')

    def init_params(self, rng: Optional[Array] = None) -> P:
        """Extract parameter values into a typed Params pytree.

        Returns:
            Typed Params dataclass with named attribute access.
        """
        hints = get_type_hints(self.__class__)
        params = {}

        for name, ann in hints.items():
            if name in SpecMeta.ATTRS:
                continue

            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, Parameter):
                params[name] = getattr(self, name)

            elif isinstance(origin, TypeVar):
                if _is_parameter_bound(origin):
                    params[name] = getattr(self, name)
                elif _is_spec_bound(origin):
                    nested_spec = getattr(self, name)
                    params[name] = nested_spec.init_params(rng)

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                params[name] = nested_spec.init_params(rng)

            elif origin is list:
                is_list, _ = _is_list_of_specs(ann)
                if is_list:
                    nested_specs = getattr(self, name)
                    params[name] = tuple(s.init_params(rng) for s in nested_specs)

        return self.__class__.Params(**params)

    def init_state(self) -> S:
        """Generate typed State pytree with None placeholders.

        Returns:
            Typed State dataclass with named attribute access.
        """
        state_cls = getattr(self.__class__, 'State')
        hints = get_type_hints(self.__class__)
        state = {}

        for field in dataclass_fields(state_cls):
            name = field.name
            ann = hints.get(name)
            if ann is None:
                state[name] = None
                continue

            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, State):
                state[name] = None

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                state[name] = nested_spec.init_state()

            elif origin is list:
                is_list, _ = _is_list_of_specs(ann)
                if is_list:
                    nested_specs = getattr(self, name)
                    state[name] = tuple(s.init_state() for s in nested_specs)

        return state_cls(**state)

    @classmethod
    def check_state(cls, state: S, *field_names: str):
        """Validate state fields and return CheckedState with narrowed types.

        Args:
            state: The State instance to validate
            *field_names: Names of fields that must not be None

        Returns:
            CheckedState instance with non-Optional field types

        Raises:
            StateValidationError: If any specified field is None
        """
        from ._requires import _validate_and_convert

        return _validate_and_convert(cls, state, field_names)

    def get_transforms(self) -> Tr:
        """Get transforms with named attribute access.

        Returns:
            Transforms dataclass with named attributes for each parameter.
        """
        return self._build_transforms()

    def _build_transforms(self) -> Any:
        """Build transforms instance for this Spec.

        For Parameter fields, uses Identity transform.
        For nested Spec/Prior fields, recurses.
        For Prior fields, the Prior's get_transforms() handles
        using the Prior's class-level transform for its value field.
        """
        transforms_cls = getattr(self.__class__, 'Transforms')
        hints = get_type_hints(self.__class__)
        values = {}

        for field in dataclass_fields(transforms_cls):
            name = field.name
            ann = hints.get(name)
            if ann is None:
                values[name] = Identity()
                continue

            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, Parameter):
                # Terminal parameter - use Identity
                values[name] = Identity()

            elif isinstance(origin, TypeVar):
                # TypeVar - resolve at runtime
                nested = getattr(self, name, None)
                if isinstance(nested, Spec):
                    values[name] = nested._build_transforms()
                else:
                    values[name] = Identity()

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                values[name] = nested_spec._build_transforms()

            elif origin is list:
                is_list, _ = _is_list_of_specs(ann)
                if is_list:
                    nested_specs = getattr(self, name)
                    values[name] = tuple(s._build_transforms() for s in nested_specs)
                else:
                    values[name] = Identity()

            else:
                values[name] = Identity()

        return transforms_cls(**values)
