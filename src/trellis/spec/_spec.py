from abc import ABC, ABCMeta
from dataclasses import fields as dataclass_fields
from dataclasses import make_dataclass

import jax.tree_util
from beartype.typing import (
    Any,
    ClassVar,
    Generic,
    Optional,
    Type,
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
from ._types import P, S, Tr


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


class SpecMeta(ABCMeta):
    """Metaclass that generates Params, State, and Transforms classes for Specs."""

    ATTRS = ('Params', 'State', 'Transforms')

    _params_cache: dict[type, type] = {}
    _state_cache: dict[type, type] = {}
    _transforms_cache: dict[type, type] = {}

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip base classes (Spec, Prior)
        if name in ('Spec', 'Prior'):
            return cls

        # Generate and attach Params/State/Transforms classes
        setattr(cls, 'Params', mcs._generate_params_class(cls))
        setattr(cls, 'State', mcs._generate_state_class(cls))
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
                if dataclass_fields(nested_transforms):
                    fields.append((name, nested_transforms))
                else:
                    fields.append((name, nested_transforms))

        class_name = f'{spec_cls.__name__}Transforms'
        TransformsClass = make_dataclass(class_name, fields, frozen=True)

        mcs._transforms_cache[spec_cls] = TransformsClass
        return TransformsClass


class Spec(ABC, Generic[P, S, Tr], metaclass=SpecMeta):
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
                setattr(self, name, kwargs[name])
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

        return self.__class__.Params(**params)

    def init_state(self) -> S:
        """Generate typed State pytree with None placeholders.

        Returns:
            Typed State dataclass with named attribute access.
        """
        state_dict = self._collect_state_dict()
        return self._dict_to_state(state_dict)

    def _collect_state_dict(self) -> dict:
        """Collect state values into a dict."""
        hints = get_type_hints(self.__class__)
        state = {}

        for name, ann in hints.items():
            # Skip class attributes
            if name in SpecMeta.ATTRS:
                continue

            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, State):
                state[name] = None

            elif isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                nested_state = nested_spec._collect_state_dict()
                if nested_state:
                    state[name] = nested_state

        return state

    def _dict_to_state(self, state_dict: dict) -> S:
        """Convert nested dict to typed State dataclass."""
        hints = get_type_hints(self.__class__)
        converted = {}

        for name, value in state_dict.items():
            ann = hints.get(name)
            origin = get_origin(ann) or ann

            if isinstance(origin, type) and issubclass(origin, Spec):
                nested_spec = getattr(self, name)
                converted[name] = nested_spec._dict_to_state(value)
            else:
                converted[name] = value

        return self.__class__.State(**converted)

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

            else:
                values[name] = Identity()

        return transforms_cls(**values)

    @classmethod
    def to_spec(cls: Type['Spec']) -> dict:
        """Extract type specification from class annotations.

        Returns:
            Dict mapping field names to their type markers (Parameter/State/nested Spec).
        """
        return {
            name: cls._from_annotation(ann)
            for name, ann in get_type_hints(cls).items()
        }

    @classmethod
    def _from_annotation(cls, ann):
        origin = get_origin(ann) or ann
        args = get_args(ann)

        if isinstance(origin, type) and issubclass(origin, (Parameter, State)):
            return origin

        if isinstance(origin, type) and issubclass(origin, Spec):
            return origin.to_structure_from_args(args)

        raise TypeError(f'Unsupported annotation {ann}')

    @classmethod
    def to_structure_from_args(cls, args):
        type_vars = getattr(cls, '__parameters__', ())
        tv_map: dict[TypeVar, Type] = dict(zip(type_vars, args))

        def substitute(ann):
            if isinstance(ann, TypeVar):
                return tv_map[ann]

            origin = get_origin(ann)
            if origin is None:
                return ann

            return origin[tuple(substitute(a) for a in get_args(ann))]

        resolved_hints = {
            name: substitute(ann) for name, ann in get_type_hints(cls).items()
        }

        return {
            name: cls._from_annotation(ann)
            for name, ann in resolved_hints.items()
        }
