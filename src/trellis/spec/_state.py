from beartype.typing import Generic, TypeVar

T = TypeVar('T')


class State(Generic[T]):
    """Type marker for cached state/computations.

    Use in type hints to declare state fields:
        class MyModel(Spec):
            cholesky: State[Float[Array, 'N N']]

    State values are not optimized - they are derived from params and data.
    """

    pass
