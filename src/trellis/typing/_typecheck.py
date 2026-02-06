from beartype.typing import Callable, TypeVar

from .._config import config

F = TypeVar('F', bound=Callable)


def typecheck(func: F) -> F:
    """
    Apply @jaxtyped(typechecker=beartype) iff TYPECHECKING_ENABLED is True.
    Otherwise return the function unchanged.
    """
    if not config.get('runtime_typechecking', True):
        return func

    from beartype import beartype
    from jaxtyping import jaxtyped

    return jaxtyped(typechecker=beartype)(func)
