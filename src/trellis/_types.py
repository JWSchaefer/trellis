from beartype.typing import TypeAlias, TypeVar
from jaxtyping import Array, Float

T = TypeVar('T', float, Float[Array, '...'], Array)
S = TypeVar('S', float, Float[Array, '...'], Array)

Scalar: TypeAlias = Float[Array, ''] | Float[Array, 'N']
