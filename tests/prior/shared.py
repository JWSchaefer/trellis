import jax.numpy as jnp
from beartype.typing import List, Tuple, Type
from jax import Array

from trellis.prior import HalfNormal, LogNormal, NoPrior, Normal, Prior
from trellis.transform import Identity, Log, Transform

shapes: List[Tuple[int, ...]] = [(1,), (5,), (2, 5), (2, 5, 6)]
transforms: List[Type[Transform]] = [Log, Identity]
prior_inits: List[Tuple[Type[Prior], dict[str, float]]] = [
    (HalfNormal, {'scale': 1e-1}),
    (LogNormal, {'loc': 0.0, 'scale': 1e-1}),
    (NoPrior, {}),
    (Normal, {'loc': 0.0, 'scale': 1e-1}),
]

values: List[float | Array] = [
    0.0,
    jnp.zeros((1,)),
    jnp.zeros((10_000,)),
]


def values_id(values: float | Array) -> str:
    return (
        str(values.shape) if isinstance(values, Array) else str(float.__name__)
    )


def prior_id(prior_and_args: Tuple[Prior, dict[str, float]]) -> str:
    prior, args = prior_and_args
    return f'{prior}-{args}'


def transform_id(transform: Type[Transform]) -> str:
    return transform.__name__


def shape_id(shape: Tuple[int, ...]) -> str:
    return str(shape)
