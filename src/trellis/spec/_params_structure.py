from dataclasses import dataclass

from beartype.typing import Any


@dataclass(frozen=True)
class ParamsStructure:
    treedef: Any
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[Any, ...]

    @staticmethod
    def _prod(shape: tuple[int, ...]) -> int:
        result = 1
        for dim in shape:
            result *= dim
        return result


@dataclass(frozen=True)
class LearnableParamsStructure(ParamsStructure):
    """Structure for flattening/unflattening learnable params only.

    Stores both learnable param metadata and fixed values for reconstruction.
    Extends ParamsStructure with fields for fixed parameter handling.
    """

    fixed_indices: tuple[int, ...]  # Positions of fixed params in leaf list
    fixed_values: tuple[Any, ...]  # Fixed param values to restore
