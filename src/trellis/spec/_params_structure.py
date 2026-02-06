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
