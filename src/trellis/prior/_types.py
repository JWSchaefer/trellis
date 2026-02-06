from __future__ import annotations

from beartype.typing import Any, TypeVar

from ._prior import Prior

MuPrior = TypeVar('MuPrior', bound=Prior[Any])
LocPrior = TypeVar('LocPrior', bound=Prior[Any])

SigmaPrior = TypeVar('SigmaPrior', bound=Prior[Any])
ScalePrior = TypeVar('ScalePrior', bound=Prior[Any])
