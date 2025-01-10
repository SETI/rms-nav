from pathlib import Path
from typing import Any, TypeVar
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from filecache import FCPath
import numpy as np
import numpy.typing as npt

NDArrayLike = npt.ArrayLike
DTypeLike = npt.DTypeLike
NDArrayBoolType = npt.NDArray[np.bool_]
NDArrayFloatType = npt.NDArray[np.floating[Any]]
NDArrayIntType = npt.NDArray[np.int_]
NPType = TypeVar('NPType', bound=np.generic, covariant=True)
NDArrayType = npt.NDArray[NPType]

PathLike = str | Path | FCPath
