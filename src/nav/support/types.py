from pathlib import Path
from typing import Any, Optional, Protocol, TypeVar

from filecache import FCPath
import numpy as np
import numpy.typing as npt

NDArrayLike = npt.ArrayLike
DTypeLike = npt.DTypeLike
NDArrayBoolType = npt.NDArray[np.bool_]
NDArrayFloatType = npt.NDArray[np.floating[Any]]
NDArrayIntType = npt.NDArray[np.integer[Any]]
NDArrayUint8Type = npt.NDArray[np.uint8]
NDArrayUint32Type = npt.NDArray[np.uint32]
NPType = TypeVar('NPType', bound=np.generic, covariant=True)
NDArrayType = npt.NDArray[NPType]

PathLike = str | Path | FCPath


class MutableStar(Protocol):
    unique_number: Optional[int]
    catalog_name: str
    pretty_name: str
    name: str

    # Image-space location and motion
    v: float
    u: float
    move_v: float
    move_u: float

    # Photometry and spectral info
    vmag: float | None
    b_v: float | None
    johnson_mag_v: float | None
    johnson_mag_b: float | None
    johnson_mag_faked: bool
    spectral_class: Optional[str]
    temperature: float | None
    temperature_faked: bool

    # Proper motion and current RA/DEC
    ra: float | None
    dec: float | None
    ra_pm: float
    dec_pm: float

    # Additional fields used during processing
    psf_size: tuple[int, int]
    dn: float
    conflicts: str
    diff_u: float
    diff_v: float

    def ra_dec_with_pm(self, tdb: float) -> tuple[float, float] | tuple[None, None]: ...
