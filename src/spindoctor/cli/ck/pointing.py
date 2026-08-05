"""One navigated image's recorded pointing, as a C-kernel writer reads it.

The navigation pipeline records, for every image it navigates, the corrected
camera attitude as a C-matrix in the SPICE camera frame convention at the
exposure midtime, beside the frame identities and the exposure epochs a
C-kernel segment needs.  This module turns that recorded metadata block into
the single input type the segment writer accepts.

The writer reads the metadata rather than sharing the pipeline's in-memory
dataclass on purpose: the pipeline's version is produced by code that imports
oops, and a kernel writer that imports oops defeats the point of writing
kernels.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

# Spelled here rather than imported from ``spindoctor.support.types`` because
# the writer must not import ``spindoctor.support``: its modules pull in oops.
NDArrayFloatType = npt.NDArray[np.floating[Any]]

# A recorded C-matrix must be a proper rotation to this tolerance.  Anything
# looser is a defect in what was recorded, not something to orthonormalize
# away, and it would be written into a kernel other tools trust.
_ROTATION_TOL = 1e-9


@dataclass(frozen=True)
class ImagePointing:
    """The corrected pointing of one navigated image.

    Parameters:
        image_name: Basename of the navigated image.  It names the segment
            written for the image.
        cmatrix: Corrected J2000-to-camera rotation at the exposure midtime,
            in the SPICE camera frame convention (``v_camera = C . v_J2000``).
        cmatrix_original: The same rotation before the correction, as the
            kernels furnished at navigation time gave it.  It identifies the
            baseline kernel the image navigated against: a candidate kernel
            belongs to this image only if it reproduces this matrix.
        camera_frame: SPICE name of the camera frame ``cmatrix`` is expressed
            in, for example ``'CASSINI_ISS_NAC'``.
        ck_frame_id: SPICE id of the object a corrected C-kernel targets --
            the bus or scan platform the existing kernels describe, not the
            camera.
        start_et: Exposure start, TDB seconds past J2000.
        stop_et: Exposure stop, TDB seconds past J2000.
        midtime_et: Exposure midtime, TDB seconds past J2000.
        exposure_s: Exposure duration in seconds.

    Raises:
        ValueError: if ``image_name`` is empty, if either C-matrix is not a
            3x3 proper orthonormal rotation, if any epoch or ``exposure_s`` is
            not finite, if the three epochs are not ordered
            ``start <= midtime <= stop``, or if ``exposure_s`` is negative.
    """

    image_name: str
    cmatrix: NDArrayFloatType
    cmatrix_original: NDArrayFloatType
    camera_frame: str
    ck_frame_id: int
    start_et: float
    stop_et: float
    midtime_et: float
    exposure_s: float

    def __post_init__(self) -> None:
        """Store the C-matrices read-only and refuse anything unusable."""
        object.__setattr__(self, 'cmatrix', _as_rotation(self.cmatrix, 'cmatrix'))
        object.__setattr__(
            self, 'cmatrix_original', _as_rotation(self.cmatrix_original, 'cmatrix_original')
        )
        if len(self.image_name) == 0:
            raise ValueError('image_name is empty; a segment must name the image it corrects')
        # Checked before the comparisons below, which a NaN would answer with
        # False rather than refuse, and which an infinite epoch would satisfy
        # outright.  Every one of these values reaches a clock encoding or a
        # record cadence, where a non-finite value stops being attributable.
        for field, value in (
            ('start_et', self.start_et),
            ('stop_et', self.stop_et),
            ('midtime_et', self.midtime_et),
            ('exposure_s', self.exposure_s),
        ):
            if not math.isfinite(value):
                raise ValueError(f'{field} is not finite for {self.image_name}: {value!r}')
        if not self.start_et <= self.midtime_et <= self.stop_et:
            raise ValueError(
                f'exposure epochs are out of order for {self.image_name}: start {self.start_et!r}, '
                f'midtime {self.midtime_et!r}, stop {self.stop_et!r}'
            )
        if self.exposure_s < 0.0:
            raise ValueError(f'exposure_s is negative for {self.image_name}: {self.exposure_s!r}')

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> 'ImagePointing':
        """Read one image's pointing out of its navigation metadata.

        The metadata is the per-image ``_metadata.json`` dict the navigation
        pipeline writes.  The fields read are ``observation.image_name``, the
        ``navigation_result.pointing`` block (``cmatrix``,
        ``cmatrix_original``, ``camera_frame``, ``ck_frame_id``) and the
        ``navigation_result.times`` block (``start_et``, ``stop_et``,
        ``midtime_et``, ``exposure_s``).  Nothing else is consulted, and no
        eligibility rule is applied here: an image whose pointing this
        constructor accepts is one the writer can express, not necessarily one
        that should be written.

        Parameters:
            metadata: The image's full navigation metadata dict.

        Returns:
            The image's ImagePointing.

        Raises:
            ValueError: if any of those fields is absent, or if the values
                present do not satisfy the ImagePointing invariants.  A
                corrected ``cmatrix`` is absent for every image that navigated
                without an offset or with a fitted camera rotation, and such an
                image cannot be given a segment.
            TypeError: if a field is present but holds a value of the wrong
                kind -- a JSON ``null`` where a number belongs, or anything
                but text where ``image_name`` or ``camera_frame`` belongs.
                That is a malformed document rather than an image without a
                solution, so it fails loudly instead of being reported as an
                omission.  Text is never coerced: ``str(None)`` is ``'None'``,
                which would name a written segment.
        """
        observation = read_section(metadata, 'observation', 'metadata')
        navigation_result = read_section(metadata, 'navigation_result', 'metadata')
        pointing = read_section(navigation_result, 'pointing', 'navigation_result')
        times = read_section(navigation_result, 'times', 'navigation_result')
        return cls(
            image_name=read_text(observation, 'image_name', 'observation'),
            cmatrix=_rotation_from_metadata(read_field(pointing, 'cmatrix', 'pointing'), 'cmatrix'),
            cmatrix_original=_rotation_from_metadata(
                read_field(pointing, 'cmatrix_original', 'pointing'), 'cmatrix_original'
            ),
            camera_frame=read_text(pointing, 'camera_frame', 'pointing'),
            ck_frame_id=int(read_field(pointing, 'ck_frame_id', 'pointing')),
            start_et=float(read_field(times, 'start_et', 'times')),
            stop_et=float(read_field(times, 'stop_et', 'times')),
            midtime_et=float(read_field(times, 'midtime_et', 'times')),
            exposure_s=float(read_field(times, 'exposure_s', 'times')),
        )


def _rotation_from_metadata(value: Any, label: str) -> NDArrayFloatType:
    """Read a recorded C-matrix, accepting only the shapes the schema writes.

    The metadata records a C-matrix as nine row-major floats, so a flat
    nine-element sequence is the canonical form and a 3x3 nesting is accepted
    as the obvious equivalent.  Reshaping whatever arrives is deliberately not
    done: ``(1, 9)`` and ``(3, 3, 1)`` also hold nine values and would reshape
    silently, so a document malformed in a way worth knowing about would be
    read as if it were well formed.

    Parameters:
        value: The recorded matrix value, as read from the metadata.
        label: Name of the field, used in the exception message.

    Returns:
        The 3x3 rotation.

    Raises:
        ValueError: if the value does not hold nine numbers as a flat
            sequence or a 3x3 nesting.
    """
    array = np.asarray(value, dtype=np.float64)
    if array.shape not in ((9,), (3, 3)):
        raise ValueError(
            f'{label} must be nine row-major floats or a 3x3 nesting; got shape {array.shape}'
        )
    return array.reshape(3, 3)


def read_text(section: dict[str, Any], key: str, where: str) -> str:
    """Return one required metadata value that must already be a string.

    ``str()`` is deliberately not used to coerce.  A JSON ``null`` coerces to
    the text ``'None'``, which is neither empty nor obviously wrong, so an
    image whose name is null would otherwise be given a segment identified as
    ``None`` and pass every check downstream of it.

    Parameters:
        section: The dict to read.
        key: The key that must be present.
        where: Name of the section, used in the exception message.

    Returns:
        The value stored under ``key``.

    Raises:
        ValueError: if ``key`` is absent.
        TypeError: if the value present is not a string.
    """
    value = read_field(section, key, where)
    if not isinstance(value, str):
        raise TypeError(f'{where} field {key!r} is {type(value).__name__}, not a string: {value!r}')
    return value


def read_field(section: dict[str, Any], key: str, where: str) -> Any:
    """Return one required metadata value.

    Parameters:
        section: The dict to read.
        key: The key that must be present.
        where: Name of the section, used in the exception message.

    Returns:
        The value stored under ``key``.

    Raises:
        ValueError: if ``key`` is absent.
    """
    if key not in section:
        raise ValueError(f'{where} has no {key!r} field; the metadata records no such value')
    return section[key]


def read_section(metadata: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    """Return one required metadata sub-dict.

    Parameters:
        metadata: The dict to read.
        key: The key that must be present and hold a dict.
        where: Name of the enclosing section, used in the exception message.

    Returns:
        The sub-dict stored under ``key``.

    Raises:
        ValueError: if ``key`` is absent or does not hold a dict.
    """
    value = read_field(metadata, key, where)
    if not isinstance(value, dict):
        raise ValueError(f'{where}.{key} is {type(value).__name__}, not a section')
    return value


def _as_rotation(matrix: NDArrayFloatType, label: str) -> NDArrayFloatType:
    """Copy a matrix into a read-only 3x3 rotation, refusing anything else.

    Parameters:
        matrix: Any 3x3 array-like of numbers.
        label: Name used in the exception messages.

    Returns:
        A read-only float64 copy.

    Raises:
        ValueError: if the input is not 3x3, holds a non-finite value, or is
            not a proper orthonormal rotation to within ``_ROTATION_TOL``.
    """
    out = np.array(matrix, dtype=np.float64)
    if out.shape != (3, 3):
        raise ValueError(f'{label} is not a 3x3 matrix; got shape {out.shape}')
    # NaN fails every inequality below, so both tolerance guards would pass a
    # non-finite matrix silently.
    if not bool(np.all(np.isfinite(out))):
        raise ValueError(f'{label} holds a non-finite value: {out.tolist()!r}')
    det = float(np.linalg.det(out))
    if abs(det - 1.0) > _ROTATION_TOL:
        raise ValueError(
            f'{label} is not a proper rotation: determinant {det!r} differs from 1 by more '
            f'than {_ROTATION_TOL}'
        )
    residual = float(np.max(np.abs(out @ out.T - np.eye(3))))
    if residual > _ROTATION_TOL:
        raise ValueError(
            f'{label} is not orthonormal: max|C C^T - I| = {residual!r} exceeds {_ROTATION_TOL}'
        )
    out.setflags(write=False)
    return out
