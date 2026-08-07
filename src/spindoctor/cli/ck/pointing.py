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

# The numpy dtype kinds a C-matrix may arrive as: signed and unsigned integers
# and floats.  Booleans ('b') and text ('U', 'S') are excluded although both
# convert to float64 without complaint, which is what would let nine ``True``
# values pass as an identity rotation.
_REAL_NUMBER_KINDS = frozenset({'i', 'u', 'f'})

# How far the recorded exposure duration may sit from the start-to-stop span
# before the two are treated as describing different exposures.  The pipeline
# derives both from one cadence, so they differ only by the rounding of adding
# a duration to an epoch -- a few nanoseconds at the epochs these missions
# observe at, five orders of magnitude under this bound and three under the
# shortest exposure any of them commands.
_EXPOSURE_SPAN_TOL_S = 1.0e-3


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
        TypeError: if either C-matrix holds values that are not real numbers.
            Booleans and numeric text convert to float64 without complaint, so
            a matrix of nine ``True`` values would otherwise be accepted as a
            flawless identity rotation.
        ValueError: if ``image_name`` or ``camera_frame`` is empty, if either
            C-matrix is not a 3x3 proper orthonormal rotation, if any epoch or
            ``exposure_s`` is not finite, if the three epochs are not ordered
            ``start <= midtime <= stop``, if ``exposure_s`` is negative, or if
            ``exposure_s`` differs from ``stop_et - start_et`` by more than a
            millisecond.  The two describe the same exposure and are used
            interchangeably, so a disagreement is a defect in what was
            recorded.
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
        if len(self.camera_frame) == 0:
            raise ValueError(
                f'camera_frame is empty for {self.image_name}; the rotation from the CK object to '
                f'the camera is read from the furnished kernels by that name'
            )
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
        # The duration and the epochs come from different fields of the
        # metadata and are used interchangeably downstream -- one decides
        # whether a segment gets interior records, the other decides how many.
        # Nothing else would notice them disagreeing.
        span_s = self.stop_et - self.start_et
        if abs(self.exposure_s - span_s) > _EXPOSURE_SPAN_TOL_S:
            raise ValueError(
                f'exposure_s disagrees with the recorded epochs for {self.image_name}: '
                f'{self.exposure_s!r} s against a start-to-stop span of {span_s!r} s'
            )

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
            ValueError: if any of those fields is absent, if an epoch is a
                whole number too large for a double, or if the values present
                do not satisfy the ImagePointing invariants.  A corrected
                ``cmatrix`` is absent for every image that navigated without an
                offset or with a fitted camera rotation, and such an image
                cannot be given a segment.
            TypeError: if a field is present but holds a value of the wrong
                kind: anything but text where ``image_name`` or
                ``camera_frame`` belongs, anything but a whole number where
                ``ck_frame_id`` belongs, or anything but a number where an
                epoch belongs.  That is a malformed document rather than an
                image without a solution, so it fails loudly instead of being
                reported as an omission.  Nothing is coerced: ``str(None)`` is
                ``'None'``, which would name a written segment, and
                ``int(-82000.9)`` is a valid Cassini bus id that the metadata
                never recorded.
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
            ck_frame_id=read_int(pointing, 'ck_frame_id', 'pointing'),
            start_et=read_number(times, 'start_et', 'times'),
            stop_et=read_number(times, 'stop_et', 'times'),
            midtime_et=read_number(times, 'midtime_et', 'times'),
            exposure_s=read_number(times, 'exposure_s', 'times'),
        )


def _rotation_from_metadata(value: Any, label: str) -> NDArrayFloatType:
    """Read a recorded C-matrix, accepting only the shapes the schema writes.

    The metadata records a C-matrix as nine row-major floats, so a flat
    nine-element sequence is the canonical form and a 3x3 nesting is accepted
    as the obvious equivalent.  Reshaping whatever arrives is deliberately not
    done: ``(1, 9)`` and ``(3, 3, 1)`` also hold nine values and would reshape
    silently, so a document malformed in a way worth knowing about would be
    read as if it were well formed.

    The elements are checked before conversion for the same reason the scalar
    fields are: ``np.asarray(..., dtype=np.float64)`` converts text and
    booleans without complaint, so a matrix of nine ``true`` and ``false``
    values becomes a flawless identity that satisfies every rotation guard
    there is.

    Parameters:
        value: The recorded matrix value, as read from the metadata.
        label: Name of the field, used in the exception message.

    Returns:
        The 3x3 rotation.

    Raises:
        ValueError: if the value does not hold nine numbers as a flat
            sequence or a 3x3 nesting.
        TypeError: if any element is not a real number.
    """
    array = np.asarray(_rotation_elements(value, label), dtype=np.float64)
    if array.shape not in ((9,), (3, 3)):
        raise ValueError(
            f'{label} must be nine row-major floats or a 3x3 nesting; got shape {array.shape}'
        )
    return array.reshape(3, 3)


def _rotation_elements(value: Any, label: str) -> Any:
    """Check that a recorded matrix holds real numbers, and return it unchanged.

    Only the shapes the schema writes are walked -- a flat sequence and one
    level of nesting -- because anything else is refused for its shape a moment
    later, and a value that is not a sequence at all is left for the shape
    check to report.

    Parameters:
        value: The recorded matrix value, as read from the metadata.
        label: Name of the field, used in the exception message.

    Returns:
        ``value`` itself.

    Raises:
        TypeError: if an element is text, a boolean, or anything else that is
            not a real number.  ``bool`` is refused although Python counts it
            as an ``int``: a matrix of JSON ``true`` and ``false`` values
            converts to a valid rotation.
    """
    if not isinstance(value, list | tuple):
        return value
    for row in value:
        elements = row if isinstance(row, list | tuple) else [row]
        for element in elements:
            if isinstance(element, list | tuple):
                # Nested deeper than the schema writes; the shape check reports
                # that, and reports it better than an element check could.
                continue
            if isinstance(element, bool) or not isinstance(element, int | float):
                raise TypeError(
                    f'{label} holds a {type(element).__name__}, not a number: {element!r}'
                )
    return value


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


def read_int(section: dict[str, Any], key: str, where: str) -> int:
    """Return one required metadata value that must already be a whole number.

    ``int()`` is deliberately not used to coerce, for the same reason
    :func:`read_text` does not use ``str()``.  ``int('-82000')`` and
    ``int(-82000.9)`` both produce a valid Cassini bus id, the second by
    truncating a value that was never that id, and ``int(True)`` produces 1.
    Every one of those would resolve a spacecraft clock, encode time tags and
    write a segment against an object the metadata never named.

    Parameters:
        section: The dict to read.
        key: The key that must be present.
        where: Name of the section, used in the exception message.

    Returns:
        The value stored under ``key``.

    Raises:
        ValueError: if ``key`` is absent.
        TypeError: if the value present is not an integer.  ``bool`` is
            refused although Python counts it as one, since a JSON ``true`` is
            not an object id.
    """
    value = read_field(section, key, where)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f'{where} field {key!r} is {type(value).__name__}, not an integer: {value!r}'
        )
    # Narrowed to int above; restated so the annotation is checkable rather
    # than inferred from a dict of Any.
    return int(value)


def read_number(section: dict[str, Any], key: str, where: str) -> float:
    """Return one required metadata value that must already be a number.

    ``float()`` is deliberately not used to coerce, for the same reason
    :func:`read_text` does not use ``str()``: ``float('0.0')`` and
    ``float(True)`` both succeed, so an epoch recorded as text or as a JSON
    ``true`` would reach a clock encoding as a plausible number.  A whole
    number is accepted and widened, since JSON writes an exact epoch that way.

    Parameters:
        section: The dict to read.
        key: The key that must be present.
        where: Name of the section, used in the exception message.

    Returns:
        The value stored under ``key``, as a float.

    Raises:
        ValueError: if ``key`` is absent, or if the value is a whole number too
            large for a double.  JSON has no bound on an integer literal, so a
            document can carry one no epoch could be; it is a malformed
            document, reported as one, rather than the ``OverflowError`` the
            widening raises, which no caller of this module expects.
        TypeError: if the value present is not a real number.
    """
    value = read_field(section, key, where)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f'{where} field {key!r} is {type(value).__name__}, not a number: {value!r}')
    try:
        return float(value)
    except OverflowError as exc:
        raise ValueError(
            f'{where} field {key!r} is a whole number too large to be a double: {value!r}'
        ) from exc


def read_optional_text(section: dict[str, Any], key: str, where: str) -> str | None:
    """Return one optional metadata value that must be text when present.

    An absent key means the metadata records no such value.  A key present but
    holding a JSON ``null`` does not: nothing the pipeline writes as text is
    ever written as null, so a null is a malformed document.  ``str()`` is
    deliberately not used to coerce it either, since ``str(None)`` is
    ``'None'``, which is neither empty nor obviously wrong -- a null camera
    would pair with the opposite camera of a simultaneous exposure and silently
    decide which of the two keeps its correction.

    Parameters:
        section: The dict to read.
        key: The key to read.
        where: Name of the section, used in the exception message.

    Returns:
        The value stored under ``key``, or ``None`` when the key is absent.

    Raises:
        TypeError: if the value present is not a string.
    """
    if key not in section:
        return None
    return read_text(section, key, where)


def read_optional_number(section: dict[str, Any], key: str, where: str) -> float | None:
    """Return one optional metadata value that must be a number when present.

    An absent key means the metadata records no such value; a key holding a
    JSON ``null`` is refused, for the same reason :func:`read_optional_text`
    refuses one.

    Parameters:
        section: The dict to read.
        key: The key to read.
        where: Name of the section, used in the exception message.

    Returns:
        The value stored under ``key`` as a float, or ``None`` when the key is
        absent.

    Raises:
        TypeError: if the value present is not a real number.
        ValueError: if the number present is not finite.  The pipeline maps a
            non-finite value onto a large finite sentinel before writing it, so
            a bare ``NaN`` or infinity here is a hand-edited document rather
            than a recorded measurement, and it would be reported as a number
            no reader could attribute.
    """
    if key not in section:
        return None
    value = read_number(section, key, where)
    if not math.isfinite(value):
        raise ValueError(f'{where} field {key!r} is not finite: {value!r}')
    return value


def read_optional_pair(section: dict[str, Any], key: str, where: str) -> tuple[float, float] | None:
    """Return one optional metadata value that must be two numbers when recorded.

    The pipeline records an offset and a per-axis sigma as a two-element
    ``[dv, du]`` list, and writes an explicit ``null`` for a sigma it has none
    of -- which is why a null is read here as "not recorded" where every other
    reader in this module refuses one.

    Parameters:
        section: The dict to read.
        key: The key to read.
        where: Name of the section, used in the exception message.

    Returns:
        The two values, or ``None`` when the key is absent or null.

    Raises:
        TypeError: if the value present is not a list or tuple, or if either
            element is not a real number.
        ValueError: if it does not hold exactly two elements, or if either is
            not finite.
    """
    if key not in section or section[key] is None:
        return None
    value = read_field(section, key, where)
    if not isinstance(value, list | tuple):
        raise TypeError(
            f'{where} field {key!r} is {type(value).__name__}, not a pair of numbers: {value!r}'
        )
    if len(value) != 2:
        raise ValueError(f'{where} field {key!r} holds {len(value)} values, not two: {value!r}')
    pair = tuple(read_number({'value': element}, 'value', where) for element in value)
    for element in pair:
        if not math.isfinite(element):
            raise ValueError(f'{where} field {key!r} holds a non-finite value: {value!r}')
    return (pair[0], pair[1])


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
        TypeError: if the values are not real numbers.  ``np.array(...,
            dtype=np.float64)`` converts booleans and numeric text without
            complaint, so nine ``True`` values would otherwise arrive here as a
            flawless identity and satisfy every guard below.
        ValueError: if the input is not 3x3, holds a non-finite value, or is
            not a proper orthonormal rotation to within ``_ROTATION_TOL``.
    """
    given = np.asarray(matrix)
    if given.dtype.kind not in _REAL_NUMBER_KINDS:
        raise TypeError(f'{label} holds {given.dtype} values, not real numbers: {given.tolist()!r}')
    out = np.array(given, dtype=np.float64)
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
