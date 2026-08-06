"""Corrected-attitude C-matrices for a navigated observation.

A C-matrix is the rotation taking a vector expressed in J2000 to the same vector
expressed in a camera frame::

    v_frame = C . v_J2000

which is what ``cspyce.pxform('J2000', frame_name, et)`` returns.  Navigation
measures a pixel offset; this module converts that offset into the corrected
attitude the camera actually had, so the same measurement can later be written
into a C-kernel and consumed by any SPICE-aware tool without applying a pixel
offset by hand.

Two matrices are produced per navigated image, both in the SPICE camera frame
convention and both at the exposure midtime:

``cmatrix_original``
    The attitude the furnished kernels gave at navigation time, before any
    correction.
``cmatrix``
    The corrected attitude, the one a kernel should carry.

The oops observation frame is **not** the SPICE camera frame.  For three of the
four supported instruments oops builds its frame with a constant flip ``R`` on
top of the SPICE frame, so that ``C_oops = R . C_spice``:

============================  ==========================
Instrument                    ``R``
============================  ==========================
Cassini ISS (NAC, WAC)        ``diag(-1, -1, +1)``
New Horizons LORRI            ``diag(+1, -1, -1)``
Galileo SSI                   identity
Voyager ISS                   identity by construction
============================  ==========================

A correction built in the oops frame and composed onto a ``pxform``-derived
matrix without conjugating through ``R`` is a proper rotation of the right
magnitude pointing the wrong way, so ``R`` is measured at runtime and checked
against the value above rather than assumed.

Voyager is the exception to everything: oops freezes the observation frame from
a tolerance-snapped ``ckgp`` lookup rather than evaluating a frame chain, so
``pxform`` at the midtime does not reproduce it.  For Voyager the observation
frame attitude already **is** the SPICE camera frame attitude, and ``R`` is the
identity by construction.

The offset-to-rotation conversion lives here, behind one entry point, so that
when oops grows its own corrected-attitude API this module's body is replaced
and its interface stays.
"""

from dataclasses import dataclass

import cspyce
import numpy as np
import oops

from spindoctor.obs import (
    ObsCassiniISS,
    ObsGalileoSSI,
    ObsNewHorizonsLORRI,
    ObsSnapshotInst,
    ObsVoyagerISS,
)
from spindoctor.spice_ids import CK_OBJECT_SCLK_ID
from spindoctor.support.exceptions import NavPointingError
from spindoctor.support.types import NDArrayFloatType

# The offset-to-attitude conversion is deliberately behind one public
# function: when oops grows its own corrected-attitude API this module's body
# is replaced and only ``compute_pointing`` has to survive the swap.  The
# helpers below it are private for that reason, not because they are trivial.
__all__ = [
    'AttitudeBaseline',
    'PointingSolution',
    'compute_pointing',
]


# The object each mission's C-kernels describe.  The correction is measured at
# the camera but written at the bus or platform the existing kernels already
# cover, so a corrected kernel targets one of these.  Voyager's is derived per
# spacecraft instead, because one instrument key serves two spacecraft.
_CASSINI_CK_FRAME_ID = -82000
_GALILEO_CK_FRAME_ID = -77001
_LORRI_CK_FRAME_ID = -98000

# What cspyce raises when the furnished kernels cannot answer: a missing frame
# or an unresolvable clock arrives as a LookupError, an unreadable kernel as
# an OSError, and a SPICE error as a RuntimeError or a ValueError.  Each is an
# environment failure rather than a defect, so it is converted to a
# NavPointingError at the call site that made it.
_SPICE_FAILURES = (LookupError, OSError, RuntimeError, ValueError)

_IDENTITY: NDArrayFloatType = np.eye(3)
_IDENTITY.setflags(write=False)

# oops rotates the Cassini ISS camera frames 180 degrees about the boresight
# because the instrument's internal coordinate system is turned that way.
_CASSINI_OOPS_FROM_SPICE: NDArrayFloatType = np.diag([-1.0, -1.0, 1.0])
_CASSINI_OOPS_FROM_SPICE.setflags(write=False)

# The LORRI SPICE boresight is -Z; oops flips Y and Z to put it on +Z.
_LORRI_OOPS_FROM_SPICE: NDArrayFloatType = np.diag([1.0, -1.0, -1.0])
_LORRI_OOPS_FROM_SPICE.setflags(write=False)

# Angular tolerance, in matrix-element terms, for the measured flip matching
# its expected constant value and for that value being epoch-independent.
_FLIP_TOL = 1e-9

# A C-matrix must be a proper rotation to this tolerance; anything looser is a
# defect in the frame chain rather than something to orthonormalize away.
_ROTATION_TOL = 1e-9

# Below this cross-product norm the corrected and uncorrected boresights are
# the same direction to sub-nanoradian precision and the correction is exactly
# the identity.
_DEGENERATE_AXIS_NORM = 1e-12


@dataclass(frozen=True)
class _FrameIdentity:
    """Per-instrument frame facts needed to place a correction in SPICE terms.

    Parameters:
        camera_frame: SPICE name of the frame the observation's boresight is
            expressed in.
        ck_frame_id: SPICE id of the object a corrected C-kernel targets.
        sclk_id: SPICE id of the spacecraft clock that object's time tags are
            encoded against.
        oops_from_spice: The constant rotation ``R`` relating the oops
            observation frame to the SPICE camera frame.
        frozen_oops_attitude: True when oops freezes the observation frame
            from a tolerance-snapped pointing lookup, so the SPICE baseline is
            the observation frame itself rather than a ``pxform`` evaluation.
    """

    camera_frame: str
    ck_frame_id: int
    sclk_id: int
    oops_from_spice: NDArrayFloatType
    frozen_oops_attitude: bool


@dataclass(frozen=True)
class AttitudeBaseline:
    """The SPICE-derived facts about one observation, independent of any offset.

    Parameters:
        cmatrix_original: Uncorrected J2000-to-camera rotation at the exposure
            midtime, in the SPICE camera frame convention.
        oops_from_spice: The measured constant rotation ``R`` satisfying
            ``C_oops = R . C_spice``.
        camera_frame: SPICE name of the camera frame.
        camera_frame_id: SPICE id of the camera frame.
        ck_frame_id: SPICE id of the object a corrected C-kernel targets.
        start_et: Exposure start, TDB seconds past J2000.
        stop_et: Exposure stop, TDB seconds past J2000.
        midtime_et: Exposure midtime, TDB seconds past J2000.
        exposure_s: Exposure duration in seconds.
        sclk_start: Spacecraft clock string at ``start_et``.
        sclk_midtime: Spacecraft clock string at ``midtime_et``.
        sclk_stop: Spacecraft clock string at ``stop_et``.
    """

    cmatrix_original: NDArrayFloatType
    oops_from_spice: NDArrayFloatType
    camera_frame: str
    camera_frame_id: int
    ck_frame_id: int
    start_et: float
    stop_et: float
    midtime_et: float
    exposure_s: float
    sclk_start: str
    sclk_midtime: str
    sclk_stop: str

    def __post_init__(self) -> None:
        """Store both matrices as read-only 3x3 float arrays."""
        object.__setattr__(self, 'cmatrix_original', _as_readonly_3x3(self.cmatrix_original))
        object.__setattr__(self, 'oops_from_spice', _as_readonly_3x3(self.oops_from_spice))


@dataclass(frozen=True)
class PointingSolution:
    """A navigated image's baseline attitude plus its corrected attitude.

    Parameters:
        baseline: The uncorrected attitude, frame identities, and times.
        cmatrix: Corrected J2000-to-camera rotation at the exposure midtime,
            in the SPICE camera frame convention.  ``None`` when the result
            carried no offset, or carried a fitted camera rotation whose pivot
            is not recorded and therefore cannot be expressed as an attitude.
    """

    baseline: AttitudeBaseline
    cmatrix: NDArrayFloatType | None

    def __post_init__(self) -> None:
        """Store the corrected matrix as a read-only 3x3 float array."""
        if self.cmatrix is not None:
            object.__setattr__(self, 'cmatrix', _as_readonly_3x3(self.cmatrix))


def _as_readonly_3x3(matrix: NDArrayFloatType) -> NDArrayFloatType:
    """Copy ``matrix`` into a read-only 3x3 float64 array.

    Parameters:
        matrix: Any 3x3 array-like of numbers.

    Returns:
        A read-only float64 copy.

    Raises:
        NavPointingError: if the input is not 3x3.
    """
    out = np.array(matrix, dtype=np.float64)
    if out.shape != (3, 3):
        raise NavPointingError(f'expected a 3x3 matrix; got shape {out.shape}')
    out.setflags(write=False)
    return out


def _validate_rotation(matrix: NDArrayFloatType, label: str) -> None:
    """Raise unless ``matrix`` is a proper rotation to ``_ROTATION_TOL``.

    Parameters:
        matrix: The 3x3 matrix to check.
        label: Name used in the exception message.

    Raises:
        NavPointingError: if the matrix holds a non-finite value, or if its
            determinant differs from 1, or it is not orthonormal, by more than
            the tolerance.
    """
    # NaN fails every inequality below, so both tolerance guards would pass a
    # NaN matrix silently; and a non-finite value serialized into the metadata
    # is written as a bare NaN / Infinity token that strict JSON parsers
    # reject.  Reject it here, where the defect is still attributable.
    if not bool(np.all(np.isfinite(matrix))):
        raise NavPointingError(f'{label} holds a non-finite value: {np.asarray(matrix).tolist()!r}')
    det = float(np.linalg.det(matrix))
    if abs(det - 1.0) > _ROTATION_TOL:
        raise NavPointingError(
            f'{label} is not a proper rotation: determinant {det!r} differs from 1 by more '
            f'than {_ROTATION_TOL}'
        )
    residual = float(np.max(np.abs(matrix @ matrix.T - np.eye(3))))
    if residual > _ROTATION_TOL:
        raise NavPointingError(
            f'{label} is not orthonormal: max|C C^T - I| = {residual!r} exceeds {_ROTATION_TOL}'
        )


def _oops_correction_matrix(fov: oops.fov.FOV, offset_px: tuple[float, float]) -> NDArrayFloatType:
    """Build the pointing correction a navigated offset implies, in oops terms.

    The offset is applied downstream as ``oops.fov.OffsetFOV(fov,
    uv_offset=(du, dv))``, which maps pixels to tangent-plane coordinates as
    ``fov.xy_from_uv(uv) - xy_offset`` with ``xy_offset =
    fov.xy_from_uv(fov.uv_los + (du, dv))``.  Under the corrected pointing the
    true direction seen by pixel ``uv`` in the *original* frame is therefore
    ``fov.los_from_xy(fov.xy_from_uv(uv) - xy_offset)``, and the corrected
    frame is the one in which the unmodified FOV holds.  Evaluating that at
    the boresight gives the returned rotation ``M``, the minimal rotation
    taking the uncorrected boresight direction to the corrected one.

    ``M`` is an active vector rotation: ``M . d`` is the corrected direction
    ``b``, where ``d`` is the direction the uncorrected FOV assigns to the
    boresight pixel.  It is exact by construction; nothing is orthonormalized.
    An exact rigid rotation is not exactly a uniform tangent-plane shift, so
    away from the boresight the rotation and the pixel offset differ at second
    order in field angle.

    Parameters:
        fov: The observation's unmodified oops FOV.
        offset_px: The navigated ``(dv, du)`` offset in pixels.

    Returns:
        The 3x3 rotation ``M`` in oops observation frame coordinates.  Exactly
        the identity when the offset moves the boresight by less than a
        sub-nanoradian.
    """
    dv, du = float(offset_px[0]), float(offset_px[1])
    uv_los = fov.uv_los
    xy_los = fov.xy_from_uv(uv_los)
    xy_offset = fov.xy_from_uv(oops.Pair((uv_los.vals[0] + du, uv_los.vals[1] + dv)))
    uncorrected = np.asarray(fov.los_from_xy(xy_los - xy_offset).unit().vals, dtype=np.float64)
    corrected = np.asarray(fov.los_from_xy(xy_los).unit().vals, dtype=np.float64)
    axis = np.cross(uncorrected, corrected)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < _DEGENERATE_AXIS_NORM:
        return _IDENTITY
    # arctan2(|d x b|, d . b) is the well-conditioned form of arccos(d . b)
    # for unit vectors: arccos loses relative precision as the angle goes to
    # zero, which is exactly the regime a sub-pixel offset lives in.
    angle = float(np.arctan2(axis_norm, float(np.dot(uncorrected, corrected))))
    correction: NDArrayFloatType = np.asarray(
        cspyce.axisar(axis / axis_norm, angle), dtype=np.float64
    )
    return correction


def _spice_cmatrix(
    cmatrix_original: NDArrayFloatType,
    correction: NDArrayFloatType,
    oops_from_spice: NDArrayFloatType,
) -> NDArrayFloatType:
    """Apply an oops-frame correction to a SPICE-convention C-matrix.

    With ``R`` the constant rotation satisfying ``C_oops = R . C_spice`` and
    ``M`` the correction expressed in oops observation frame coordinates, the
    corrected attitude in the SPICE convention is ``(R^T . M . R) . C``.
    Skipping the conjugation yields a proper rotation of the right magnitude
    pointing the wrong way.

    Parameters:
        cmatrix_original: Uncorrected J2000-to-camera rotation, SPICE
            convention.
        correction: The oops-frame correction ``M``.
        oops_from_spice: The constant rotation ``R``.

    Returns:
        The corrected 3x3 J2000-to-camera rotation in the SPICE convention.
        Bit-identical to ``cmatrix_original`` when the correction is exactly
        the identity, since no correction must mean no change.

    Raises:
        NavPointingError: if ``cmatrix_original`` or the corrected result is
            not a proper orthonormal rotation.
    """
    original = np.asarray(cmatrix_original, dtype=np.float64)
    _validate_rotation(original, 'cmatrix_original')
    if np.array_equal(correction, np.eye(3)):
        return original
    rotation = np.asarray(oops_from_spice, dtype=np.float64)
    conjugated = rotation.T @ np.asarray(correction, dtype=np.float64) @ rotation
    corrected: NDArrayFloatType = conjugated @ original
    _validate_rotation(corrected, 'cmatrix')
    return corrected


def _build_pointing_solution(
    baseline: AttitudeBaseline,
    fov: oops.fov.FOV,
    *,
    offset_px: tuple[float, float] | None,
    rotation_fitted: bool,
) -> PointingSolution:
    """Combine a SPICE baseline and a navigated offset into a PointingSolution.

    The baseline is always carried through.  A corrected ``cmatrix`` is
    produced only when the navigation yielded an offset and did not fit a
    camera rotation: a fitted rotation turns about a per-technique pivot that
    no result records, so the correction it implies is not expressible from
    the recorded data and no corrected attitude is claimed.

    Parameters:
        baseline: The observation's uncorrected attitude, frames, and times.
        fov: The observation's unmodified oops FOV.
        offset_px: The navigated ``(dv, du)`` offset, or ``None`` when the
            navigation produced no offset.
        rotation_fitted: True when the result carries a fitted camera
            rotation.

    Returns:
        A PointingSolution whose ``cmatrix`` is ``None`` unless an offset was
        supplied with no fitted rotation.

    Raises:
        NavPointingError: if the baseline or the corrected matrix is not a
            proper orthonormal rotation.
    """
    if offset_px is None or rotation_fitted:
        _validate_rotation(np.asarray(baseline.cmatrix_original, np.float64), 'cmatrix_original')
        return PointingSolution(baseline=baseline, cmatrix=None)
    correction = _oops_correction_matrix(fov, offset_px)
    cmatrix = _spice_cmatrix(baseline.cmatrix_original, correction, baseline.oops_from_spice)
    return PointingSolution(baseline=baseline, cmatrix=cmatrix)


def compute_pointing(
    obs: ObsSnapshotInst,
    *,
    offset_px: tuple[float, float] | None,
    rotation_fitted: bool,
) -> PointingSolution | None:
    """Compute the recorded and corrected attitudes for one navigated image.

    This is the single entry point for the offset-to-attitude conversion.  It
    reads the observation's own frame and FOV, measures the constant rotation
    between the oops observation frame and the SPICE camera frame, checks that
    rotation against the constant expected for the instrument, and returns
    both attitudes in the SPICE convention along with the frame identities and
    the exposure times a C-kernel writer needs.

    Every expected failure is reported as a ``NavPointingError``, so a caller
    that must survive an attitude the environment cannot supply absorbs that
    one exception and nothing else: any other exception escaping this function
    is a defect in it.

    Parameters:
        obs: The navigated observation.
        offset_px: The navigated ``(dv, du)`` offset, or ``None``.
        rotation_fitted: True when the result carries a fitted camera
            rotation.

    Returns:
        A PointingSolution, or ``None`` when the observation's host is not one
        whose SPICE frames this module knows (a simulated image, for example).

    Raises:
        NavPointingError: if the furnished kernels cannot supply the attitude,
            the camera frame or the spacecraft clock; if the resolved
            spacecraft clock is not the one expected for the mission; if the
            measured flip between the oops and SPICE frames differs from the
            constant expected for the instrument or varies across the
            exposure; or if either C-matrix is not a proper orthonormal
            rotation.
    """
    identity = _frame_identity(obs)
    if identity is None:
        return None
    baseline = _attitude_baseline(obs, identity)
    return _build_pointing_solution(
        baseline, obs.fov, offset_px=offset_px, rotation_fitted=rotation_fitted
    )


def _ck_object_sclk_id(ck_frame_id: int) -> int:
    """Return the spacecraft clock a CK object's time tags are encoded against.

    The value is read from the shared mapping the C-kernel writer reads too,
    so that the clock a correction is timed with and the clock it is written
    with cannot disagree.

    Parameters:
        ck_frame_id: SPICE id of the object a corrected C-kernel targets.

    Returns:
        The spacecraft clock id recorded for that object.

    Raises:
        NavPointingError: if no clock is recorded for the object.
    """
    if ck_frame_id not in CK_OBJECT_SCLK_ID:
        raise NavPointingError(
            f'CK object {ck_frame_id} has no recorded spacecraft clock; expected one of '
            f'{sorted(CK_OBJECT_SCLK_ID)}'
        )
    return CK_OBJECT_SCLK_ID[ck_frame_id]


def _frame_identity(obs: ObsSnapshotInst) -> _FrameIdentity | None:
    """Return the SPICE frame facts for an observation's instrument.

    Parameters:
        obs: The observation to identify.

    Returns:
        The instrument's ``_FrameIdentity``, or ``None`` for a host with no
        SPICE camera frame this module knows.

    Raises:
        NavPointingError: if the instrument's CK object has no recorded
            spacecraft clock.
    """
    if isinstance(obs, ObsCassiniISS):
        return _FrameIdentity(
            camera_frame=f'CASSINI_ISS_{obs.camera}',
            ck_frame_id=_CASSINI_CK_FRAME_ID,
            sclk_id=_ck_object_sclk_id(_CASSINI_CK_FRAME_ID),
            oops_from_spice=_CASSINI_OOPS_FROM_SPICE,
            frozen_oops_attitude=False,
        )
    if isinstance(obs, ObsGalileoSSI):
        return _FrameIdentity(
            camera_frame='GLL_SCAN_PLATFORM',
            ck_frame_id=_GALILEO_CK_FRAME_ID,
            sclk_id=_ck_object_sclk_id(_GALILEO_CK_FRAME_ID),
            oops_from_spice=_IDENTITY,
            frozen_oops_attitude=False,
        )
    if isinstance(obs, ObsNewHorizonsLORRI):
        return _FrameIdentity(
            camera_frame='NH_LORRI',
            ck_frame_id=_LORRI_CK_FRAME_ID,
            sclk_id=_ck_object_sclk_id(_LORRI_CK_FRAME_ID),
            oops_from_spice=_LORRI_OOPS_FROM_SPICE,
            frozen_oops_attitude=False,
        )
    if isinstance(obs, ObsVoyagerISS):
        digit = obs.spacecraft_digit
        # One instrument key serves two spacecraft, so the CK object is derived
        # from which spacecraft this is; its clock then follows from the object
        # rather than from the same digit, so a wrong derivation is refused
        # instead of producing a self-consistent wrong pair.
        ck_frame_id = -(30 + int(digit)) * 1000 - 100
        # The Voyager FK spells the cameras ISSNA and ISSWA, so the oops
        # detector names NAC and WAC contribute only their first letter.
        return _FrameIdentity(
            camera_frame=f'VG{digit}_ISS{obs.camera[0]}A',
            ck_frame_id=ck_frame_id,
            sclk_id=_ck_object_sclk_id(ck_frame_id),
            oops_from_spice=_IDENTITY,
            frozen_oops_attitude=True,
        )
    return None


def _observation_attitude(obs: ObsSnapshotInst, et: float) -> NDArrayFloatType:
    """Return the observation frame's J2000-to-frame rotation at one epoch.

    Parameters:
        obs: The observation whose frame is evaluated.
        et: TDB seconds past J2000.

    Returns:
        The 3x3 rotation in the oops observation frame convention.

    Raises:
        NavPointingError: if the furnished kernels cannot place the frame at
            this epoch.
    """
    try:
        transform = obs.frame.wrt(oops.frame.Frame.J2000).transform_at_time(et)
    except _SPICE_FAILURES as exc:
        raise NavPointingError(
            f'the observation frame has no attitude at et {et!r}: {exc}'
        ) from exc
    return np.asarray(transform.matrix.vals, dtype=np.float64).reshape(3, 3)


def _attitude_baseline(obs: ObsSnapshotInst, identity: _FrameIdentity) -> AttitudeBaseline:
    """Read the uncorrected attitude, frame ids, and exposure times from SPICE.

    Parameters:
        obs: The observation to read.
        identity: The instrument's frame facts.

    Returns:
        The observation's AttitudeBaseline.

    Raises:
        NavPointingError: if the furnished kernels cannot supply the attitude,
            the camera frame id or the clock strings; if the resolved
            spacecraft clock is not the instrument's; or if the measured flip
            between the oops observation frame and the SPICE camera frame
            differs from the instrument's expected constant or varies across
            the exposure.
    """
    start_et = float(obs.time[0])
    stop_et = float(obs.time[1])
    midtime_et = float(obs.midtime)
    if identity.frozen_oops_attitude:
        # oops froze this frame from a tolerance-snapped pointing lookup, so a
        # pxform at the midtime does not reproduce it; the observation frame
        # attitude already is the SPICE camera attitude that was navigated.
        cmatrix_original = _observation_attitude(obs, midtime_et)
        oops_from_spice = _IDENTITY
    else:
        cmatrix_original = _pxform(identity.camera_frame, midtime_et)
        oops_from_spice = _observation_attitude(obs, midtime_et) @ cmatrix_original.T
        _check_flip(oops_from_spice, identity)
        for et in (start_et, stop_et):
            at_epoch = _observation_attitude(obs, et) @ _pxform(identity.camera_frame, et).T
            if not np.allclose(at_epoch, oops_from_spice, rtol=0.0, atol=_FLIP_TOL):
                raise NavPointingError(
                    f'the rotation between the oops and SPICE {identity.camera_frame} frames '
                    f'is not constant across the exposure: it differs by up to '
                    f'{float(np.max(np.abs(at_epoch - oops_from_spice)))!r} between et {et!r} '
                    f'and the midtime {midtime_et!r}'
                )
    sclk_id = _sclk_id(identity)
    return AttitudeBaseline(
        cmatrix_original=cmatrix_original,
        oops_from_spice=oops_from_spice,
        camera_frame=identity.camera_frame,
        camera_frame_id=_camera_frame_id(identity.camera_frame),
        ck_frame_id=identity.ck_frame_id,
        start_et=start_et,
        stop_et=stop_et,
        midtime_et=midtime_et,
        exposure_s=float(obs.texp),
        sclk_start=_sclk_string(sclk_id, start_et),
        sclk_midtime=_sclk_string(sclk_id, midtime_et),
        sclk_stop=_sclk_string(sclk_id, stop_et),
    )


def _sclk_id(identity: _FrameIdentity) -> int:
    """Resolve the spacecraft clock for a CK object and check it is the right one.

    ``cspyce.ckmeta`` computes a clock id from a CK object id rather than
    validating it: it answers ``-999`` for the nonexistent object ``-999999``
    and ``-12`` for ``-12345``, raising for neither.  A CK object that is
    wrong for any reason therefore yields a plausible clock id, and every
    clock string built from it encodes the wrong spacecraft's time while
    ``sce2s`` reports success.  The resolved id is checked against the one the
    mission actually uses before any clock string is built.

    Parameters:
        identity: The instrument's frame facts, carrying both the CK object
            and the spacecraft clock expected for it.

    Returns:
        The resolved spacecraft clock id, equal to ``identity.sclk_id``.

    Raises:
        NavPointingError: if no clock resolves for the CK object, or if the
            one that resolves is not the instrument's.
    """
    try:
        resolved = int(cspyce.ckmeta(identity.ck_frame_id, 'SCLK'))
    except _SPICE_FAILURES as exc:
        raise NavPointingError(
            f'no spacecraft clock resolves for CK object {identity.ck_frame_id}: {exc}'
        ) from exc
    if resolved != identity.sclk_id:
        raise NavPointingError(
            f'CK object {identity.ck_frame_id} resolves to spacecraft clock {resolved}, not the '
            f'{identity.sclk_id} the {identity.camera_frame} camera is tagged against; every '
            f'clock string built from it would encode the wrong spacecraft'
        )
    # The recorded id is returned, not the one ``ckmeta`` computed, even though
    # the check above has just proved them equal.  ``ckmeta`` answers for
    # objects that do not exist, so it is a cross-check here and never the
    # source: if this check is ever weakened, the clock strings still come from
    # the recorded table rather than from whatever ``ckmeta`` returned.
    return identity.sclk_id


def _sclk_string(sclk_id: int, et: float) -> str:
    """Encode one epoch as a spacecraft clock string.

    Parameters:
        sclk_id: SPICE id of the spacecraft clock.
        et: TDB seconds past J2000.

    Returns:
        The clock string ``sce2s`` produces for that epoch.

    Raises:
        NavPointingError: if the furnished kernels cannot encode the epoch.
    """
    try:
        return str(cspyce.sce2s(sclk_id, et))
    except _SPICE_FAILURES as exc:
        raise NavPointingError(
            f'spacecraft clock {sclk_id} cannot encode et {et!r}: {exc}'
        ) from exc


def _camera_frame_id(camera_frame: str) -> int:
    """Look up the SPICE id of a named frame.

    Parameters:
        camera_frame: SPICE name of the camera frame.

    Returns:
        The frame's SPICE id.

    Raises:
        NavPointingError: if the name resolves to no id in the kernel pool.
    """
    try:
        return int(cspyce.namfrm(camera_frame))
    except _SPICE_FAILURES as exc:
        raise NavPointingError(f'the frame {camera_frame} has no SPICE id: {exc}') from exc


def _pxform(camera_frame: str, et: float) -> NDArrayFloatType:
    """Evaluate the J2000-to-camera rotation from the furnished kernels.

    Parameters:
        camera_frame: SPICE name of the camera frame.
        et: TDB seconds past J2000.

    Returns:
        The 3x3 rotation ``pxform('J2000', camera_frame, et)`` as float64.

    Raises:
        NavPointingError: if the furnished kernels cannot supply the rotation
            at this epoch.
    """
    try:
        matrix = cspyce.pxform('J2000', camera_frame, et)
    except _SPICE_FAILURES as exc:
        raise NavPointingError(
            f'the furnished kernels cannot supply the J2000 to {camera_frame} rotation at '
            f'et {et!r}: {exc}'
        ) from exc
    return np.asarray(matrix, dtype=np.float64)


def _check_flip(measured: NDArrayFloatType, identity: _FrameIdentity) -> None:
    """Raise unless the measured oops-to-SPICE flip is the expected constant.

    Parameters:
        measured: The rotation ``R`` measured from the observation.
        identity: The instrument's frame facts, carrying the expected ``R``.

    Raises:
        NavPointingError: if the two differ by more than ``_FLIP_TOL``.
    """
    expected = np.asarray(identity.oops_from_spice, dtype=np.float64)
    if not np.allclose(measured, expected, rtol=0.0, atol=_FLIP_TOL):
        raise NavPointingError(
            f'the rotation between the oops observation frame and the SPICE '
            f'{identity.camera_frame} frame is {measured.tolist()!r}, which differs from the '
            f'expected {expected.tolist()!r} by up to '
            f'{float(np.max(np.abs(measured - expected)))!r}'
        )
