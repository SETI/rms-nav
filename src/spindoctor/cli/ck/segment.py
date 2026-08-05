"""Type-3 C-kernel segments carrying one navigated image's corrected pointing.

Navigation measures the correction at the camera, but the C-kernels that exist
describe a bus or a scan platform, and that is where the correction has to be
written.  With ``F = pxform(ck_frame, camera_frame, midtime)`` -- always
computed, never assumed, because Cassini's ``F`` is a permutation-like matrix
nowhere near the identity -- the corrected attitude of the CK object and the
correction expressed in that object's own coordinates are::

    C_ck_corrected(mid) = F^-1 . cmatrix
    delta               = C_ck_corrected(mid) . C_ck_original(mid)^T

with ``C_ck_original`` read from the baseline kernels the caller has furnished.
Across the exposure the correction is held body-fixed::

    C_ck_corrected(t) = delta . C_ck_original(t)

which is the physical model -- the spacecraft is pointed slightly wrong and the
error turns with it -- so attitude still varies correctly inside the exposure
and smear geometry stays right.

Voyager is the exception.  Its navigated attitude came from a frozen,
tolerance-snapped pointing lookup rather than a time-varying frame chain, so a
Voyager segment carries the single corrected attitude, constant across its
window; writing time-varying pointing there would disagree with what was
navigated.

Angular velocity is copied unchanged and never rotated.  CK angular velocity is
expressed in the segment's base reference frame (J2000), and two frames rigidly
attached to each other have identical angular velocity in that frame, so
rotating it through ``delta`` -- superficially the thorough treatment -- writes
a vector in no frame at all.  A segment carries one angular velocity flag for
all of its records, so a baseline that lacks angular velocity at any record --
including an exposure straddling one baseline segment that has it and one that
does not -- yields a segment that carries none, and consumers fall back to
``ckgp``.

The caller owns the kernel pool: the supporting kernels (LSK, SCLK, FK) and the
one baseline C-kernel whose attitude the image was navigated against must be
furnished before ``build_segment`` is called, and the SCLK must be the one
navigation used, since a different one is a silent time-tag error.
"""

import math
from dataclasses import dataclass

import cspyce
import numpy as np

from spindoctor.cli.ck.pointing import ImagePointing, NDArrayFloatType
from spindoctor.spice_ids import CK_OBJECT_SCLK_ID

# The CK objects whose navigated attitude was a frozen, tolerance-snapped
# lookup rather than an evaluated frame chain.  Their segments carry one
# constant attitude across the exposure.
_FROZEN_ATTITUDE_CK_IDS = frozenset({-31100, -32100})

# Every segment is written relative to J2000, which is what the recorded
# C-matrices are referenced to.
_BASE_FRAME = 'J2000'

# Records go at the exposure start, midtime and stop.  An exposure longer than
# this gets additional records at the cadence below, so that a long exposure's
# attitude history is not reduced to three points.
_LONG_EXPOSURE_S = 10.0
_RECORD_CADENCE_S = 1.0

# Baseline pointing is read at the record epoch itself, not at whatever nearby
# epoch a tolerance would admit.
_LOOKUP_TOL_TICKS = 0.0

# One interpolation interval per segment, spanning the exposure exactly.
_INTERVAL_COUNT = 1

# SPICE segment identifiers hold at most this many characters.
_SEGID_MAX_CHARS = 40


@dataclass(frozen=True)
class CkSegment:
    """One type-3 C-kernel segment, ready to be written.

    Parameters:
        ck_frame_id: SPICE id of the object the segment describes.
        segid: Segment identifier, at most 40 characters.
        sclkdp: Encoded SCLK time tags, shape ``(n,)``, strictly increasing.
        quats: SPICE quaternions of the attitude at each time tag, shape
            ``(n, 4)``, sign-continuous from one record to the next.
        avvs: Angular velocity at each time tag in the segment's base frame
            (J2000), shape ``(n, 3)``, or ``None`` when the segment carries no
            angular velocity.

    Raises:
        ValueError: if the arrays disagree on the record count, have the wrong
            width, hold no records, or if the time tags are not strictly
            increasing.
    """

    ck_frame_id: int
    segid: str
    sclkdp: NDArrayFloatType
    quats: NDArrayFloatType
    avvs: NDArrayFloatType | None

    def __post_init__(self) -> None:
        """Refuse a record set SPICE would reject, and store it read-only.

        The arrays are stored as read-only float64 copies, so the invariants
        checked here still hold when the segment is written: a caller that
        keeps a reference to the array it passed in cannot edit the records
        out from under them.
        """
        sclkdp = np.array(self.sclkdp, dtype=np.float64)
        quats = np.array(self.quats, dtype=np.float64)
        if sclkdp.ndim != 1 or sclkdp.size == 0:
            raise ValueError(f'sclkdp must hold at least one time tag; got shape {sclkdp.shape}')
        count = sclkdp.shape[0]
        if quats.shape != (count, 4):
            raise ValueError(f'quats must have shape {(count, 4)}; got {quats.shape}')
        avvs = None if self.avvs is None else np.array(self.avvs, dtype=np.float64)
        if avvs is not None and avvs.shape != (count, 3):
            raise ValueError(f'avvs must have shape {(count, 3)}; got {avvs.shape}')
        if len(self.segid) > _SEGID_MAX_CHARS:
            raise ValueError(
                f'segment id {self.segid!r} is longer than the {_SEGID_MAX_CHARS} characters '
                f'SPICE stores'
            )
        gaps = np.diff(sclkdp)
        if count > 1 and float(np.min(gaps)) <= 0.0:
            raise ValueError(
                f'encoded SCLK time tags are not strictly increasing: smallest step is '
                f'{float(np.min(gaps))!r}'
            )
        sclkdp.setflags(write=False)
        quats.setflags(write=False)
        object.__setattr__(self, 'sclkdp', sclkdp)
        object.__setattr__(self, 'quats', quats)
        if avvs is not None:
            avvs.setflags(write=False)
            object.__setattr__(self, 'avvs', avvs)

    @property
    def record_count(self) -> int:
        """Number of records in the segment."""
        return int(self.sclkdp.shape[0])

    @property
    def has_angular_velocity(self) -> bool:
        """True when the segment carries angular velocity."""
        return self.avvs is not None

    @property
    def begtim(self) -> float:
        """Encoded SCLK at which the segment's coverage begins.

        This is the first record's time tag, and the segment advertises no
        coverage before it.
        """
        return float(self.sclkdp[0])

    @property
    def endtim(self) -> float:
        """Encoded SCLK at which the segment's coverage ends.

        This is the last record's time tag, and the segment advertises no
        coverage after it.
        """
        return float(self.sclkdp[-1])


def resolve_sclk_id(ck_frame_id: int) -> int:
    """Return the spacecraft clock a CK object's time tags are encoded against.

    The id comes from ``cspyce.ckmeta`` and is then checked against the clock
    recorded for the object, because ``ckmeta`` computes rather than validates:
    it answers for objects that do not exist, so an unnoticed wrong CK id would
    produce a wrong clock, a successful encoding, and silently wrong time tags.
    The recorded clocks are the ones the attitude computation checks against
    too, so the two cannot drift apart.

    Parameters:
        ck_frame_id: SPICE id of the object a corrected C-kernel targets.

    Returns:
        The spacecraft clock id, for example -82 for the Cassini bus (-82000).

    Raises:
        ValueError: if the object is not one this writer knows, or if the id
            ``ckmeta`` resolves is not the one expected for it.
    """
    if ck_frame_id not in CK_OBJECT_SCLK_ID:
        raise ValueError(
            f'CK object {ck_frame_id} is not one this writer knows; expected one of '
            f'{sorted(CK_OBJECT_SCLK_ID)}'
        )
    expected = CK_OBJECT_SCLK_ID[ck_frame_id]
    sclk_id = int(cspyce.ckmeta(ck_frame_id, 'SCLK'))
    if sclk_id != expected:
        raise ValueError(
            f'CK object {ck_frame_id} resolves to spacecraft clock {sclk_id}, not the expected '
            f'{expected}'
        )
    return sclk_id


def build_segment(pointing: ImagePointing) -> CkSegment:
    """Build the corrected type-3 segment for one navigated image.

    The caller must already have furnished the supporting kernels: the
    spacecraft clock navigation used, and the frame kernel defining the CK
    object's frame and the camera frame.  Every object but a frozen-attitude
    one also needs the baseline C-kernel the image navigated against, read at
    the exposure midtime and at every record epoch; a frozen-attitude object's
    segment carries one constant attitude and never reads the baseline.

    Records go at the exposure start, midtime and stop, plus a 1 s cadence when
    the exposure is longer than 10 s, each encoded with ``cspyce.sce2c``.  Time
    tags that do not strictly increase are dropped, and epochs that all encode
    to one tick yield a single record at the midtime.  Since ``sce2c`` encodes
    a fractional tick, that happens only for an exposure whose start, midtime
    and stop are one floating-point value, not merely for an exposure shorter
    than a tick.

    Parameters:
        pointing: The image's recorded corrected pointing.

    Returns:
        The segment to write, carrying the baseline's angular velocity
        unchanged when the baseline has it at every record, and none when the
        baseline lacks it anywhere.

    Raises:
        ValueError: if the CK object is not one this writer knows, if the
            resolved spacecraft clock is not the expected one, or if the image
            name does not fit a SPICE segment identifier.
        OSError: if the furnished kernels provide no pointing for the CK object
            at the exposure midtime or at a record epoch.
        KeyError: if the CK object has no frame name in the furnished kernels.
    """
    segid = _segment_id(pointing.image_name)
    sclk_id = resolve_sclk_id(pointing.ck_frame_id)
    ticks = _record_ticks(pointing, sclk_id)
    corrected_midtime = _corrected_attitude_at_midtime(pointing)
    attitudes: list[NDArrayFloatType]
    avvs: NDArrayFloatType | None
    if pointing.ck_frame_id in _FROZEN_ATTITUDE_CK_IDS:
        # The navigated model assumed one snapped attitude across the whole
        # exposure, so the segment says exactly that.  It carries no angular
        # velocity: a constant attitude has none, and the rigid-attachment
        # argument that lets a body-fixed segment copy the baseline's angular
        # velocity does not hold for a segment that deliberately drops the
        # baseline's time variation.
        attitudes = [corrected_midtime] * len(ticks)
        avvs = None
    else:
        baseline_midtime = _baseline_attitude(
            pointing.ck_frame_id, float(cspyce.sce2c(sclk_id, pointing.midtime_et))
        )
        delta = corrected_midtime @ baseline_midtime.T
        attitudes, avvs = _corrected_history(pointing.ck_frame_id, ticks, delta)
    return CkSegment(
        ck_frame_id=pointing.ck_frame_id,
        segid=segid,
        sclkdp=np.asarray(ticks, dtype=np.float64),
        quats=_quaternion_sequence(attitudes),
        avvs=avvs,
    )


def write_segment(handle: int, segment: CkSegment) -> None:
    """Add one type-3 segment to an open C-kernel.

    The segment is descriptor-bounded and interpolation-bounded by its own
    records: its begin and end times are the first and last time tag, and its
    one interpolation interval starts at the first.  A consumer asking
    ``ckcov`` what the file covers is therefore told exactly the exposure, and
    is never handed interpolated pointing outside the window the correction was
    measured over.

    Parameters:
        handle: Handle of a C-kernel opened for writing, from ``cspyce.ckopn``.
        segment: The segment to add.

    Raises:
        OSError: if SPICE refuses the write, for example because the handle is
            not open for writing or the file cannot be extended.  The record
            set itself is already valid: ``CkSegment`` enforces the count,
            width and strictly-increasing invariants when it is constructed.
    """
    avvs = segment.avvs
    if avvs is None:
        # ckw03 wants an array either way; with avflag false it is ignored.
        avvs = np.zeros((segment.record_count, 3), dtype=np.float64)
    cspyce.ckw03(
        handle,
        segment.begtim,
        segment.endtim,
        segment.ck_frame_id,
        _BASE_FRAME,
        segment.has_angular_velocity,
        segment.segid,
        np.asarray(segment.sclkdp, dtype=np.float64),
        np.asarray(segment.quats, dtype=np.float64),
        np.asarray(avvs, dtype=np.float64),
        _INTERVAL_COUNT,
        [segment.begtim],
    )


def _segment_id(image_name: str) -> str:
    """Return the segment identifier naming one image.

    Parameters:
        image_name: Basename of the navigated image.

    Returns:
        The identifier to store in the segment.

    Raises:
        ValueError: if the name is longer than a SPICE segment identifier,
            since truncating it would silently lose the image's identity.
    """
    if len(image_name) > _SEGID_MAX_CHARS:
        raise ValueError(
            f'image name {image_name!r} is longer than the {_SEGID_MAX_CHARS} characters a '
            f'SPICE segment identifier holds'
        )
    return image_name


def _record_epochs(pointing: ImagePointing) -> list[float]:
    """Return the epochs a segment carries records at, in increasing order.

    Parameters:
        pointing: The image's recorded corrected pointing.

    Returns:
        The exposure start, midtime and stop, plus interior epochs at
        ``_RECORD_CADENCE_S`` when the exposure exceeds ``_LONG_EXPOSURE_S``.
    """
    epochs = [pointing.start_et, pointing.midtime_et, pointing.stop_et]
    if pointing.exposure_s > _LONG_EXPOSURE_S:
        steps = math.floor((pointing.stop_et - pointing.start_et) / _RECORD_CADENCE_S)
        epochs.extend(pointing.start_et + step * _RECORD_CADENCE_S for step in range(1, steps + 1))
    return sorted(epochs)


def _record_ticks(pointing: ImagePointing, sclk_id: int) -> list[float]:
    """Encode the record epochs, keeping only strictly increasing time tags.

    Parameters:
        pointing: The image's recorded corrected pointing.
        sclk_id: The spacecraft clock the tags are encoded against.

    Returns:
        The encoded SCLK time tags, strictly increasing.  A single tag at the
        exposure midtime when the epochs all encode to one tick, which type 3
        permits.
    """
    ticks: list[float] = []
    for epoch in _record_epochs(pointing):
        tick = float(cspyce.sce2c(sclk_id, epoch))
        if len(ticks) == 0 or tick > ticks[-1]:
            ticks.append(tick)
    if len(ticks) < 2:
        return [float(cspyce.sce2c(sclk_id, pointing.midtime_et))]
    return ticks


def _corrected_attitude_at_midtime(pointing: ImagePointing) -> NDArrayFloatType:
    """Return the corrected attitude of the CK object at the exposure midtime.

    The recorded C-matrix is the corrected attitude of the *camera*; the
    segment describes the object the baseline kernels describe.  The fixed
    rotation between them is read from the furnished frame kernels rather than
    assumed, since for Cassini it is nowhere near the identity.

    Parameters:
        pointing: The image's recorded corrected pointing.

    Returns:
        The 3x3 J2000-to-CK-object rotation at the midtime.

    Raises:
        KeyError: if the CK object has no frame name in the furnished kernels.
    """
    ck_frame = str(cspyce.frmnam(pointing.ck_frame_id))
    camera_from_ck = np.asarray(
        cspyce.pxform(ck_frame, pointing.camera_frame, pointing.midtime_et), dtype=np.float64
    )
    corrected: NDArrayFloatType = camera_from_ck.T @ np.asarray(pointing.cmatrix, dtype=np.float64)
    return corrected


def _baseline_attitude(ck_frame_id: int, tick: float) -> NDArrayFloatType:
    """Read the baseline attitude of a CK object at one encoded SCLK time.

    Parameters:
        ck_frame_id: SPICE id of the object.
        tick: Encoded SCLK time tag.

    Returns:
        The 3x3 J2000-to-CK-object rotation the furnished kernels give.

    Raises:
        OSError: if the furnished kernels provide no pointing there.
    """
    cmat, _clkout = cspyce.ckgp(ck_frame_id, tick, _LOOKUP_TOL_TICKS, _BASE_FRAME)
    return np.asarray(cmat, dtype=np.float64)


def _baseline_has_angular_velocity(ck_frame_id: int, tick: float) -> bool:
    """Probe whether the furnished baseline carries angular velocity.

    This is the fast path only.  It answers for one time tag, and a segment's
    flag has to hold for every record, so ``_baseline_history`` samples all of
    them and its result, not this probe, decides what the segment claims.

    Parameters:
        ck_frame_id: SPICE id of the object.
        tick: Encoded SCLK time tag to probe.

    Returns:
        True when angular velocity is available there.
    """
    try:
        cspyce.ckgpav(ck_frame_id, tick, _LOOKUP_TOL_TICKS, _BASE_FRAME)
    except OSError:
        # SPICE reports "this segment carries no angular velocity" and "no
        # pointing here at all" as the same insufficient-data error, so the two
        # are not distinguishable from the exception.  Demoting the second to a
        # segment without angular velocity hides nothing: the attitude lookups
        # that follow use ckgp at the same time tags and raise for pointing
        # that genuinely is not there.
        return False
    return True


def _sample_with_angular_velocity(
    ck_frame_id: int, ticks: list[float]
) -> tuple[list[NDArrayFloatType], NDArrayFloatType] | None:
    """Sample the baseline attitude and angular velocity at every record.

    Parameters:
        ck_frame_id: SPICE id of the object.
        ticks: The encoded SCLK time tags to sample.

    Returns:
        The attitudes and angular velocity vectors, or ``None`` when any
        record has no angular velocity available.  ``None`` also comes back
        when a record has no pointing at all, which the attitude-only pass
        then raises on.
    """
    attitudes: list[NDArrayFloatType] = []
    velocities: list[NDArrayFloatType] = []
    for tick in ticks:
        try:
            cmat, av, _clkout = cspyce.ckgpav(ck_frame_id, tick, _LOOKUP_TOL_TICKS, _BASE_FRAME)
        except OSError:
            return None
        attitudes.append(np.asarray(cmat, dtype=np.float64))
        velocities.append(np.asarray(av, dtype=np.float64))
    return attitudes, np.vstack(velocities)


def _baseline_history(
    ck_frame_id: int, ticks: list[float]
) -> tuple[list[NDArrayFloatType], NDArrayFloatType | None]:
    """Sample the baseline attitude at each record, with angular velocity if all have it.

    A segment carries one angular velocity flag for all of its records, so an
    exposure straddling a baseline segment that has angular velocity and one
    that does not cannot claim it: the corrected segment then carries none at
    all rather than inventing vectors for the records that lack them.

    Parameters:
        ck_frame_id: SPICE id of the object.
        ticks: The encoded SCLK time tags to sample.

    Returns:
        The baseline attitudes and its angular velocity vectors, or ``None``
        for the angular velocity when any record has none.

    Raises:
        OSError: if the furnished kernels provide no pointing at a record.
    """
    if _baseline_has_angular_velocity(ck_frame_id, ticks[0]):
        sampled = _sample_with_angular_velocity(ck_frame_id, ticks)
        if sampled is not None:
            return sampled
    return [_baseline_attitude(ck_frame_id, tick) for tick in ticks], None


def _corrected_history(
    ck_frame_id: int, ticks: list[float], delta: NDArrayFloatType
) -> tuple[list[NDArrayFloatType], NDArrayFloatType | None]:
    """Apply a body-fixed correction to the baseline attitude at each record.

    Parameters:
        ck_frame_id: SPICE id of the object.
        ticks: The encoded SCLK time tags to sample.
        delta: The correction in the CK object's own coordinates.

    Returns:
        The corrected attitudes and the baseline's angular velocity vectors,
        or ``None`` for the angular velocity when the baseline does not carry
        it at every record.

    Raises:
        OSError: if the furnished kernels provide no pointing at a record.
    """
    attitudes, velocities = _baseline_history(ck_frame_id, ticks)
    # The angular velocity is passed through untouched: it is expressed in the
    # segment's base frame, which the correction does not rotate, and two
    # frames rigidly attached to each other share it.
    return [delta @ attitude for attitude in attitudes], velocities


def _quaternion_sequence(attitudes: list[NDArrayFloatType]) -> NDArrayFloatType:
    """Convert attitudes to SPICE quaternions, keeping the sequence continuous.

    ``cspyce.m2q`` fixes the scalar component non-negative, so a sequence whose
    rotation angle crosses 180 degrees comes back with a sign flip between
    adjacent records even though the attitude moved barely at all.  Each record
    is negated as needed to keep a non-negative dot product with its
    predecessor, which leaves the attitude it represents unchanged.

    Parameters:
        attitudes: The 3x3 rotations, one per record, in record order.

    Returns:
        The quaternions, shape ``(n, 4)``.
    """
    quats: list[NDArrayFloatType] = []
    for attitude in attitudes:
        quat = np.asarray(cspyce.m2q(attitude), dtype=np.float64)
        if len(quats) > 0 and float(np.dot(quat, quats[-1])) < 0.0:
            quat = -quat
        quats.append(quat)
    return np.vstack(quats)
