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
error turns with it -- so the attitude inside the exposure varies as the
baseline's does.  It varies as finely as the records resolve it and no finer:
the segment holds the exposure start, midtime and stop, plus a one-second
cadence once the exposure reaches ten seconds, and SPICE interpolates between
them.  Three records over a ten-second exposure reproduce a real Cassini
baseline to about 2 NAC pixels at the median and 15 at the worst, so the
in-exposure history is an approximation at that cadence rather than an exact
reproduction; the record epochs themselves are exact.

Voyager is the exception.  Its navigated attitude came from a frozen,
tolerance-snapped pointing lookup rather than a time-varying frame chain, so a
Voyager segment carries the single corrected attitude, constant across its
window; writing time-varying pointing there would disagree with what was
navigated.

Angular velocity is copied unchanged and never rotated.  CK angular velocity is
expressed in the segment's base reference frame (J2000), and two frames rigidly
attached to each other have identical angular velocity in that frame, so
rotating it through ``delta`` -- superficially the thorough treatment -- writes
a vector in no frame at all.

Every segment written here carries angular velocity, because a segment that
declares none is not read as a segment whose angular velocity is unknown.
SPICE skips such a segment outright for ``ckgpav`` and for ``sxform`` and
answers from the next loaded kernel that does carry angular velocity for the
same object and epoch -- with that kernel's *uncorrected* attitude.  A
corrected segment declaring no angular velocity would therefore deliver its
correction to ``ckgp`` and ``pxform`` and silently withhold it from ``sxform``,
which is the call oops makes.  So a frozen segment writes zeros, which is a
constant attitude's true angular velocity, and an exposure whose baseline does
not supply angular velocity at every record is refused rather than written
without it: writing zeros there would claim a parked platform the baseline
never measured, and a consumer cannot tell an invented zero from a measured
one.

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
from spindoctor.spice_ids import CK_OBJECT_SCLK_ID, FROZEN_ATTITUDE_CK_IDS

# Every segment is written relative to J2000, which is what the recorded
# C-matrices are referenced to.
BASE_FRAME = 'J2000'

# Records go at the exposure start, midtime and stop.  An exposure of this
# length or longer gets additional records at the cadence below, so that a long
# exposure's attitude history is not reduced to three points.  The bound is
# inclusive because ten seconds exactly is an ordinary commanded ISS exposure
# and it is where the three-record fidelity has already gone: measured over 40
# random epochs on a reconstructed Cassini kernel, sampled across the window
# and expressed in NAC pixels, a ten-second exposure at three records leaves a
# mean error of 0.43 px, a 99th percentile of 5.2 px and a worst case of 9.9 px,
# with 23% of samples beyond a tenth of a pixel; the same exposure at the
# one-second cadence leaves 0.02 px, 0.44 px and 5.9 px, with 3% beyond a tenth.
_LONG_EXPOSURE_S = 10.0
_RECORD_CADENCE_S = 1.0

# The most records one segment may hold.  At the cadence above that is an
# exposure of nearly three hours, which no supported instrument commands, so
# reaching it means the recorded epochs are not an exposure at all -- and the
# arithmetic that expands them has no bound of its own: a span of 1e7 s asks
# for ten million records and one of 1e9 s exhausts memory.
_MAX_RECORDS = 10_000

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
            (J2000), shape ``(n, 3)``, or ``None`` for a segment that declares
            none.  Declaring none is not the same as declaring zero: SPICE
            skips such a segment for ``ckgpav`` and ``sxform`` and answers
            those from the next loaded kernel that does carry angular velocity
            for the same object and epoch.  :func:`build_segment` therefore
            never produces one.

    Raises:
        ValueError: if the arrays disagree on the record count, have the wrong
            width, hold no records, hold a non-finite value, or if the time
            tags are not strictly increasing.
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
        # Before the ordering check below, which a non-finite value defeats
        # rather than fails: every comparison against a NaN is False, so a NaN
        # time tag would read as strictly increasing, and an infinite one
        # satisfies the ordering outright.  Both would then reach ckw03 as
        # record data.
        _reject_non_finite(sclkdp, 'sclkdp')
        _reject_non_finite(quats, 'quats')
        if avvs is not None:
            _reject_non_finite(avvs, 'avvs')
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


def _reject_non_finite(values: NDArrayFloatType, label: str) -> None:
    """Raise unless every value in a record array is finite.

    A non-finite record is not something SPICE reports back: ``ckw03`` stores
    what it is handed, and a NaN quaternion or time tag becomes a kernel that
    answers nonsense to every consumer that furnishes it.  It also cannot be
    caught by the comparisons that validate ordering, since a NaN answers False
    to all of them.

    Parameters:
        values: The record array to check.
        label: Name of the array, used in the exception message.

    Raises:
        ValueError: if any value is not finite, naming the first such index.
    """
    offenders = np.argwhere(~np.isfinite(values))
    if offenders.shape[0] == 0:
        return
    index = tuple(int(position) for position in offenders[0])
    raise ValueError(f'{label} holds a non-finite value at index {index}: {float(values[index])!r}')


def resolve_sclk_id(ck_frame_id: int) -> int:
    """Return the spacecraft clock a CK object's time tags are encoded against.

    The id comes from the recorded CK-object-to-clock mapping, and
    ``cspyce.ckmeta`` is then required to agree with it.  It is deliberately
    that way round rather than the reverse, because ``ckmeta`` computes rather
    than validates: it answers for objects that do not exist, so taking its
    word would turn an unnoticed wrong CK id into a wrong clock, a successful
    encoding, and silently wrong time tags.  The returned value is the
    recorded one even though the check has just proved the two equal, so that
    weakening the check later cannot quietly make ``ckmeta`` the source.  The
    recorded clocks are the ones the attitude computation checks against too,
    so the two cannot drift apart.

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
    # The recorded id is returned, not the one ``ckmeta`` computed, even though
    # the check above has just proved them equal.  ``ckmeta`` answers for
    # objects that do not exist, so it is a cross-check here and never the
    # source: if this check is ever weakened, the time tags still come from the
    # recorded table rather than from whatever ``ckmeta`` returned.
    return expected


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
        The segment to write.  It always carries angular velocity: the
        baseline's own vectors unchanged, or zeros for a frozen-attitude
        object, whose attitude is constant.

    Raises:
        ValueError: if the CK object is not one this writer knows, if the
            resolved spacecraft clock is not the expected one, if the image
            name does not fit a SPICE segment identifier, if the baseline
            supplies angular velocity at only some of the record epochs, or if
            a record epoch lies outside the furnished spacecraft clock's
            coverage, which ``sce2c`` refuses rather than extrapolating.
        OSError: if the furnished kernels provide no pointing for the CK object
            at the exposure midtime or at a record epoch.
        KeyError: if the furnished kernels do not define one of the two frames
            the correction is expressed between: the CK object has no frame
            name, or the recorded camera frame is not a frame SPICE knows.
    """
    segid = _segment_id(pointing.image_name)
    sclk_id = resolve_sclk_id(pointing.ck_frame_id)
    ticks = _record_ticks(pointing, sclk_id)
    corrected_midtime = _corrected_attitude_at_midtime(pointing)
    attitudes: list[NDArrayFloatType]
    avvs: NDArrayFloatType
    if pointing.ck_frame_id in FROZEN_ATTITUDE_CK_IDS:
        # The navigated model assumed one snapped attitude across the whole
        # exposure, so the segment says exactly that, and its angular velocity
        # is zero because that is what a constant attitude's angular velocity
        # is.  Zeros rather than no angular velocity at all: SPICE skips a
        # segment carrying none for ckgpav and sxform and answers from the next
        # kernel that has some, so declaring none would hand an sxform caller
        # another kernel's uncorrected attitude.  The baseline's own vectors
        # are not copied here, since the rigid-attachment argument that
        # licenses copying them does not hold for a segment that deliberately
        # drops the baseline's time variation.
        attitudes = [corrected_midtime] * len(ticks)
        avvs = np.zeros((len(ticks), 3), dtype=np.float64)
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
            not open for writing or the file cannot be extended.
        ValueError: if SPICE refuses the record set itself: a segment
            identifier holding a non-printing character, or a quaternion of
            magnitude zero.  ``CkSegment`` enforces the count, width,
            finiteness and strictly-increasing invariants when it is
            constructed, but not these two, which only ``ckw03`` knows about.
    """
    avvs = segment.avvs
    if avvs is None:
        # ckw03 wants an array either way; with avflag false it is ignored.
        # Only a segment built by hand reaches this, since build_segment always
        # supplies angular velocity.
        avvs = np.zeros((segment.record_count, 3), dtype=np.float64)
    cspyce.ckw03(
        handle,
        segment.begtim,
        segment.endtim,
        segment.ck_frame_id,
        BASE_FRAME,
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

    Both the decision to add interior records and their number are taken from
    the same quantity, the span from the exposure start to its stop.  That is
    what the records have to cover, and ``ImagePointing`` has already required
    the recorded duration to agree with it, so the two cannot answer for
    different exposures.

    Three records reproduce the attitude at the start, the midtime and the stop
    exactly and interpolate everything between them; the cadence buys back the
    interpolation error over a long window, which is otherwise unbounded.

    Parameters:
        pointing: The image's recorded corrected pointing.

    Returns:
        The exposure start, midtime and stop, plus interior epochs at
        ``_RECORD_CADENCE_S`` when the span reaches ``_LONG_EXPOSURE_S``.

    Raises:
        ValueError: if the span would need more than ``_MAX_RECORDS`` records.
    """
    epochs = [pointing.start_et, pointing.midtime_et, pointing.stop_et]
    span_s = pointing.stop_et - pointing.start_et
    if span_s < _LONG_EXPOSURE_S:
        return sorted(epochs)
    steps = math.floor(span_s / _RECORD_CADENCE_S)
    if steps + len(epochs) > _MAX_RECORDS:
        raise ValueError(
            f'the exposure of {pointing.image_name} spans {span_s!r} s, which needs '
            f'{steps + len(epochs)} records at a {_RECORD_CADENCE_S} s cadence, more than the '
            f'{_MAX_RECORDS} a segment may hold'
        )
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
        KeyError: if the CK object has no frame name in the furnished kernels,
            or if the recorded camera frame is not one they define.
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
    cmat, _clkout = cspyce.ckgp(ck_frame_id, tick, _LOOKUP_TOL_TICKS, BASE_FRAME)
    return np.asarray(cmat, dtype=np.float64)


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
            cmat, av, _clkout = cspyce.ckgpav(ck_frame_id, tick, _LOOKUP_TOL_TICKS, BASE_FRAME)
        except OSError:
            return None
        attitudes.append(np.asarray(cmat, dtype=np.float64))
        velocities.append(np.asarray(av, dtype=np.float64))
    return attitudes, np.vstack(velocities)


def _baseline_history(
    ck_frame_id: int, ticks: list[float]
) -> tuple[list[NDArrayFloatType], NDArrayFloatType]:
    """Sample the baseline attitude and angular velocity at every record.

    A segment carries one angular velocity flag for all of its records, so an
    exposure straddling a baseline segment that has angular velocity and one
    that does not cannot claim it -- and cannot decline it either, since a
    segment declaring none is skipped by ``ckgpav`` and ``sxform`` in favor of
    whatever other kernel answers there.  Such an exposure is refused.

    Parameters:
        ck_frame_id: SPICE id of the object.
        ticks: The encoded SCLK time tags to sample.

    Returns:
        The baseline attitudes and its angular velocity vectors.

    Raises:
        OSError: if the furnished kernels provide no pointing at a record.
        ValueError: if they provide pointing at every record but angular
            velocity at only some of them.
    """
    sampled = _sample_with_angular_velocity(ck_frame_id, ticks)
    if sampled is not None:
        return sampled
    # ``ckgpav`` reports "no angular velocity here" and "no pointing here at
    # all" as one insufficient-data error, so which of the two happened is
    # decided by reading the attitude alone: a coverage gap raises out of this
    # loop as itself, and anything that survives it lacked only the angular
    # velocity.
    for tick in ticks:
        _baseline_attitude(ck_frame_id, tick)
    raise ValueError(
        f'the furnished baseline supplies pointing at every record of CK object {ck_frame_id} '
        f'but angular velocity at only some of them; a segment carries one angular velocity flag '
        f'for all of its records, and one declaring none would be skipped by ckgpav and sxform in '
        f'favor of another kernel answering with an uncorrected attitude'
    )


def _corrected_history(
    ck_frame_id: int, ticks: list[float], delta: NDArrayFloatType
) -> tuple[list[NDArrayFloatType], NDArrayFloatType]:
    """Apply a body-fixed correction to the baseline attitude at each record.

    Parameters:
        ck_frame_id: SPICE id of the object.
        ticks: The encoded SCLK time tags to sample.
        delta: The correction in the CK object's own coordinates.

    Returns:
        The corrected attitudes and the baseline's angular velocity vectors.

    Raises:
        OSError: if the furnished kernels provide no pointing at a record.
        ValueError: if the baseline does not supply angular velocity at every
            record.
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
