"""Which baseline kernel each image's correction belongs to, and where it goes.

A corrected C-kernel is an overlay on exactly one original: it carries the
segments of the images whose attitude that original supplied, under the
original's own name with ``_nav`` before the extension.  Deciding which
original that is cannot be a guess.  The metadata records sorted kernel
basenames with no load order, accumulated across a batch, so several of them
may describe the right object over the right epochs while only one of them is
the file the navigation actually read.

So the pairing is made by reproduction.  Each candidate is furnished on its own
and asked for the attitude the image navigated against; a candidate that
answers the recorded ``cmatrix_original`` to within a nanoradian supplied that
baseline, and one that does not, did not.  An image no candidate reproduces
receives no segment at all: that is also the detector for a kernel set that has
changed since navigation ran, since a corrected segment measured against a
baseline that no longer exists is worse than no segment.

How the attitude is asked for follows how it was navigated.  Most instruments
evaluate a frame chain at the exposure midtime, which ``pxform`` reproduces.
Voyager does not: its observation frame is frozen from a single
tolerance-snapped pointing lookup, so reproducing it means making the same
lookup, at the same whole clock tick, with the same tolerance, and composing
the same fixed platform-to-camera rotation on top.

The caller owns the kernel pool.  The supporting kernels -- leapseconds, the
spacecraft clock navigation used, and the frame kernel defining the camera and
CK object frames -- must be furnished before assignment runs, and no C-kernel
may be, since a stray one answers the same lookups and would make the
reproduction test meaningless.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cspyce
import numpy as np

from spindoctor.cli.ck.images import ImageEntry, OmissionReason, botsim_losers
from spindoctor.cli.ck.index import CK_SUFFIXES, CkFile, CkIndex
from spindoctor.cli.ck.pointing import ImagePointing, NDArrayFloatType
from spindoctor.cli.ck.segment import BASE_FRAME, FROZEN_ATTITUDE_CK_IDS, resolve_sclk_id

# A candidate reproduces a recorded attitude when the rotation between the two
# is smaller than this.  It is a reproduction bound, not a navigation bound:
# the same kernels evaluated the same way answer to floating-point noise, so
# anything larger is a different kernel.
REPRODUCTION_TOL_RAD = 1e-9

# The tolerance oops uses for the snapped pointing lookup that freezes a
# Voyager observation frame, in spacecraft clock ticks: a fixed part plus a
# term in the exposure.  When that lookup finds nothing, oops falls back to a
# frame registered at a far wider tolerance, so a baseline is tried at both.
_SNAPPED_TOL_TICKS = 800.0
_SNAPPED_TOL_TICKS_PER_EXPOSURE_S = 1.0 / 48.0
_SNAPPED_FALLBACK_TOL_TICKS = 80000.0

# The fixed rotation from a frozen-attitude object's frame to the camera frame
# is read at this epoch, as oops reads it: the frame kernel defines it as a
# constant, so the epoch only has to match.
_FIXED_ROTATION_ET = 0.0

# What cspyce raises when the furnished kernels cannot answer a lookup: a
# missing frame connection or an unresolvable clock arrives as a LookupError,
# a kernel it cannot read as an OSError, and other SPICE errors as a
# RuntimeError or a ValueError.  A candidate that cannot answer is a candidate
# that did not supply the baseline, which is an ordinary outcome of asking.
_SPICE_LOOKUP_FAILURES = (LookupError, OSError, RuntimeError, ValueError)

# What a corrected file's name carries beyond the original's, before the
# extension.
OUTPUT_NAME_MARKER = '_nav'


@dataclass(frozen=True)
class Assignment:
    """One image's disposition: the baseline it corrects, or why it has none.

    Parameters:
        entry: The image.
        baseline: The C-kernel file that supplied its uncorrected attitude, or
            ``None`` when it receives no segment.
        omission_reason: Why it receives no segment, or ``None`` when it does.

    Raises:
        ValueError: if a baseline is paired with a reason there is none, if
            neither is present, or if a baseline is assigned to an image that
            carries no pointing.
    """

    entry: ImageEntry
    baseline: CkFile | None
    omission_reason: OmissionReason | None

    def __post_init__(self) -> None:
        """Refuse an assignment that both writes a segment and does not."""
        if (self.baseline is None) == (self.omission_reason is None):
            raise ValueError(
                f'{self.entry.image_name} must be assigned either a baseline kernel or a reason '
                f'it has none, not both and not neither'
            )
        if self.baseline is not None and self.entry.pointing is None:
            raise ValueError(
                f'{self.entry.image_name} is assigned the baseline {self.baseline.basename} but '
                f'carries no pointing to write'
            )

    @property
    def image_name(self) -> str:
        """Basename of the image this assignment is for."""
        return self.entry.image_name

    @property
    def output_name(self) -> str | None:
        """Basename of the corrected file carrying the segment, or None."""
        if self.baseline is None:
            return None
        return output_basename(self.baseline.basename)


@dataclass(frozen=True)
class OutputGroup:
    """One corrected C-kernel file and the images whose segments it carries.

    Parameters:
        baseline: The original C-kernel this file mirrors.
        name: Basename of the corrected file.
        assignments: The images it carries, in the order they were assigned.
    """

    baseline: CkFile
    name: str
    assignments: tuple[Assignment, ...]


def output_basename(basename: str) -> str:
    """Return the name of the corrected file mirroring one original.

    The corrected file carries the original's name with ``_nav`` before the
    extension, so that the pairing is legible without opening either file:
    ``03236_04002ra.bc`` becomes ``03236_04002ra_nav.bc``.

    Parameters:
        basename: Basename of the original C-kernel.

    Returns:
        The corrected file's basename.

    Raises:
        ValueError: if the name is not a bare C-kernel basename -- if it is
            empty, carries a directory component, has no name before its
            extension, or does not end in a C-kernel extension -- or if it is
            already a corrected file's name, which would otherwise correct a
            correction and write it to a third name.
    """
    path = Path(basename)
    if len(path.parts) != 1 or basename in ('.', '..'):
        raise ValueError(f'{basename!r} is not a bare C-kernel basename')
    if path.suffix.lower() not in CK_SUFFIXES:
        raise ValueError(
            f'{basename!r} does not end in a C-kernel extension; expected one of '
            f'{sorted(CK_SUFFIXES)}'
        )
    stem = path.stem
    if len(stem) == 0:
        raise ValueError(f'{basename!r} has no name before its extension')
    if stem.endswith(OUTPUT_NAME_MARKER):
        raise ValueError(
            f'{basename!r} is already a corrected kernel; correcting it again would measure a '
            f'correction against a corrected baseline'
        )
    return f'{stem}{OUTPUT_NAME_MARKER}{path.suffix}'


def rotation_angle_rad(first: NDArrayFloatType, second: NDArrayFloatType) -> float:
    """Return the angle of the rotation carrying one attitude onto another.

    The angle is taken through a quaternion rather than through the trace,
    because the trace form loses half its digits as the angle goes to zero,
    which is the regime a reproduction test lives in.  It is zero if and only
    if the two matrices are equal, so it measures a wrong direction as readily
    as a wrong magnitude.

    Parameters:
        first: A 3x3 rotation.
        second: A 3x3 rotation.

    Returns:
        The angle in radians, in ``[0, pi]``.

    Raises:
        ValueError: if either matrix is not 3x3 or holds a non-finite value.
            A NaN would otherwise answer every comparison with False, which
            reads as agreement in one direction and disagreement in the other
            depending only on how the test is written.
    """
    relative = _as_finite_3x3(first, 'first') @ _as_finite_3x3(second, 'second').T
    quaternion = np.asarray(cspyce.m2q(relative), dtype=np.float64)
    return float(2.0 * np.arctan2(float(np.linalg.norm(quaternion[1:])), abs(float(quaternion[0]))))


def _as_finite_3x3(matrix: NDArrayFloatType, label: str) -> NDArrayFloatType:
    """Return one attitude as a finite 3x3 float array.

    Parameters:
        matrix: The matrix to check.
        label: Name used in the exception messages.

    Returns:
        The matrix as float64.

    Raises:
        ValueError: if it is not 3x3 or holds a non-finite value.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(f'{label} attitude is not a 3x3 matrix; got shape {array.shape}')
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f'{label} attitude holds a non-finite value: {array.tolist()!r}')
    return array


def attitudes_reproduce(recorded: NDArrayFloatType, evaluated: NDArrayFloatType) -> bool:
    """Report whether an evaluated attitude reproduces a recorded one.

    Parameters:
        recorded: The attitude the metadata recorded.
        evaluated: The attitude a candidate kernel gives.

    Returns:
        True when the rotation between them is at most
        ``REPRODUCTION_TOL_RAD``.

    Raises:
        ValueError: if either matrix is not a finite 3x3.
    """
    return rotation_angle_rad(recorded, evaluated) <= REPRODUCTION_TOL_RAD


def reproduces_baseline(pointing: ImagePointing) -> bool:
    """Report whether the furnished kernels reproduce an image's baseline.

    Exactly one candidate C-kernel must be furnished, along with the supporting
    kernels; this asks it for the uncorrected attitude the image navigated
    against and compares it against the recorded one.

    A frozen-attitude object is asked the way its navigated frame was built: a
    pointing lookup at the truncated clock tick of the exposure midtime, with
    the tolerance that lookup used, composed with the fixed rotation from the
    object's frame to the camera frame.  When that finds nothing the wider
    tolerance of the frame oops falls back to is tried, since an image
    navigated through that fallback reproduces only there.  Every other object
    is asked for the camera frame's attitude at the midtime directly.

    Parameters:
        pointing: The image's recorded pointing.

    Returns:
        True when the furnished kernels answer the recorded
        ``cmatrix_original`` to within ``REPRODUCTION_TOL_RAD``.  False when
        they answer something else, and False when they cannot answer at all,
        which is what a candidate covering the wrong epochs does.

    Raises:
        ValueError: if the CK object is not one this writer knows, or if the
            spacecraft clock resolved for it is not the expected one.
        KeyError: if a frozen-attitude object has no frame name in the
            furnished kernels, which is a missing frame kernel rather than a
            candidate that does not reproduce.
    """
    for evaluated in _baseline_attitudes(pointing):
        if attitudes_reproduce(pointing.cmatrix_original, evaluated):
            return True
    return False


def _baseline_attitudes(pointing: ImagePointing) -> Iterator[NDArrayFloatType]:
    """Yield the attitudes the furnished kernels give for one image's baseline.

    Parameters:
        pointing: The image's recorded pointing.

    Yields:
        One attitude per way the navigated frame could have been built: one for
        an evaluated frame chain, and for a frozen-attitude object one per
        lookup tolerance, in the order oops would have tried them.  A lookup
        the kernels cannot answer yields nothing rather than raising.
    """
    if pointing.ck_frame_id not in FROZEN_ATTITUDE_CK_IDS:
        chained = _evaluated_attitude(pointing.camera_frame, pointing.midtime_et)
        if chained is not None:
            yield chained
        return
    sclk_id = resolve_sclk_id(pointing.ck_frame_id)
    snapped_tol = _SNAPPED_TOL_TICKS + pointing.exposure_s * _SNAPPED_TOL_TICKS_PER_EXPOSURE_S
    # oops encodes the midtime as a whole tick for the primary lookup and
    # continuously for the wider fallback; both are reproduced as
    # they are made, since a truncated tick and a fractional one can land on
    # different pointing records.
    for tick, tolerance in (
        (float(cspyce.sce2t(sclk_id, pointing.midtime_et)), snapped_tol),
        (float(cspyce.sce2c(sclk_id, pointing.midtime_et)), _SNAPPED_FALLBACK_TOL_TICKS),
    ):
        snapped = _snapped_attitude(pointing, tick, tolerance)
        if snapped is not None:
            yield snapped


def _evaluated_attitude(camera_frame: str, et: float) -> NDArrayFloatType | None:
    """Evaluate the J2000-to-camera rotation from the furnished kernels.

    Parameters:
        camera_frame: SPICE name of the camera frame.
        et: TDB seconds past J2000.

    Returns:
        The 3x3 rotation, or ``None`` when the furnished kernels cannot supply
        it -- which for a candidate under test means it is not the baseline.
    """
    try:
        matrix = cspyce.pxform(BASE_FRAME, camera_frame, et)
    except _SPICE_LOOKUP_FAILURES:
        return None
    return np.asarray(matrix, dtype=np.float64)


def _snapped_attitude(
    pointing: ImagePointing, tick: float, tolerance_ticks: float
) -> NDArrayFloatType | None:
    """Read a frozen observation frame's attitude the way oops built it.

    Parameters:
        pointing: The image's recorded pointing.
        tick: Encoded spacecraft clock time to ask at.
        tolerance_ticks: How far from that tick a pointing record may be found.

    Returns:
        The 3x3 J2000-to-camera rotation the lookup gives, composed with the
        fixed rotation from the CK object's frame to the camera frame, or
        ``None`` when the furnished kernels have no pointing within the
        tolerance.
    """
    try:
        j2000_to_object, _clkout = cspyce.ckgp(
            pointing.ck_frame_id, tick, tolerance_ticks, BASE_FRAME
        )
    except _SPICE_LOOKUP_FAILURES:
        return None
    object_frame = str(cspyce.frmnam(pointing.ck_frame_id))
    object_to_camera = np.asarray(
        cspyce.pxform(object_frame, pointing.camera_frame, _FIXED_ROTATION_ET), dtype=np.float64
    )
    attitude: NDArrayFloatType = object_to_camera @ np.asarray(j2000_to_object, dtype=np.float64)
    return attitude


@contextmanager
def _furnished(path: Path) -> Iterator[None]:
    """Furnish one kernel for the duration of a block and unload it after.

    Parameters:
        path: The kernel to furnish.

    Yields:
        Nothing; the kernel is furnished for the body of the block.
    """
    cspyce.furnsh(str(path))
    try:
        yield
    finally:
        cspyce.unload(str(path))


def assign_images(entries: Sequence[ImageEntry], index: CkIndex) -> tuple[Assignment, ...]:
    """Decide, for every image a run considered, what becomes of its pointing.

    Each eligible image is paired with the indexed C-kernel that reproduces its
    recorded uncorrected attitude; every other image is given a reason.
    Candidates are tried one at a time with nothing else furnished, and images
    sharing a candidate set are tested together, so the kernel pool changes
    once per set rather than once per image.

    Parameters:
        entries: The images the run considered.
        index: The pre-indexed C-kernels their baselines may be among.

    Returns:
        One assignment per image, in the order the images were given.

    Raises:
        ValueError: if two entries name the same image, whose assignments
            could not then be told apart, or if a C-kernel is already
            furnished, which would answer the reproduction lookups alongside
            the candidate under test.
        OSError: if a candidate kernel cannot be furnished.
    """
    names = [entry.image_name for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError('two images have the same name; their assignments cannot be told apart')
    _require_no_furnished_ck()
    losers = botsim_losers(entries)
    testable = [
        entry for entry in entries if entry.pointing is not None and entry.image_name not in losers
    ]
    baselines = _reproducing_baselines(testable, index)
    return tuple(_assignment_for(entry, losers, baselines) for entry in entries)


def _assignment_for(
    entry: ImageEntry, losers: frozenset[str], baselines: dict[str, tuple[CkFile, ...]]
) -> Assignment:
    """Decide one image's disposition from what the run has established.

    Parameters:
        entry: The image.
        losers: The images that yielded to a simultaneous exposure.
        baselines: The reproducing candidates of every image that was tested.

    Returns:
        The image's assignment.

    Raises:
        KeyError: if an image that is eligible and did not yield was never
            tested, which cannot happen for the test set the assignment step
            builds from these same two facts.
    """
    if entry.ineligibility_reason is not None:
        return Assignment(entry=entry, baseline=None, omission_reason=entry.ineligibility_reason)
    if entry.image_name in losers:
        return Assignment(entry=entry, baseline=None, omission_reason=OmissionReason.BOTSIM_LOSER)
    candidates = baselines[entry.image_name]
    if len(candidates) == 0:
        return Assignment(
            entry=entry, baseline=None, omission_reason=OmissionReason.NO_REPRODUCING_BASELINE
        )
    return Assignment(entry=entry, baseline=candidates[0], omission_reason=None)


def _reproducing_baselines(
    entries: Sequence[ImageEntry], index: CkIndex
) -> dict[str, tuple[CkFile, ...]]:
    """Find, for each eligible image, every candidate that reproduces its baseline.

    Parameters:
        entries: The eligible images.  An image carrying no pointing is not
            tested and does not appear in the result.
        index: The pre-indexed C-kernels.

    Returns:
        One entry per tested image, holding its reproducing candidates in
        preference order, so the first is the one to use and an empty tuple
        means no candidate reproduced.

    Raises:
        OSError: if a candidate kernel cannot be furnished.
    """
    tested = [(entry, entry.pointing) for entry in entries if entry.pointing is not None]
    groups: dict[tuple[Path, ...], list[tuple[ImageEntry, ImagePointing]]] = {}
    candidates_by_key: dict[tuple[Path, ...], tuple[CkFile, ...]] = {}
    reproducing: dict[str, list[CkFile]] = {}
    for entry, pointing in tested:
        candidates = index.candidates(
            basenames=entry.kernel_basenames,
            ck_frame_id=pointing.ck_frame_id,
            et=pointing.midtime_et,
        )
        key = tuple(candidate.path for candidate in candidates)
        candidates_by_key[key] = candidates
        groups.setdefault(key, []).append((entry, pointing))
        reproducing[entry.image_name] = []
    for key, group in groups.items():
        for candidate in candidates_by_key[key]:
            with _furnished(candidate.path):
                for entry, pointing in group:
                    if reproduces_baseline(pointing):
                        reproducing[entry.image_name].append(candidate)
    return {name: tuple(found) for name, found in reproducing.items()}


def _require_no_furnished_ck() -> None:
    """Refuse to run the reproduction test with a C-kernel already furnished.

    Raises:
        ValueError: if the kernel pool holds any C-kernel.  Reproduction asks
            the pool for an attitude with one candidate furnished, so anything
            else covering the same object and epoch would answer instead, and
            every image would be paired with whichever kernel happened to be
            loaded first.
    """
    furnished = int(cspyce.ktotal('CK'))
    if furnished != 0:
        raise ValueError(
            f'{furnished} C-kernel(s) are already furnished; the reproduction test needs a pool '
            f'holding only the supporting kernels, since any other C-kernel answers the same '
            f'lookups as the candidate under test'
        )


def group_for_output(assignments: Sequence[Assignment]) -> tuple[OutputGroup, ...]:
    """Group assigned images by the corrected file that will carry them.

    Each corrected file mirrors exactly one original, so its size stays
    proportional to that original's and regeneration is per original file.  An
    original that no image reproduces yields no group and therefore no file:
    SPICE refuses to close a C-kernel holding no segments, and an empty
    corrected kernel would claim a correction it does not carry.

    Parameters:
        assignments: The assignments of a run, in any order.  Those with no
            baseline are skipped.

    Returns:
        One group per corrected file, ordered by name, each holding its
        assignments in the order given.

    Raises:
        ValueError: if two different originals would be written to the same
            corrected name, which happens when two directories hold the same
            basename and different images reproduce each.  One would overwrite
            the other.
    """
    grouped: dict[str, list[Assignment]] = {}
    baselines: dict[str, CkFile] = {}
    for assignment in assignments:
        baseline = assignment.baseline
        if baseline is None:
            continue
        name = output_basename(baseline.basename)
        if name in baselines and baselines[name].path != baseline.path:
            raise ValueError(
                f'{baselines[name].path} and {baseline.path} would both be corrected to {name!r}'
            )
        baselines[name] = baseline
        grouped.setdefault(name, []).append(assignment)
    return tuple(
        OutputGroup(baseline=baselines[name], name=name, assignments=tuple(grouped[name]))
        for name in sorted(grouped)
    )
