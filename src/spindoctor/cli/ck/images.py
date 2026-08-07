"""Which navigated images get a corrected segment, and why the rest do not.

The generator considers every image a navigation run produced and writes a
segment for some of them.  This module reads what it needs out of one image's
metadata and answers the first half of that question -- whether the image is
eligible at all, judged from its own record -- while the second half, which
baseline kernel the correction belongs to, needs SPICE and lives beside the
assignment step.

An image is eligible when it navigated to a status of ``success`` or
``conflicted`` and carries a corrected C-matrix.  There is no confidence or
rank threshold: a consumer filters on the status, status reason, confidence and
rank the report and the segment comments carry.  Everything else is reported
with a reason, and the reasons are a closed set, so that every image considered
appears exactly once in the report with either a source file or a reason it has
none.

One rule needs more than one image to apply.  Cassini can expose its narrow and
wide angle cameras together, and the two frames then share one bus attitude
that cannot honor two different corrections; the pair is detected here and the
wide angle member is the one that yields.
"""

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from spindoctor.cli.ck.pointing import (
    ImagePointing,
    read_field,
    read_optional_text,
    read_section,
    read_text,
)

# The navigation statuses whose pointing is written into a kernel.  A
# conflicted result is written because the conflict is reported alongside it,
# not because it is trusted; nothing else navigated.
ELIGIBLE_STATUSES = frozenset({'success', 'conflicted'})

# Cassini's label value for an exposure taken on both cameras at once, the two
# cameras that take one, and the one whose correction is kept when they are
# paired.  Both cameras are named rather than treating everything that is not
# the winner as the loser: the rule is about these two sharing a bus attitude,
# and any other camera reaching it would be reported as yielding to a pair it
# was never part of.
BOTSIM_SHUTTER_MODE = 'BOTSIM'
BOTSIM_WINNING_CAMERA = 'NAC'
BOTSIM_YIELDING_CAMERA = 'WAC'
BOTSIM_CAMERAS = frozenset({BOTSIM_WINNING_CAMERA, BOTSIM_YIELDING_CAMERA})

# Two exposures this far apart or closer, taken in the shutter mode above on
# opposite cameras, are the same event seen twice.
BOTSIM_WINDOW_S = 1.0


class OmissionReason(Enum):
    """Why an image the generator considered received no corrected segment.

    Every image appears in the report exactly once, with either the file
    carrying its segment or one of these reasons.  The set is closed: a new
    reason is a schema change for every consumer of the report, and every
    member is one the generator emits, so a consumer that handles all five
    handles every report this tool writes.  A reason no run can produce would
    be worse than a missing one, since it asks every consumer to write dead
    code against a case that will never arrive.

    Two of them are about the baseline and are not the same condition.
    ``NO_REPRODUCING_BASELINE`` means no candidate kernel answers the attitude
    the image navigated against, which is how a kernel set that changed since
    navigation is detected.  ``BASELINE_COVERAGE_GAP`` is the opposite
    situation: a baseline that did reproduce, at the exposure midtime, and then
    supplied no pointing at one of the segment's other record epochs.

    An image whose pointing the writer cannot express as a segment at all --
    an exposure needing more records than a segment holds, or one whose
    baseline supplies angular velocity at only some of its records -- is not
    one of these.  It stops the run, because the run has found something about
    the kernels or the metadata that the operator has to see, and reporting it
    as one image's omission would bury it.
    """

    NOT_ELIGIBLE = 'not_eligible'
    BOTSIM_LOSER = 'botsim_loser'
    ROTATION_UNSUPPORTED = 'rotation_unsupported'
    NO_REPRODUCING_BASELINE = 'no_reproducing_baseline'
    BASELINE_COVERAGE_GAP = 'baseline_coverage_gap'


@dataclass(frozen=True)
class ImageEntry:
    """One navigated image, as the generator reads it out of its metadata.

    Parameters:
        image_name: Basename of the image.
        status: The navigation status recorded for it.
        camera: The camera that took it, or ``None`` when the metadata records
            none.
        shutter_mode: The shutter mode it was taken in, or ``None`` when the
            metadata records none.
        rotation_fitted: True when the navigation fitted a camera rotation,
            whose pivot no result records and which therefore cannot be
            expressed as an attitude.
        kernel_basenames: The SPICE kernel basenames recorded for the run that
            navigated it, empty for an image with no pointing to place.
        pointing: Its recorded corrected pointing, or ``None`` when the image
            is not eligible for a segment.
        ineligibility_reason: Why it is not eligible, or ``None`` when it is.

    Raises:
        ValueError: if a pointing solution is paired with a reason it is
            ineligible, or if neither is present.
    """

    image_name: str
    status: str
    camera: str | None
    shutter_mode: str | None
    rotation_fitted: bool
    kernel_basenames: tuple[str, ...]
    pointing: ImagePointing | None
    ineligibility_reason: OmissionReason | None

    def __post_init__(self) -> None:
        """Refuse an entry that is both eligible and not."""
        if (self.pointing is None) == (self.ineligibility_reason is None):
            raise ValueError(
                f'{self.image_name} must carry either a pointing solution or a reason it has '
                f'none, not both and not neither'
            )

    @property
    def is_eligible(self) -> bool:
        """True when the image carries pointing a segment can be built from."""
        return self.pointing is not None

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> 'ImageEntry':
        """Read one image's eligibility and pointing out of its metadata.

        The metadata is the per-image ``_metadata.json`` dict the navigation
        pipeline writes, including the document written for an image that
        failed to load, which carries no ``navigation_result`` at all and is
        simply not eligible.

        The fields read are the top-level ``status``, the ``observation`` block
        (``image_name``, ``camera``, ``shutter_mode``), the presence of
        ``navigation_result.rotation_deg``, and, for an eligible image, its
        ``navigation_result.provenance.spice_kernels`` list and the pointing
        and times blocks.  An ineligible image's pointing is not read at all,
        since an image with no corrected matrix has nothing there to read.

        Parameters:
            metadata: The image's full navigation metadata dict.

        Returns:
            The entry.

        Raises:
            ValueError: if a required field is absent, or if an eligible
                image's pointing does not satisfy the ImagePointing
                invariants.
            TypeError: if a field is present but holds a value of the wrong
                kind: a JSON ``null`` where text belongs, a kernel list that is
                not a list of text, or a section that is not a section.  That
                is a malformed document rather than an image without a
                solution, so it fails loudly instead of being reported as an
                omission.
        """
        observation = read_section(metadata, 'observation', 'metadata')
        image_name = read_text(observation, 'image_name', 'observation')
        status = read_text(metadata, 'status', 'metadata')
        camera = read_optional_text(observation, 'camera', 'observation')
        shutter_mode = read_optional_text(observation, 'shutter_mode', 'observation')
        navigation_result: dict[str, Any] = {}
        if 'navigation_result' in metadata:
            navigation_result = read_section(metadata, 'navigation_result', 'metadata')
        rotation_fitted = 'rotation_deg' in navigation_result
        reason = _ineligibility_reason(
            status=status, rotation_fitted=rotation_fitted, result=navigation_result
        )
        pointing = None if reason is not None else ImagePointing.from_metadata(metadata)
        basenames: tuple[str, ...] = ()
        if pointing is not None:
            basenames = _kernel_basenames(navigation_result)
        return cls(
            image_name=image_name,
            status=status,
            camera=camera,
            shutter_mode=shutter_mode,
            rotation_fitted=rotation_fitted,
            kernel_basenames=basenames,
            pointing=pointing,
            ineligibility_reason=reason,
        )


def _ineligibility_reason(
    *, status: str, rotation_fitted: bool, result: dict[str, Any]
) -> OmissionReason | None:
    """Judge one image's eligibility from its own record.

    The status is tested first, so an image that did not navigate is reported
    as not eligible whatever else its record holds; a fitted rotation is
    reported as such only for an image that would otherwise have been written,
    where it is the reason the correction cannot be expressed.

    Parameters:
        status: The navigation status recorded for the image.
        rotation_fitted: True when the result carries a fitted camera rotation.
        result: The image's ``navigation_result`` block, empty for a document
            that has none.

    Returns:
        The reason the image gets no segment, or ``None`` when it is eligible.
    """
    if status not in ELIGIBLE_STATUSES:
        return OmissionReason.NOT_ELIGIBLE
    if rotation_fitted:
        return OmissionReason.ROTATION_UNSUPPORTED
    if 'pointing' not in result:
        return OmissionReason.NOT_ELIGIBLE
    # Read as a section rather than tested for a key, so a pointing block that
    # is not a block at all is a malformed document rather than an image
    # quietly reported as having navigated without a correction.
    if 'cmatrix' not in read_section(result, 'pointing', 'navigation_result'):
        return OmissionReason.NOT_ELIGIBLE
    return None


def _kernel_basenames(result: dict[str, Any]) -> tuple[str, ...]:
    """Read the SPICE kernel basenames recorded for a navigated image.

    Parameters:
        result: The image's ``navigation_result`` block.

    Returns:
        The recorded basenames, in the order recorded.  The list is sorted and
        holds no directories, and in a batch run it accumulates kernels earlier
        images needed, so it is a superset of the ones this image used.

    Raises:
        ValueError: if the provenance block or its kernel list is absent, or
            if the list is empty.  An image carrying a corrected attitude was
            navigated against kernels, so a record naming none of them is a
            defect in the record, and it is refused exactly as a missing
            provenance block is: routing it into the report instead would say
            the image's baseline had drifted, which is the one thing that
            report is meant to mean.
        TypeError: if the kernel list is not a list, or holds anything but
            text.
    """
    provenance = read_section(result, 'provenance', 'navigation_result')
    kernels = read_field(provenance, 'spice_kernels', 'provenance')
    if not isinstance(kernels, list):
        raise TypeError(
            f"provenance field 'spice_kernels' is {type(kernels).__name__}, not a list: {kernels!r}"
        )
    for kernel in kernels:
        if not isinstance(kernel, str):
            raise TypeError(
                f"provenance field 'spice_kernels' holds a {type(kernel).__name__}, not a "
                f'string: {kernel!r}'
            )
    if len(kernels) == 0:
        raise ValueError(
            "provenance field 'spice_kernels' is empty; an image with a corrected attitude was "
            'navigated against kernels, so recording none of them is a defect in the record'
        )
    return tuple(kernels)


def botsim_losers(entries: Sequence[ImageEntry]) -> frozenset[str]:
    """Name the images that yield their correction to a simultaneous exposure.

    Two Cassini frames exposed together share one bus attitude, and a corrected
    kernel describes that bus: it cannot carry both corrections.  Two eligible
    images pair when both record the ``BOTSIM`` shutter mode, one was taken by
    the narrow angle camera and the other by the wide angle camera, both
    correct the same CK object, and their exposures start within one second of
    each other.  The narrow angle camera keeps its correction, so a wide angle
    frame pairing with any eligible narrow angle frame is named here.  A wide
    angle frame whose narrow angle partner did not navigate pairs with nothing
    and keeps its own correction, and a frame from any other instrument is not
    part of this rule at all.

    Parameters:
        entries: The images the run considered, in any order.

    Returns:
        The basenames of the images that yield.  An ineligible image never
        appears: it is already omitted for its own reason.
    """
    members = _botsim_members(entries)
    winners: dict[int, list[float]] = {}
    for entry, pointing in members:
        if entry.camera == BOTSIM_WINNING_CAMERA:
            winners.setdefault(pointing.ck_frame_id, []).append(pointing.start_et)
    for starts in winners.values():
        starts.sort()
    losers: set[str] = set()
    for entry, pointing in members:
        # Every member is one of the two cameras, so not the winner is the one
        # that yields; nothing else reaches this loop.
        if entry.camera == BOTSIM_WINNING_CAMERA:
            continue
        starts = winners.get(pointing.ck_frame_id, [])
        if _has_start_within(starts, pointing.start_et, BOTSIM_WINDOW_S):
            losers.add(entry.image_name)
    return frozenset(losers)


def _botsim_members(entries: Sequence[ImageEntry]) -> list[tuple[ImageEntry, ImagePointing]]:
    """Select the images that can take part in a simultaneous-exposure pair.

    Parameters:
        entries: The images the run considered.

    Returns:
        Each eligible image that records the simultaneous shutter mode and was
        taken by one of the two cameras that share a bus attitude, paired with
        its pointing.  An image whose camera is absent, empty, or anything
        else is left out and keeps its own correction.
    """
    return [
        (entry, entry.pointing)
        for entry in entries
        if entry.pointing is not None
        and entry.shutter_mode == BOTSIM_SHUTTER_MODE
        and entry.camera in BOTSIM_CAMERAS
    ]


def _has_start_within(starts: Sequence[float], start_et: float, window_s: float) -> bool:
    """Report whether a sorted list of epochs holds one close to another epoch.

    Parameters:
        starts: Exposure start epochs, sorted ascending.
        start_et: The epoch to look near.
        window_s: The largest separation that still counts as simultaneous.

    Returns:
        True when some entry of ``starts`` is within ``window_s`` of
        ``start_et``, the separation itself included.
    """
    at = bisect_left(starts, start_et - window_s)
    return at < len(starts) and starts[at] <= start_et + window_s
