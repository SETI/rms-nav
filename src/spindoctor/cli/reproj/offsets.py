"""Pointing selection and application for the reprojection and backplane readers.

Read the ``_metadata.json`` navigation record for an image, decide which
recorded pointing it carries, and apply that pointing to the observation.  The
recorded ``cmatrix`` -- the corrected camera attitude -- is the senior form
and is applied by replacing the observation's frame; the pixel ``offset`` is
its first-order approximation and remains the documented mechanism for every
record that carries no usable C-matrix (a fitted-rotation result, a simulated
image, a malformed pointing block, or a record that fails the reader's
gates).  This module serves both the mosaic drivers (``sd_mosaic`` and its
cloud-task worker) and the backplane stage.

:func:`select_pointing` is the whole classifier and takes a parsed record, so
it is equally the classifier for a record that never was a file.  Which storage
a program reads its records from is chosen by
:mod:`spindoctor.cli.reproj.pointing_source`, and that module states the few
places where a record rebuilt from an index row is classified differently from
the document it was ingested from.

Which values of a record are usable at all is not decided here either: that is
:mod:`spindoctor.support.nav_record`, and the results index stores what those
functions return.  The classifier therefore never has to ask whether the record
in front of it came from a document or a row, because a value that reached it
through a row is a value a document could have carried unchanged.

Failing to load a pointing does not stop the product; it proceeds on the
camera's uncorrected pointing, and the product it writes carries no sign of
that.  So the fact is reported in both places, and the two say different
things.  The *detailed* account -- which file was missing, what the
navigation status was, what the malformed field contained -- belongs to the
image and is written to its log here.  That the image was processed degraded
at all belongs to the run: an expected degradation (no C-matrix recorded, no
offset recorded) is counted by the caller from the returned reason, while a
degradation that indicates a defect or a changed environment (a malformed
pointing block, a failed gate) is also warned to the run log here, since it
means the same thing at every call site.  A cloud-task worker has no run
log, so its caller returns the counts and a per-reason tally in the task
result instead.
"""

import enum
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import oops
from filecache import FCPath

from spindoctor.config import IMAGE_LOGGER, MAIN_LOGGER
from spindoctor.dataset.dataset import ImageFile
from spindoctor.obs import ObsSnapshotInst

# The record validator is deliberately imported from the one module that owns
# the C-matrix conventions, rather than duplicated here: the reader and the
# selection must refuse exactly the same malformed records.
from spindoctor.support.cmatrix import (
    CMATRIX_BASELINE_MISMATCH,
    MALFORMED_POINTING,
    CmatrixApplication,
    apply_cmatrix_to_obs,
    validated_record_rotation,
)
from spindoctor.support.exceptions import NavPointingError

# Which values of a record a reader can use at all is decided in one place, and
# the results index stores what that place returns, so a record rebuilt from a
# row supplies the pointing its document supplies.  Importing the domain rather
# than restating it is what keeps the two from drifting apart.
from spindoctor.support.nav_record import (
    INVALID_OFFSET_TYPE,
    MALFORMED_OFFSET,
    MISSING_OFFSET_KEY,
    NON_FINITE_OFFSET,
    NULL_OFFSET,
    finite_float,
    record_offset,
    record_rotation_matrix,
    record_status,
)
from spindoctor.support.types import NDArrayFloatType

# The degraded-selection reasons this module classifies, beyond the gate and
# malformed-record reasons ``spindoctor.support.cmatrix`` stamps on its
# refusals.  ``no_cmatrix_rotation_fitted`` keys on the mechanism -- a result
# that fitted a camera rotation records no corrected attitude -- not on any
# mission.
NO_CMATRIX_ROTATION_FITTED = 'no_cmatrix_rotation_fitted'
NO_POINTING_BLOCK = 'no_pointing_block'

NO_METADATA = 'no_metadata'
"""Nothing recorded this image at all, however the records are stored."""

NO_METADATA_MESSAGE = 'No navigation record for %s in %s; using uncorrected pointing.'
"""What an image's log says when nothing recorded it.

Shared by every way of looking a record up, so an image with no record reads the
same in its log whether the records were sought as documents or as index rows.
The second value names the storage that was searched, from
:func:`storage_description`: "nothing ever navigated this image" and "the
storage searched does not hold it yet" read alike, and which one it is can only
be judged by someone who knows what was searched.
"""


def storage_description(nav_results_root: str | FCPath) -> str:
    """Name the documents a record was sought among, for a message about not finding one.

    Parameters:
        nav_results_root: Root the navigator wrote its documents under.

    Returns:
        The phrase naming that storage.
    """
    return f'the navigation results under {nav_results_root}'


class PointingMechanism(enum.Enum):
    """Which recorded pointing a metadata record supplies.

    ``CMATRIX`` applies the recorded corrected attitude by frame replacement;
    ``OFFSET`` applies the recorded pixel offset via ``oops.fov.OffsetFOV``;
    ``NONE`` leaves the observation's pointing uncorrected.
    """

    CMATRIX = 'cmatrix'
    OFFSET = 'offset'
    NONE = 'none'


@dataclass(frozen=True)
class PointingSelection:
    """The pointing one image's navigation record supplies, classified.

    Parameters:
        mechanism: Which mechanism the record selects.
        cmatrix: The recorded corrected attitude as a 3x3 rotation, or None
            when the mechanism is not ``CMATRIX``.
        cmatrix_original: The recorded uncorrected attitude, or None likewise.
        midtime_et: The recorded exposure midtime, or None likewise.
        offset: The recorded ``(dv, du)`` offset in pixels when one was
            usable, or None.  Carried even under the ``CMATRIX`` mechanism,
            because a gate failure at apply time degrades to it.
        reason: Why the selection is degraded, or None for a clean ``CMATRIX``
            selection -- and also None when nothing was asked for, since a
            pointing nobody wanted is not missing.  The detailed account is in
            the image's log; this is the short form a run-level report and
            count use.
    """

    mechanism: PointingMechanism
    cmatrix: NDArrayFloatType | None
    cmatrix_original: NDArrayFloatType | None
    midtime_et: float | None
    offset: tuple[float, float] | None
    reason: str | None


@dataclass(frozen=True)
class AppliedPointing:
    """What ``apply_pointing_to_obs`` actually did to the observation.

    Parameters:
        source: The pointing the observation now carries: ``'cmatrix'`` when
            the frame was replaced with the recorded corrected attitude,
            ``'pool'`` when the furnished kernel pool already answered it and
            nothing was applied, ``'offset'`` when the pixel offset was
            applied via ``OffsetFOV``, and ``'none'`` when the pointing was
            left uncorrected.
        reason: Why the source is not ``'cmatrix'``, in the short per-reason
            form run-level tallies use, or None for a clean application (and
            for a ``'none'`` outcome nobody asked to correct).
    """

    source: Literal['cmatrix', 'pool', 'offset', 'none']
    reason: str | None


def none_selection(reason: str | None) -> PointingSelection:
    """Build the selection for a record that supplies no pointing at all.

    Parameters:
        reason: Why, or None when nothing was asked for.

    Returns:
        The selection.
    """
    return PointingSelection(
        mechanism=PointingMechanism.NONE,
        cmatrix=None,
        cmatrix_original=None,
        midtime_et=None,
        offset=None,
        reason=reason,
    )


def resolved_nav_metadata_path(
    nav_results_root: str | FCPath,
    image_file: ImageFile,
) -> FCPath | None:
    """Resolve ``<nav_results_root>/<stub>_metadata.json`` and ensure it stays under root.

    Rejects null bytes, absolute ``results_path_stub`` fragments, and any resolved
    path that escapes ``nav_results_root`` (e.g. ``..`` segments in ``stub``).
    Every reader of a navigation document resolves its path through here, so one
    rule decides which paths a results root may be read at rather than each
    reader deciding for itself.

    Parameters:
        nav_results_root: Root the navigator wrote its documents under.
        image_file: The image whose document is wanted.

    Returns:
        The resolved path, or None when the stub does not name one this root
        may be read at, with the reason written to the image's log.
    """
    rel_name = f'{image_file.results_path_stub}_metadata.json'
    if '\x00' in rel_name:
        IMAGE_LOGGER.warning(
            'nav_results_root: metadata path contains null byte; refusing pointing load for %s.',
            image_file.image_file_url,
        )
        return None
    if Path(rel_name).is_absolute():
        IMAGE_LOGGER.warning(
            'nav_results_root: metadata path fragment is absolute; refusing pointing load for %s.',
            image_file.image_file_url,
        )
        return None
    root = FCPath(nav_results_root).expanduser().resolve()
    candidate = (root / rel_name).resolve()
    if not candidate.is_relative_to(root):
        IMAGE_LOGGER.warning(
            'nav_results_root: resolved metadata path %s is outside root %s; refusing '
            'pointing load for %s (check results_path_stub for path traversal).',
            candidate,
            root,
            image_file.image_file_url,
        )
        return None
    return candidate


_OFFSET_REASON_MESSAGES = {
    MISSING_OFFSET_KEY: 'Nav metadata for %s has no offset field; using uncorrected pointing.',
    NULL_OFFSET: 'Nav metadata for %s has null offset; using uncorrected pointing.',
    INVALID_OFFSET_TYPE: (
        'Nav metadata for %s has invalid offset type; using uncorrected pointing.'
    ),
    NON_FINITE_OFFSET: 'Nav metadata for %s has non-finite offset; using uncorrected pointing.',
    MALFORMED_OFFSET: (
        'Nav metadata for %s has malformed offset field; using uncorrected pointing.'
    ),
}


def _parse_pointing_values(
    nav_metadata: dict[str, Any],
) -> tuple[NDArrayFloatType, NDArrayFloatType, float] | str:
    """Read the ``pointing`` and ``times`` blocks the C-matrix mechanism needs.

    Parameters:
        nav_metadata: The parsed metadata record.

    Returns:
        The ``(cmatrix, cmatrix_original, midtime_et)`` triple when the record
        carries a usable one, or the selection reason that classifies why it
        does not: :data:`NO_POINTING_BLOCK` when there is no
        ``navigation_result`` or no ``pointing`` block at all,
        :data:`NO_CMATRIX_ROTATION_FITTED` when the block exists with no
        ``cmatrix`` key, and :data:`MALFORMED_POINTING` when a ``cmatrix`` is
        present but the block cannot be used (a malformed matrix, an absent or
        malformed ``cmatrix_original``, or an absent or non-finite
        ``times.midtime_et`` -- the gates cannot run).  This function owns the
        record-shape distinctions; the caller does not walk the record again.
    """
    nav_result = nav_metadata.get('navigation_result')
    if not isinstance(nav_result, dict):
        return NO_POINTING_BLOCK
    pointing = nav_result.get('pointing')
    if not isinstance(pointing, dict):
        return NO_POINTING_BLOCK
    if 'cmatrix' not in pointing:
        return NO_CMATRIX_ROTATION_FITTED
    times = nav_result.get('times')
    try:
        cmatrix = _parse_record_rotation(pointing['cmatrix'], 'cmatrix')
        if 'cmatrix_original' not in pointing:
            raise NavPointingError(
                'the pointing block has no cmatrix_original', reason=MALFORMED_POINTING
            )
        cmatrix_original = _parse_record_rotation(pointing['cmatrix_original'], 'cmatrix_original')
    except NavPointingError:
        return MALFORMED_POINTING
    if not isinstance(times, dict) or 'midtime_et' not in times:
        return MALFORMED_POINTING
    # Read through the one function that decides which recorded numbers a
    # reader can use: ``float()`` on its own accepts text and booleans, a NaN
    # midtime would defeat the reader's midtime gate in both directions, and an
    # integer of several hundred digits raises out of it.
    midtime = finite_float(times['midtime_et'])
    if midtime is None:
        return MALFORMED_POINTING
    return cmatrix, cmatrix_original, midtime


def _parse_record_rotation(value: Any, label: str) -> NDArrayFloatType:
    """Read one recorded C-matrix as the one 3x3 matrix of numbers it denotes.

    The matrix is assembled by the one function the results index stores
    rotations through, so a matrix this accepts is a matrix that survives
    ingest and one it refuses is stored as nothing.  Whether the matrix that
    survives is a proper orthonormal rotation is delegated to the reader's own
    validator, so the selection and the application refuse exactly the same
    records.

    A producer writes the nine values row-major and a 3x3 nesting of them
    denotes the same nine, but what is accepted is every nesting an array
    library reconciles into one 3x3 -- nine rows of one among them.  That is
    deliberate and it is wider than the two written shapes: the recorded value
    denotes exactly one matrix in any of them, and a reader that took the
    typography for the value would classify a record by how its nine numbers
    were bracketed.  The domain is stated once, in
    :func:`~spindoctor.support.nav_record.record_rotation_matrix`, because the
    store reads it through the same function.

    Parameters:
        value: The recorded value.
        label: Name used in refusal messages.

    Returns:
        The 3x3 rotation.

    Raises:
        NavPointingError: with reason ``malformed_pointing`` when the value is
            unusable.
    """
    matrix = record_rotation_matrix(value)
    if matrix is None:
        raise NavPointingError(
            f'{label} is not one 3x3 matrix of finite real numbers',
            reason=MALFORMED_POINTING,
        )
    return validated_record_rotation(matrix, label)


def select_pointing(nav_metadata: dict[str, Any], *, subject: str = '') -> PointingSelection:
    """Classify which recorded pointing one image's metadata record supplies.

    The ladder, in order: a usable ``pointing.cmatrix`` selects the C-matrix
    mechanism (the exact form, and what a consumer of the corrected kernels
    sees); a record without one -- a fitted-rotation result
    (``no_cmatrix_rotation_fitted``), a record with no ``pointing`` block at
    all (``no_pointing_block``), or a pointing block the gates cannot run on
    (``malformed_pointing``) -- selects the offset mechanism when a usable
    offset exists; and a record with no usable pointing of either kind
    selects none, with the reason (``navigation_did_not_succeed``,
    ``missing_offset_key``, ``null_offset``, ``invalid_offset_type``,
    ``non_finite_offset``, ``malformed_offset``).

    The detailed account of every degraded selection is written to the
    image's log; a malformed pointing block additionally puts one line in the
    run log, because it indicates a defective record rather than an expected
    record class.

    Parameters:
        nav_metadata: The parsed metadata record.
        subject: What the record describes (the image URL or path), used in
            log messages.

    Returns:
        The classified selection.
    """
    status = record_status(nav_metadata)
    if status != 'success':
        IMAGE_LOGGER.warning(
            'Nav metadata for %s has status=%r; using uncorrected pointing.', subject, status
        )
        return none_selection('navigation_did_not_succeed')

    classified_offset = record_offset(nav_metadata)
    offset, offset_reason = classified_offset.pair, classified_offset.reason
    pointing_values = _parse_pointing_values(nav_metadata)

    if isinstance(pointing_values, tuple):
        cmatrix, cmatrix_original, midtime_et = pointing_values
        if offset is None and offset_reason is not None:
            # The C-matrix path needs no offset, but a record defective in its
            # offset field too is still a doubly-defective record, and this is
            # the only place the second defect is visible: a later gate
            # refusal would find no offset to fall back to and never say why.
            IMAGE_LOGGER.info(
                'Nav metadata for %s carries an unusable offset (%s); the C-matrix path needs '
                'no fallback, but none exists if a gate refuses this record.',
                subject,
                offset_reason,
            )
        return PointingSelection(
            mechanism=PointingMechanism.CMATRIX,
            cmatrix=cmatrix,
            cmatrix_original=cmatrix_original,
            midtime_et=midtime_et,
            offset=offset,
            reason=None,
        )

    reason = pointing_values
    if reason == MALFORMED_POINTING:
        IMAGE_LOGGER.warning(
            'Nav metadata for %s has a malformed pointing block; falling back to the offset path.',
            subject,
        )
        MAIN_LOGGER.warning(
            '%s: malformed pointing block in nav metadata; using the offset path.', subject
        )

    if offset is not None:
        if reason in (NO_CMATRIX_ROTATION_FITTED, NO_POINTING_BLOCK):
            IMAGE_LOGGER.info(
                'Nav metadata for %s records no corrected attitude (%s); the offset path applies.',
                subject,
                reason,
            )
        return PointingSelection(
            mechanism=PointingMechanism.OFFSET,
            cmatrix=None,
            cmatrix_original=None,
            midtime_et=None,
            offset=offset,
            reason=reason,
        )

    if reason == MALFORMED_POINTING:
        # The malformed pointing block is the more diagnostic classification;
        # the offset shortfall is secondary once the record itself is defective.
        final_reason: str | None = MALFORMED_POINTING
    else:
        final_reason = offset_reason
        if offset_reason is not None:
            IMAGE_LOGGER.warning(_OFFSET_REASON_MESSAGES[offset_reason], subject)
    return PointingSelection(
        mechanism=PointingMechanism.NONE,
        cmatrix=None,
        cmatrix_original=None,
        midtime_et=None,
        offset=None,
        reason=final_reason,
    )


def load_pointing_if_any(
    nav_results_root: str | FCPath | None,
    image_file: ImageFile,
) -> PointingSelection:
    """Load and classify one image's recorded pointing from its metadata file.

    Parameters:
        nav_results_root: Root directory written by ``sd_offset``.  If ``None``,
            no pointing is looked for and none is reported missing.
        image_file: The image to look up.

    Returns:
        A :class:`PointingSelection`.  When the metadata file exists, is valid
        JSON and a JSON object, the record is classified by
        :func:`select_pointing`; in every other case -- including a
        ``results_path_stub`` that would resolve outside ``nav_results_root``
        -- the selection is ``NONE`` and carries the reason
        (``unusable_metadata_path``, ``no_metadata``, ``unreadable_metadata``,
        ``invalid_json``, ``metadata_not_an_object``), which the caller
        reports to the run and counts.  Processing with uncorrected pointing
        is not an error, but it is the difference between a product being
        registered and only looking registered, so it is not silent either.
    """
    if nav_results_root is None:
        # Nothing was asked for, so nothing is missing.
        return none_selection(None)

    metadata_path = resolved_nav_metadata_path(nav_results_root, image_file)
    if metadata_path is None:
        return none_selection('unusable_metadata_path')

    try:
        text = metadata_path.read_text()
    except FileNotFoundError:
        IMAGE_LOGGER.warning(
            NO_METADATA_MESSAGE,
            image_file.image_file_url,
            storage_description(nav_results_root),
        )
        return none_selection(NO_METADATA)
    except (OSError, UnicodeDecodeError) as exc:
        IMAGE_LOGGER.warning(
            'Could not read metadata for %s (%s); using uncorrected pointing.',
            image_file.image_file_url,
            exc,
        )
        return none_selection('unreadable_metadata')

    try:
        nav_metadata = json.loads(text)
    except json.JSONDecodeError as exc:
        IMAGE_LOGGER.warning(
            'Invalid JSON in metadata for %s (%s); using uncorrected pointing.',
            image_file.image_file_url,
            exc,
        )
        return none_selection('invalid_json')

    if not isinstance(nav_metadata, dict):
        IMAGE_LOGGER.warning(
            'Nav metadata for %s is not a JSON object '
            '(type=%s, value=%r); using uncorrected pointing.',
            image_file.image_file_url,
            type(nav_metadata).__name__,
            nav_metadata,
        )
        return none_selection('metadata_not_an_object')

    return select_pointing(nav_metadata, subject=str(image_file.image_file_url))


def apply_pointing_to_obs(
    obs: ObsSnapshotInst,
    selection: PointingSelection,
    *,
    subject: str = '',
) -> AppliedPointing:
    """Apply one classified pointing selection to an observation, in place.

    A ``CMATRIX`` selection replaces the observation's frame with the
    recorded corrected attitude via
    :func:`spindoctor.support.cmatrix.apply_cmatrix_to_obs`, leaving the FOV
    untouched.  When that application reports the kernel pool already answers
    the corrected attitude, nothing is applied at all -- the observation is
    already right, and the offset path would double-correct -- and the
    outcome is ``source='pool'`` with reason ``pool_already_corrected``.
    When a gate refuses the record (a foreign midtime, a changed baseline, a
    malformed matrix that survived selection), the refusal is warned to both
    logs and the selection's offset is applied instead; with no usable offset
    the pointing is left uncorrected.  An ``OFFSET`` selection wraps the FOV
    in ``oops.fov.OffsetFOV`` exactly as the offset path always has.

    After either mechanism mutates the observation, ``obs.reset_all()``
    clears every cached ``Backplane`` and ``Meshgrid``, so all downstream
    geometry -- including caches primed while ``from_file`` ran -- is built
    on the corrected observation.

    Parameters:
        obs: The observation to point, mutated in place.
        selection: The classified pointing to apply.
        subject: What the observation shows (the image URL or path), used in
            log messages.

    Returns:
        An :class:`AppliedPointing` naming which pointing the observation now
        carries and, when it is not the clean C-matrix application, the
        per-reason short form the caller counts:  ``('cmatrix', None)`` for a
        clean frame replacement; ``('pool', 'pool_already_corrected')`` for
        the deliberate no-op on an already-corrected pool;  ``('offset',
        reason)`` when the pixel offset was applied, whether selected or as a
        gate fallback; and ``('none', reason)`` when nothing was applied
        (with ``reason=None`` only when nothing was asked for).

    Raises:
        ValueError: if the selection's mechanism does not carry the values it
            promises, which is a defect in the caller, never in the record.
    """
    if selection.mechanism is PointingMechanism.CMATRIX:
        if (
            selection.cmatrix is None
            or selection.cmatrix_original is None
            or selection.midtime_et is None
        ):
            raise ValueError('a CMATRIX selection must carry cmatrix, original, and midtime')
        try:
            outcome = apply_cmatrix_to_obs(
                obs, selection.cmatrix, selection.cmatrix_original, selection.midtime_et
            )
        except NavPointingError as exc:
            # An expected refusal always carries a reason; an unexplained one
            # degrades under the unexplained-mismatch reason, per the rule
            # that a cmatrix that failed a gate is never applied.
            reason = exc.reason if exc.reason is not None else CMATRIX_BASELINE_MISMATCH
            IMAGE_LOGGER.warning(
                'Recorded C-matrix for %s not applied (%s): %s', subject, reason, exc
            )
            if selection.offset is not None:
                MAIN_LOGGER.warning(
                    '%s: recorded C-matrix not applied (%s); using the offset path.',
                    subject,
                    reason,
                )
                _apply_offset(obs, selection.offset)
                return AppliedPointing(source='offset', reason=reason)
            MAIN_LOGGER.warning(
                '%s: recorded C-matrix not applied (%s) and no usable offset; '
                'using uncorrected pointing.',
                subject,
                reason,
            )
            return AppliedPointing(source='none', reason=reason)
        if outcome is CmatrixApplication.POOL_ALREADY_CORRECTED:
            IMAGE_LOGGER.info(
                'The furnished kernel pool already answers the corrected attitude for %s; '
                'nothing applied.',
                subject,
            )
            return AppliedPointing(
                source='pool', reason=CmatrixApplication.POOL_ALREADY_CORRECTED.value
            )
        IMAGE_LOGGER.debug(
            'Applied the recorded C-matrix to %s: frame replaced, FOV untouched.', subject
        )
        obs.reset_all()
        return AppliedPointing(source='cmatrix', reason=None)

    if selection.mechanism is PointingMechanism.OFFSET:
        if selection.offset is None:
            raise ValueError('an OFFSET selection must carry the offset')
        IMAGE_LOGGER.info(
            'Applied the recorded (dv, du) = (%s, %s) px offset to %s via OffsetFOV (%s).',
            selection.offset[0],
            selection.offset[1],
            subject,
            selection.reason,
        )
        _apply_offset(obs, selection.offset)
        return AppliedPointing(source='offset', reason=selection.reason)

    return AppliedPointing(source='none', reason=selection.reason)


def _apply_offset(obs: ObsSnapshotInst, offset: tuple[float, float]) -> None:
    """Wrap the observation's FOV in an ``OffsetFOV`` and reset its caches.

    Parameters:
        obs: The observation, mutated in place.
        offset: The ``(dv, du)`` offset in pixels.
    """
    dv, du = offset
    obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=(float(du), float(dv)))
    obs.reset_all()
