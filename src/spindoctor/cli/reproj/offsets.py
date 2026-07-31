"""Helpers for loading pre-computed navigation offsets from sd_offset results.

Read the ``_metadata.json`` file for an image and apply the stored ``(dv, du)``
offset to the observation's FOV via ``oops.fov.OffsetFOV``.

Every reason an offset could not be loaded describes one image, so it is
reported to that image's log rather than to the run's.  These are the records
that say a reprojection used uncorrected pointing, which is the difference
between a product being what it claims and being quietly misregistered; they
belong with the image they qualify, where they are still there to be found once
the run that produced them is over.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import oops
from filecache import FCPath

from spindoctor.config import IMAGE_LOGGER
from spindoctor.dataset.dataset import ImageFile
from spindoctor.obs import ObsSnapshotInst


@dataclass(frozen=True)
class OffsetLookup:
    """The outcome of looking for one image's navigation offset.

    Parameters:
        offset: The ``(dv, du)`` offset in pixels, or None when none was
            loaded.
        reason: Why no offset was loaded, or None when one was -- and also
            None when none was asked for, since an offset nobody wanted is
            not missing.  The detailed account is in the image's log; this is
            the short form a run-level report and count can use.
    """

    offset: tuple[float, float] | None
    reason: str | None


def _resolved_nav_metadata_path(
    nav_results_root: str | FCPath,
    image_file: ImageFile,
) -> FCPath | None:
    """Resolve ``<nav_results_root>/<stub>_metadata.json`` and ensure it stays under root.

    Rejects null bytes, absolute ``results_path_stub`` fragments, and any resolved
    path that escapes ``nav_results_root`` (e.g. ``..`` segments in ``stub``).
    """
    rel_name = f'{image_file.results_path_stub}_metadata.json'
    if '\x00' in rel_name:
        IMAGE_LOGGER.warning(
            'nav_results_root: metadata path contains null byte; refusing offset load for %s.',
            image_file.image_file_url,
        )
        return None
    if Path(rel_name).is_absolute():
        IMAGE_LOGGER.warning(
            'nav_results_root: metadata path fragment is absolute; refusing offset load for %s.',
            image_file.image_file_url,
        )
        return None
    root = FCPath(nav_results_root).expanduser().resolve()
    candidate = (root / rel_name).resolve()
    if not candidate.is_relative_to(root):
        IMAGE_LOGGER.warning(
            'nav_results_root: resolved metadata path %s is outside root %s; refusing '
            'offset load for %s (check results_path_stub for path traversal).',
            candidate,
            root,
            image_file.image_file_url,
        )
        return None
    return candidate


def _parse_nav_offset_pair(offset: object) -> tuple[float, float] | None:
    """Parse ``offset`` from spindoctor metadata JSON into ``(dv, du)`` floats.

    Returns:
        A pair of floats on success, or ``None`` if ``offset`` is not a two-element
        sequence (excluding strings/bytes) or values are not convertible to float.

    Raises:
        TypeError: If either element is a ``bool`` (booleans are not valid pixel offsets).
        ValueError: If either element converts to a non-finite float (NaN or Infinity).
    """
    if offset is None or isinstance(offset, (str, bytes)):
        return None
    if not isinstance(offset, Sequence):
        return None
    if len(offset) != 2:
        return None
    try:
        dv_raw, du_raw = offset[0], offset[1]
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    if isinstance(dv_raw, bool) or isinstance(du_raw, bool):
        raise TypeError(f'Offset elements must not be bool; got dv={dv_raw!r}, du={du_raw!r}')
    try:
        dv = float(dv_raw)
        du = float(du_raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(dv) or not math.isfinite(du):
        raise ValueError(f'Offset elements must be finite floats; got dv={dv!r}, du={du!r}')
    return dv, du


def load_offset_if_any(
    nav_results_root: str | FCPath | None,
    image_file: ImageFile,
) -> OffsetLookup:
    """Return the ``(dv, du)`` offset from a sd_offset metadata file, if available.

    Parameters:
        nav_results_root: Root directory written by ``sd_offset``.  If ``None``,
            no offset is looked for and none is reported missing.
        image_file: The image to look up.

    Returns:
        An :class:`OffsetLookup` carrying the ``(dv, du)`` offset when the
        metadata file exists, is valid JSON, and has ``status == 'success'``
        with a non-null ``offset`` field.  In every other case -- including a
        ``results_path_stub`` that would resolve outside ``nav_results_root``
        -- it carries no offset and the reason there is none, which the caller
        reports to the run and counts.  Reprojecting with uncorrected pointing
        is not an error, but it is the difference between a product being
        registered and only looking registered, so it is not silent either.
    """
    if nav_results_root is None:
        # Nothing was asked for, so nothing is missing.
        return OffsetLookup(offset=None, reason=None)

    metadata_path = _resolved_nav_metadata_path(nav_results_root, image_file)
    if metadata_path is None:
        return OffsetLookup(offset=None, reason='unusable_metadata_path')

    try:
        text = metadata_path.read_text()
    except FileNotFoundError:
        IMAGE_LOGGER.warning(
            'nav_results_root provided but no metadata found for %s; using uncorrected pointing.',
            image_file.image_file_url,
        )
        return OffsetLookup(offset=None, reason='no_metadata')
    except (OSError, UnicodeDecodeError) as exc:
        IMAGE_LOGGER.warning(
            'Could not read metadata for %s (%s); using uncorrected pointing.',
            image_file.image_file_url,
            exc,
        )
        return OffsetLookup(offset=None, reason='unreadable_metadata')

    try:
        nav_metadata = json.loads(text)
    except json.JSONDecodeError as exc:
        IMAGE_LOGGER.warning(
            'Invalid JSON in metadata for %s (%s); using uncorrected pointing.',
            image_file.image_file_url,
            exc,
        )
        return OffsetLookup(offset=None, reason='invalid_json')

    if not isinstance(nav_metadata, dict):
        IMAGE_LOGGER.warning(
            'Nav metadata for %s is not a JSON object '
            '(type=%s, value=%r); using uncorrected pointing.',
            image_file.image_file_url,
            type(nav_metadata).__name__,
            nav_metadata,
        )
        return OffsetLookup(offset=None, reason='metadata_not_an_object')

    status = nav_metadata.get('status')
    if status != 'success':
        IMAGE_LOGGER.warning(
            'Nav metadata for %s has status=%r; using uncorrected pointing.',
            image_file.image_file_url,
            status,
        )
        return OffsetLookup(offset=None, reason='navigation_did_not_succeed')

    offset = nav_metadata.get('offset')
    if offset is None:
        IMAGE_LOGGER.warning(
            'Nav metadata for %s has null offset; using uncorrected pointing.',
            image_file.image_file_url,
        )
        return OffsetLookup(offset=None, reason='null_offset')

    try:
        parsed = _parse_nav_offset_pair(offset)
    except TypeError as exc:
        IMAGE_LOGGER.warning(
            'Nav metadata for %s has invalid offset type (%s); using uncorrected pointing.',
            image_file.image_file_url,
            exc,
        )
        return OffsetLookup(offset=None, reason='invalid_offset_type')
    except ValueError as exc:
        IMAGE_LOGGER.warning(
            'Nav metadata for %s has non-finite offset (%s); using uncorrected pointing.',
            image_file.image_file_url,
            exc,
        )
        return OffsetLookup(offset=None, reason='non_finite_offset')
    if parsed is None:
        IMAGE_LOGGER.warning(
            'Nav metadata for %s has malformed offset field; using uncorrected pointing.',
            image_file.image_file_url,
        )
        return OffsetLookup(offset=None, reason='malformed_offset')
    return OffsetLookup(offset=parsed, reason=None)


def apply_offset_to_obs(obs: ObsSnapshotInst, dv: float, du: float) -> None:
    """Apply a navigation offset in-place to an observation's FOV.

    Parameters:
        obs: The observation whose FOV should be adjusted.
        dv: Vertical (row) offset in pixels.
        du: Horizontal (column) offset in pixels.
    """
    obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=(float(du), float(dv)))
