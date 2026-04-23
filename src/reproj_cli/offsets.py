"""Helpers for loading pre-computed navigation offsets from nav_offset results.

The pattern mirrors ``src/backplanes/backplanes.py`` lines 71-79:
read the ``_metadata.json`` file for an image and apply the stored
``(dv, du)`` offset to the observation's FOV via ``oops.fov.OffsetFOV``.
"""

import json
from typing import Any, cast

import oops
from filecache import FCPath

from nav.config import MAIN_LOGGER
from nav.dataset.dataset import ImageFile
from nav.obs import ObsSnapshotInst


def load_offset_if_any(
    nav_results_root: str | FCPath | None,
    image_file: ImageFile,
) -> tuple[float, float] | None:
    """Return the ``(dv, du)`` offset from a nav_offset metadata file, if available.

    Parameters:
        nav_results_root: Root directory written by ``nav_offset``.  If ``None``,
            returns ``None`` immediately.
        image_file: The image to look up.

    Returns:
        ``(dv, du)`` as floats when the metadata file exists, is valid JSON,
        and has ``status == 'success'`` with a non-null ``offset`` field.
        Returns ``None`` (with a warning) in all other cases.
    """
    if nav_results_root is None:
        return None

    metadata_path = FCPath(nav_results_root) / (image_file.results_path_stub + '_metadata.json')

    try:
        text = metadata_path.read_text()
    except FileNotFoundError:
        MAIN_LOGGER.warning(
            'nav_results_root provided but no metadata found for %s; using uncorrected pointing.',
            image_file.image_file_url,
        )
        return None
    except Exception as exc:
        MAIN_LOGGER.warning(
            'Could not read metadata for %s (%s); using uncorrected pointing.',
            image_file.image_file_url,
            exc,
        )
        return None

    try:
        nav_metadata = cast(dict[str, Any], json.loads(text))
    except json.JSONDecodeError as exc:
        MAIN_LOGGER.warning(
            'Invalid JSON in metadata for %s (%s); using uncorrected pointing.',
            image_file.image_file_url,
            exc,
        )
        return None

    status = nav_metadata.get('status')
    if status != 'success':
        MAIN_LOGGER.warning(
            'Nav metadata for %s has status=%r; using uncorrected pointing.',
            image_file.image_file_url,
            status,
        )
        return None

    offset = nav_metadata.get('offset')
    if offset is None:
        MAIN_LOGGER.warning(
            'Nav metadata for %s has null offset; using uncorrected pointing.',
            image_file.image_file_url,
        )
        return None

    dv, du = offset
    return (float(dv), float(du))


def apply_offset_to_obs(obs: ObsSnapshotInst, dv: float, du: float) -> None:
    """Apply a navigation offset in-place to an observation's FOV.

    Parameters:
        obs: The observation whose FOV should be adjusted.
        dv: Vertical (row) offset in pixels.
        du: Horizontal (column) offset in pixels.
    """
    obs.fov = oops.fov.OffsetFOV(obs.fov, uv_offset=(float(du), float(dv)))
