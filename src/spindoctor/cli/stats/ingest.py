"""Ingest per-image navigation metadata JSON files into the statistics database."""

import argparse
import json
import math
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from filecache import FCPath

from spindoctor.cli.stats.classify import date_from_image_et
from spindoctor.cli.stats.schema import open_stats_db, upsert_image
from spindoctor.config import MAIN_LOGGER

__all__ = ['ingest_metadata_files', 'main_ingest', 'rows_from_metadata']


def _finite_or_none(value: Any) -> float | None:
    """Coerce a JSON value to a finite float, or None."""
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _str_or_none(value: Any) -> str | None:
    """Coerce a JSON value to a non-empty string, or None."""
    if isinstance(value, str) and value:
        return value
    return None


def _image_shape(value: Any) -> tuple[int | None, int | None]:
    """Per-axis ``(v, u)`` pixel dimensions from a metadata image_shape list."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None, None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None, None


def _sigma_from_covariance(covariance: Any) -> tuple[float | None, float | None]:
    """Per-axis 1-sigma pair from a curated covariance matrix (or None pair)."""
    if not isinstance(covariance, list) or len(covariance) < 2:
        return None, None
    try:
        var_dv = float(covariance[0][0])
        var_du = float(covariance[1][1])
    except (TypeError, ValueError, IndexError):
        return None, None
    sigma_dv = math.sqrt(var_dv) if var_dv >= 0.0 else None
    sigma_du = math.sqrt(var_du) if var_du >= 0.0 else None
    return sigma_dv, sigma_du


def _source_names_from_feature_ids(feature_ids: Any) -> list[str]:
    """Body / ring / catalog names parsed from curated feature ids.

    Feature ids follow ``kind:NAME[...]`` (``body_disc:IAPETUS``,
    ``ring_edge:SATURN:feature_135_ieg:IEG``, ``star:UCAC4:10230452``).
    """
    names: set[str] = set()
    if not isinstance(feature_ids, list):
        return []
    for feature_id in feature_ids:
        if not isinstance(feature_id, str):
            continue
        parts = feature_id.split(':')
        if len(parts) >= 2 and parts[1]:
            names.add(parts[1])
    return sorted(names)


def rows_from_metadata(
    metadata: dict[str, Any], *, source_file: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten one metadata JSON document into database rows.

    Parameters:
        metadata: Parsed metadata JSON as written by ``navigate_image_files``.
        source_file: Path or URL the document was read from (recorded for
            provenance).

    Returns:
        ``(image_row, technique_rows, source_rows)`` ready for
        :func:`~spindoctor.cli.stats.schema.upsert_image`.

    Raises:
        ValueError: If the document lacks the observation image name or the
            observation instrument.
    """
    observation = metadata.get('observation') or {}
    image_name = observation.get('image_name')
    if not image_name:
        raise ValueError(f'{source_file}: metadata lacks observation.image_name')
    instrument = observation.get('instrument')
    if not isinstance(instrument, str) or not instrument:
        raise ValueError(f'{source_file}: metadata lacks observation.instrument')
    nav = metadata.get('navigation_result') or {}
    provenance = nav.get('provenance') or {}
    classifier = nav.get('image_classifier') or {}
    offset = nav.get('offset_px') or [None, None]
    sigma = nav.get('sigma_px') or [None, None]
    # A navigated image's epoch comes from its observation (provenance); an
    # image that never loaded has no provenance, so the navigator records the
    # epoch it read from the index under ``observation.image_et``.  Either way
    # every image is placed in time.
    image_et = _finite_or_none(provenance.get('image_et'))
    if image_et is None:
        image_et = _finite_or_none(observation.get('image_et'))
    per_technique = nav.get('per_technique') or []
    timing = metadata.get('timing') or {}
    shape_v, shape_u = _image_shape(observation.get('image_shape'))

    image_row: dict[str, Any] = {
        'image_name': image_name,
        'instrument': instrument,
        # Absent for images that failed to load (no observation was built),
        # which also have no offset to attribute to a camera.
        'camera': _str_or_none(observation.get('camera')),
        'image_path': observation.get('image_path'),
        'image_et': image_et,
        'image_date': date_from_image_et(image_et),
        'status': str(metadata.get('status') or nav.get('status') or 'unknown'),
        'status_reason': nav.get('status_reason') or metadata.get('status_error'),
        'offset_dv': _finite_or_none(offset[0]),
        'offset_du': _finite_or_none(offset[1]),
        'sigma_dv': _finite_or_none(sigma[0]),
        'sigma_du': _finite_or_none(sigma[1]),
        'confidence': _finite_or_none(nav.get('confidence')),
        'confidence_rank': nav.get('confidence_rank'),
        'n_techniques': len(per_technique),
        'excluded_from_consensus': json.dumps(sorted(nav.get('excluded_from_consensus') or [])),
        'image_class': classifier.get('class'),
        'noise_sigma': _finite_or_none(classifier.get('noise_sigma')),
        'image_shape_v': shape_v,
        'image_shape_u': shape_u,
        'run_start': _str_or_none(timing.get('start_iso8601')),
        'run_end': _str_or_none(timing.get('end_iso8601')),
        'elapsed_s': _finite_or_none(timing.get('elapsed_s')),
        'config_hash': provenance.get('config_hash'),
        'git_sha': provenance.get('spindoctor_git_sha'),
        'pipeline_run': provenance.get('pipeline_run_iso8601'),
        'source_file': source_file,
    }

    technique_rows: list[dict[str, Any]] = []
    for entry in per_technique:
        tech_offset = entry.get('offset_px') or [None, None]
        sigma_dv, sigma_du = _sigma_from_covariance(entry.get('covariance_px2'))
        technique_rows.append(
            {
                'technique_name': entry.get('technique_name', 'unknown'),
                'offset_dv': _finite_or_none(tech_offset[0]),
                'offset_du': _finite_or_none(tech_offset[1]),
                'sigma_dv': sigma_dv,
                'sigma_du': sigma_du,
                'confidence': _finite_or_none(entry.get('confidence')),
                'spurious': int(bool(entry.get('spurious'))),
                'at_edge': int(bool(entry.get('at_edge'))),
                'source_names': json.dumps(
                    _source_names_from_feature_ids(entry.get('feature_ids'))
                ),
                'diagnostics': json.dumps(entry.get('diagnostics') or {}, sort_keys=True),
            }
        )

    # Aggregate the feature inventory by (feature_type, source_model, name).
    counts: dict[tuple[str, str, str], list[int]] = {}
    for entry in nav.get('feature_inventory') or []:
        feature_id = str(entry.get('feature_id', ''))
        parts = feature_id.split(':')
        name = parts[1] if len(parts) >= 2 and parts[1] else '(none)'
        key = (
            str(entry.get('feature_type', 'unknown')),
            str(entry.get('source_model', 'unknown')),
            name,
        )
        tally = counts.setdefault(key, [0, 0])
        tally[0] += 1
        tally[1] += int(bool(entry.get('gated')))
    source_rows = [
        {
            'feature_type': feature_type,
            'source_model': source_model,
            'source_name': source_name,
            'n_features': tally[0],
            'n_gated': tally[1],
        }
        for (feature_type, source_model, source_name), tally in sorted(counts.items())
    ]
    return image_row, technique_rows, source_rows


def _iter_metadata_files(root: FCPath) -> Iterator[FCPath]:
    """Yield every ``*_metadata.json`` under ``root``."""
    yield from root.rglob('*_metadata.json')


def ingest_metadata_files(
    conn: sqlite3.Connection,
    roots: list[str],
) -> tuple[int, int]:
    """Ingest every metadata file under the given roots.

    The whole scan runs in a single database transaction, committed on
    success (per-image commits do not scale to archive-wide runs).

    Parameters:
        conn: Open statistics database connection.
        roots: Local directories or FCPath-compatible URLs to scan
            recursively for ``*_metadata.json`` files.

    Returns:
        ``(n_ingested, n_errors)`` counts.
    """
    n_ingested = 0
    n_errors = 0
    with conn:
        for root in roots:
            files = sorted(_iter_metadata_files(FCPath(root)), key=lambda p: p.as_posix())
            for metadata_path in files:
                source = metadata_path.as_posix()
                try:
                    local = cast(Path, metadata_path.get_local_path())
                    metadata = json.loads(local.read_text(encoding='utf-8'))
                    image_row, technique_rows, source_rows = rows_from_metadata(
                        metadata, source_file=source
                    )
                except (OSError, ValueError) as exc:
                    MAIN_LOGGER.warning('Skipping %s: %s', source, exc)
                    n_errors += 1
                    continue
                upsert_image(
                    conn, image_row, technique_rows=technique_rows, source_rows=source_rows
                )
                n_ingested += 1
    return n_ingested, n_errors


def main_ingest(cmdline: list[str] | None = None) -> int:
    """Entry point for ``sd_stats_ingest``.

    Parameters:
        cmdline: Argument list; None uses ``sys.argv``.

    Returns:
        Process exit code (0 on success, 1 when nothing was ingested).
    """
    parser = argparse.ArgumentParser(
        description='Ingest navigation metadata JSON files into a SQLite statistics database.'
    )
    parser.add_argument(
        'roots',
        nargs='+',
        help='One or more navigation-results roots (local directories or URLs) '
        'scanned recursively for *_metadata.json files',
    )
    parser.add_argument(
        '--db',
        default='nav_stats.sqlite3',
        help='SQLite database path (created if missing; default: %(default)s)',
    )
    arguments = parser.parse_args(cmdline)

    conn = open_stats_db(arguments.db)
    try:
        n_ingested, n_errors = ingest_metadata_files(conn, arguments.roots)
    finally:
        conn.close()
    MAIN_LOGGER.info(
        'Ingested %d metadata file(s) into %s (%d error(s))',
        n_ingested,
        arguments.db,
        n_errors,
    )
    return 0 if n_ingested > 0 else 1
