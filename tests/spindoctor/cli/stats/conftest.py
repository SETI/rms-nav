"""Shared document factories and index helpers for the statistics tests.

The factories build metadata documents in the shape ``navigate_image_files``
writes, so a test that cares about one field does not have to restate the
surrounding document.  The index helpers build a real index over a real tree,
because the ingest guarantees that matter -- what is keyed by what, what is
read a second time -- are properties of the walk and the writer together.
"""

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from tests.spindoctor.results_index.conftest import (
    postgres_schema,
    postgres_server_url,
    postgres_url,
)

from spindoctor.cli.stats.ingest import IngestCounts, ingest_metadata_files
from spindoctor.cli.stats.report import build_report
from spindoctor.results_index import open_index

# The statistics postgres tier runs against a schema of its own, exactly as the
# results-index tier does; re-exporting rather than restating keeps one
# definition of how that schema is created and dropped.
__all__ = ['postgres_schema', 'postgres_server_url', 'postgres_url']

DATA_DIR = Path(__file__).resolve().parent / 'data'
"""Directory holding the fixture results tree and the frozen report output."""

RESULTS_TREE = DATA_DIR / 'results_tree'
"""Fixture results tree the report regression is measured over."""

GOLDEN_DIR = DATA_DIR / 'golden'
"""Report and CSV output this tree produced before the move onto the index."""


def index_url(path: Path) -> str:
    """Return the SQLite URL naming an index file.

    Parameters:
        path: The database file's path.

    Returns:
        The URL.
    """
    return f'sqlite:///{path.as_posix()}'


@pytest.fixture
def quiet_logger() -> pdslogger.PdsLogger:
    """Return a logger that keeps ingest chatter out of the test output.

    Returns:
        A logger of its own, so raising its level cannot affect another test.
        The name carries a token that is unique for the life of the process:
        an object's address is not, since the object it belonged to is already
        collected and the next allocation is free to reuse it.
    """
    logger = pdslogger.PdsLogger(f'stats_test_{uuid.uuid4().hex}')
    logger.set_level('ERROR')
    return logger


def technique(
    name: str,
    offset: tuple[float, float],
    *,
    confidence: float = 0.7,
    spurious: bool = False,
    at_edge: bool = False,
) -> dict[str, Any]:
    """Build one ``per_technique`` entry.

    Parameters:
        name: Technique class name.
        offset: The technique's ``(dv, du)`` estimate.
        confidence: The technique's calibrated confidence.
        spurious: Whether the technique flagged its own result as spurious.
        at_edge: Whether the fit landed at the edge of its search space.

    Returns:
        The entry.
    """
    return {
        'technique_name': name,
        'feature_ids': [f'{name.lower()}:IAPETUS'],
        'offset_px': list(offset),
        'covariance_px2': [[0.01, 0.0], [0.0, 0.01]],
        'confidence': confidence,
        'spurious': spurious,
        'at_edge': at_edge,
        'diagnostics': {'a': 1},
    }


def metadata_document(
    *,
    image_name: str = 'N1454725799_1_CALIB.IMG',
    instrument: str | None = 'coiss',
    camera: str | None = 'NAC',
    status: str = 'success',
    status_reason: str | None = 'ok',
    status_error: str | None = None,
    offset: list[float] | None = None,
    confidence: float = 0.8,
    confidence_rank: str = 'high',
    per_technique: list[dict[str, Any]] | None = None,
    excluded: list[str] | None = None,
    image_et: float | None = 0.0,
    image_shape: list[int] | None = None,
    elapsed_s: float | None = 3.25,
) -> dict[str, Any]:
    """Build a metadata document in the ``navigate_image_files`` shape.

    Parameters:
        image_name: Recorded ``observation.image_name``.
        instrument: Recorded ``observation.instrument``; None omits the field,
            which models a file that is not a navigation document.
        camera: Recorded ``observation.camera``; None omits the field, as
            happens for an image that never loaded.
        status: Top-level status.
        status_reason: The navigator's explanation; None omits the field.
        status_error: The fatal error; None omits the field.
        offset: The authoritative top-level offset; None omits it, and a
            successful document defaults to one.
        confidence: Top-level confidence.
        confidence_rank: Recorded confidence tier.
        per_technique: Technique entries, from :func:`technique`.
        excluded: Technique names the ensemble excluded.
        image_et: Recorded provenance epoch.
        image_shape: Recorded ``observation.image_shape``; None omits it.
        elapsed_s: Recorded run time; None omits the whole timing section.

    Returns:
        The document.
    """
    if offset is None and status == 'success':
        offset = [1.5, -2.5]
    observation: dict[str, Any] = {
        'image_path': f'/holdings/{image_name}',
        'image_name': image_name,
    }
    if instrument is not None:
        observation['instrument'] = instrument
    if camera is not None:
        observation['camera'] = camera
    if image_shape is not None:
        observation['image_shape'] = image_shape
    navigation_result: dict[str, Any] = {
        'status': status,
        'offset_px': offset,
        'sigma_px': [0.1, 0.2] if offset else None,
        'confidence': confidence,
        'confidence_rank': confidence_rank,
        'covariance_px2': [[0.01, 0.0], [0.0, 0.04]] if offset else None,
        'techniques_used': sorted({t['technique_name'] for t in per_technique or []}),
        'excluded_from_consensus': excluded or [],
        'per_technique': per_technique or [],
        'feature_inventory': [
            {
                'feature_id': 'body_disc:IAPETUS',
                'feature_type': 'BODY_DISC',
                'source_model': 'body:IAPETUS',
                'gated': False,
            },
            {
                'feature_id': 'star:UCAC4:10230452',
                'feature_type': 'STAR',
                'source_model': 'stars',
                'gated': True,
            },
        ],
        'image_classifier': {'class': 'clean', 'noise_sigma': 1.0, 'max_dn': 255.0},
        'provenance': {
            'spindoctor_git_sha': 'abc1234',
            'config_hash': 'deadbeef',
            'image_et': image_et,
            'pipeline_run_iso8601': '2026-07-11T00:00:00Z',
        },
    }
    if status_reason is not None:
        navigation_result['status_reason'] = status_reason
    document: dict[str, Any] = {
        'status': status,
        'observation': observation,
        'navigation_result': navigation_result,
        'confidence': confidence,
    }
    if offset is not None:
        document['offset'] = list(offset)
    if status_error is not None:
        document['status_error'] = status_error
    if elapsed_s is not None:
        document['timing'] = {
            'start_iso8601': '2026-07-11T00:00:00Z',
            'end_iso8601': '2026-07-11T00:00:03.250000Z',
            'elapsed_s': elapsed_s,
        }
    return document


def write_metadata(root: Path, stub: str, document: dict[str, Any]) -> Path:
    """Write one metadata document into a results tree.

    Parameters:
        root: The results root.
        stub: The document's results path stub under that root.
        document: The document to write.

    Returns:
        The path written.
    """
    path = root / f'{stub}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding='utf-8')
    return path


def write_summary_png(root: Path, stub: str) -> Path:
    """Write a stand-in summary PNG beside a document.

    Only its name matters: the walk records that a summary exists, and never
    opens one.

    Parameters:
        root: The results root.
        stub: The image's results path stub.

    Returns:
        The path written.
    """
    path = root / f'{stub}_summary.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\x89PNG\r\n\x1a\n')
    return path


def ingest_tree(
    url: str, roots: list[Path], *, logger: pdslogger.PdsLogger, force: bool = False
) -> IngestCounts:
    """Create an index and ingest one or more results trees into it.

    Parameters:
        url: The index URL to create or add to.
        roots: The results roots to walk.
        logger: Logger the ingest reports through.
        force: Whether to re-read every document.

    Returns:
        What the pass did.
    """
    engine = open_index(url, create=True)
    try:
        return ingest_metadata_files(
            engine, [root.as_posix() for root in roots], force=force, logger=logger
        )
    finally:
        engine.dispose()


@pytest.fixture
def indexed_tree(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> Iterator[sqlalchemy.Connection]:
    """Yield a connection to an index built from the frozen fixture tree.

    Parameters:
        tmp_path: Directory the index file is written into.
        quiet_logger: Logger the ingest reports through.

    Yields:
        An open connection to the index.
    """
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [RESULTS_TREE], logger=quiet_logger)
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            yield connection
    finally:
        engine.dispose()


def report_from_tree(url: str, out: Path, *, logger: pdslogger.PdsLogger, **options: Any) -> Path:
    """Ingest the fixture tree into an index and write one report from it.

    One definition of the whole cycle -- ingest, open, build, dispose -- so that
    a change to the report's signature is made once rather than once per backend.

    Parameters:
        url: The index URL to create and ingest into.
        out: Directory receiving the report.
        logger: Logger the ingest reports through.
        options: Report options, passed through to ``build_report``.

    Returns:
        The directory the report was written into.
    """
    ingest_tree(url, [RESULTS_TREE], logger=logger)
    out.mkdir(parents=True, exist_ok=True)
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            build_report(connection, out, **options)
    finally:
        engine.dispose()
    return out
