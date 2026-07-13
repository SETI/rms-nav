"""Tests for the navigation statistics system (ingest + report)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from spindoctor.cli.stats.classify import date_from_image_et
from spindoctor.cli.stats.ingest import ingest_metadata_files, main_ingest, rows_from_metadata
from spindoctor.cli.stats.report import build_report, main_report
from spindoctor.cli.stats.report_common import image_number_from_name
from spindoctor.cli.stats.report_sections import resolve_offset_limit
from spindoctor.cli.stats.schema import open_stats_db, upsert_image


def _metadata(
    *,
    image_name: str = 'N1454725799_1_CALIB.IMG',
    instrument: str | None = 'coiss',
    status: str = 'success',
    status_reason: str = 'ok',
    offset: list[float] | None = None,
    confidence: float = 0.8,
    confidence_rank: str = 'high',
    per_technique: list[dict[str, Any]] | None = None,
    excluded: list[str] | None = None,
    image_et: float = 0.0,
    image_shape: list[int] | None = None,
    elapsed_s: float | None = 3.25,
) -> dict[str, Any]:
    """Build a minimal metadata document in the navigate_image_files shape.

    ``instrument=None`` omits the ``observation.instrument`` field to model
    a malformed document.  ``elapsed_s=None`` omits the ``timing`` section;
    ``image_shape=None`` omits ``observation.image_shape``.
    """
    if offset is None and status == 'success':
        offset = [1.5, -2.5]
    observation: dict[str, Any] = {
        'image_path': f'/holdings/{image_name}',
        'image_name': image_name,
    }
    if instrument is not None:
        observation['instrument'] = instrument
    if image_shape is not None:
        observation['image_shape'] = image_shape
    doc: dict[str, Any] = {
        'status': status,
        'observation': observation,
        'navigation_result': {
            'status': status,
            'status_reason': status_reason,
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
        },
    }
    if elapsed_s is not None:
        doc['timing'] = {
            'start_iso8601': '2026-07-11T00:00:00Z',
            'end_iso8601': '2026-07-11T00:00:03.250000Z',
            'elapsed_s': elapsed_s,
        }
    return doc


def _technique(
    name: str,
    offset: tuple[float, float],
    *,
    confidence: float = 0.7,
    spurious: bool = False,
) -> dict[str, Any]:
    return {
        'technique_name': name,
        'feature_ids': [f'{name.lower()}:IAPETUS'],
        'offset_px': list(offset),
        'covariance_px2': [[0.01, 0.0], [0.0, 0.01]],
        'confidence': confidence,
        'spurious': spurious,
        'at_edge': False,
        'diagnostics': {'a': 1},
    }


def _write_metadata(root: Path, name: str, doc: dict[str, Any]) -> Path:
    path = root / f'{name}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding='utf-8')
    return path


def _upsert(conn: sqlite3.Connection, doc: dict[str, Any]) -> None:
    """Flatten a metadata document and upsert it into the database."""
    image_row, technique_rows, source_rows = rows_from_metadata(doc, source_file='x.json')
    with conn:
        upsert_image(conn, image_row, technique_rows=technique_rows, source_rows=source_rows)


# --- classification helpers ---


def test_date_from_image_et_j2000() -> None:
    assert date_from_image_et(0.0) == '2000-01-01'


def test_date_from_image_et_none() -> None:
    assert date_from_image_et(None) is None


@pytest.mark.parametrize(
    ('image_name', 'expected'),
    [
        ('N1454725799_1_CALIB.IMG', 1454725799),
        ('/some/dir/W1728613298_8.IMG', 1728613298),
        ('lor_0003103486_0x630_sci.fit', 3103486),
        ('1454725799', 1454725799),
        ('no-digits-here', None),
        (None, None),
    ],
)
def test_image_number_from_name(image_name: str | None, expected: int | None) -> None:
    assert image_number_from_name(image_name) == expected


# --- flattening ---


def test_rows_from_metadata_success_document() -> None:
    doc = _metadata(
        per_technique=[
            _technique('BodyDiscCorrelateNav', (1.5, -2.5)),
            _technique('BodyLimbNav', (1.6, -2.4)),
        ],
        excluded=['BodyTerminatorNav'],
    )
    image_row, technique_rows, source_rows = rows_from_metadata(doc, source_file='x.json')
    assert image_row['image_name'] == 'N1454725799_1_CALIB.IMG'
    assert image_row['instrument'] == 'coiss'
    assert image_row['status'] == 'success'
    assert image_row['offset_dv'] == pytest.approx(1.5)
    assert image_row['offset_du'] == pytest.approx(-2.5)
    assert image_row['image_date'] == '2000-01-01'
    assert json.loads(image_row['excluded_from_consensus']) == ['BodyTerminatorNav']
    assert len(technique_rows) == 2
    assert technique_rows[0]['sigma_dv'] == pytest.approx(0.1)
    assert json.loads(technique_rows[0]['source_names']) == ['IAPETUS']
    assert len(source_rows) == 2
    star_row = next(r for r in source_rows if r['source_model'] == 'stars')
    assert star_row['n_features'] == 1
    assert star_row['n_gated'] == 1


def test_rows_from_metadata_requires_image_name() -> None:
    with pytest.raises(ValueError, match=r'lacks observation\.image_name'):
        rows_from_metadata({'observation': {}}, source_file='x.json')


def test_rows_from_metadata_requires_instrument() -> None:
    """A document without observation.instrument is rejected."""
    doc = _metadata(instrument=None)
    with pytest.raises(ValueError, match=r'lacks observation\.instrument'):
        rows_from_metadata(doc, source_file='x.json')


def test_rows_from_metadata_uses_recorded_instrument() -> None:
    """The recorded observation.instrument is stored verbatim."""
    doc = _metadata(instrument='sim')
    image_row, _, _ = rows_from_metadata(doc, source_file='x.json')
    assert image_row['instrument'] == 'sim'


def test_rows_from_metadata_null_status_falls_back() -> None:
    """An explicit null top-level status falls through to the nav status."""
    doc = _metadata()
    doc['status'] = None
    image_row, _, _ = rows_from_metadata(doc, source_file='x.json')
    assert image_row['status'] == 'success'


def test_rows_from_metadata_records_timing_and_shape() -> None:
    """Timing and image-shape metadata land in the image row."""
    doc = _metadata(image_shape=[1024, 512], elapsed_s=7.5)
    image_row, _, _ = rows_from_metadata(doc, source_file='x.json')
    assert image_row['run_start'] == '2026-07-11T00:00:00Z'
    assert image_row['run_end'] == '2026-07-11T00:00:03.250000Z'
    assert image_row['elapsed_s'] == pytest.approx(7.5)
    assert image_row['image_shape_v'] == 1024
    assert image_row['image_shape_u'] == 512


def test_rows_from_metadata_tolerates_missing_timing_and_shape() -> None:
    """Documents without timing or image_shape ingest with NULL columns."""
    doc = _metadata(elapsed_s=None, image_shape=None)
    image_row, _, _ = rows_from_metadata(doc, source_file='x.json')
    assert image_row['run_start'] is None
    assert image_row['run_end'] is None
    assert image_row['elapsed_s'] is None
    assert image_row['image_shape_v'] is None
    assert image_row['image_shape_u'] is None


def test_upsert_image_rejects_unknown_column(tmp_path: Path) -> None:
    """A row key that is not a schema column raises rather than reaching SQL."""
    conn = open_stats_db(tmp_path / 'stats.sqlite3')
    image_row, technique_rows, source_rows = rows_from_metadata(_metadata(), source_file='x.json')
    image_row['nonsense; DROP TABLE images'] = 1
    with pytest.raises(ValueError, match='unknown images column'):
        upsert_image(conn, image_row, technique_rows=technique_rows, source_rows=source_rows)
    conn.close()


def test_open_stats_db_rejects_mismatched_schema(tmp_path: Path) -> None:
    """A database with a different images column set is rejected, not migrated."""
    db_path = tmp_path / 'stats.sqlite3'
    conn = sqlite3.connect(str(db_path))
    conn.execute('CREATE TABLE images (image_name TEXT PRIMARY KEY, status TEXT)')
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match='Delete the database file'):
        open_stats_db(db_path)


# --- ingestion ---


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / 'results'
    _write_metadata(root / 'VOL1', 'N1454725799_1_CALIB', _metadata())
    _write_metadata(
        root / 'VOL1',
        'N1454725800_1_CALIB',
        _metadata(
            image_name='N1454725800_1_CALIB.IMG',
            status='failed',
            status_reason='no_features_extracted',
            offset=None,
            confidence=0.0,
            confidence_rank='failed',
        ),
    )
    conn = open_stats_db(tmp_path / 'stats.sqlite3')
    n_ingested, n_errors = ingest_metadata_files(conn, [str(root)])
    assert n_ingested == 2
    assert n_errors == 0
    # Second pass updates rather than duplicates.
    n_ingested, n_errors = ingest_metadata_files(conn, [str(root)])
    assert n_ingested == 2
    count = conn.execute('SELECT COUNT(*) FROM images').fetchone()[0]
    assert count == 2
    statuses = dict(conn.execute('SELECT image_name, status FROM images'))
    assert statuses['N1454725800_1_CALIB.IMG'] == 'failed'
    conn.close()


def test_ingest_skips_malformed_file(tmp_path: Path) -> None:
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'bad_metadata.json').write_text('not json', encoding='utf-8')
    _write_metadata(root, 'N1454725799_1_CALIB', _metadata())
    conn = open_stats_db(tmp_path / 'stats.sqlite3')
    n_ingested, n_errors = ingest_metadata_files(conn, [str(root)])
    assert n_ingested == 1
    assert n_errors == 1
    conn.close()


def test_ingest_counts_missing_instrument_as_error(tmp_path: Path) -> None:
    """A document without observation.instrument is skipped and counted."""
    root = tmp_path / 'results'
    _write_metadata(root, 'N1454725799_1_CALIB', _metadata(instrument=None))
    _write_metadata(root, 'N1454725800_1_CALIB', _metadata(image_name='N1454725800_1_CALIB.IMG'))
    conn = open_stats_db(tmp_path / 'stats.sqlite3')
    n_ingested, n_errors = ingest_metadata_files(conn, [str(root)])
    assert n_ingested == 1
    assert n_errors == 1
    names = [r[0] for r in conn.execute('SELECT image_name FROM images')]
    assert names == ['N1454725800_1_CALIB.IMG']
    conn.close()


def test_upsert_replaces_children(tmp_path: Path) -> None:
    conn = open_stats_db(tmp_path / 'stats.sqlite3')
    _upsert(conn, _metadata(per_technique=[_technique('BodyLimbNav', (1.0, 1.0))]))
    _upsert(conn, _metadata(per_technique=[]))
    count = conn.execute('SELECT COUNT(*) FROM techniques').fetchone()[0]
    assert count == 0
    conn.close()


def test_main_ingest_cli(tmp_path: Path) -> None:
    root = tmp_path / 'results'
    _write_metadata(root, 'N1454725799_1_CALIB', _metadata())
    db = tmp_path / 'stats.sqlite3'
    exit_code = main_ingest([str(root), '--db', str(db)])
    assert exit_code == 0
    conn = sqlite3.connect(db)
    assert conn.execute('SELECT COUNT(*) FROM images').fetchone()[0] == 1
    conn.close()


# --- offset-limit resolution ---


def test_resolve_offset_limit_coiss_nac_by_size() -> None:
    """Cassini NAC CALIB limits come from the per-size margin table."""
    limit = resolve_offset_limit('coiss', 'N1454725799_1_CALIB.IMG', 1024)
    assert limit == (50.0, 140.0)


def test_resolve_offset_limit_coiss_wac() -> None:
    """Cassini WAC limits use the wac detector block."""
    limit = resolve_offset_limit('coiss', 'W1454725799_1_CALIB.IMG', 512)
    assert limit == (5.0, 10.0)


def test_resolve_offset_limit_requires_shape_for_size_tables() -> None:
    """A size-keyed margin table cannot resolve without a recorded shape."""
    result = resolve_offset_limit('coiss', 'N1454725799_1_CALIB.IMG', None)
    assert result == 'image shape not recorded in the database'


def test_resolve_offset_limit_unknown_instrument() -> None:
    result = resolve_offset_limit('mystery', 'X123.IMG', 1024)
    assert isinstance(result, str)
    assert 'no configured search limit' in result


def test_resolve_offset_limit_missing_size_entry() -> None:
    """A size with no margin entry reports the failure instead of guessing."""
    result = resolve_offset_limit('vgiss', 'C3250013_GEOMED.IMG', 1024)
    assert isinstance(result, str)
    assert 'no extfov_margin_vu entry for image size 1024' in result


# --- reporting ---


def _populated_db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_stats_db(tmp_path / 'stats.sqlite3')
    docs = [
        _metadata(
            image_name='N1000000001_1_CALIB.IMG',
            image_shape=[1024, 1024],
            per_technique=[
                _technique('BodyDiscCorrelateNav', (1.5, -2.5)),
                _technique('BodyLimbNav', (1.7, -2.3)),
            ],
        ),
        _metadata(
            image_name='N1000000002_1_CALIB.IMG',
            image_shape=[1024, 1024],
            confidence=0.3,
            confidence_rank='medium',
            per_technique=[_technique('StarUniqueMatchNav', (0.1, 0.2))],
            excluded=['BodyBlobNav'],
        ),
        _metadata(
            image_name='C3250013_GEOMED.IMG',
            instrument='vgiss',
            status='failed',
            status_reason='no_features_extracted',
            offset=None,
            confidence=0.0,
            confidence_rank='failed',
            image_et=-6.0e8,
            image_shape=[1000, 1000],
        ),
    ]
    for doc in docs:
        _upsert(conn, doc)
    return conn


def test_build_report_writes_markdown_and_charts(tmp_path: Path) -> None:
    conn = _populated_db(tmp_path)
    out = tmp_path / 'report'
    report_path = build_report(conn, out)
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    assert '| success | 2 |' in text
    assert '| failed | 1 |' in text
    assert 'no_features_extracted' in text
    assert 'BodyDiscCorrelateNav' in text
    assert 'IAPETUS' in text
    assert 'BodyDiscCorrelateNav vs BodyLimbNav' in text
    assert 'BodyBlobNav' in text  # ensemble exclusion section
    assert (out / 'status_counts.png').exists()
    assert (out / 'technique_usage.png').exists()
    assert (out / 'offsets_hist.png').exists()


def test_build_report_accepts_fcpath_output_dir(tmp_path: Path) -> None:
    """Every report artifact (markdown, charts, filelists, CSV) writes via FCPath."""
    conn = _populated_db(tmp_path)
    out = FCPath(str(tmp_path / 'report'))
    report_path = build_report(conn, out, top_n=2, filelists=True, csv_export=True)
    conn.close()
    assert 'Total images: 3' in report_path.read_text(encoding='utf-8')
    assert (tmp_path / 'report' / 'status_counts.png').exists()
    assert (tmp_path / 'report' / 'images.csv').exists()
    filelists = sorted(p.name for p in (tmp_path / 'report' / 'filelists').glob('*.txt'))
    assert len(filelists) > 0


def test_build_report_instrument_filter(tmp_path: Path) -> None:
    conn = _populated_db(tmp_path)
    out = tmp_path / 'report'
    report_path = build_report(conn, out, instrument='vgiss')
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    assert '| failed | 1 |' in text
    assert 'success' not in text.split('## Failure taxonomy')[0].split('## Success')[1]


def test_build_report_date_filter_excludes_out_of_range(tmp_path: Path) -> None:
    conn = _populated_db(tmp_path)
    out = tmp_path / 'report'
    report_path = build_report(conn, out, start_date='1999-01-01', end_date='2001-01-01')
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    # The Voyager frame (image_et = -6.0e8, year 1980) is outside the range.
    assert 'Total images: 2' in text


def test_build_report_min_image_filter(tmp_path: Path) -> None:
    """min_image keeps only images whose numeric portion is at or above it."""
    conn = _populated_db(tmp_path)
    out = tmp_path / 'report'
    report_path = build_report(conn, out, min_image='N1000000002')
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    assert 'Total images: 1' in text
    assert 'image number >= 1000000002' in text


def test_build_report_max_image_filter(tmp_path: Path) -> None:
    """max_image keeps only images whose numeric portion is at or below it."""
    conn = _populated_db(tmp_path)
    out = tmp_path / 'report'
    report_path = build_report(conn, out, max_image='5000000')
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    # Only the Voyager frame (3250013) falls at or below 5000000.
    assert 'Total images: 1' in text
    assert '| failed | 1 |' in text


def test_build_report_rejects_digitless_image_bound(tmp_path: Path) -> None:
    conn = _populated_db(tmp_path)
    with pytest.raises(ValueError, match='contains no digits'):
        build_report(conn, tmp_path / 'report', min_image='nodigits')
    conn.close()


def test_report_failure_taxonomy_section(tmp_path: Path) -> None:
    """Failed images classify by content; bodies get failure shares."""
    conn = _populated_db(tmp_path)
    out = tmp_path / 'report'
    report_path = build_report(conn, out)
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    assert '## Failure taxonomy by image content' in text
    # The failed Voyager frame recorded one body (IAPETUS) and stars.
    assert '| single-body | 1 |' in text
    assert '| single-body | no_features_extracted | 1 |' in text
    assert '### Per-body failure shares' in text
    assert '| IAPETUS | 1 | 2 | 0.333 |' in text


def test_report_offset_by_group_section(tmp_path: Path) -> None:
    """Offsets additionally break down by (instrument, image size)."""
    conn = _populated_db(tmp_path)
    out = tmp_path / 'report'
    report_path = build_report(conn, out)
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    assert '### By instrument and image size' in text
    assert '| coiss | 1024x1024 | 2 ' in text


def test_report_suspect_offset_section(tmp_path: Path) -> None:
    """An offset near the NAC search margin is flagged as suspect."""
    conn = _populated_db(tmp_path)
    # NAC 1024 CALIB margin is (50, 140); |dV| = 49 is 0.98 of the limit.
    _upsert(
        conn,
        _metadata(
            image_name='N1000000003_1_CALIB.IMG',
            image_shape=[1024, 1024],
            offset=[49.0, 10.0],
        ),
    )
    out = tmp_path / 'report'
    report_path = build_report(conn, out)
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    assert '## Suspect offsets (near the search limit)' in text
    assert 'Suspect images: 1 of 3 screened.' in text
    assert '| N1000000003_1_CALIB.IMG | coiss | 49.000 | 10.000 |' in text
    assert '(50.0, 140.0)' in text


def test_report_suspect_offset_reports_unresolved_limits(tmp_path: Path) -> None:
    """Images whose search limit cannot be resolved are called out."""
    conn = open_stats_db(tmp_path / 'stats.sqlite3')
    _upsert(conn, _metadata(image_name='X9999999.IMG', instrument='mystery'))
    out = tmp_path / 'report'
    report_path = build_report(conn, out)
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    assert 'Search limit could not be resolved for some images:' in text
    assert "mystery: no configured search limit for instrument 'mystery' (1 image(s))" in text


def _botsim_db(tmp_path: Path) -> sqlite3.Connection:
    """Two BOTSIM pairs: one consistent, one with a 2 px NAC-scale residual."""
    conn = open_stats_db(tmp_path / 'stats.sqlite3')
    docs = [
        _metadata(
            image_name='N1454725799_1_CALIB.IMG', image_shape=[1024, 1024], offset=[10.0, -20.0]
        ),
        _metadata(
            image_name='W1454725799_1_CALIB.IMG', image_shape=[1024, 1024], offset=[1.0, -2.0]
        ),
        _metadata(
            image_name='N1454725900_1_CALIB.IMG', image_shape=[1024, 1024], offset=[12.0, -20.0]
        ),
        _metadata(
            image_name='W1454725900_1_CALIB.IMG', image_shape=[1024, 1024], offset=[1.0, -2.0]
        ),
        # A pair whose WAC frame failed: identified but not compared.
        _metadata(
            image_name='N1454726000_1_CALIB.IMG', image_shape=[1024, 1024], offset=[1.0, 1.0]
        ),
        _metadata(
            image_name='W1454726000_1_CALIB.IMG',
            status='failed',
            status_reason='no_features_extracted',
            offset=None,
            confidence=0.0,
            confidence_rank='failed',
        ),
    ]
    for doc in docs:
        _upsert(conn, doc)
    return conn


def test_report_botsim_section(tmp_path: Path) -> None:
    conn = _botsim_db(tmp_path)
    out = tmp_path / 'report'
    report_path = build_report(conn, out, top_n=5)
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    assert '## BOTSIM pair consistency (Cassini ISS)' in text
    assert '| pairs identified | 3 |' in text
    assert '| pairs with both navigated | 2 |' in text
    # Residuals are 0.0 (consistent pair) and 2.0 (12 - 10*1), median 1.0.
    assert '| median residual (px) | 1.000 |' in text
    assert '| p95 residual (px) | 2.000 |' in text
    # Worst-pairs table leads with the 2 px pair.
    assert (
        '| 1454725900 | N1454725900_1_CALIB.IMG | W1454725900_1_CALIB.IMG '
        '| 2.000 | 0.000 | 2.000 |' in text
    )


def test_report_runtime_section(tmp_path: Path) -> None:
    conn = _populated_db(tmp_path)
    out = tmp_path / 'report'
    report_path = build_report(conn, out, top_n=2)
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    assert '## Run-time statistics' in text
    assert '| images with timing | 3 |' in text
    assert '| total (s) | 9.750 |' in text
    assert '| median (s) | 3.250 |' in text
    assert 'Slowest 2 image(s):' in text
    assert (out / 'runtime_hist.png').exists()


def test_report_runtime_section_skipped_without_timing(tmp_path: Path) -> None:
    """No ingested timing data: the run-time section is omitted."""
    conn = open_stats_db(tmp_path / 'stats.sqlite3')
    _upsert(conn, _metadata(elapsed_s=None))
    out = tmp_path / 'report'
    report_path = build_report(conn, out)
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    assert '## Run-time statistics' not in text
    assert not (out / 'runtime_hist.png').exists()


def test_report_top_n_lists_examples(tmp_path: Path) -> None:
    """top_n adds example image names to the categorical sections."""
    conn = _populated_db(tmp_path)
    out = tmp_path / 'report'
    report_path = build_report(conn, out, top_n=5)
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    assert 'Examples (up to 5 per reason):' in text
    assert '- no_features_extracted: C3250013_GEOMED.IMG' in text
    assert '- BodyBlobNav: N1000000002_1_CALIB.IMG' in text


def test_report_filelists_written(tmp_path: Path) -> None:
    """filelists writes one full image-name list per category."""
    conn = _populated_db(tmp_path)
    out = tmp_path / 'report'
    report_path = build_report(conn, out, filelists=True)
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    reason_list = out / 'filelists' / 'failure_reason_no_features_extracted.txt'
    assert reason_list.exists()
    assert reason_list.read_text(encoding='utf-8') == 'C3250013_GEOMED.IMG\n'
    excluded_list = out / 'filelists' / 'excluded_BodyBlobNav.txt'
    assert excluded_list.exists()
    assert excluded_list.read_text(encoding='utf-8') == 'N1000000002_1_CALIB.IMG\n'
    assert 'filelists/failure_reason_no_features_extracted.txt' in text


def test_report_csv_export(tmp_path: Path) -> None:
    """csv_export writes a one-row-per-image CSV with aggregate counts."""
    conn = _populated_db(tmp_path)
    out = tmp_path / 'report'
    report_path = build_report(conn, out, csv_export=True)
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    assert 'images.csv (3 row(s))' in text
    csv_lines = (out / 'images.csv').read_text(encoding='utf-8').splitlines()
    assert len(csv_lines) == 4
    header = csv_lines[0].split(',')
    assert header[0] == 'image_name'
    assert 'elapsed_s' in header
    assert header[-4:] == ['n_technique_rows', 'n_feature_sources', 'n_features', 'n_gated']
    # Rows are ordered by image name; the Voyager frame sorts first.
    assert csv_lines[1].startswith('C3250013_GEOMED.IMG,vgiss,')


def test_report_is_deterministic(tmp_path: Path) -> None:
    conn = _populated_db(tmp_path)
    out_a = tmp_path / 'a'
    out_b = tmp_path / 'b'
    text_a = build_report(conn, out_a, top_n=3, filelists=True, csv_export=True).read_text(
        encoding='utf-8'
    )
    text_b = build_report(conn, out_b, top_n=3, filelists=True, csv_export=True).read_text(
        encoding='utf-8'
    )
    conn.close()
    assert text_a == text_b


def test_main_report_cli(tmp_path: Path) -> None:
    conn = _populated_db(tmp_path)
    conn.close()
    out = tmp_path / 'report'
    exit_code = main_report(['--db', str(tmp_path / 'stats.sqlite3'), '--output-dir', str(out)])
    assert exit_code == 0
    assert (out / 'report.md').exists()


def test_main_report_cli_new_flags(tmp_path: Path) -> None:
    """The drill-down, range, suspect, and CSV flags parse and take effect."""
    conn = _populated_db(tmp_path)
    conn.close()
    out = tmp_path / 'report'
    exit_code = main_report(
        [
            '--db',
            str(tmp_path / 'stats.sqlite3'),
            '--output-dir',
            str(out),
            '--top-n',
            '3',
            '--filelists',
            '--csv',
            '--suspect-fraction',
            '0.8',
            '--min-image',
            '1',
        ]
    )
    assert exit_code == 0
    assert (out / 'images.csv').exists()
    assert (out / 'filelists').is_dir()
    text = (out / 'report.md').read_text(encoding='utf-8')
    assert 'at least 0.80 of the per-axis maximum expected pointing' in text
    assert 'image number >= 1' in text
