"""Tests for the navigation statistics system (ingest + report)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from spindoctor.cli.stats.classify import date_from_image_et, instrument_from_image_name
from spindoctor.cli.stats.ingest import ingest_metadata_files, main_ingest, rows_from_metadata
from spindoctor.cli.stats.report import build_report, main_report
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
) -> dict[str, Any]:
    """Build a minimal metadata document in the navigate_image_files shape.

    ``instrument=None`` omits the ``observation.instrument`` field to model
    a metadata document that predates the field.
    """
    if offset is None and status == 'success':
        offset = [1.5, -2.5]
    observation: dict[str, Any] = {
        'image_path': f'/holdings/{image_name}',
        'image_name': image_name,
    }
    if instrument is not None:
        observation['instrument'] = instrument
    return {
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
                    'source_model': 'body',
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


# --- classification ---


@pytest.mark.parametrize(
    ('image_name', 'expected'),
    [
        ('N1454725799_1_CALIB.IMG', 'coiss'),
        ('W1728613298_8.IMG', 'coiss'),
        ('C3250013_GEOMED.IMG', 'vgiss'),
        ('C0349632000R.IMG', 'gossi'),
        ('lor_0003103486_0x630_sci.fit', 'nhlorri'),
        ('mystery.dat', 'unknown'),
    ],
)
def test_instrument_from_image_name(image_name: str, expected: str) -> None:
    assert instrument_from_image_name(image_name) == expected


def test_date_from_image_et_j2000() -> None:
    assert date_from_image_et(0.0) == '2000-01-01'


def test_date_from_image_et_none() -> None:
    assert date_from_image_et(None) is None


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


def test_rows_from_metadata_prefers_recorded_instrument() -> None:
    """The recorded observation.instrument wins over the filename shape."""
    doc = _metadata(instrument='sim')
    image_row, _, _ = rows_from_metadata(doc, source_file='x.json')
    assert image_row['instrument'] == 'sim'


def test_rows_from_metadata_instrument_falls_back_to_filename() -> None:
    """A document without observation.instrument classifies by filename."""
    doc = _metadata(instrument=None)
    image_row, _, _ = rows_from_metadata(doc, source_file='x.json')
    assert image_row['instrument'] == 'coiss'


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


def test_upsert_replaces_children(tmp_path: Path) -> None:
    conn = open_stats_db(tmp_path / 'stats.sqlite3')
    doc = _metadata(per_technique=[_technique('BodyLimbNav', (1.0, 1.0))])
    upsert_image(conn, *rows_from_metadata(doc, source_file='x.json'))
    doc = _metadata(per_technique=[])
    upsert_image(conn, *rows_from_metadata(doc, source_file='x.json'))
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


# --- reporting ---


def _populated_db(tmp_path: Path) -> sqlite3.Connection:
    conn = open_stats_db(tmp_path / 'stats.sqlite3')
    docs = [
        _metadata(
            image_name='N1000000001_1_CALIB.IMG',
            per_technique=[
                _technique('BodyDiscCorrelateNav', (1.5, -2.5)),
                _technique('BodyLimbNav', (1.7, -2.3)),
            ],
        ),
        _metadata(
            image_name='N1000000002_1_CALIB.IMG',
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
        ),
    ]
    for doc in docs:
        upsert_image(conn, *rows_from_metadata(doc, source_file='x.json'))
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


def test_build_report_instrument_filter(tmp_path: Path) -> None:
    conn = _populated_db(tmp_path)
    out = tmp_path / 'report'
    report_path = build_report(conn, out, instrument='vgiss')
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    assert '| failed | 1 |' in text
    assert 'success' not in text.split('## Technique usage')[0].split('## Success')[1]


def test_build_report_date_filter_excludes_out_of_range(tmp_path: Path) -> None:
    conn = _populated_db(tmp_path)
    out = tmp_path / 'report'
    report_path = build_report(conn, out, start_date='1999-01-01', end_date='2001-01-01')
    conn.close()
    text = report_path.read_text(encoding='utf-8')
    # The Voyager frame (image_et = -6.0e8, year 1980) is outside the range.
    assert 'Total images: 2' in text


def test_report_is_deterministic(tmp_path: Path) -> None:
    conn = _populated_db(tmp_path)
    out_a = tmp_path / 'a'
    out_b = tmp_path / 'b'
    text_a = build_report(conn, out_a).read_text(encoding='utf-8')
    text_b = build_report(conn, out_b).read_text(encoding='utf-8')
    conn.close()
    assert text_a == text_b


def test_main_report_cli(tmp_path: Path) -> None:
    conn = _populated_db(tmp_path)
    conn.close()
    out = tmp_path / 'report'
    exit_code = main_report(['--db', str(tmp_path / 'stats.sqlite3'), '--output-dir', str(out)])
    assert exit_code == 0
    assert (out / 'report.md').exists()
