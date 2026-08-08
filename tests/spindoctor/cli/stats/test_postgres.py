"""The statistics programs against a real PostgreSQL server.

SQLite accepts almost anything: it will subtract a boolean from an integer,
count distinct values of whatever it is handed, and treat a JSON column as
text.  A report that runs on SQLite therefore proves very little about the
backend an index shared across machines actually runs on.  These re-run the
frozen comparison, and the ingest that feeds it, against a server that enforces
the type discipline the schema declares.

The tier is opt-in: it is excluded by the default marker filter and skips itself
when ``SPINDOCTOR_TEST_POSTGRES_URL`` is unset, so a checkout with no server
still runs a green suite.  Each test gets a schema of its own from the
``postgres_url`` fixture, so a repeated run, or two workers of a parallel run,
never share a table.
"""

import csv
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from tests.spindoctor.cli.stats.conftest import (
    GOLDEN_DIR,
    RESULTS_TREE,
    ingest_tree,
    metadata_document,
    technique,
    write_metadata,
)

from spindoctor.cli.stats.report import build_report
from spindoctor.results_index import IMAGES, TECHNIQUES, open_index

pytestmark = pytest.mark.postgres

_FULL_VARIANT: dict[str, Any] = {'top_n': 5, 'filelists': True, 'csv_export': True}
"""The unfiltered report invocation the frozen output was produced by."""


def _report_from_tree(url: str, out: Path, logger: pdslogger.PdsLogger, **options: Any) -> Path:
    """Ingest the fixture tree into a server index and write the report.

    Parameters:
        url: The PostgreSQL URL to ingest into.
        out: Directory receiving the report.
        logger: Logger the ingest reports through.
        options: Report options.

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


def test_the_report_is_byte_identical_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The same tree, the same report, on the backend the queries were written for.

    Parameters:
        postgres_url: URL of a schema of this test's own.
        tmp_path: Directory the report is written into.
        quiet_logger: Logger the ingest reports through.
    """
    out = _report_from_tree(postgres_url, tmp_path / 'full', quiet_logger, **_FULL_VARIANT)
    frozen = (GOLDEN_DIR / 'full' / 'report.md').read_text(encoding='utf-8')
    assert (out / 'report.md').read_text(encoding='utf-8') == frozen


def test_the_csv_carries_the_same_images_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The export orders and covers the same images, on the same key.

    Parameters:
        postgres_url: URL of a schema of this test's own.
        tmp_path: Directory the report is written into.
        quiet_logger: Logger the ingest reports through.
    """
    out = _report_from_tree(postgres_url, tmp_path / 'full', quiet_logger, **_FULL_VARIANT)
    produced = [
        row['image_name']
        for row in csv.DictReader((out / 'images.csv').read_text(encoding='utf-8').splitlines())
    ]
    frozen_text = (GOLDEN_DIR / 'full' / 'images.csv').read_text(encoding='utf-8')
    frozen = [row['image_name'] for row in csv.DictReader(frozen_text.splitlines())]
    assert produced == frozen


def test_the_json_columns_round_trip_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A JSONB column decodes to a Python value, and the CSV re-encodes one.

    Parameters:
        postgres_url: URL of a schema of this test's own.
        tmp_path: Directory the report is written into.
        quiet_logger: Logger the ingest reports through.
    """
    out = _report_from_tree(postgres_url, tmp_path / 'full', quiet_logger, **_FULL_VARIANT)
    frozen_text = (GOLDEN_DIR / 'full' / 'images.csv').read_text(encoding='utf-8')
    frozen = {
        row['image_name']: row['excluded_from_consensus']
        for row in csv.DictReader(frozen_text.splitlines())
    }
    produced = {
        row['image_name']: row['excluded_from_consensus']
        for row in csv.DictReader((out / 'images.csv').read_text(encoding='utf-8').splitlines())
    }
    assert produced == frozen


def test_the_unrounded_offset_survives_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Fifteen significant digits through a double-precision column.

    Parameters:
        postgres_url: URL of a schema of this test's own.
        tmp_path: Directory the index and the tree live under.
        quiet_logger: Logger the ingest reports through.
    """
    root = tmp_path / 'results'
    write_metadata(
        root,
        'VOL/N1454725799_1_CALIB',
        metadata_document(offset=[3.14159265358979, -2.71828182845905]),
    )
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    engine = open_index(postgres_url)
    with engine.connect() as connection:
        found = list(connection.execute(sqlalchemy.select(IMAGES.c.offset_dv)))
    engine.dispose()
    assert found[0][0] == 3.14159265358979


def test_a_boolean_column_holds_a_boolean_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An integer flag would be a type error the moment the report read it.

    Parameters:
        postgres_url: URL of a schema of this test's own.
        tmp_path: Directory the index and the tree live under.
        quiet_logger: Logger the ingest reports through.
    """
    root = tmp_path / 'results'
    write_metadata(
        root,
        'VOL/N1454725799_1_CALIB',
        metadata_document(per_technique=[technique('BodyLimbNav', (1.0, 1.0), spurious=True)]),
    )
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    engine = open_index(postgres_url)
    with engine.connect() as connection:
        found = list(connection.execute(sqlalchemy.select(TECHNIQUES.c.spurious)))
    engine.dispose()
    assert found[0][0] is True


def test_an_unchanged_file_is_skipped_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The incremental skip reads the metrics it stored, on either backend.

    Parameters:
        postgres_url: URL of a schema of this test's own.
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the ingest reports through.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    counts = ingest_tree(postgres_url, [root], logger=quiet_logger)
    assert counts.files_skipped == 1


def test_two_volumes_with_one_basename_produce_two_rows_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The composite primary key is a server-enforced constraint here.

    Parameters:
        postgres_url: URL of a schema of this test's own.
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the ingest reports through.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'COISS_2001/data/N1294561202_1_CALIB', metadata_document())
    write_metadata(root, 'COISS_2002/data/N1294561202_1_CALIB', metadata_document())
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    engine = open_index(postgres_url)
    with engine.connect() as connection:
        found = list(
            connection.execute(sqlalchemy.select(IMAGES.c.volume).order_by(IMAGES.c.volume))
        )
    engine.dispose()
    assert [row.volume for row in found] == ['COISS_2001', 'COISS_2002']
