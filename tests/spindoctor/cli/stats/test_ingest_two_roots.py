"""Tests for the root half of the key, on every query that writes or removes.

An image is keyed by ``(root_url, results_path_stub)``, and two results trees
routinely hold the same stub: one volume's images are navigated twice, into a
production tree and a rescue tree.  Every delete an ingest issues names both
halves, and on a fixture holding one root a delete that named only the stub
would behave identically -- which is how a root-blind query ships.  So each of
them is exercised here with a second root present that holds the same stub, and
what is asserted is that the other root's row is still there afterwards.  The
count a pass reports of what its root's refusals amount to is a read of one
root, and is here for the same reason: two roots that disagree about it are what
tells a count of one from a count of the index.

The other reads are covered where they are used: a share's lookup in
``test_ingest_cloud_tasks``, the report's queries in ``test_report``.
"""

from pathlib import Path

import pdslogger
import pytest
import sqlalchemy

from spindoctor.results_index import FAILED_FILES, IMAGES, normalize_root_url, open_index

from .conftest import (
    FIRST_STUB,
    REFUSAL_REPORT_LEAD,
    index_url,
    ingest_tree,
    metadata_document,
    recorded_lines,
    refusal_report,
    write_metadata,
    write_refusal,
)

SECOND_STUB = 'VOL/N1454725800_1_CALIB'
"""A second stub, for the root that has to differ from the other in its count."""


def stubs_under(url: str, table: sqlalchemy.Table, root: Path) -> list[str]:
    """Return the stubs one table holds under one root.

    Parameters:
        url: The index URL.
        table: The table to read, ``images`` or ``failed_files``.
        root: The results root to ask about.

    Returns:
        The stubs, in name order.
    """
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            found = connection.execute(
                sqlalchemy.select(table.c.results_path_stub).where(
                    table.c.root_url == normalize_root_url(root)
                )
            )
            return sorted(str(row.results_path_stub) for row in found)
    finally:
        engine.dispose()


def test_ingesting_a_document_keeps_another_roots_refusal_of_that_stub(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Writing an image clears the refusal that stub used to carry, per root.

    A refusal is what stops the next pass paying to download and parse a file it
    has already refused, so one cleared by another root's ingest costs that
    download on every pass from then on, and the recorded reason for it with it.
    """
    refusing = tmp_path / 'refusing'
    holding = tmp_path / 'holding'
    write_refusal(refusing, FIRST_STUB)
    write_metadata(holding, FIRST_STUB, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [refusing], logger=quiet_logger)
    ingest_tree(url, [holding], logger=quiet_logger)
    assert stubs_under(url, FAILED_FILES, refusing) == [FIRST_STUB]


def test_refusing_a_file_keeps_another_roots_row_for_that_stub(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Recording a refusal removes the image row that stub used to have, per root.

    A document that stopped reading must not answer for its image any more, so
    its row goes as the refusal is written.  Removed on the evidence of the stub
    alone, another root's navigated image goes with it -- and every consumer
    reads the absence as "this image was never navigated" while that root's own
    ingest run says it completed.
    """
    holding = tmp_path / 'holding'
    refusing = tmp_path / 'refusing'
    write_metadata(holding, FIRST_STUB, metadata_document())
    write_refusal(refusing, FIRST_STUB)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [holding], logger=quiet_logger)
    ingest_tree(url, [refusing], logger=quiet_logger)
    assert stubs_under(url, IMAGES, holding) == [FIRST_STUB]


def test_refusing_a_file_keeps_another_roots_refusal_of_that_stub(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A refusal replaces the one recorded for its own root and no other's."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_refusal(first, FIRST_STUB)
    write_refusal(second, FIRST_STUB)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first], logger=quiet_logger)
    ingest_tree(url, [second], logger=quiet_logger)
    assert stubs_under(url, FAILED_FILES, first) == [FIRST_STUB]


def test_a_prune_keeps_another_roots_row_for_a_stub_it_removed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A document leaving one tree says nothing about the same stub in another.

    The prune is the one delete that acts on a whole root at once, so a
    root-blind one takes every other root's row for that stub with it -- while
    those roots' runs go on saying their ingest completed.
    """
    staying = tmp_path / 'staying'
    emptying = tmp_path / 'emptying'
    write_metadata(staying, FIRST_STUB, metadata_document())
    document = write_metadata(emptying, FIRST_STUB, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [staying, emptying], logger=quiet_logger)
    document.unlink()
    ingest_tree(url, [emptying], logger=quiet_logger)
    assert stubs_under(url, IMAGES, staying) == [FIRST_STUB]


def test_a_prune_keeps_another_roots_refusal_of_a_stub_it_removed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The prune reads both tables, so both of its deletes are keyed on the root."""
    staying = tmp_path / 'staying'
    emptying = tmp_path / 'emptying'
    write_refusal(staying, FIRST_STUB)
    refused = write_refusal(emptying, FIRST_STUB)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [staying, emptying], logger=quiet_logger)
    refused.unlink()
    ingest_tree(url, [emptying], logger=quiet_logger)
    assert stubs_under(url, FAILED_FILES, staying) == [FIRST_STUB]


def test_a_pass_reports_the_refusals_of_the_root_it_walked(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The standing count is a read, and a read of one root is keyed on it too.

    Read without its root, one number answers for the whole index: an operator
    measuring how short an error filter comes on this root is handed the sum of
    every root anybody ingested, and a root holding no refusal at all reports
    another root's.  The two roots are built to disagree about it, which a count
    over both cannot report as either.
    """
    written = recorded_lines(quiet_logger, monkeypatch)
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_refusal(first, FIRST_STUB)
    write_refusal(second, FIRST_STUB)
    write_refusal(second, SECOND_STUB)
    ingest_tree(index_url(tmp_path / 'index.sqlite3'), [first, second], logger=quiet_logger)
    assert [line for line in written if line.startswith(REFUSAL_REPORT_LEAD)] == [
        refusal_report(first, 1),
        refusal_report(second, 2),
    ]
