"""What one root's rows are allowed to decide about another root's files.

A row is keyed by ``(root_url, results_path_stub)``, and one index serves
several roots.  Two of them holding copies of one tree is the ordinary result of
a mirror, a restored backup or a rescue root beside a primary: the same stubs,
the same lengths and the same modification times under two roots.

So every question the ingest asks the index before it reads a file has to be
asked about the root it is walking.  A pass that answered from the other root's
rows would skip a file it has never read, and skipping is what leaves no row at
all -- neither an image row nor a refusal -- for a file that exists.  A consumer
reads that as an image nobody navigated, offers it for navigation again, and no
later pass corrects it, because the skip lasts as long as the file does not
change.  A single-root fixture cannot see any of this happen, which is why both
arms of the question are asked here with a second root stocked to change the
answer.
"""

import shutil
from pathlib import Path

import pdslogger
import sqlalchemy

from spindoctor.results_index import FAILED_FILES, normalize_root_url, open_index

from .conftest import index_url, ingest_tree, metadata_document, write_metadata

INGESTED = 'COISS_2001/data/a/N1454725799_1_CALIB'
"""Stub of the document both roots hold and the ingest reads."""

REFUSED = 'COISS_2001/data/a/edges'
"""Stub of the file both roots hold and the ingest refuses."""


def _one_tree_under_two_roots(tmp_path: Path) -> tuple[Path, Path]:
    """Write one results tree and copy it whole to a second root.

    The copy preserves each file's length and modification time, which is what
    ``rsync -a``, ``cp -a`` and a restored backup all produce, and which is
    everything the ingest compares before deciding it has already read a file.

    Parameters:
        tmp_path: Directory both roots live under.

    Returns:
        The root ingested first, and the copy of it ingested second.
    """
    first = tmp_path / 'primary'
    write_metadata(first, INGESTED, metadata_document())
    (first / f'{REFUSED}_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    second = tmp_path / 'rescue'
    shutil.copytree(first, second, copy_function=shutil.copy2)
    return first, second


def _refusals_under(url: str, root: Path) -> list[str]:
    """Return the stubs the index records a refusal of under one root.

    Parameters:
        url: The index to read.
        root: The results root to read them for.

    Returns:
        The recorded stubs, in the order the index yields them.
    """
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                sqlalchemy.select(FAILED_FILES.c.results_path_stub).where(
                    FAILED_FILES.c.root_url == normalize_root_url(root.as_posix())
                )
            )
            return [str(row[0]) for row in rows]
    finally:
        engine.dispose()


def test_a_document_read_under_one_root_is_read_again_under_another(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The second root's document has no row of its own until this pass writes one."""
    first, second = _one_tree_under_two_roots(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first], logger=quiet_logger)
    counts = ingest_tree(url, [second], logger=quiet_logger)
    assert counts.files_ingested == 1


def test_a_file_refused_under_one_root_is_read_again_under_another(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A refusal is evidence about one root's file, and the other root's is another file.

    The two are indistinguishable by everything the skip compares, so a refusal
    read without its root makes this pass decline to read a file it has never
    seen.
    """
    first, second = _one_tree_under_two_roots(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first], logger=quiet_logger)
    counts = ingest_tree(url, [second], logger=quiet_logger)
    assert counts.files_failed == 1


def test_a_file_refused_under_one_root_is_recorded_under_the_other_too(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A file that was skipped leaves no row, and no row is what "never navigated" reads as.

    The count above says the file was read; this says what reading it left
    behind, which is what a selection filter answers from.  A pass that skipped
    it would write neither an image row nor a refusal, so the root would answer
    that a file it holds does not exist.
    """
    first, second = _one_tree_under_two_roots(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first], logger=quiet_logger)
    ingest_tree(url, [second], logger=quiet_logger)
    assert _refusals_under(url, second) == [REFUSED]
