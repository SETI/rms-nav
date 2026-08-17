"""The seam every program reads its navigation records through.

Both storages answer both shapes -- one image by its stub, one mission in bulk --
so a consumer written against either shape works over either storage.  What is
tested here is the seam itself: the document-backed side of it, which is the
side no program had before the two consumers were unified, and the ceremony every
index-backed consumer opens an index through.

The index-backed side is exercised where its consumers are: over both storages
against the kernel writer's own readers in ``tests/spindoctor/cli/ck``, and
per-image against the pointing classifier in ``tests/spindoctor/cli/reproj``.
"""

import json
from pathlib import Path
from typing import Any

import pdslogger
import pytest
from filecache import FCPath
from sqlalchemy.engine import Engine
from tests.spindoctor.cli.stats.conftest import index_url, ingest_tree, write_metadata

from spindoctor.results_index import (
    IMAGES,
    IndexRecordSource,
    TreeRecordSource,
    build_record_source,
    open_index,
    open_index_for_roots,
    roots,
)
from spindoctor.support.nav_document import METADATA_SUFFIX


class _WatchedEngine:
    """An engine that counts how many times it was disposed of.

    Everything else is the engine's own: a caller of this stands in for the
    ceremony's caller, which connects and queries as usual.

    Parameters:
        engine: The engine to stand in front of.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self.disposals = 0

    def __getattr__(self, name: str) -> Any:
        """Return whatever the engine has under that name.

        Parameters:
            name: The attribute wanted.

        Returns:
            The engine's own attribute, so this stands in for one.
        """
        return getattr(self._engine, name)

    def dispose(self) -> None:
        """Dispose of the engine, and count it."""
        self.disposals += 1
        self._engine.dispose()


COLUMNS = (IMAGES.c.status, IMAGES.c.offset_dv, IMAGES.c.offset_du)
"""A consumer's columns, standing in for any consumer's."""

STUB = 'COISS_2001/N1454725799'
"""The image every test below reads."""


@pytest.fixture
def null_logger() -> pdslogger.PdsLogger:
    """Return a logger that writes nowhere.

    Returns:
        The logger, so an ingest driven by a test says nothing.
    """
    return pdslogger.NullLogger()


def _tree(tmp_path: Path) -> Path:
    """Write a results tree holding one navigated image.

    Parameters:
        tmp_path: Directory the tree is written under.

    Returns:
        The results root.
    """
    root = tmp_path / 'nav'
    write_metadata(
        root,
        STUB,
        {
            'status': 'success',
            'offset': [1.5, -2.5],
            'observation': {'image_name': 'N1454725799_1.IMG', 'instrument': 'coiss'},
        },
    )
    return root


def test_the_document_source_reads_one_image_by_its_stub(tmp_path: Path) -> None:
    """The per-image shape over the documents, which is what a stub is a key for."""
    source = TreeRecordSource(FCPath(_tree(tmp_path)))
    assert source.read_record(STUB)['offset'] == [1.5, -2.5]


def test_the_document_source_reads_one_mission_in_bulk(tmp_path: Path) -> None:
    """The bulk shape over the same tree, from the same source."""
    source = TreeRecordSource(FCPath(_tree(tmp_path)))
    records, unreadable = source.read_records('coiss')
    assert [record.stub for record in records] == [STUB]
    assert unreadable == []


def test_an_image_with_no_document_is_reported_as_one_nothing_recorded(
    tmp_path: Path,
) -> None:
    """The exception a caller reads as "this image was never navigated"."""
    source = TreeRecordSource(FCPath(_tree(tmp_path)))
    with pytest.raises(FileNotFoundError, match='COISS_2001/N9999999999'):
        source.read_record('COISS_2001/N9999999999')


def test_a_stub_that_escapes_the_root_is_refused_rather_than_read(tmp_path: Path) -> None:
    """A stub is a key, and a key holding ``..`` is a file outside the root.

    Refused as an unreadable record rather than as a path resolution, because
    that is what the caller can do something about, and the refusal names which
    rule was broken.
    """
    outside = tmp_path / f'elsewhere{METADATA_SUFFIX}'
    outside.write_text(json.dumps({'status': 'success'}))
    source = TreeRecordSource(FCPath(_tree(tmp_path)))
    with pytest.raises(FileNotFoundError, match='outside root'):
        source.read_record('../elsewhere')


def test_a_document_that_is_not_an_object_fails_the_image(tmp_path: Path) -> None:
    """A file that is not a document is a different failure from a missing one."""
    root = _tree(tmp_path)
    (root / 'COISS_2001' / f'N1111111111{METADATA_SUFFIX}').write_text('[1, 2]')
    source = TreeRecordSource(FCPath(root))
    with pytest.raises(ValueError, match='not a JSON object'):
        source.read_record('COISS_2001/N1111111111')


def test_the_document_source_says_where_it_read_from(tmp_path: Path) -> None:
    """A run log has to say which storage answered, not just how many answered."""
    root = _tree(tmp_path)
    assert TreeRecordSource(FCPath(root)).describe() == FCPath(root).as_posix()


def test_the_two_storages_answer_the_per_image_shape_alike(
    tmp_path: Path, null_logger: pdslogger.PdsLogger
) -> None:
    """The point of the seam, at the shape the reprojection stages read.

    Held on the record itself rather than on a classification of it, because a
    classifier agreeing about two different records is a weaker statement than
    the records being the same.
    """
    root = _tree(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger)
    index = build_record_source(FCPath(root), results_db_url=url, columns=COLUMNS)
    try:
        from_index = index.read_record(STUB)
    finally:
        index.close()
    from_tree = TreeRecordSource(FCPath(root)).read_record(STUB)
    assert from_index['offset'] == from_tree['offset']


def test_the_index_source_carries_only_the_columns_its_consumer_named(
    tmp_path: Path, null_logger: pdslogger.PdsLogger
) -> None:
    """Which is what makes a row cheaper than a document, and is a real difference.

    A document carries every field it has whatever the consumer asked for, so the
    two records are equal only in what was selected.  Asserted so that a consumer
    reading an unselected field is a known consequence rather than a surprise.
    """
    root = _tree(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger)
    source = build_record_source(FCPath(root), results_db_url=url, columns=COLUMNS)
    try:
        rebuilt = source.read_record(STUB)
    finally:
        source.close()
    assert 'observation' not in rebuilt


def test_opening_for_a_root_nobody_ingested_is_refused(tmp_path: Path) -> None:
    """Absence of a row means nothing under a root no pass has walked."""
    root = _tree(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    open_index(url, create=True).dispose()
    with pytest.raises(ValueError, match='no completed ingest'):
        open_index_for_roots(url, [FCPath(root).absolute().as_posix()])


def test_a_refused_open_disposes_the_engine_it_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal happens after the open, so the ceremony has to clean up.

    A caller that never received an engine cannot dispose of one, and a leaked
    pool per refused run is a connection nothing closes.  The disposal is watched
    directly rather than inferred from the database file: on this platform a file
    can be unlinked while it is still open, so a test that unlinked it would pass
    whether or not anything was disposed of.
    """
    root = _tree(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    watched = _WatchedEngine(open_index(url, create=True))
    monkeypatch.setattr(roots, 'open_index', lambda *_args, **_kwargs: watched)
    with pytest.raises(ValueError, match='no completed ingest'):
        open_index_for_roots(url, [FCPath(root).absolute().as_posix()])
    assert watched.disposals == 1


def test_no_index_url_reads_the_documents(tmp_path: Path) -> None:
    """Reading the documents is every program's default, and opens no database."""
    source = build_record_source(FCPath(_tree(tmp_path)), results_db_url=None, columns=COLUMNS)
    try:
        assert isinstance(source, TreeRecordSource)
    finally:
        source.close()


def test_an_index_url_reads_the_index(tmp_path: Path) -> None:
    """The other half of the choice, so the default above is a choice.

    Parameters:
        tmp_path: Directory the tree and the index are written under.
    """
    root = _tree(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=pdslogger.NullLogger())
    source = build_record_source(FCPath(root), results_db_url=url, columns=COLUMNS)
    try:
        assert isinstance(source, IndexRecordSource)
    finally:
        source.close()
