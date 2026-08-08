"""What the index holds between one ingest pass and the next.

A pass has to leave the index a true account of the tree, and has to leave the
next pass able to tell what it has already paid to read.  Three things follow.
A file that is not a navigation document is recorded as refused rather than
re-downloaded forever.  A document that has left the tree takes its row with
it, because absence of a row is what every consumer reads as "this image was
never navigated" and presence therefore has to mean the reverse.  And a root
the walk could not list at all is not a root that is empty, so its ingest run
is deliberately left unfinished.

The transactions are pinned here too.  A crash costs one chunk rather than a
whole run, and one image's delete and inserts share one transaction; both are
properties of what survives a failure, and neither shows up in a count of rows
after a run that went well.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from filecache import FCPath

from spindoctor.cli.stats import ingest as ingest_module
from spindoctor.cli.stats.ingest import METADATA_SUFFIX
from spindoctor.cli.stats.ingest_rows import (
    MetadataDocumentError,
    MetadataSource,
    rows_from_metadata,
)
from spindoctor.results_index import FAILED_FILES, IMAGES, INGEST_RUNS, TECHNIQUES, open_index

from .conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    technique,
    write_metadata,
    write_summary_png,
)

SOURCE = MetadataSource(
    root_url='/data/nav-results',
    results_path_stub='COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB',
    source_file='/data/nav-results/x_metadata.json',
    mtime_ns=1234567890123456789,
    size_bytes=4096,
    has_summary_png=True,
)


def _rows(connection: sqlalchemy.Connection, statement: Any) -> list[Any]:
    """Execute a statement and return its rows.

    Parameters:
        connection: An open connection.
        statement: The statement to run.

    Returns:
        The rows.
    """
    return list(connection.execute(statement))


# ---------------------------------------------------------------------------
# Documents of some other shape
# ---------------------------------------------------------------------------


def _with_navigation(**fields: Any) -> dict[str, Any]:
    """Return a valid document with fields of its navigation result replaced.

    Parameters:
        fields: Keys of ``navigation_result`` to overwrite.

    Returns:
        The document.
    """
    document = metadata_document()
    document['navigation_result'].update(fields)
    return document


def _malformed_documents() -> dict[str, dict[str, Any]]:
    """Return one document per container shape the schema does not allow.

    Every one of these reached a bare ``AttributeError`` or ``TypeError`` out of
    the middle of the converter before the shapes were checked, which ended the
    whole run and cost every other file in the tree.

    Returns:
        A name for each shape, and the document carrying it.
    """
    string_observation = metadata_document()
    string_observation['observation'] = 'N1454725799_1_CALIB.IMG'
    listed_navigation = metadata_document()
    listed_navigation['navigation_result'] = ['BodyLimbNav']
    string_timing = metadata_document()
    string_timing['timing'] = '3.25 seconds'
    return {
        'observation-is-a-string': string_observation,
        'navigation-result-is-a-list': listed_navigation,
        'timing-is-a-string': string_timing,
        'per-technique-is-a-string': _with_navigation(per_technique='BodyLimbNav'),
        'per-technique-holds-scalars': _with_navigation(per_technique=[1, 2]),
        'feature-inventory-holds-scalars': _with_navigation(feature_inventory=['BODY_DISC']),
        'feature-inventory-is-a-string': _with_navigation(feature_inventory='BODY_DISC'),
        'excluded-mixes-types': _with_navigation(excluded_from_consensus=['BodyBlobNav', 3]),
        'excluded-is-a-string': _with_navigation(excluded_from_consensus='BodyBlobNav'),
        'provenance-is-a-list': _with_navigation(provenance=['abc1234']),
        'times-is-a-number': _with_navigation(times=170000000.5),
        'pointing-is-a-list': _with_navigation(pointing=[1.0, 0.0, 0.0]),
        'image-classifier-is-a-string': _with_navigation(image_classifier='clean'),
        'diagnostics-is-a-string': _with_navigation(
            per_technique=[{'technique_name': 'BodyLimbNav', 'diagnostics': 'fine'}]
        ),
    }


_MALFORMED_PARAMS = [
    pytest.param(document, id=name) for name, document in _malformed_documents().items()
]


@pytest.mark.parametrize('document', _MALFORMED_PARAMS)
def test_a_document_of_another_shape_is_refused(document: dict[str, Any]) -> None:
    """A container the schema declares holds what the schema says, or nothing.

    Parameters:
        document: A document carrying one disallowed container shape.
    """
    with pytest.raises(MetadataDocumentError, match='not a current-schema navigation document'):
        rows_from_metadata(document, SOURCE)


@pytest.mark.parametrize('document', _MALFORMED_PARAMS)
def test_a_document_of_another_shape_costs_only_itself(
    document: dict[str, Any], tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One file of the wrong shape must not cost the rest of the tree.

    Parameters:
        document: A document carrying one disallowed container shape.
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725798_1_CALIB', document)
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert (counts.files_ingested, counts.files_failed) == (1, 1)


def test_an_unenumerated_failure_costs_only_its_own_file(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shapes are checked, and whatever nobody thought of is caught anyway."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725798_1_CALIB', metadata_document())
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    calls: list[int] = []
    real_rows = rows_from_metadata

    def occasionally_exploding(metadata: Any, source: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise ZeroDivisionError('a shape nobody enumerated')
        return real_rows(metadata, source)

    monkeypatch.setattr(ingest_module, 'rows_from_metadata', occasionally_exploding)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert (counts.files_ingested, counts.files_failed) == (1, 1)


def test_a_reason_says_the_file_was_never_a_navigation_result(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Several hundred of these are ordinary, and the tally has to read that way."""
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.failures_by_reason == {
        'not a current-schema navigation document (no observation.image_name)': 1
    }


def test_a_reason_carries_one_example_file(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A field-level diagnosis is only a judgement once one real file is named."""
    root = tmp_path / 'results'
    root.mkdir()
    path = root / 'edges_metadata.json'
    path.write_text('{"edges": []}', encoding='utf-8')
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert list(counts.example_by_reason.values()) == [path.as_posix()]


# ---------------------------------------------------------------------------
# What a refused file costs the next pass
# ---------------------------------------------------------------------------


def _counting_retrievals(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every batched retrieval the next ingest performs.

    Parameters:
        monkeypatch: Fixture the retrieval is wrapped through.

    Returns:
        The list each call appends its sub-path argument to.
    """
    retrievals: list[Any] = []
    real_retrieve = FCPath.retrieve

    def counted(self: FCPath, sub_path: Any = None, **kwargs: Any) -> Any:
        retrievals.append(sub_path)
        return real_retrieve(self, sub_path, **kwargs)

    monkeypatch.setattr(FCPath, 'retrieve', counted)
    return retrievals


def _tree_with_a_refused_file(tmp_path: Path) -> Path:
    """Write a results tree holding one document and one file that is not one.

    Parameters:
        tmp_path: Directory the tree lives under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    return root


def test_a_refused_file_is_not_read_again(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file with no row is still a file this ingest has already paid to read.

    A real tree holds hundreds that are not navigation documents.  Re-reading
    them every pass is, on a cloud root, hundreds of paid downloads per run --
    exactly the cost the index exists to remove.
    """
    root = _tree_with_a_refused_file(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    retrievals = _counting_retrievals(monkeypatch)
    ingest_tree(url, [root], logger=quiet_logger)
    assert retrievals == []


def test_a_refused_file_is_counted_as_skipped(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The pass has to account for it, since it accounted for it as seen."""
    root = _tree_with_a_refused_file(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert (counts.files_seen, counts.files_skipped) == (2, 2)


def test_force_re_reads_a_refused_file(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """The escape hatch reaches every file, not only the ones that ingested."""
    root = _tree_with_a_refused_file(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    counts = ingest_tree(url, [root], logger=quiet_logger, force=True)
    assert counts.files_failed == 1


def test_a_changed_refusal_is_read_again(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A file that was refused and then rewritten is a file nobody has read."""
    root = tmp_path / 'results'
    root.mkdir()
    path = root / 'thing_metadata.json'
    path.write_text('{"edges": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    path.write_text(json.dumps(metadata_document()), encoding='utf-8')
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert counts.files_ingested == 1


def test_a_refusal_that_becomes_a_document_leaves_no_refusal_behind(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Otherwise the next pass would skip a file that now has a row to write."""
    root = tmp_path / 'results'
    root.mkdir()
    path = root / 'thing_metadata.json'
    path.write_text('{"edges": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    path.write_text(json.dumps(metadata_document()), encoding='utf-8')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(FAILED_FILES.c.results_path_stub))
    engine.dispose()
    assert found == []


def test_a_document_that_stops_reading_loses_its_row(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A row nothing backs would answer confidently for an image nothing produced."""
    root = tmp_path / 'results'
    stub = 'VOL/N1454725799_1_CALIB'
    write_metadata(root, stub, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    (root / f'{stub}{METADATA_SUFFIX}').write_text('{"edges": []}', encoding='utf-8')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.results_path_stub))
    engine.dispose()
    assert found == []


def test_a_summary_png_written_after_the_document_is_noticed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The flag comes from the walk, so the walk is what has to be compared."""
    root = tmp_path / 'results'
    stub = 'VOL/N1454725799_1_CALIB'
    write_metadata(root, stub, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    write_summary_png(root, stub)
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.has_summary_png))
    engine.dispose()
    assert [bool(row.has_summary_png) for row in found] == [True]


# ---------------------------------------------------------------------------
# What leaving the tree costs
# ---------------------------------------------------------------------------


def test_a_deleted_document_loses_its_row(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Absence of a row means "never navigated", so presence must mean the reverse."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    gone = write_metadata(root, 'VOL/N1454725800_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    gone.unlink()
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(
            connection,
            sqlalchemy.select(IMAGES.c.results_path_stub).order_by(IMAGES.c.results_path_stub),
        )
    engine.dispose()
    assert [row.results_path_stub for row in found] == ['VOL/N1454725799_1_CALIB']


def test_a_deleted_document_is_counted_as_removed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A silent deletion of rows is not something an operator should discover later."""
    root = tmp_path / 'results'
    gone = write_metadata(root, 'VOL/N1454725800_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    gone.unlink()
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert counts.files_removed == 1


def test_a_deleted_document_takes_its_child_rows_with_it(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A technique row belonging to no image is a count nothing can explain."""
    root = tmp_path / 'results'
    gone = write_metadata(
        root,
        'VOL/N1454725800_1_CALIB',
        metadata_document(per_technique=[technique('BodyLimbNav', (1.0, 1.0))]),
    )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    gone.unlink()
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(TECHNIQUES.c.technique_name))
    engine.dispose()
    assert found == []


def test_another_roots_rows_are_left_alone(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A pass over one root is evidence about that root and no other."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_metadata(first, 'VOL/N1454725799_1_CALIB', metadata_document())
    write_metadata(second, 'VOL/N1454725800_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first, second], logger=quiet_logger)
    ingest_tree(url, [first], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES))
    engine.dispose()
    assert found[0][0] == 2


def test_a_partly_listed_root_removes_nothing(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory that could not be listed is not a directory that is empty."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL1/N1454725799_1_CALIB', metadata_document())
    write_metadata(root, 'VOL2/N1454725800_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)

    real_iterdir = FCPath.iterdir_metadata

    def unlistable_vol2(self: FCPath) -> Any:
        if self.name == 'VOL2':
            raise FileNotFoundError(self.as_posix())
        yield from real_iterdir(self)

    monkeypatch.setattr(FCPath, 'iterdir_metadata', unlistable_vol2)
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES))
    engine.dispose()
    assert found[0][0] == 2


def test_the_prune_refuses_a_listing_of_part_of_a_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A worker holding a share of a root would otherwise delete its peers' rows."""
    url = index_url(tmp_path / 'index.sqlite3')
    engine = open_index(url, create=True)
    listing = ingest_module._RootListing(directory_missed=True)
    try:
        with pytest.raises(ValueError, match='complete listing'):
            ingest_module._prune_missing(
                engine, '/data/nav-results', listing, {}, logger=quiet_logger
            )
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# A root nobody can list
# ---------------------------------------------------------------------------


def _finish_times(url: str) -> list[Any]:
    """Return the finish time of every ingest run in an index.

    Parameters:
        url: The index URL.

    Returns:
        One entry per run, in insertion order.
    """
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            return [
                row.finished_utc
                for row in _rows(connection, sqlalchemy.select(INGEST_RUNS.c.finished_utc))
            ]
    finally:
        engine.dispose()


def test_a_root_that_is_not_there_leaves_its_run_unfinished(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A completed run over a mistyped root answers "not navigated" for every image."""
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [tmp_path / 'absent'], logger=quiet_logger)
    assert _finish_times(url) == [None]


def test_a_root_that_is_not_there_is_counted(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The count is what the driver reports, since nothing else went wrong."""
    counts = ingest_tree(
        index_url(tmp_path / 'index.sqlite3'), [tmp_path / 'absent'], logger=quiet_logger
    )
    assert counts.roots_unreadable == 1


def test_an_empty_root_completes_its_run(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A root that exists and holds nothing has been ingested, and holds nothing."""
    root = tmp_path / 'results'
    root.mkdir()
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    assert [time is not None for time in _finish_times(url)] == [True]


def test_a_root_that_is_not_there_keeps_its_rows(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An unmounted root is not an emptied one, and must cost no row."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    for path in sorted(root.rglob('*_metadata.json')):
        path.unlink()
    (root / 'VOL').rmdir()
    root.rmdir()
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES))
    engine.dispose()
    assert found[0][0] == 1


# ---------------------------------------------------------------------------
# What one transaction holds
# ---------------------------------------------------------------------------


def _seven_images(tmp_path: Path) -> Path:
    """Write seven documents, enough to cross a chunk boundary twice.

    Parameters:
        tmp_path: Directory the tree lives under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    for index in range(7):
        write_metadata(
            root,
            f'VOL/N145472579{index}_1_CALIB',
            metadata_document(image_name=f'N145472579{index}_1_CALIB.IMG'),
        )
    return root


def test_a_crash_mid_run_costs_one_chunk_and_no_more(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chunking is what a crash costs, and counting rows afterwards is what says so.

    Seven images in chunks of three, failing on the fifth write: the first chunk
    is committed and the second is not, so exactly three rows survive.  One
    transaction for the whole run would leave none, and a commit per image would
    leave four.
    """
    monkeypatch.setattr(ingest_module, 'INGEST_COMMIT_CHUNK_SIZE', 3)
    root = _seven_images(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    written: list[Any] = []
    real_write = ingest_module._write_image

    def failing(connection: Any, rows: Any) -> Any:
        written.append(rows)
        if len(written) == 5:
            raise RuntimeError('the writer died')
        return real_write(connection, rows)

    monkeypatch.setattr(ingest_module, '_write_image', failing)
    with pytest.raises(RuntimeError, match='the writer died'):
        ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES))
    engine.dispose()
    assert found[0][0] == 3


def _statement_transactions(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, Any]]:
    """Ingest one image, recording the transaction each statement ran inside.

    The transaction objects are kept rather than their identities, so comparing
    two of them is a comparison of live objects rather than of addresses a
    garbage collector is free to reuse.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the execute wrapper is installed through.

    Returns:
        The opening line of each statement, with the transaction it ran inside.
    """
    root = tmp_path / 'results'
    write_metadata(
        root,
        'VOL/N1454725799_1_CALIB',
        metadata_document(per_technique=[technique('BodyLimbNav', (1.0, 1.0))]),
    )
    seen: list[tuple[str, Any]] = []
    real_execute = sqlalchemy.Connection.execute

    def watching(self: Any, statement: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append((str(statement).split('\n')[0], self.get_transaction()))
        return real_execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(sqlalchemy.Connection, 'execute', watching)
    ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    return seen


def _transaction_of(seen: list[tuple[str, Any]], opening: str) -> Any:
    """Return the transaction the first statement with this opening ran inside.

    Parameters:
        seen: What :func:`_statement_transactions` recorded.
        opening: Text the statement begins with.

    Returns:
        The transaction object.
    """
    return next(transaction for text, transaction in seen if text.startswith(opening))


def test_an_image_write_runs_inside_a_transaction(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two statements outside any transaction would compare equal and prove nothing."""
    seen = _statement_transactions(tmp_path, quiet_logger, monkeypatch)
    assert _transaction_of(seen, 'DELETE FROM images') is not None


def test_the_image_delete_and_its_insert_share_one_transaction(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent worker must never see the gap between the two."""
    seen = _statement_transactions(tmp_path, quiet_logger, monkeypatch)
    assert _transaction_of(seen, 'DELETE FROM images') is _transaction_of(
        seen, 'INSERT INTO images'
    )


def test_the_image_delete_and_its_child_inserts_share_one_transaction(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The delete cascades to the children, so their inserts belong with it."""
    seen = _statement_transactions(tmp_path, quiet_logger, monkeypatch)
    assert _transaction_of(seen, 'DELETE FROM images') is _transaction_of(
        seen, 'INSERT INTO techniques'
    )


# ---------------------------------------------------------------------------
# Absent is not empty
# ---------------------------------------------------------------------------


def _stored_json(database: Path, column: str) -> list[Any]:
    """Read one JSON column of every image row with a plain SQLite reader.

    Parameters:
        database: The index file.
        column: The column to read.

    Returns:
        One entry per row: the stored text, or None for SQL NULL.
    """
    connection = sqlite3.connect(database)
    try:
        return [row[0] for row in connection.execute(f'SELECT {column} FROM images')]
    finally:
        connection.close()


def test_an_absent_cmatrix_is_sql_null(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """``WHERE cmatrix IS NOT NULL`` has to find the rows that carry a matrix.

    Stored as the JSON value ``null`` the column would satisfy that test on every
    row ever written, and the CSV export would carry the text ``null``.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    database = tmp_path / 'index.sqlite3'
    ingest_tree(index_url(database), [root], logger=quiet_logger)
    assert _stored_json(database, 'cmatrix') == [None]


def test_a_recorded_cmatrix_is_not_sql_null(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """And the reverse, so the test tells the two cases apart in both directions."""
    document = metadata_document()
    document['navigation_result']['pointing'] = {
        'cmatrix': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', document)
    database = tmp_path / 'index.sqlite3'
    ingest_tree(index_url(database), [root], logger=quiet_logger)
    assert json.loads(str(_stored_json(database, 'cmatrix')[0])) == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def test_an_empty_exclusion_set_is_stored_as_an_empty_list(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Nothing excluded is a statement, and is not the same as nothing recorded."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document(excluded=[]))
    database = tmp_path / 'index.sqlite3'
    ingest_tree(index_url(database), [root], logger=quiet_logger)
    assert json.loads(str(_stored_json(database, 'excluded_from_consensus')[0])) == []
