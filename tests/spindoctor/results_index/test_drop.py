"""Removing the results index's own tables, and nothing else.

Three things are asked here.  That the drop removes exactly the tables the
schema declares and no other object of the database.  That it works on the
databases no opener will accept, since a version the gate refuses is the case
the drop exists for.  And that whatever it leaves behind is a state the version
gate reads as "not an index" rather than as a broken one, which is what the
order the tables go in is for.
"""

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from tests.spindoctor.results_index.conftest import image_row, opened, sqlite_url_for

from spindoctor.results_index import (
    IMAGES,
    INGEST_RUNS,
    METADATA,
    SCHEMA_META,
    SCHEMA_VERSION,
    drop_index_tables,
    index_contents,
    index_table_names,
    open_database,
    open_index,
)

FOREIGN_TABLE = 'somebody_elses_table'
"""A table of the same database that SpinDoctor did not create.

A PostgreSQL server is routinely shared, and a SQLite file is free to hold
anything, so the drop's promise is about the tables it names rather than about
the database around them.
"""


def _built(path: Path) -> str:
    """Create an index at a path and return its URL.

    Parameters:
        path: The database file's path.

    Returns:
        The URL of the created index.
    """
    url = sqlite_url_for(path)
    with opened(url, create=True):
        pass
    return url


def _table_names(url: str) -> list[str]:
    """Return every table a database holds, whoever created it.

    Parameters:
        url: The database URL.

    Returns:
        The table names, sorted.
    """
    engine = sqlalchemy.create_engine(url)
    try:
        return sorted(sqlalchemy.inspect(engine).get_table_names())
    finally:
        engine.dispose()


def _dropped(url: str) -> tuple[str, ...]:
    """Open a database, drop the index tables from it, and close it.

    Parameters:
        url: The database URL.

    Returns:
        The names of the tables that were dropped.
    """
    engine = open_database(url)
    try:
        return drop_index_tables(engine)
    finally:
        engine.dispose()


def _contents(url: str) -> Any:
    """Open a database and read what of the index it holds.

    Parameters:
        url: The database URL.

    Returns:
        The contents.
    """
    engine = open_database(url)
    try:
        return index_contents(engine)
    finally:
        engine.dispose()


def _add_foreign_table(url: str) -> None:
    """Create a table in the database that the index does not own.

    Parameters:
        url: The database URL.
    """
    engine = sqlalchemy.create_engine(url)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(f'CREATE TABLE {FOREIGN_TABLE} (x INTEGER)')
    finally:
        engine.dispose()


# The order the tables go in


def test_the_drop_covers_every_table_the_schema_declares() -> None:
    """The names come from the metadata, so a table added to it is dropped too.

    A hand-written list would be right on the day it was written and wrong on
    the day the next table was added, and the table it forgot is the one left
    standing over a database that reads as empty.
    """
    assert sorted(index_table_names()) == sorted(METADATA.tables)


def test_the_stamp_is_dropped_before_anything_it_stamps() -> None:
    """Every state an interrupted drop can leave is then one with no stamp.

    A stamp still standing over tables that have gone is the one state that must
    not be left: the version gate reads it as a healthy index of this version,
    and every consumer then fails inside its first query instead of being told
    to ingest.
    """
    assert index_table_names()[0] == SCHEMA_META.name


@pytest.mark.parametrize('child', ['techniques', 'feature_sources'])
def test_a_child_table_goes_before_the_table_it_references(child: str) -> None:
    """A table another one references cannot be dropped while that reference stands.

    Parameters:
        child: Name of a table carrying a foreign key into ``images``.
    """
    order = index_table_names()
    assert order.index(child) < order.index(IMAGES.name)


# What a drop removes


def test_the_drop_removes_every_table_of_the_index(tmp_path: Path) -> None:
    """The whole schema goes, not the part a query happened to name.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _dropped(url)
    assert _table_names(url) == []


def test_the_drop_names_what_it_removed(tmp_path: Path) -> None:
    """An operator reads the answer, so the answer is the table names.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    assert _dropped(url) == index_table_names()


def test_a_table_the_index_does_not_own_survives_the_drop(tmp_path: Path) -> None:
    """A shared database is the case this promise is about.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _add_foreign_table(url)
    _dropped(url)
    assert _table_names(url) == [FOREIGN_TABLE]


def _statements_of_a_drop(url: str) -> list[str]:
    """Return every statement a drop issued, in the order it issued them.

    Parameters:
        url: The database URL.

    Returns:
        The statements, stripped of surrounding space.
    """
    statements: list[str] = []
    engine = open_database(url)
    sqlalchemy.event.listen(
        engine,
        'before_cursor_execute',
        lambda conn, cursor, statement, *rest: statements.append(statement.strip()),
    )
    try:
        drop_index_tables(engine)
    finally:
        engine.dispose()
    return statements


def test_the_drop_issues_one_drop_table_per_index_table(tmp_path: Path) -> None:
    """Read off the statements themselves, rather than off what survived.

    A database this test can see the whole of would pass on its contents alone
    even if the drop had reached for the schema around them; what says it did
    not is that the only destructive statements it issued were these six, each
    naming one table by name.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    issued = _statements_of_a_drop(url)
    destructive = [statement for statement in issued if statement.upper().startswith('DROP')]
    assert destructive == [f'DROP TABLE {name}' for name in index_table_names()]


def test_the_drop_never_utters_a_table_it_does_not_own(tmp_path: Path) -> None:
    """Including in the statements it issues to find out what is there.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _add_foreign_table(url)
    named = [statement for statement in _statements_of_a_drop(url) if FOREIGN_TABLE in statement]
    assert named == []


def test_a_second_drop_removes_nothing(tmp_path: Path) -> None:
    """An idempotent drop has to be visibly idempotent, not silently so.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _dropped(url)
    assert _dropped(url) == ()


def test_a_second_drop_leaves_a_foreign_table_alone(tmp_path: Path) -> None:
    """A database holding none of the index is not written at all.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _add_foreign_table(url)
    _dropped(url)
    _dropped(url)
    assert _table_names(url) == [FOREIGN_TABLE]


# The databases no opener accepts


def test_an_index_stamped_with_another_version_can_be_dropped(tmp_path: Path) -> None:
    """The case the drop exists for: the gate's own remedy, made reachable.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    _dropped(url)
    assert _table_names(url) == []


def test_the_contents_report_a_version_this_code_does_not_read(tmp_path: Path) -> None:
    """What is about to go is named by the version it was written under.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    assert _contents(url).schema_version == SCHEMA_VERSION + 1


def test_a_database_holding_part_of_a_schema_can_be_dropped(tmp_path: Path) -> None:
    """An interrupted creation leaves one, and no opener will touch it.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    engine = sqlalchemy.create_engine(url)
    try:
        IMAGES.create(engine)
    finally:
        engine.dispose()
    assert _dropped(url) == (IMAGES.name,)


def test_a_stamp_whose_columns_are_not_this_version_does_not_stop_the_drop(
    tmp_path: Path,
) -> None:
    """The stamp is a fact reported, not a fact required.

    A ``schema_meta`` left over from a schema whose columns were different is
    exactly the shape an old index has, and refusing to say what a database
    holds because its stamp would not come out is withholding the drop from one
    of the cases it is for.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    engine = sqlalchemy.create_engine(url)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(f'CREATE TABLE {SCHEMA_META.name} (something_else INTEGER)')
    finally:
        engine.dispose()
    assert _contents(url).schema_version is None


# What the drop leaves behind


def test_a_database_with_no_stamp_reads_as_one_nobody_ingested(tmp_path: Path) -> None:
    """The state an interrupted drop leaves, put to the gate that reads it.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(sqlalchemy.schema.DropTable(SCHEMA_META))
    with pytest.raises(ValueError, match='this is not a results index'):
        open_index(url)


def test_a_dropped_index_is_rebuilt_by_the_next_creating_open(tmp_path: Path) -> None:
    """Left usable, rather than left absent: the next ingest builds it again.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _dropped(url)
    with opened(url, create=True):
        pass
    assert _table_names(url) == sorted(METADATA.tables)


def test_a_rebuilt_index_carries_this_version(tmp_path: Path) -> None:
    """And is stamped, so the gate lets a consumer in again.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    _dropped(url)
    with opened(url, create=True) as engine, engine.connect() as connection:
        stamped = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).scalar()
    assert stamped == SCHEMA_VERSION


def test_a_rebuilt_index_holds_none_of_the_dropped_rows(tmp_path: Path) -> None:
    """Starting from scratch is the point, so nothing may survive the rebuild.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(IMAGES.insert(), image_row())
    _dropped(url)
    with opened(url, create=True) as engine, engine.connect() as connection:
        remaining = connection.execute(
            sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES)
        ).scalar()
    assert remaining == 0


# What a drop says it is about to remove


def test_the_contents_count_the_rows_of_each_table(tmp_path: Path) -> None:
    """What is at stake is rows, so rows are what the account is in.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(IMAGES.insert(), image_row())
    counted = {table.name: table.rows for table in _contents(url).tables}
    assert counted[IMAGES.name] == 1


def test_the_contents_total_the_rows_across_the_tables(tmp_path: Path) -> None:
    """One number for a person deciding, beside the per-table breakdown.

    The stamp row is one of them, so a freshly created index holding one image
    totals two.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(IMAGES.insert(), image_row())
    assert _contents(url).rows == 2


def test_the_contents_report_the_stamp(tmp_path: Path) -> None:
    """A version that is not this one is what makes a drop the right move.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    assert _contents(url).schema_version == SCHEMA_VERSION


def test_the_contents_of_a_database_holding_no_index_are_empty(tmp_path: Path) -> None:
    """Which is what lets a drop answer without writing anything.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    _add_foreign_table(url)
    assert _contents(url).tables == ()


def test_the_contents_count_an_unfinished_ingest_run(tmp_path: Path) -> None:
    """A drop is allowed under one, so the person answering is told about it.

    Nothing in the index tells a pass that is writing it now from one that died,
    so the count is reported rather than acted on.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(
            INGEST_RUNS.insert(),
            {
                'root_url': '/data/nav-results',
                'started_utc': '2026-08-09T00:00:00+00:00',
                'finished_utc': None,
                'schema_version': SCHEMA_VERSION,
            },
        )
    assert _contents(url).unfinished_runs == 1


def test_a_finished_ingest_run_is_not_counted_as_unfinished(tmp_path: Path) -> None:
    """Otherwise every drop would carry the warning and none would mean it.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(
            INGEST_RUNS.insert(),
            {
                'root_url': '/data/nav-results',
                'started_utc': '2026-08-09T00:00:00+00:00',
                'finished_utc': '2026-08-09T00:01:00+00:00',
                'schema_version': SCHEMA_VERSION,
            },
        )
    assert _contents(url).unfinished_runs == 0


def test_the_unfinished_count_is_withheld_at_another_schema_version(tmp_path: Path) -> None:
    """The question is phrased in a column, and a version may have changed it.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = _built(tmp_path / 'index.sqlite3')
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    assert _contents(url).unfinished_runs is None


# The opener the drop uses


def test_a_sqlite_path_that_is_not_there_is_refused(tmp_path: Path) -> None:
    """A database that is not there is refused on both backends alike.

    A PostgreSQL database that does not exist is refused by the server, and a
    typed path deserves the same answer rather than a quiet success over
    nothing.

    Parameters:
        tmp_path: Directory the path names a file in.
    """
    with pytest.raises(ValueError, match='there is no results index at'):
        open_database(sqlite_url_for(tmp_path / 'absent.sqlite3'))


def test_the_absent_refusal_says_nothing_was_dropped(tmp_path: Path) -> None:
    """Rather than telling an operator who is deleting one to build one.

    Parameters:
        tmp_path: Directory the path names a file in.
    """
    with pytest.raises(ValueError, match='Nothing was dropped'):
        open_database(sqlite_url_for(tmp_path / 'absent.sqlite3'))


def test_a_refused_path_is_not_created(tmp_path: Path) -> None:
    """A mistyped URL must not leave an empty database where it pointed.

    Parameters:
        tmp_path: Directory the path names a file in.
    """
    path = tmp_path / 'absent.sqlite3'
    with pytest.raises(ValueError):
        open_database(sqlite_url_for(path))
    assert not path.exists()


def test_a_read_only_database_is_refused_before_a_table_goes(tmp_path: Path) -> None:
    """Refusing beats half-completing, so the question is asked at the open.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    path = tmp_path / 'index.sqlite3'
    url = _built(path)
    path.chmod(0o444)
    try:
        with pytest.raises(ValueError, match='dropping the index has to write it'):
            open_database(url)
    finally:
        path.chmod(0o644)


def test_the_read_only_refusal_does_not_prescribe_ingesting_a_copy(tmp_path: Path) -> None:
    """The remedy is per operation: a copy answers nothing for a drop.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    path = tmp_path / 'index.sqlite3'
    url = _built(path)
    path.chmod(0o444)
    try:
        with pytest.raises(ValueError) as excinfo:
            open_database(url)
    finally:
        path.chmod(0o644)
    assert 'Ingest a writable copy' not in str(excinfo.value)


def test_a_url_no_driver_reads_is_refused_as_a_value_error() -> None:
    """The drop's opener keeps the one-exception-type contract of the other.

    Parameters:
        None.
    """
    with pytest.raises(ValueError, match='no database driver for this URL scheme'):
        open_database('nosuchbackend://host/db')
