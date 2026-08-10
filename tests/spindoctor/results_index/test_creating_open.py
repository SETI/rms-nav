"""What a creating open will and will not build an index on top of.

One rule is under test: an ingest never stamps a schema that already holds
tables SpinDoctor did not create.  The stamp is what every later reading treats
as proof that the tables beside it are the index's -- it is what the drop
destroys on the strength of -- so a stamp written over a stranger's table is
what makes somebody else's data indistinguishable from ours.

Four answers make up the rule, and each is asked here: a schema holding nothing
is built in, one carrying a stamp of SpinDoctor's is gone on with whatever
version it names, one holding tables of the index's own names with no such stamp
is refused, and one holding any table the index does not own is refused whether
it is stamped or not.  A refusal must also leave the schema exactly as it found
it, which is the half a message cannot assert on its own.
"""

from pathlib import Path

import pytest
import sqlalchemy
from tests.spindoctor.results_index.conftest import opened, sqlite_url_for

from spindoctor.results_index import (
    IMAGES,
    INGEST_RUNS,
    SCHEMA_META,
    SCHEMA_VERSION,
    index_table_names,
    open_index,
)

FOREIGN_TABLE = 'somebody_elses_table'
"""A table of the same schema that SpinDoctor did not create.

Named nothing like the index's own tables, so that what refuses it is its
being there rather than what it is called.
"""

COLLIDING_TABLE = IMAGES.name
"""A table SpinDoctor did not create, under a name SpinDoctor also uses.

``images`` is among the commonest table names there are, so a database is free
to hold one for a reason of its own, and a creating open that built over it
would write its stamp beside somebody else's rows.
"""

SQLITE_SCHEMA = 'main'
"""The one namespace a SQLite database has, which a refusal names."""


def _execute(path: Path, *statements: str) -> None:
    """Run statements against a database as somebody other than the index.

    Parameters:
        path: The database file's path.
        statements: The statements to run, in order.
    """
    engine = sqlalchemy.create_engine(sqlite_url_for(path))
    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.exec_driver_sql(statement)
    finally:
        engine.dispose()


def _tables(path: Path) -> list[str]:
    """Return every table a database holds, whoever created it.

    Parameters:
        path: The database file's path.

    Returns:
        The table names, sorted.
    """
    engine = sqlalchemy.create_engine(sqlite_url_for(path))
    try:
        return sorted(sqlalchemy.inspect(engine).get_table_names())
    finally:
        engine.dispose()


def _rows_of(path: Path, table: str) -> int:
    """Return how many rows a table holds.

    Parameters:
        path: The database file's path.
        table: The table to count.

    Returns:
        The row count.
    """
    engine = sqlalchemy.create_engine(sqlite_url_for(path))
    try:
        with engine.connect() as connection:
            counted = connection.exec_driver_sql(f'SELECT count(*) FROM {table}').scalar()
    finally:
        engine.dispose()
    return 0 if counted is None else int(counted)


def _refusal_of(path: Path) -> str:
    """Return the message a creating open of a database is refused with.

    Parameters:
        path: The database file's path.

    Returns:
        The refusal message.
    """
    with pytest.raises(ValueError) as excinfo:
        open_index(sqlite_url_for(path), create=True)
    return str(excinfo.value)


# ---------------------------------------------------------------------------
# A schema holding nothing
# ---------------------------------------------------------------------------


def test_a_database_holding_nothing_is_built_in(tmp_path: Path) -> None:
    """The ordinary first ingest, which every other answer is measured against.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True):
        pass
    assert _tables(path) == sorted(index_table_names())


def test_a_database_holding_nothing_is_stamped(tmp_path: Path) -> None:
    """A schema built in is a schema stamped, in the same open.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True) as engine, engine.connect() as connection:
        stamped = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).scalar()
    assert stamped == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# A schema carrying a stamp of SpinDoctor's
# ---------------------------------------------------------------------------


def test_an_index_of_this_version_is_opened_again(tmp_path: Path) -> None:
    """Every ingest after the first opens a schema that is already full.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True):
        pass
    with opened(sqlite_url_for(path), create=True):
        pass
    assert _tables(path) == sorted(index_table_names())


def test_a_stamp_of_another_version_reaches_the_version_gate(tmp_path: Path) -> None:
    """A stamp is evidence of ownership, not of a column set.

    The database a version bump leaves behind carries our marks and our names
    and neither our version nor our columns.  It is the index's own, and the
    answer it is owed is the version gate's, which prescribes the drop.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    path = tmp_path / 'index.sqlite3'
    _execute(
        path,
        f'CREATE TABLE {SCHEMA_META.name} '
        f'(singleton INTEGER PRIMARY KEY, schema_version INTEGER, created_utc TEXT)',
        f"INSERT INTO {SCHEMA_META.name} VALUES (1, {SCHEMA_VERSION - 1}, 'then')",
        f'CREATE TABLE {COLLIDING_TABLE} (root_url TEXT, a_column_of_an_older_version TEXT)',
    )
    assert f'schema version {SCHEMA_VERSION - 1} is not the version' in _refusal_of(path)


def test_a_stamp_of_another_version_is_not_built_over(tmp_path: Path) -> None:
    """Refused before anything is created, so the drop still has one version to remove.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    path = tmp_path / 'index.sqlite3'
    _execute(
        path,
        f'CREATE TABLE {SCHEMA_META.name} '
        f'(singleton INTEGER PRIMARY KEY, schema_version INTEGER, created_utc TEXT)',
        f"INSERT INTO {SCHEMA_META.name} VALUES (1, {SCHEMA_VERSION - 1}, 'then')",
        f'CREATE TABLE {COLLIDING_TABLE} (root_url TEXT)',
    )
    _refusal_of(path)
    assert _tables(path) == sorted([SCHEMA_META.name, COLLIDING_TABLE])


def test_a_stamp_with_no_row_in_it_is_completed(tmp_path: Path) -> None:
    """An interrupted creation left the marks, so the schema is the index's own.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    path = tmp_path / 'index.sqlite3'
    _execute(
        path,
        f'CREATE TABLE {SCHEMA_META.name} '
        f'(singleton INTEGER PRIMARY KEY, schema_version INTEGER, created_utc TEXT)',
    )
    with opened(sqlite_url_for(path), create=True):
        pass
    assert _tables(path) == sorted(index_table_names())


# ---------------------------------------------------------------------------
# A schema holding our names with no stamp of ours
# ---------------------------------------------------------------------------


def test_a_table_of_one_of_our_names_with_no_stamp_is_refused(tmp_path: Path) -> None:
    """The name is not evidence, so nothing is written beside it that would be.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    path = tmp_path / 'theirs.sqlite3'
    _execute(path, f'CREATE TABLE {COLLIDING_TABLE} (id INTEGER PRIMARY KEY, caption TEXT)')
    assert 'no schema_meta of SpinDoctor' in _refusal_of(path)


def test_that_refusal_names_the_table_it_stopped_on(tmp_path: Path) -> None:
    """An operator who has typed the wrong URL has to be able to tell from the message.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    path = tmp_path / 'theirs.sqlite3'
    _execute(path, f'CREATE TABLE {COLLIDING_TABLE} (id INTEGER PRIMARY KEY)')
    assert COLLIDING_TABLE in _refusal_of(path)


def test_that_refusal_names_the_schema_it_looked_at(tmp_path: Path) -> None:
    """One schema was examined, and which one is not something the URL said.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    path = tmp_path / 'theirs.sqlite3'
    _execute(path, f'CREATE TABLE {COLLIDING_TABLE} (id INTEGER PRIMARY KEY)')
    assert f'schema {SQLITE_SCHEMA}' in _refusal_of(path)


def test_that_refusal_names_the_url(tmp_path: Path) -> None:
    """A run resolves its index URL from three places, so the message says which one.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    path = tmp_path / 'theirs.sqlite3'
    _execute(path, f'CREATE TABLE {COLLIDING_TABLE} (id INTEGER PRIMARY KEY)')
    assert sqlite_url_for(path) in _refusal_of(path)


def test_a_refused_schema_is_not_stamped(tmp_path: Path) -> None:
    """The stamp is what a later drop destroys on the strength of.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    path = tmp_path / 'theirs.sqlite3'
    _execute(path, f'CREATE TABLE {COLLIDING_TABLE} (id INTEGER PRIMARY KEY)')
    _refusal_of(path)
    assert _tables(path) == [COLLIDING_TABLE]


def test_a_refused_schema_keeps_the_rows_it_had(tmp_path: Path) -> None:
    """Left exactly as it was found, which is the half the message cannot assert.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    path = tmp_path / 'theirs.sqlite3'
    _execute(
        path,
        f'CREATE TABLE {COLLIDING_TABLE} (id INTEGER PRIMARY KEY, caption TEXT)',
        f"INSERT INTO {COLLIDING_TABLE} (caption) VALUES ('their cat'), ('their dog')",
    )
    _refusal_of(path)
    assert _rows_of(path, COLLIDING_TABLE) == 2


def test_a_stamp_carrying_only_the_version_mark_is_not_evidence(tmp_path: Path) -> None:
    """One mark is what any migration table carries, and is not enough to build over.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    path = tmp_path / 'theirs.sqlite3'
    _execute(
        path,
        f'CREATE TABLE {SCHEMA_META.name} (schema_version INTEGER)',
        f'CREATE TABLE {COLLIDING_TABLE} (id INTEGER)',
    )
    assert 'no schema_meta of SpinDoctor' in _refusal_of(path)


def test_a_stamp_carrying_only_the_singleton_mark_is_not_evidence(tmp_path: Path) -> None:
    """The other mark alone is no better, so the pair is what is required.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    path = tmp_path / 'theirs.sqlite3'
    _execute(
        path,
        f'CREATE TABLE {SCHEMA_META.name} (singleton INTEGER PRIMARY KEY)',
        f'CREATE TABLE {COLLIDING_TABLE} (id INTEGER)',
    )
    assert 'no schema_meta of SpinDoctor' in _refusal_of(path)


def test_a_view_of_one_of_our_names_is_refused(tmp_path: Path) -> None:
    """A view of that name is what a create finds already there, and is not ours.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    path = tmp_path / 'theirs.sqlite3'
    _execute(path, f'CREATE VIEW {INGEST_RUNS.name} AS SELECT 1 AS run_id')
    assert INGEST_RUNS.name in _refusal_of(path)


# ---------------------------------------------------------------------------
# A schema holding a table the index does not own
# ---------------------------------------------------------------------------


def test_a_table_the_index_does_not_own_is_refused(tmp_path: Path) -> None:
    """A results index owns the schema it lives in, so anything else is a wrong URL.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    path = tmp_path / 'theirs.sqlite3'
    _execute(path, f'CREATE TABLE {FOREIGN_TABLE} (x INTEGER)')
    assert 'does not own' in _refusal_of(path)


def test_that_refusal_names_the_foreign_table(tmp_path: Path) -> None:
    """What the schema holds is what says the URL names somewhere else.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    path = tmp_path / 'theirs.sqlite3'
    _execute(path, f'CREATE TABLE {FOREIGN_TABLE} (x INTEGER)')
    assert FOREIGN_TABLE in _refusal_of(path)


def test_a_foreign_table_is_not_built_around(tmp_path: Path) -> None:
    """Nothing is created, so a schema that was somebody else's stays only theirs.

    Parameters:
        tmp_path: Directory the database file is written into.
    """
    path = tmp_path / 'theirs.sqlite3'
    _execute(path, f'CREATE TABLE {FOREIGN_TABLE} (x INTEGER)')
    _refusal_of(path)
    assert _tables(path) == [FOREIGN_TABLE]


def test_a_foreign_table_beside_a_stamped_index_is_refused(tmp_path: Path) -> None:
    """A stamp says the tables of our names are ours; it says nothing about the rest.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True):
        pass
    _execute(path, f'CREATE TABLE {FOREIGN_TABLE} (x INTEGER)')
    assert FOREIGN_TABLE in _refusal_of(path)


def test_a_foreign_table_beside_a_stamped_index_leaves_the_index(tmp_path: Path) -> None:
    """The refusal creates nothing and removes nothing, on a schema of either kind.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True):
        pass
    _execute(path, f'CREATE TABLE {FOREIGN_TABLE} (x INTEGER)')
    _refusal_of(path)
    assert _tables(path) == sorted([*index_table_names(), FOREIGN_TABLE])
