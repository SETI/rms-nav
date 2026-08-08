"""Tests for the results-index opener, its version gate, and its SQLite settings.

Every one of these is about a failure an operator meets on a bad day: an index
that was never built, one built by different code, a driver that is not
installed, a file on a filesystem that cannot lock. Each has to say what went
wrong and what to do about it, because the alternative is a stack trace from
inside a database driver, or -- worse -- a run that quietly reads nothing.
"""

import sqlite3
from pathlib import Path

import pytest
import sqlalchemy
from tests.spindoctor.results_index.conftest import image_row, opened

from spindoctor.results_index import IMAGES, SCHEMA_META, SCHEMA_VERSION, open_index
from spindoctor.results_index import engine as engine_module

MISSING_DRIVER_URL = 'postgresql+psycopg2://user:pw@localhost:5432/spindoctor'


def _url_for(path: Path) -> str:
    """Return the SQLite URL naming a filesystem path.

    Parameters:
        path: The database file's path.

    Returns:
        The URL.
    """
    return f'sqlite:///{path}'


# ---------------------------------------------------------------------------
# A database that is not there
# ---------------------------------------------------------------------------


def test_a_consumer_refuses_a_database_that_does_not_exist(tmp_path: Path) -> None:
    """Absence is an error, not an empty answer.

    Parameters:
        tmp_path: Directory the database would have lived in.
    """
    missing = tmp_path / 'index.sqlite3'
    with pytest.raises(ValueError, match='there is no results index at'):
        open_index(_url_for(missing))


def test_the_missing_database_message_names_the_ingest_program(tmp_path: Path) -> None:
    """The reader is told the one command that fixes it.

    Parameters:
        tmp_path: Directory the database would have lived in.
    """
    missing = tmp_path / 'index.sqlite3'
    with pytest.raises(ValueError, match='sd_stats_ingest'):
        open_index(_url_for(missing))


def test_a_consumer_does_not_create_the_database_it_refused(tmp_path: Path) -> None:
    """A consumer that created an empty index would answer every later run wrongly.

    Parameters:
        tmp_path: Directory the database would have lived in.
    """
    missing = tmp_path / 'index.sqlite3'
    with pytest.raises(ValueError, match='there is no results index at'):
        open_index(_url_for(missing))
    assert missing.exists() is False


def test_an_ingest_run_does_create_the_database(tmp_path: Path) -> None:
    """The creating flag is what separates a builder from a reader.

    Parameters:
        tmp_path: Directory the database is created in.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True):
        pass
    assert path.exists() is True


# ---------------------------------------------------------------------------
# A database that is there but is not an index
# ---------------------------------------------------------------------------


def test_a_database_with_no_schema_meta_table_is_refused(tmp_path: Path) -> None:
    """A file that happens to be SQLite is not therefore a results index.

    Parameters:
        tmp_path: Directory holding the unrelated database.
    """
    path = tmp_path / 'index.sqlite3'
    connection = sqlite3.connect(path)
    connection.execute('CREATE TABLE unrelated (x INTEGER)')
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match='not a results index'):
        open_index(_url_for(path))


def test_a_database_with_no_schema_meta_row_is_refused(tmp_path: Path) -> None:
    """An interrupted creation leaves tables with nothing stamped in them.

    Parameters:
        tmp_path: Directory holding the half-built database.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.delete())
    with pytest.raises(ValueError, match='no schema_meta row'):
        open_index(_url_for(path))


def test_an_in_memory_database_is_refused_by_a_consumer() -> None:
    """An in-memory URL starts empty every time, so it is never an index."""
    with pytest.raises(ValueError, match='not a results index'):
        open_index('sqlite://')


# ---------------------------------------------------------------------------
# The version gate
# ---------------------------------------------------------------------------


def test_a_database_stamped_with_another_version_is_refused(tmp_path: Path) -> None:
    """A column whose meaning changed reads as valid until someone checks.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    with pytest.raises(ValueError, match=f'schema version {SCHEMA_VERSION + 1}'):
        open_index(_url_for(path))


def test_the_version_message_names_the_version_this_code_reads(tmp_path: Path) -> None:
    """Both numbers appear, so the reader can tell which side is stale.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    with pytest.raises(ValueError, match=f'version {SCHEMA_VERSION} this code reads'):
        open_index(_url_for(path))


def test_the_version_message_says_to_delete_and_re_ingest(tmp_path: Path) -> None:
    """There are no migrations, so the instruction is the whole remedy.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    with pytest.raises(ValueError, match='delete the database and re-run sd_stats_ingest'):
        open_index(_url_for(path))


def test_the_version_gate_applies_to_a_creating_open_too(tmp_path: Path) -> None:
    """Ingest reads an existing database as much as it writes one.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    with pytest.raises(ValueError, match='is not the version'):
        open_index(_url_for(path), create=True)


def test_creating_twice_keeps_one_version_row(tmp_path: Path) -> None:
    """A second ingest opens the same database; it does not re-stamp it.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True):
        pass
    with opened(_url_for(path), create=True) as engine, engine.connect() as connection:
        rows = connection.execute(
            sqlalchemy.select(sqlalchemy.func.count()).select_from(SCHEMA_META)
        ).scalar()
    assert rows == 1


def test_creating_twice_keeps_the_rows_already_ingested(tmp_path: Path) -> None:
    """Creation is idempotent: an incremental ingest depends on it.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True) as engine, engine.begin() as connection:
        connection.execute(IMAGES.insert(), image_row())
    with opened(_url_for(path), create=True) as engine, engine.connect() as connection:
        rows = connection.execute(
            sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES)
        ).scalar()
    assert rows == 1


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_a_postgres_url_without_its_driver_names_the_extra() -> None:
    """An import traceback would not tell an operator what to install."""
    with pytest.raises(ValueError, match=r'rms-spindoctor\[postgres\]'):
        open_index(MISSING_DRIVER_URL)


def test_the_missing_driver_message_names_the_url() -> None:
    """A program may resolve its URL from three places; the message says which."""
    with pytest.raises(ValueError, match='postgresql\\+psycopg2'):
        open_index(MISSING_DRIVER_URL)


def test_a_missing_driver_for_another_backend_is_not_disguised() -> None:
    """Only PostgreSQL ships as an extra; other backends are not supported.

    Reporting a MySQL driver as a missing SpinDoctor extra would send the reader
    to install something that would still not work.
    """
    with pytest.raises(ModuleNotFoundError, match='MySQLdb'):
        open_index('mysql+mysqldb://user@localhost/spindoctor')


def test_engine_echo_is_off(tmp_path: Path) -> None:
    """Engine echo writes SQL through stdlib logging, which nothing here configures.

    The unset value is None rather than False, so the assertion is on the flag
    being off rather than on the identity of the default.

    Parameters:
        tmp_path: Directory holding the database.
    """
    with opened(_url_for(tmp_path / 'index.sqlite3'), create=True) as engine:
        echo = engine.echo
    assert bool(echo) is False


# ---------------------------------------------------------------------------
# SQLite connect-time settings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('pragma', 'expected'),
    [
        ('foreign_keys', 1),
        ('busy_timeout', engine_module.SQLITE_BUSY_TIMEOUT_MS),
        ('journal_mode', 'wal'),
    ],
)
def test_every_sqlite_connection_carries_the_pragma(
    tmp_path: Path, pragma: str, expected: object
) -> None:
    """The settings are applied per connection, not once at open.

    A pool hands out connections opened long after the engine was built, and a
    connection without these is a connection without the cascade, without
    concurrent readers, and without the wait that keeps two writers from failing.

    Parameters:
        tmp_path: Directory holding the database.
        pragma: The pragma to read back.
        expected: Its expected value.
    """
    with opened(_url_for(tmp_path / 'index.sqlite3'), create=True) as engine:
        # A fresh connection, so a value set only on the first one would not show.
        with engine.connect():
            pass
        with engine.connect() as connection:
            found = connection.exec_driver_sql(f'PRAGMA {pragma}').scalar()
    assert found == expected


def test_a_locked_database_file_is_refused_at_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe takes a write lock at open so a bad location fails early.

    A filesystem that cannot honor SQLite locking corrupts the file under
    concurrent writers instead of erroring, so the lock is taken while there is
    still something useful to say. Here the lock is denied by an exclusive holder
    rather than by a filesystem, which is the same refusal by a cheaper route.

    Parameters:
        tmp_path: Directory holding the database.
        monkeypatch: Fixture used to shorten the busy timeout.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True):
        pass
    monkeypatch.setattr(engine_module, 'SQLITE_BUSY_TIMEOUT_MS', 50)
    blocker = sqlite3.connect(path)
    try:
        blocker.execute('BEGIN EXCLUSIVE')
        with pytest.raises(ValueError, match='could not take a SQLite write lock'):
            open_index(_url_for(path))
    finally:
        blocker.rollback()
        blocker.close()


def test_the_lock_failure_names_postgresql_as_the_shared_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader who put the file on a network share needs the alternative.

    Parameters:
        tmp_path: Directory holding the database.
        monkeypatch: Fixture used to shorten the busy timeout.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True):
        pass
    monkeypatch.setattr(engine_module, 'SQLITE_BUSY_TIMEOUT_MS', 50)
    blocker = sqlite3.connect(path)
    try:
        blocker.execute('BEGIN EXCLUSIVE')
        with pytest.raises(ValueError, match='postgresql\\+psycopg'):
            open_index(_url_for(path))
    finally:
        blocker.rollback()
        blocker.close()


def test_a_refused_open_leaves_no_usable_engine_behind(tmp_path: Path) -> None:
    """A failed open disposes its engine rather than leaking the connection.

    Proven by deleting the file afterwards: a pooled connection still holding it
    open would keep a Windows-style lock, and on any platform an undisposed pool
    is a file descriptor nobody closes.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    with pytest.raises(ValueError, match='is not the version'):
        open_index(_url_for(path))
    path.unlink()
    assert path.exists() is False
