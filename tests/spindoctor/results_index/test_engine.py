"""Tests for the results-index opener, its version gate, and its SQLite settings.

Every one of these is about a failure an operator meets on a bad day: an index
that was never built, one built by different code, a driver that is not
installed, a file on a filesystem that cannot lock. Each has to say what went
wrong and what to do about it, because the alternative is a stack trace from
inside a database driver, or -- worse -- a run that quietly reads nothing.
"""

import builtins
import dataclasses
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from tests.spindoctor.results_index.conftest import image_row, opened

from spindoctor.results_index import IMAGES, SCHEMA_META, SCHEMA_VERSION, open_index
from spindoctor.results_index import engine as engine_module

MISSING_DRIVER_URL = 'postgresql+psycopg://user:pw@localhost:5432/spindoctor'

UNSUPPORTED_BACKEND_URL = 'mysql+mysqldb://user@localhost/spindoctor'

UNKNOWN_BACKEND_URL = 'frobnicate://user@localhost/spindoctor'

MALFORMED_URL = 'this is not a connection url :::'

UNREACHABLE_SERVER_URL = 'postgresql+psycopg://spindoctor:pw@127.0.0.1:1/spindoctor'


def _url_for(path: Path) -> str:
    """Return the SQLite URL naming a filesystem path.

    Parameters:
        path: The database file's path.

    Returns:
        The URL.
    """
    return f'sqlite:///{path}'


def _without_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make one module unimportable, as it is on a machine without it installed.

    Asserting on a driver that merely happens to be absent from the current
    virtual environment is a test that stops testing the moment something pulls
    that driver in as a transitive dependency.

    Parameters:
        monkeypatch: Fixture the import hook is installed through.
        name: Dotted name of the module to hide, together with its submodules.
    """
    real_import = builtins.__import__

    def blocked(module_name: str, *args: Any, **kwargs: Any) -> Any:
        if module_name == name or module_name.startswith(f'{name}.'):
            raise ModuleNotFoundError(f'No module named {name!r}', name=name)
        return real_import(module_name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked)


def _index_stamped_with_another_version(tmp_path: Path) -> Path:
    """Build an index stamped with a version this code does not read.

    Parameters:
        tmp_path: Directory the database is created in.

    Returns:
        Path of the database file.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    return path


def _table_names(path: Path) -> set[str]:
    """Return the names of the tables a SQLite file holds.

    Read with the standard library rather than through the opener, so a database
    the opener refuses can still be inspected.

    Parameters:
        path: Path of the database file.

    Returns:
        The table names.
    """
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        return {str(name) for (name,) in rows}
    finally:
        connection.close()


def _read_only_index(tmp_path: Path) -> Path:
    """Build an index file SQLite can read but will never write.

    The journal mode is moved off write-ahead logging first: SQLite creates a
    file beside a write-ahead-logged database even to read it, so one that is
    read-only in a read-only directory cannot be read at all, which is a
    different case from this one.

    Parameters:
        tmp_path: Directory the database is created in.

    Returns:
        Path of the read-only database file.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True):
        pass
    connection = sqlite3.connect(path)
    try:
        connection.execute('PRAGMA journal_mode = DELETE')
    finally:
        connection.close()
    path.chmod(0o444)
    if os.access(path, os.W_OK):
        pytest.skip('this user can write a file whose mode forbids writing')
    return path


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
    path = _index_stamped_with_another_version(tmp_path)
    with pytest.raises(ValueError, match='is not the version'):
        open_index(_url_for(path), create=True)


def test_a_creating_open_writes_nothing_to_the_version_it_refuses(tmp_path: Path) -> None:
    """The gate runs before creation, so a refused open leaves the file alone.

    Otherwise a version bump writes this version's tables into a database
    stamped as another version, on the way to reporting that it will not read it.

    The refusal is checked against a table removed beforehand: table creation
    skips what is already there, so only a missing one shows whether it ran.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = _index_stamped_with_another_version(tmp_path)
    connection = sqlite3.connect(path)
    try:
        connection.execute('DROP TABLE ingest_runs')
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ValueError, match='is not the version'):
        open_index(_url_for(path), create=True)
    assert 'ingest_runs' not in _table_names(path)


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


def test_a_postgres_url_without_its_driver_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An import traceback would not tell an operator what to install.

    Parameters:
        monkeypatch: Fixture used to hide the driver.
    """
    _without_module(monkeypatch, 'psycopg')
    with pytest.raises(ValueError, match=r'rms-spindoctor\[postgres\]'):
        open_index(MISSING_DRIVER_URL)


def test_the_missing_driver_message_names_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A program may resolve its URL from three places; the message says which.

    Parameters:
        monkeypatch: Fixture used to hide the driver.
    """
    _without_module(monkeypatch, 'psycopg')
    with pytest.raises(ValueError, match=r'postgresql\+psycopg'):
        open_index(MISSING_DRIVER_URL)


def test_a_missing_driver_for_another_backend_is_a_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One exception type covers every refusal, whichever backend was named.

    Parameters:
        monkeypatch: Fixture used to hide the driver.
    """
    _without_module(monkeypatch, 'MySQLdb')
    with pytest.raises(ValueError, match=r'mysql\+mysqldb'):
        open_index(UNSUPPORTED_BACKEND_URL)


def test_a_missing_driver_for_another_backend_names_the_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reader is told which import failed, not merely that one did.

    Parameters:
        monkeypatch: Fixture used to hide the driver.
    """
    _without_module(monkeypatch, 'MySQLdb')
    with pytest.raises(ValueError, match='MySQLdb'):
        open_index(UNSUPPORTED_BACKEND_URL)


def test_a_missing_driver_for_another_backend_is_not_disguised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only PostgreSQL ships as an extra; other backends are not supported.

    Reporting a MySQL driver as a missing SpinDoctor extra would send the reader
    to install something that would still not work.

    Parameters:
        monkeypatch: Fixture used to hide the driver.
    """
    _without_module(monkeypatch, 'MySQLdb')
    with pytest.raises(ValueError) as excinfo:
        open_index(UNSUPPORTED_BACKEND_URL)
    assert 'rms-spindoctor[postgres]' not in str(excinfo.value)


def test_an_unknown_backend_is_a_value_error() -> None:
    """A URL scheme no driver claims is a configuration error like any other."""
    with pytest.raises(ValueError, match='no database driver for this URL scheme'):
        open_index(UNKNOWN_BACKEND_URL)


def test_the_unknown_backend_message_names_the_url_forms_that_work() -> None:
    """Naming the two supported forms is the whole remedy for a typed scheme."""
    with pytest.raises(ValueError, match=r'postgresql\+psycopg: URL naming a server'):
        open_index(UNKNOWN_BACKEND_URL)


def test_a_malformed_url_is_a_value_error() -> None:
    """A URL the parser rejects reaches the caller as the one type it catches."""
    with pytest.raises(ValueError, match='could not open the results index'):
        open_index(MALFORMED_URL)


def test_the_malformed_url_message_names_the_url() -> None:
    """The text the operator typed is what identifies which setting was wrong."""
    with pytest.raises(ValueError, match='this is not a connection url'):
        open_index(MALFORMED_URL)


def test_a_server_that_refuses_the_connection_is_a_value_error() -> None:
    """The common operational failure -- server down, wrong password -- is one type.

    A driver's own exception escaping here would defeat every consumer that
    reports the cause instead of crashing, and it is the failure they meet most.
    """
    pytest.importorskip('psycopg')
    with pytest.raises(ValueError, match='could not open the results index'):
        open_index(UNREACHABLE_SERVER_URL)


def test_the_refused_connection_message_names_the_url() -> None:
    """A driver's connection error names the server, never which setting supplied it."""
    pytest.importorskip('psycopg')
    with pytest.raises(ValueError, match=re.escape('127.0.0.1:1')):
        open_index(UNREACHABLE_SERVER_URL)


def test_a_translated_failure_keeps_the_driver_error_as_its_cause() -> None:
    """Translating the type must not throw away what the driver actually said."""
    pytest.importorskip('psycopg')
    with pytest.raises(ValueError) as excinfo:
        open_index(UNREACHABLE_SERVER_URL)
    assert isinstance(excinfo.value.__cause__, sqlalchemy.exc.SQLAlchemyError)


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


def test_a_locked_database_file_is_refused_by_a_creating_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer meets the same refusal a reader does, for the same reason.

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
            open_index(_url_for(path), create=True)
    finally:
        blocker.rollback()
        blocker.close()


# ---------------------------------------------------------------------------
# A read-only index
# ---------------------------------------------------------------------------


def test_a_read_only_index_is_opened_by_a_consumer(tmp_path: Path) -> None:
    """A reader cannot corrupt anything, so read-only media serve it.

    An archived copy, or one on a read-only mount, honors locking perfectly; the
    write it refuses is a write nobody on this path was going to make.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = _read_only_index(tmp_path)
    with opened(_url_for(path)) as engine, engine.connect() as connection:
        stamped = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).scalar()
    assert stamped == SCHEMA_VERSION


def test_a_read_only_index_is_refused_by_a_creating_open(tmp_path: Path) -> None:
    """Ingest writes, so a database it can never write is refused at open.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = _read_only_index(tmp_path)
    with pytest.raises(ValueError, match='this SQLite database is read-only'):
        open_index(_url_for(path), create=True)


def test_the_read_only_refusal_does_not_blame_the_filesystem(tmp_path: Path) -> None:
    """The message names read-only rather than the filesystem's locking.

    Sending an operator to fix locking on a filesystem that locks correctly costs
    the whole diagnosis.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = _read_only_index(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        open_index(_url_for(path), create=True)
    assert 'honors locking' not in str(excinfo.value)


def test_a_write_ahead_logged_index_in_a_read_only_directory_cannot_be_read(
    tmp_path: Path,
) -> None:
    """SQLite reads such a database through a file it creates beside it.

    A copy whose directory forbids writing therefore cannot be read either, and
    saying so beats letting a consumer's first query fail with a write error.

    Parameters:
        tmp_path: Directory the read-only directory is created in.
    """
    directory = tmp_path / 'read-only'
    directory.mkdir()
    path = directory / 'index.sqlite3'
    with opened(_url_for(path), create=True):
        pass
    path.chmod(0o444)
    directory.chmod(0o555)
    try:
        if os.access(path, os.W_OK):
            pytest.skip('this user can write a file whose mode forbids writing')
        with pytest.raises(ValueError, match='cannot be read either'):
            open_index(_url_for(path))
    finally:
        directory.chmod(0o755)
        path.chmod(0o644)


# ---------------------------------------------------------------------------
# Engine disposal
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _RefusedOpen:
    """What a refused open left behind.

    Attributes:
        engine: The engine the refused open built.
        pool: The pool that engine held before the open failed.
    """

    engine: Engine
    pool: QueuePool


def _refuse_an_open(path: Path, monkeypatch: pytest.MonkeyPatch) -> _RefusedOpen:
    """Refuse one open on a version mismatch and capture the engine it built.

    A caller of the opener never sees the engine of an open that raised, so it is
    recorded as it is created; the pool is recorded with it, because disposal
    replaces the one the engine holds.

    Parameters:
        path: Path of an index stamped with a version this code does not read.
        monkeypatch: Fixture the recording hook is installed through.

    Returns:
        The recorded engine and its pool.
    """
    engines: list[Engine] = []
    pools: list[QueuePool] = []
    real_create_engine = sqlalchemy.create_engine

    def recording(*args: Any, **kwargs: Any) -> Engine:
        made = real_create_engine(*args, **kwargs)
        engines.append(made)
        # A file-backed SQLite engine pools its connections; the assertion states
        # that, because only a pooling implementation can report what it holds.
        assert isinstance(made.pool, QueuePool)
        pools.append(made.pool)
        return made

    monkeypatch.setattr(sqlalchemy, 'create_engine', recording)
    with pytest.raises(ValueError, match='is not the version'):
        open_index(_url_for(path))
    return _RefusedOpen(engine=engines[0], pool=pools[0])


def test_a_refused_open_disposes_the_pool_it_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed open closes its connections rather than leaking them.

    The version gate reads through a connection, which returns to the pool; an
    undisposed pool is that connection's descriptor with no owner left to close
    it, and a caller that retries in a loop runs out of them.

    Parameters:
        tmp_path: Directory holding the database.
        monkeypatch: Fixture the recording hook is installed through.
    """
    refused = _refuse_an_open(_index_stamped_with_another_version(tmp_path), monkeypatch)
    assert refused.pool.checkedin() == 0


def test_a_refused_open_does_not_hand_back_the_pool_it_disposed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disposal replaces the pool, which is the observable proof it happened.

    Parameters:
        tmp_path: Directory holding the database.
        monkeypatch: Fixture the recording hook is installed through.
    """
    refused = _refuse_an_open(_index_stamped_with_another_version(tmp_path), monkeypatch)
    assert refused.engine.pool is not refused.pool
