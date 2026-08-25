"""Tests for what the results-index opener will and will not open.

Every one of these is about a failure an operator meets on a bad day: an index
that was never built, one built by different code, a driver that is not
installed, a server that will not answer. Each has to say what went wrong and
what to do about it, because the alternative is a stack trace from inside a
database driver, or -- worse -- a run that quietly reads nothing.

What the opener does with a SQLite file specifically is in
``test_engine_sqlite.py``; how it names a URL without naming its password is in
``test_masking.py``.
"""

import dataclasses
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from tests.spindoctor.results_index.conftest import (
    EXPLODING_FACTORY_MESSAGE,
    exploding_factory,
    image_row,
    opened,
    sqlite_url_for,
    without_module,
)

from spindoctor.results_index import (
    IMAGES,
    SCHEMA_META,
    SCHEMA_VERSION,
    open_database,
    open_index,
    reporting_a_failed_read,
)

MISSING_DRIVER_URL = 'postgresql+psycopg://user@localhost:5432/spindoctor'

UNSUPPORTED_BACKEND_URL = 'mysql+mysqldb://user@localhost/spindoctor'

UNKNOWN_BACKEND_URL = 'frobnicate://user@localhost/spindoctor'

MALFORMED_URL = 'this is not a connection url :::'

UNREACHABLE_SERVER_URL = 'postgresql+psycopg://spindoctor@127.0.0.1:1/spindoctor'


def _index_stamped_with_another_version(tmp_path: Path) -> Path:
    """Build an index stamped with a version this code does not read.

    Parameters:
        tmp_path: Directory the database is created in.

    Returns:
        Path of the database file.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True) as engine, engine.begin() as connection:
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
        open_index(sqlite_url_for(missing))


def test_the_missing_database_message_names_the_ingest_program(tmp_path: Path) -> None:
    """The reader is told the one command that fixes it.

    Parameters:
        tmp_path: Directory the database would have lived in.
    """
    missing = tmp_path / 'index.sqlite3'
    with pytest.raises(ValueError, match='sd_results_index'):
        open_index(sqlite_url_for(missing))


def test_a_consumer_does_not_create_the_database_it_refused(tmp_path: Path) -> None:
    """A consumer that created an empty index would answer every later run wrongly.

    Parameters:
        tmp_path: Directory the database would have lived in.
    """
    missing = tmp_path / 'index.sqlite3'
    with pytest.raises(ValueError, match='there is no results index at'):
        open_index(sqlite_url_for(missing))
    assert missing.exists() is False


def test_an_ingest_run_does_create_the_database(tmp_path: Path) -> None:
    """The creating flag is what separates a builder from a reader.

    Parameters:
        tmp_path: Directory the database is created in.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True):
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
        open_index(sqlite_url_for(path))


def test_a_database_with_no_schema_meta_row_is_refused(tmp_path: Path) -> None:
    """An interrupted creation leaves tables with nothing stamped in them.

    Parameters:
        tmp_path: Directory holding the half-built database.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.delete())
    with pytest.raises(ValueError, match='no schema_meta row'):
        open_index(sqlite_url_for(path))


def test_an_in_memory_database_is_refused_by_a_consumer() -> None:
    """An in-memory URL starts empty every time, so it is never an index."""
    with pytest.raises(ValueError, match='names an in-memory SQLite database'):
        open_index('sqlite://')


def test_an_in_memory_database_is_refused_by_a_drop() -> None:
    """Every connection to one makes a different empty database.

    A drop answering "nothing to drop" over one would report the state of a
    database that came into being to be asked and goes when the answer is
    given.
    """
    with pytest.raises(ValueError, match='names an in-memory SQLite database'):
        open_database('sqlite://')


# ---------------------------------------------------------------------------
# The version gate
# ---------------------------------------------------------------------------


def test_a_database_stamped_with_another_version_is_refused(tmp_path: Path) -> None:
    """A column whose meaning changed reads as valid until someone checks.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = _index_stamped_with_another_version(tmp_path)
    with pytest.raises(ValueError, match=f'schema version {SCHEMA_VERSION + 1}'):
        open_index(sqlite_url_for(path))


def test_the_version_message_names_the_version_this_code_reads(tmp_path: Path) -> None:
    """Both numbers appear, so the reader can tell which side is stale.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = _index_stamped_with_another_version(tmp_path)
    with pytest.raises(ValueError, match=f'version {SCHEMA_VERSION} this code reads'):
        open_index(sqlite_url_for(path))


def test_the_version_message_says_to_delete_and_re_ingest(tmp_path: Path) -> None:
    """There are no migrations, so the instruction is the whole remedy.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = _index_stamped_with_another_version(tmp_path)
    with pytest.raises(ValueError, match='empty the database with sd_results_index --drop-index'):
        open_index(sqlite_url_for(path))


def test_the_version_gate_applies_to_a_creating_open_too(tmp_path: Path) -> None:
    """Ingest reads an existing database as much as it writes one.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = _index_stamped_with_another_version(tmp_path)
    with pytest.raises(ValueError, match='is not the version'):
        open_index(sqlite_url_for(path), create=True)


def test_a_creating_open_creates_no_table_in_the_version_it_refuses(tmp_path: Path) -> None:
    """The gate runs before creation, so a refused open builds nothing.

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
        open_index(sqlite_url_for(path), create=True)
    assert 'ingest_runs' not in _table_names(path)


def test_creating_twice_keeps_one_version_row(tmp_path: Path) -> None:
    """A second ingest opens the same database; it does not re-stamp it.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True):
        pass
    with opened(sqlite_url_for(path), create=True) as engine, engine.connect() as connection:
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
    with opened(sqlite_url_for(path), create=True) as engine, engine.begin() as connection:
        connection.execute(IMAGES.insert(), image_row())
    with opened(sqlite_url_for(path), create=True) as engine, engine.connect() as connection:
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
    without_module(monkeypatch, 'psycopg')
    with pytest.raises(ValueError, match=r'rms-spindoctor\[postgres\]'):
        open_index(MISSING_DRIVER_URL)


def test_the_missing_driver_message_names_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A program may resolve its URL from three places; the message says which.

    Parameters:
        monkeypatch: Fixture used to hide the driver.
    """
    without_module(monkeypatch, 'psycopg')
    with pytest.raises(ValueError, match=r'postgresql\+psycopg'):
        open_index(MISSING_DRIVER_URL)


def test_a_missing_driver_for_another_backend_is_a_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One exception type covers every refusal, whichever backend was named.

    Parameters:
        monkeypatch: Fixture used to hide the driver.
    """
    without_module(monkeypatch, 'MySQLdb')
    with pytest.raises(ValueError, match=r'mysql\+mysqldb'):
        open_index(UNSUPPORTED_BACKEND_URL)


def test_a_missing_driver_for_another_backend_names_the_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reader is told which import failed, not merely that one did.

    Parameters:
        monkeypatch: Fixture used to hide the driver.
    """
    without_module(monkeypatch, 'MySQLdb')
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
    without_module(monkeypatch, 'MySQLdb')
    with pytest.raises(ValueError, match='ships no driver for that backend') as excinfo:
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


def test_an_unexpected_failure_inside_the_driver_is_a_value_error_naming_the_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The translation is a catch-all, because the escapes are not enumerable.

    A dialect coerces its own connect arguments and reports a bad one as a bare
    ``ValueError`` naming nothing; only catching everything keeps the promise
    that a caller reporting the cause has one type to catch, a URL to name, and
    the driver's own exception still attached to debug from.

    Parameters:
        tmp_path: Directory the database would have lived in.
        monkeypatch: Fixture the failing engine factory is installed through.
    """
    path = tmp_path / 'index.sqlite3'
    monkeypatch.setattr(sqlalchemy, 'create_engine', exploding_factory)
    with pytest.raises(ValueError, match=EXPLODING_FACTORY_MESSAGE) as excinfo:
        open_index(sqlite_url_for(path), create=True)
    assert str(path) in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_engine_echo_is_off(tmp_path: Path) -> None:
    """Engine echo writes SQL through stdlib logging, which nothing here configures.

    The unset value is None rather than False, so the assertion is on the flag
    being off rather than on the identity of the default.

    Parameters:
        tmp_path: Directory holding the database.
    """
    with opened(sqlite_url_for(tmp_path / 'index.sqlite3'), create=True) as engine:
        echo = engine.echo
    assert bool(echo) is False


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
        open_index(sqlite_url_for(path))
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


def _timed_out() -> sqlalchemy.exc.SQLAlchemyError:
    """Return a database failure that carries no driver exception under it.

    Every failure a dropped table provokes is a statement a driver answered, so
    it has an ``orig`` to report.  A pool that ran out of connections, a
    connection used after it closed, and a result read after it was consumed do
    not: they are raised by the database layer itself, before or after any
    driver was asked anything, and they are what a production index under load
    produces.

    Returns:
        The failure, with the sentence it carries as its own.
    """
    return sqlalchemy.exc.TimeoutError(
        'QueuePool limit of size 5 overflow 10 reached, connection timed out'
    )


def test_a_failure_with_no_driver_message_reports_the_sentence_it_has(tmp_path: Path) -> None:
    """The report carries the driver's sentence, and this failure's own is all there is."""
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    with (
        pytest.raises(ValueError, match='connection timed out'),
        reporting_a_failed_read(url),
    ):
        raise _timed_out()


def test_a_failure_with_no_driver_message_does_not_report_the_word_none(tmp_path: Path) -> None:
    """``str(None)`` is ``'None'``, which reads as a driver that answered that."""
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    with (
        pytest.raises(ValueError) as excinfo,
        reporting_a_failed_read(url),
    ):
        raise _timed_out()
    assert 'None' not in str(excinfo.value)
