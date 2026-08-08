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
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import QueuePool
from tests.spindoctor.results_index.conftest import image_row, opened

from spindoctor.results_index import IMAGES, SCHEMA_META, SCHEMA_VERSION, open_index
from spindoctor.results_index import engine as engine_module

PASSWORD = 'sup3rs3cr3t'
"""A password distinctive enough that finding it anywhere is proof of a leak."""

MISSING_DRIVER_URL = 'postgresql+psycopg://user@localhost:5432/spindoctor'

UNSUPPORTED_BACKEND_URL = 'mysql+mysqldb://user@localhost/spindoctor'

UNKNOWN_BACKEND_URL = 'frobnicate://user@localhost/spindoctor'

MALFORMED_URL = 'this is not a connection url :::'

UNREACHABLE_SERVER_URL = 'postgresql+psycopg://spindoctor@127.0.0.1:1/spindoctor'


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


@pytest.fixture(params=['wal', 'delete'])
def read_only_index(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Path]:
    """Yield an index file SQLite can read but will never write.

    Both journal modes are covered, because the two answer a would-be writer
    differently and neither answer may be the one the opener depends on.
    Write-ahead logging is the mode this opener always leaves behind, and it is
    the permissive one: the journal-mode selection and a write lock both succeed
    on a file SQLite will never write. The rollback journal, which an index
    arrives in after a tool that checkpoints and vacuums it, refuses the journal
    mode outright.

    Parameters:
        request: Fixture carrying the journal mode of this case.
        tmp_path: Directory the database is created in.

    Yields:
        Path of the read-only database file.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True):
        pass
    connection = sqlite3.connect(path)
    try:
        connection.execute(f'PRAGMA journal_mode = {request.param}')
    finally:
        connection.close()
    path.chmod(0o444)
    if os.access(path, os.W_OK):
        pytest.skip('this user can write a file whose mode forbids writing')
    try:
        yield path
    finally:
        # Restored so the temporary directory can be removed by a later run.
        path.chmod(0o644)


@dataclasses.dataclass(frozen=True)
class _Route:
    """One way a URL is refused, and what its refusal has to say.

    Attributes:
        url: The URL to open, carrying a password.
        message: Pattern the refusal message must match.
        identifies: Text of the URL the message must keep, so a reader can tell
            which of the resolution levels supplied the value.
        cause: Exception type the refusal keeps as its ``__cause__``.
        hidden_module: Module to make unimportable first, or None.
        needs_psycopg: Whether the route reaches the PostgreSQL driver itself.
    """

    url: str
    message: str
    identifies: str
    cause: type[BaseException]
    hidden_module: str | None = None
    needs_psycopg: bool = False


REFUSAL_ROUTES = [
    pytest.param(
        _Route(
            url=f'postgresql+psycopg://user:{PASSWORD}@localhost:5432/spindoctor',
            message=r'rms-spindoctor\[postgres\]',
            identifies='localhost:5432',
            cause=ModuleNotFoundError,
            hidden_module='psycopg',
        ),
        id='driver-not-installed',
    ),
    pytest.param(
        _Route(
            url=f'mysql+mysqldb://user:{PASSWORD}@localhost/spindoctor',
            message='MySQLdb',
            identifies='mysql+mysqldb',
            cause=ModuleNotFoundError,
            hidden_module='MySQLdb',
        ),
        id='unsupported-backend',
    ),
    pytest.param(
        _Route(
            url=f'frobnicate://user:{PASSWORD}@localhost/spindoctor',
            message='no database driver for this URL scheme',
            identifies='frobnicate',
            cause=sqlalchemy.exc.NoSuchModuleError,
        ),
        id='unknown-scheme',
    ),
    pytest.param(
        _Route(
            url=f'postgresql+psycopg://user:{PASSWORD}@localhost:notaport/spindoctor',
            message='could not open the results index',
            identifies='localhost:notaport',
            cause=ValueError,
        ),
        id='unparseable-port',
    ),
    pytest.param(
        _Route(
            url=f'postgresql+psycopg://spindoctor:{PASSWORD}@127.0.0.1:1/spindoctor',
            message='could not open the results index',
            identifies='127.0.0.1:1',
            cause=sqlalchemy.exc.OperationalError,
            needs_psycopg=True,
        ),
        id='server-refuses-the-connection',
    ),
]
"""Every route by which a URL carrying a password reaches a refusal."""


def _refusal_of(route: _Route, monkeypatch: pytest.MonkeyPatch) -> ValueError:
    """Open a route's URL and return the refusal it raised.

    Parameters:
        route: The route to drive.
        monkeypatch: Fixture the import hook is installed through.

    Returns:
        The refusal, for assertions on what it says.
    """
    if route.needs_psycopg:
        pytest.importorskip('psycopg')
    if route.hidden_module is not None:
        _without_module(monkeypatch, route.hidden_module)
    with pytest.raises(ValueError, match=route.message) as excinfo:
        open_index(route.url)
    return excinfo.value


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


@pytest.mark.parametrize('route', REFUSAL_ROUTES)
def test_a_translated_failure_keeps_the_driver_error_as_its_cause(
    route: _Route, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Translating the type must not throw away what the driver actually said.

    Parameters:
        route: The refusal route under test.
        monkeypatch: Fixture the import hook is installed through.
    """
    assert isinstance(_refusal_of(route, monkeypatch).__cause__, route.cause)


@pytest.mark.parametrize('route', REFUSAL_ROUTES)
def test_a_refusal_does_not_repeat_the_password(
    route: _Route, monkeypatch: pytest.MonkeyPatch
) -> None:
    """These messages reach run logs and operators; a password reaches neither.

    Parameters:
        route: The refusal route under test.
        monkeypatch: Fixture the import hook is installed through.
    """
    assert PASSWORD not in str(_refusal_of(route, monkeypatch))


@pytest.mark.parametrize('route', REFUSAL_ROUTES)
def test_a_masked_refusal_still_names_the_rest_of_the_url(
    route: _Route, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Masking a password must not cost the identification the message is for.

    A program resolves its URL from a command line, a configuration file or the
    environment, and the URL is what says which of them supplied this one.

    Parameters:
        route: The refusal route under test.
        monkeypatch: Fixture the import hook is installed through.
    """
    assert route.identifies in str(_refusal_of(route, monkeypatch))


def test_an_unexpected_failure_inside_the_driver_is_still_a_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The translation is a catch-all, because the escapes are not enumerable.

    A dialect coerces its own connect arguments and reports a bad one as a bare
    ``ValueError`` naming nothing; only catching everything keeps the promise
    that a caller reporting the cause has one type to catch and a URL to name.

    Parameters:
        tmp_path: Directory the database would have lived in.
        monkeypatch: Fixture the failing engine factory is installed through.
    """

    def exploding(*args: Any, **kwargs: Any) -> Engine:
        raise RuntimeError('the dialect exploded')

    monkeypatch.setattr(sqlalchemy, 'create_engine', exploding)
    with pytest.raises(ValueError, match='the dialect exploded'):
        open_index(_url_for(tmp_path / 'index.sqlite3'), create=True)


def test_an_unexpected_failure_names_the_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal that names no URL leaves the reader with nothing to correct.

    Parameters:
        tmp_path: Directory the database would have lived in.
        monkeypatch: Fixture the failing engine factory is installed through.
    """

    def exploding(*args: Any, **kwargs: Any) -> Engine:
        raise RuntimeError('the dialect exploded')

    monkeypatch.setattr(sqlalchemy, 'create_engine', exploding)
    with pytest.raises(ValueError, match=re.escape(str(tmp_path / 'index.sqlite3'))):
        open_index(_url_for(tmp_path / 'index.sqlite3'), create=True)


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
    # Both connections held at once, so the pool cannot hand the second request
    # the connection it made for the first: it has to open another one.
    with (
        opened(_url_for(tmp_path / 'index.sqlite3'), create=True) as engine,
        engine.connect() as first,
        engine.connect() as second,
    ):
        reused = first.connection.dbapi_connection is second.connection.dbapi_connection
        found = second.exec_driver_sql(f'PRAGMA {pragma}').scalar()
    assert reused is False
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
# What SQLite could not open, and why
# ---------------------------------------------------------------------------


def test_a_path_that_is_a_directory_is_not_reported_as_a_locking_failure(tmp_path: Path) -> None:
    """SQLite refuses a directory with the code it refuses a missing one with.

    Parameters:
        tmp_path: Directory the misnamed path is created in.
    """
    directory = tmp_path / 'index.sqlite3'
    directory.mkdir()
    with pytest.raises(ValueError, match='is not a file'):
        open_index(_url_for(directory))


def test_a_file_that_is_not_a_database_says_so(tmp_path: Path) -> None:
    """A file at the index path is not therefore an index, or even a database.

    Parameters:
        tmp_path: Directory holding the file.
    """
    path = tmp_path / 'index.sqlite3'
    path.write_text('this is a note somebody left here\n')
    with pytest.raises(ValueError, match='not a SQLite database'):
        open_index(_url_for(path))


def test_a_file_that_is_not_a_database_is_not_blamed_on_the_filesystem(tmp_path: Path) -> None:
    """Moving to PostgreSQL would not make this file a database.

    Parameters:
        tmp_path: Directory holding the file.
    """
    path = tmp_path / 'index.sqlite3'
    path.write_text('this is a note somebody left here\n')
    with pytest.raises(ValueError) as excinfo:
        open_index(_url_for(path))
    assert 'honors locking' not in str(excinfo.value)


def test_an_ingest_into_a_directory_that_does_not_exist_names_the_directory(
    tmp_path: Path,
) -> None:
    """The likeliest first-run error there is: ingest before the tree exists.

    Parameters:
        tmp_path: Directory the missing directory would have lived in.
    """
    path = tmp_path / 'results' / 'index.sqlite3'
    with pytest.raises(ValueError, match=re.escape(f'{tmp_path / "results"} does not exist')):
        open_index(_url_for(path), create=True)


def test_a_directory_that_does_not_exist_is_not_answered_with_postgresql(
    tmp_path: Path,
) -> None:
    """Prescribing a database server for a directory nobody created is a wrong remedy.

    Parameters:
        tmp_path: Directory the missing directory would have lived in.
    """
    path = tmp_path / 'results' / 'index.sqlite3'
    with pytest.raises(ValueError) as excinfo:
        open_index(_url_for(path), create=True)
    assert 'postgresql' not in str(excinfo.value)


def test_a_file_this_user_cannot_open_names_the_permissions(tmp_path: Path) -> None:
    """An unreadable file is a permissions problem, not a locking one.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True):
        pass
    path.chmod(0o000)
    try:
        if os.access(path, os.R_OK):
            pytest.skip('this user can read a file whose mode forbids reading')
        with pytest.raises(ValueError, match='Check that this user may read'):
            open_index(_url_for(path))
    finally:
        path.chmod(0o644)


def test_a_driver_error_carrying_no_result_code_keeps_the_lock_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Classifying by result code must not fail on an exception that carries none.

    Parameters:
        tmp_path: Directory holding the database.
        monkeypatch: Fixture the failing statement execution is installed through.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True):
        pass

    def refusing(self: Any, statement: Any, *args: Any, **kwargs: Any) -> Any:
        raise sqlalchemy.exc.OperationalError(str(statement), None, Exception('nothing to go on'))

    monkeypatch.setattr(Connection, 'exec_driver_sql', refusing)
    with pytest.raises(ValueError, match='could not take a SQLite write lock'):
        open_index(_url_for(path))


# ---------------------------------------------------------------------------
# A SQLite URL is a plain path
# ---------------------------------------------------------------------------


def test_a_sqlite_url_with_a_query_string_is_refused(tmp_path: Path) -> None:
    """The driver opens a file named after the query, not the file named here.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(_url_for(path), create=True):
        pass
    with pytest.raises(ValueError, match='carries a query string'):
        open_index(f'{_url_for(path)}?uri=true&mode=ro')


def test_a_refused_query_string_leaves_no_file_behind(tmp_path: Path) -> None:
    """Such a URL passes the existence check and then creates a second file.

    The stray file is named for the query, so it is neither the database the
    operator meant nor an index anything can read.

    Parameters:
        tmp_path: Directory the database would have lived in.
    """
    path = tmp_path / 'index.sqlite3'
    with pytest.raises(ValueError, match='carries a query string'):
        open_index(f'{_url_for(path)}?uri=true&mode=ro', create=True)
    assert list(tmp_path.iterdir()) == []


def test_a_query_argument_the_driver_would_reject_is_refused_as_a_query_string(
    tmp_path: Path,
) -> None:
    """The driver coerces its connect arguments and reports a bad one bare.

    ``?timeout=abc`` reaches a float conversion inside the dialect, whose
    ``ValueError`` names neither the URL nor the setting; the rule that a SQLite
    index URL is a plain path refuses it before it gets there.

    Parameters:
        tmp_path: Directory the database would have lived in.
    """
    with pytest.raises(ValueError, match='carries a query string'):
        open_index(f'{_url_for(tmp_path / "index.sqlite3")}?timeout=abc', create=True)


# ---------------------------------------------------------------------------
# A read-only index
# ---------------------------------------------------------------------------


def test_a_read_only_index_is_opened_by_a_consumer(read_only_index: Path) -> None:
    """A reader cannot corrupt anything, so read-only media serve it.

    An archived copy, or one on a read-only mount, honors locking perfectly; the
    write it refuses is a write nobody on this path was going to make.

    Parameters:
        read_only_index: A database file this user cannot write.
    """
    with opened(_url_for(read_only_index)) as engine, engine.connect() as connection:
        stamped = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).scalar()
    assert stamped == SCHEMA_VERSION


def test_a_read_only_index_is_refused_by_a_creating_open(read_only_index: Path) -> None:
    """Ingest writes, so a database it can never write is refused at open.

    Parameters:
        read_only_index: A database file this user cannot write.
    """
    with pytest.raises(ValueError, match='this SQLite database is read-only'):
        open_index(_url_for(read_only_index), create=True)


def test_the_read_only_refusal_does_not_blame_the_filesystem(read_only_index: Path) -> None:
    """The message names read-only rather than the filesystem's locking.

    Sending an operator to fix locking on a filesystem that locks correctly costs
    the whole diagnosis.

    Parameters:
        read_only_index: A database file this user cannot write.
    """
    with pytest.raises(ValueError) as excinfo:
        open_index(_url_for(read_only_index), create=True)
    assert 'honors locking' not in str(excinfo.value)


def test_a_read_only_index_is_refused_before_anything_is_opened(read_only_index: Path) -> None:
    """The refusal comes from the filesystem, so no side file is left behind.

    A write-ahead-logged database that has been connected to leaves a
    shared-memory index beside itself; refusing before the engine connects is
    what keeps an ingest that will not run from touching the directory at all.

    Parameters:
        read_only_index: A database file this user cannot write.
    """
    with pytest.raises(ValueError, match='this SQLite database is read-only'):
        open_index(_url_for(read_only_index), create=True)
    assert sorted(path.name for path in read_only_index.parent.iterdir()) == ['index.sqlite3']


def test_an_ingest_into_a_read_only_directory_is_refused(tmp_path: Path) -> None:
    """SQLite writes its write-ahead log beside the database, so the directory counts.

    A writable file in a directory that permits nothing is a database ingest
    still cannot write, and SQLite grants the write lock on it anyway.

    Parameters:
        tmp_path: Directory the read-only directory is created in.
    """
    directory = tmp_path / 'archive'
    directory.mkdir()
    path = directory / 'index.sqlite3'
    with opened(_url_for(path), create=True):
        pass
    directory.chmod(0o555)
    try:
        if os.access(directory, os.W_OK):
            pytest.skip('this user can write a directory whose mode forbids writing')
        with pytest.raises(ValueError, match='is read-only, and ingest has to write the'):
            open_index(_url_for(path), create=True)
    finally:
        directory.chmod(0o755)


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
