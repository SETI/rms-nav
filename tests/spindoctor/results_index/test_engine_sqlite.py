"""Tests for what the results-index opener does with a SQLite database file.

A SQLite index is a local file, and almost everything that can go wrong with one
is a property of the filesystem rather than of the database: a directory nobody
created, a file that is not a database, a mount that will not honor a write lock,
an archived copy nothing may write. SQLite reports most of those through one
exception type, and only one of them is a reason to move the index to a server.
These pin which message each cause gets, and pin the per-connection settings the
concurrency model depends on.
"""

import os
import re
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy.engine import Connection
from tests.spindoctor.results_index.conftest import opened, sqlite_url_for

from spindoctor.results_index import SCHEMA_META, SCHEMA_VERSION, open_index
from spindoctor.results_index import engine as engine_module


class _DriverError(Exception):
    """A driver exception carrying the result code SQLite would have set on it.

    The codes worth testing are the ones that are hard to provoke on demand: a
    disk that filled during an ingest, a file that corrupted under a crash, an
    I/O error from a mount that stopped answering. The classification reads the
    code off the driver's exception, so an exception carrying the code is the
    whole of what it needs.

    Attributes:
        sqlite_errorname: The result-code name, under the attribute the SQLite
            driver publishes it as.
    """

    def __init__(self, message: str, error_name: object) -> None:
        """Build the refusal.

        Parameters:
            message: What the driver would have said.
            error_name: The result-code name to carry, or any other value, since
                the classification has to survive an exception carrying
                something that is not a name at all.
        """
        super().__init__(message)
        self.sqlite_errorname = error_name


def _refusing_with(original: Exception) -> Callable[..., Any]:
    """Return a statement executor that fails every statement with one error.

    Parameters:
        original: The driver exception to wrap, standing in for what SQLite
            itself raised.

    Returns:
        A replacement for ``Connection.exec_driver_sql``.
    """

    def refusing(self: Any, statement: Any, *args: Any, **kwargs: Any) -> Any:
        raise sqlalchemy.exc.OperationalError(str(statement), None, original)

    return refusing


def _sqlite_error(message: str, error_name: str) -> sqlite3.Error:
    """Return a driver error carrying a result-code name.

    Parameters:
        message: What the driver would have said.
        error_name: The result-code name to carry.

    Returns:
        The error, of the type the connect handler catches.
    """
    error = sqlite3.OperationalError(message)
    error.sqlite_errorname = error_name
    return error


class _FailingCursor:
    """A cursor that records its statements and refuses the journal-mode one.

    Attributes:
        statements: Every statement handed to it, in order.
    """

    def __init__(self, error: sqlite3.Error, statements: list[str]) -> None:
        """Build the cursor.

        Parameters:
            error: What to raise for the journal-mode selection.
            statements: The list to record statements into.
        """
        self._error = error
        self.statements = statements

    def execute(self, statement: str) -> None:
        """Record a statement, refusing the journal-mode selection.

        Parameters:
            statement: The statement to run.

        Raises:
            sqlite3.Error: For the journal-mode selection, and only it.
        """
        self.statements.append(statement)
        if 'journal_mode' in statement:
            raise self._error

    def close(self) -> None:
        """Close the cursor, which holds nothing to release."""


class _FailingConnection:
    """A stand-in for the DBAPI connection the pool hands the connect handler.

    The driver's own cursor type is immutable and a failing disk cannot be
    arranged, so the handler is given a connection whose cursor fails instead.

    Attributes:
        statements: Every statement its cursors were given, in order.
    """

    def __init__(self, error: sqlite3.Error) -> None:
        """Build the connection.

        Parameters:
            error: What its cursor raises for the journal-mode selection.
        """
        self._error = error
        self.statements: list[str] = []

    def cursor(self) -> _FailingCursor:
        """Return a cursor recording into this connection.

        Returns:
            The cursor.
        """
        return _FailingCursor(self._error, self.statements)


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
    with opened(sqlite_url_for(path), create=True):
        pass
    connection = sqlite3.connect(path)
    try:
        connection.execute(f'PRAGMA journal_mode = {request.param}')
    finally:
        connection.close()
    path.chmod(0o444)
    try:
        if os.access(path, os.W_OK):
            pytest.skip('this user can write a file whose mode forbids writing')
        yield path
    finally:
        # Every file the fixture is responsible for, not only the database:
        # SQLite copies the database's mode onto the write-ahead log and the
        # shared-memory index it creates beside it, so a test that reads this
        # database leaves those behind read-only too. The skip is inside the
        # try for the same reason: a run on a user who can write anything must
        # still leave the directory as it found it.
        for made_read_only in path.parent.glob(f'{path.name}*'):
            made_read_only.chmod(0o644)


# ---------------------------------------------------------------------------
# Connect-time settings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('pragma', 'expected'),
    [
        ('foreign_keys', 1),
        ('busy_timeout', engine_module.SQLITE_BUSY_TIMEOUT_MS),
    ],
)
def test_every_sqlite_connection_carries_the_pragma(
    tmp_path: Path, pragma: str, expected: object
) -> None:
    """The settings are applied per connection, not once at open.

    A pool hands out connections opened long after the engine was built, and a
    connection without these is a connection without the cascade and without the
    wait that keeps two writers from failing. Both are per-connection state that
    a connection the settings missed simply does not have, which is what makes
    reading them back on a second connection a real test. The journal mode is
    not: it is a property of the database file, so it reads back the same
    whether or not this connection asked for it, and it is asserted separately
    as the persistent thing it is.

    Parameters:
        tmp_path: Directory holding the database.
        pragma: The pragma to read back.
        expected: Its expected value.
    """
    # Both connections held at once, so the pool cannot hand the second request
    # the connection it made for the first: it has to open another one.
    with (
        opened(sqlite_url_for(tmp_path / 'index.sqlite3'), create=True) as engine,
        engine.connect() as first,
        engine.connect() as second,
    ):
        reused = first.connection.dbapi_connection is second.connection.dbapi_connection
        found = second.exec_driver_sql(f'PRAGMA {pragma}').scalar()
    assert reused is False
    assert found == expected


def test_the_opener_leaves_the_database_write_ahead_logged(tmp_path: Path) -> None:
    """Write-ahead logging is what lets a reader and a writer work at once.

    The journal mode lives in the database header rather than in a connection,
    so it is read back from the closed file with the standard library: that is
    the state a later ingest, and every consumer, actually meets.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True):
        pass
    connection = sqlite3.connect(path)
    try:
        (mode,) = connection.execute('PRAGMA journal_mode').fetchone()
    finally:
        connection.close()
    assert mode == 'wal'


def test_a_connect_time_failure_that_is_not_read_only_is_not_swallowed() -> None:
    """Only a read-only refusal of the journal-mode selection is tolerated.

    The connect handler ignores the refusal a database SQLite will never write
    gives that selection, because such a database has no writers for a journal to
    protect. Widening the tolerance to every driver error would swallow a failing
    disk at the one moment the opener could still report it. The handler is
    called directly, because the driver's own cursor type cannot be replaced and
    a real disk cannot be made to fail on demand.
    """
    refusing = _FailingConnection(_sqlite_error('disk I/O error', 'SQLITE_IOERR_WRITE'))
    with pytest.raises(sqlite3.Error, match='disk I/O error'):
        engine_module._sqlite_on_connect(refusing, None)


def test_the_connect_time_settings_are_all_issued() -> None:
    """The journal mode is selected last, after the two per-connection settings.

    A handler that stopped early would leave a connection without the cascade or
    without the wait, which no read of the database file can show.
    """
    connection = _FailingConnection(_sqlite_error('database is locked', 'SQLITE_READONLY'))
    engine_module._sqlite_on_connect(connection, None)
    assert connection.statements == [
        'PRAGMA foreign_keys = ON',
        f'PRAGMA busy_timeout = {engine_module.SQLITE_BUSY_TIMEOUT_MS}',
        'PRAGMA journal_mode = WAL',
    ]


# ---------------------------------------------------------------------------
# The lock probe
# ---------------------------------------------------------------------------


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
    with opened(sqlite_url_for(path), create=True):
        pass
    monkeypatch.setattr(engine_module, 'SQLITE_BUSY_TIMEOUT_MS', 50)
    blocker = sqlite3.connect(path)
    try:
        blocker.execute('BEGIN EXCLUSIVE')
        with pytest.raises(ValueError, match='could not take a SQLite write lock'):
            open_index(sqlite_url_for(path))
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
    with opened(sqlite_url_for(path), create=True):
        pass
    monkeypatch.setattr(engine_module, 'SQLITE_BUSY_TIMEOUT_MS', 50)
    blocker = sqlite3.connect(path)
    try:
        blocker.execute('BEGIN EXCLUSIVE')
        with pytest.raises(ValueError, match='postgresql\\+psycopg'):
            open_index(sqlite_url_for(path))
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
    with opened(sqlite_url_for(path), create=True):
        pass
    monkeypatch.setattr(engine_module, 'SQLITE_BUSY_TIMEOUT_MS', 50)
    blocker = sqlite3.connect(path)
    try:
        blocker.execute('BEGIN EXCLUSIVE')
        with pytest.raises(ValueError, match='could not take a SQLite write lock'):
            open_index(sqlite_url_for(path), create=True)
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
        open_index(sqlite_url_for(directory))


def test_a_file_that_is_not_a_database_says_so(tmp_path: Path) -> None:
    """A file at the index path is not therefore an index, or even a database.

    Parameters:
        tmp_path: Directory holding the file.
    """
    path = tmp_path / 'index.sqlite3'
    path.write_text('this is a note somebody left here\n')
    with pytest.raises(ValueError, match='not a SQLite database'):
        open_index(sqlite_url_for(path))


def test_a_file_that_is_not_a_database_is_not_blamed_on_the_filesystem(tmp_path: Path) -> None:
    """Moving to PostgreSQL would not make this file a database.

    Parameters:
        tmp_path: Directory holding the file.
    """
    path = tmp_path / 'index.sqlite3'
    path.write_text('this is a note somebody left here\n')
    with pytest.raises(ValueError, match='not a SQLite database') as excinfo:
        open_index(sqlite_url_for(path))
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
        open_index(sqlite_url_for(path), create=True)


def test_a_directory_that_does_not_exist_is_not_answered_with_postgresql(
    tmp_path: Path,
) -> None:
    """Prescribing a database server for a directory nobody created is a wrong remedy.

    Parameters:
        tmp_path: Directory the missing directory would have lived in.
    """
    path = tmp_path / 'results' / 'index.sqlite3'
    with pytest.raises(ValueError, match='does not exist') as excinfo:
        open_index(sqlite_url_for(path), create=True)
    assert 'postgresql' not in str(excinfo.value)


def test_a_file_this_user_cannot_open_names_the_permissions(tmp_path: Path) -> None:
    """An unreadable file is a permissions problem, not a locking one.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True):
        pass
    path.chmod(0o000)
    try:
        if os.access(path, os.R_OK):
            pytest.skip('this user can read a file whose mode forbids reading')
        with pytest.raises(ValueError, match='Check that this user may read'):
            open_index(sqlite_url_for(path))
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
    with opened(sqlite_url_for(path), create=True):
        pass
    refusing = _refusing_with(Exception('nothing to go on'))
    monkeypatch.setattr(Connection, 'exec_driver_sql', refusing)
    with pytest.raises(ValueError, match='could not take a SQLite write lock'):
        open_index(sqlite_url_for(path))


def test_a_result_code_that_is_not_a_name_keeps_the_lock_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A code that is not text at all is as unclassifiable as no code.

    The attribute is read off whatever exception a driver raised, and nothing
    guarantees a string is what it holds. Comparing a non-string against the
    prefixes would raise from inside the code that exists to report a failure.

    Parameters:
        tmp_path: Directory holding the database.
        monkeypatch: Fixture the failing statement execution is installed through.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True):
        pass
    refusing = _refusing_with(_DriverError('disk I/O error', 5386))
    monkeypatch.setattr(Connection, 'exec_driver_sql', refusing)
    with pytest.raises(ValueError, match='could not take a SQLite write lock'):
        open_index(sqlite_url_for(path))


@pytest.mark.parametrize(
    ('error_name', 'detail', 'message'),
    [
        pytest.param(
            'SQLITE_IOERR_WRITE',
            'disk I/O error',
            'could not take a SQLite write lock',
            id='an-extended-io-error',
        ),
        pytest.param(
            'SQLITE_BUSY_SNAPSHOT',
            'database is locked',
            'could not take a SQLite write lock',
            id='an-extended-busy',
        ),
        pytest.param(
            'SQLITE_CANTOPEN_ISDIR',
            'unable to open database file',
            'Check that this user may read',
            id='an-extended-cannot-open',
        ),
    ],
)
def test_an_extended_result_code_gets_the_remedy_of_its_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_name: str, detail: str, message: str
) -> None:
    """SQLite refines several codes into extended forms naming the same cause.

    ``SQLITE_CANTOPEN`` alone has six of them and ``SQLITE_IOERR`` has more than
    twenty. Matching a code by equality rather than by prefix would send every
    one of those to the fall-through, which prescribes nothing, and would lose
    the diagnosis the family is entitled to. Which extended form a platform
    happens to produce is not something a test can arrange, so the codes are
    handed to the classification directly.

    Parameters:
        tmp_path: Directory holding the database.
        monkeypatch: Fixture the failing statement execution is installed through.
        error_name: The extended result code SQLite reports.
        detail: What the driver says alongside it.
        message: Text the refusal must carry, which is its family's remedy.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True):
        pass
    monkeypatch.setattr(
        Connection, 'exec_driver_sql', _refusing_with(_DriverError(detail, error_name))
    )
    with pytest.raises(ValueError, match=message):
        open_index(sqlite_url_for(path))


@pytest.mark.parametrize(
    ('error_name', 'detail'),
    [
        ('SQLITE_FULL', 'database or disk is full'),
        ('SQLITE_CORRUPT', 'database disk image is malformed'),
    ],
)
def test_an_unclassified_result_code_is_reported_as_what_sqlite_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_name: str, detail: str
) -> None:
    """A code with no classification gets SQLite's own words, not a guess.

    Parameters:
        tmp_path: Directory holding the database.
        monkeypatch: Fixture the failing statement execution is installed through.
        error_name: The result code SQLite reports.
        detail: What the driver says alongside it.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True):
        pass
    monkeypatch.setattr(
        Connection, 'exec_driver_sql', _refusing_with(_DriverError(detail, error_name))
    )
    with pytest.raises(ValueError, match=f'SQLite refused .*{error_name}'):
        open_index(sqlite_url_for(path))


@pytest.mark.parametrize(
    ('error_name', 'detail'),
    [
        ('SQLITE_FULL', 'database or disk is full'),
        ('SQLITE_CORRUPT', 'database disk image is malformed'),
    ],
)
def test_an_unclassified_result_code_is_not_answered_with_postgresql(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_name: str, detail: str
) -> None:
    """A full disk and a corrupt file are not reasons to rebuild on a server.

    That remedy belongs to a filesystem that will not honor locking, and to
    nothing else; prescribing it for every unrecognized code is how an operator
    ends up migrating a deployment over a disk that needed emptying.

    Parameters:
        tmp_path: Directory holding the database.
        monkeypatch: Fixture the failing statement execution is installed through.
        error_name: The result code SQLite reports.
        detail: What the driver says alongside it.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True):
        pass
    monkeypatch.setattr(
        Connection, 'exec_driver_sql', _refusing_with(_DriverError(detail, error_name))
    )
    with pytest.raises(ValueError, match='SQLite refused') as excinfo:
        open_index(sqlite_url_for(path))
    assert 'postgresql' not in str(excinfo.value)


def test_an_in_memory_database_that_cannot_be_opened_says_only_that(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-memory URL names no path, so the message that names one cannot be used.

    Parameters:
        monkeypatch: Fixture the failing statement execution is installed through.
    """
    monkeypatch.setattr(
        Connection,
        'exec_driver_sql',
        _refusing_with(_DriverError('unable to open database file', 'SQLITE_CANTOPEN')),
    )
    with pytest.raises(ValueError, match='could not open this database'):
        open_index('sqlite://')


# ---------------------------------------------------------------------------
# A SQLite URL is a plain path
# ---------------------------------------------------------------------------


def test_a_sqlite_url_with_a_query_string_is_refused(tmp_path: Path) -> None:
    """The driver opens a file named after the query, not the file named here.

    Parameters:
        tmp_path: Directory holding the database.
    """
    path = tmp_path / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True):
        pass
    with pytest.raises(ValueError, match='carries a query string'):
        open_index(f'{sqlite_url_for(path)}?uri=true&mode=ro')


def test_a_refused_query_string_leaves_no_file_behind(tmp_path: Path) -> None:
    """Such a URL passes the existence check and then creates a second file.

    The stray file is named for the query, so it is neither the database the
    operator meant nor an index anything can read.

    Parameters:
        tmp_path: Directory the database would have lived in.
    """
    path = tmp_path / 'index.sqlite3'
    with pytest.raises(ValueError, match='carries a query string'):
        open_index(f'{sqlite_url_for(path)}?uri=true&mode=ro', create=True)
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
        open_index(f'{sqlite_url_for(tmp_path / "index.sqlite3")}?timeout=abc', create=True)


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
    with opened(sqlite_url_for(read_only_index)) as engine, engine.connect() as connection:
        stamped = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).scalar()
    assert stamped == SCHEMA_VERSION


def test_a_read_only_index_is_refused_by_a_creating_open(read_only_index: Path) -> None:
    """Ingest writes, so a database it can never write is refused at open.

    Parameters:
        read_only_index: A database file this user cannot write.
    """
    with pytest.raises(ValueError, match='read-only, and ingest has to write it'):
        open_index(sqlite_url_for(read_only_index), create=True)


def test_the_read_only_refusal_does_not_blame_the_filesystem(read_only_index: Path) -> None:
    """The message names read-only rather than the filesystem's locking.

    Sending an operator to fix locking on a filesystem that locks correctly costs
    the whole diagnosis.

    Parameters:
        read_only_index: A database file this user cannot write.
    """
    with pytest.raises(ValueError, match='read-only, and ingest has to write it') as excinfo:
        open_index(sqlite_url_for(read_only_index), create=True)
    assert 'honors locking' not in str(excinfo.value)


def test_a_read_only_index_is_refused_before_anything_is_opened(read_only_index: Path) -> None:
    """The refusal comes from the filesystem, so no side file is left behind.

    A write-ahead-logged database that has been connected to leaves a
    shared-memory index beside itself; refusing before the engine connects is
    what keeps an ingest that will not run from touching the directory at all.

    Parameters:
        read_only_index: A database file this user cannot write.
    """
    with pytest.raises(ValueError, match='read-only, and ingest has to write it'):
        open_index(sqlite_url_for(read_only_index), create=True)
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
    with opened(sqlite_url_for(path), create=True):
        pass
    directory.chmod(0o555)
    try:
        if os.access(directory, os.W_OK):
            pytest.skip('this user can write a directory whose mode forbids writing')
        with pytest.raises(ValueError, match='is read-only, and ingest has to write the'):
            open_index(sqlite_url_for(path), create=True)
    finally:
        directory.chmod(0o755)


def test_an_ingest_is_refused_by_sqlite_when_the_mode_bits_said_it_could_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A network filesystem answers from mode bits it does not itself enforce.

    The filesystem check ahead of the engine is the ordinary diagnosis, and here
    it is made to answer wrongly on purpose, because that is what an NFS or SMB
    mount does: it reports a database this user may write, and SQLite is the one
    that finds out otherwise. The refusal has to say read-only anyway, rather
    than sending the operator to fix locking on a filesystem that locks fine.

    Parameters:
        tmp_path: Directory the read-only directory is created in.
        monkeypatch: Fixture the lying access check is installed through.
    """
    directory = tmp_path / 'archive'
    directory.mkdir()
    path = directory / 'index.sqlite3'
    with opened(sqlite_url_for(path), create=True):
        pass
    directory.chmod(0o555)
    try:
        if os.access(directory, os.W_OK):
            pytest.skip('this user can write a directory whose mode forbids writing')
        monkeypatch.setattr(os, 'access', lambda *args, **kwargs: True)
        with pytest.raises(ValueError, match='read-only, and ingest has to write it'):
            open_index(sqlite_url_for(path), create=True)
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
    with opened(sqlite_url_for(path), create=True):
        pass
    path.chmod(0o444)
    directory.chmod(0o555)
    try:
        if os.access(path, os.W_OK):
            pytest.skip('this user can write a file whose mode forbids writing')
        with pytest.raises(ValueError, match='cannot be read either'):
            open_index(sqlite_url_for(path))
    finally:
        directory.chmod(0o755)
        path.chmod(0o644)
