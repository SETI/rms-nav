"""Engine factory and version gate for the navigation results index.

:func:`open_index` is the only opener.  It selects the backend from the
connection URL, applies the SQLite settings the concurrency model depends on,
and refuses a database whose schema version is not the one this code reads.
Every refusal is a ``ValueError`` naming the URL, including the ones a database
driver raises, so a caller that reports failures catches one type.

Two URL forms are supported::

    sqlite:////data/nav-results/index.sqlite3
    postgresql+psycopg://user@host/spindoctor

A ``sqlite:`` URL names a **local filesystem path**.  It is the one path in this
system that is not a cloud-capable location: the C library opens it directly, so
a network filesystem that cannot honor its locking is refused at open rather than
corrupted later.  A read-only database is not that case and is not refused for
it: a consumer reads one, and only an opener meaning to write is turned away.
PostgreSQL is the option for sharing one index across machines, and its driver
ships as an optional extra.
"""

import datetime
import sqlite3
from pathlib import Path
from typing import Any

import sqlalchemy
from sqlalchemy.engine import URL, Engine

from spindoctor.results_index.schema import METADATA, SCHEMA_META, SCHEMA_VERSION

__all__ = ['SQLITE_BUSY_TIMEOUT_MS', 'open_index']

SQLITE_BUSY_TIMEOUT_MS = 30000
"""How long a SQLite connection waits for a competing writer before failing.

Multiple local writer processes are an ordinary case: several ingest workers on
one machine share one file.  With write-ahead logging and short transactions they
only contend briefly, so waiting is nearly always the right answer.
"""

_SQLITE_MEMORY = ':memory:'

_POSTGRES_BACKEND = 'postgresql'

_SQLITE_BACKEND = 'sqlite'

_SQLITE_READONLY_PREFIX = 'SQLITE_READONLY'
"""Prefix shared by every SQLite result code meaning "this database is read-only"."""

_SQLITE_READ_ONLY_KEY = 'spindoctor_sqlite_read_only'
"""Key under which a connection records whether SQLite will write its database."""

_SUPPORTED_URL_FORMS = (
    'A results index is either a sqlite: URL naming a local path, or a '
    'postgresql+psycopg: URL naming a server.'
)


def _sqlite_refused_a_write(exc: BaseException | None) -> bool:
    """Report whether a SQLite error says the database itself will not be written.

    SQLite distinguishes a database it may never write -- a read-only file, or one
    whose directory it cannot create its side files in -- from a write it could
    not take at this moment, such as a busy lock or an I/O error.  Every result
    code in the first group shares one name prefix, and only that group describes
    a database a reader can still use.

    Parameters:
        exc: The driver exception to classify, or None.

    Returns:
        True when the error says the database is read-only.
    """
    return isinstance(exc, sqlite3.Error) and exc.sqlite_errorname.startswith(
        _SQLITE_READONLY_PREFIX
    )


def _sqlite_on_connect(dbapi_connection: Any, connection_record: Any) -> None:
    """Apply the per-connection SQLite settings the index depends on.

    Registered as a dialect connect-time event rather than issued as a query, so
    every connection the pool hands out carries them, including ones opened long
    after :func:`open_index` returned.

    Foreign keys are off by default in SQLite, and without them the cascade that
    makes an image's child rows disappear with it does nothing.  Write-ahead
    logging lets a reader and a writer work at the same time, and the busy
    timeout lets two writers queue instead of failing.

    Selecting the journal mode writes the database header, so a read-only
    database refuses it.  That refusal is the cheapest read-only test there is,
    so it is recorded on the connection rather than raised: a database SQLite
    will never write has no writers for a journal to protect, and
    :func:`open_index` decides from the record whether this caller needed one.

    Parameters:
        dbapi_connection: The freshly opened DBAPI connection.
        connection_record: The pool's bookkeeping record, which carries the
            read-only answer back to the opener.
    """
    read_only = False
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute('PRAGMA foreign_keys = ON')
        cursor.execute(f'PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}')
        try:
            cursor.execute('PRAGMA journal_mode = WAL')
        except sqlite3.Error as exc:
            if not _sqlite_refused_a_write(exc):
                raise
            read_only = True
    finally:
        cursor.close()
    connection_record.info[_SQLITE_READ_ONLY_KEY] = read_only


def _sqlite_path(parsed: URL) -> Path | None:
    """Return the local filesystem path a SQLite URL names.

    Parameters:
        parsed: A parsed URL whose backend is SQLite.

    Returns:
        The path, or None when the URL names an in-memory database.
    """
    database = parsed.database
    if database is None or database == '' or database == _SQLITE_MEMORY:
        return None
    # pathlib rather than FCPath: SQLite is opened by the C library, which knows
    # nothing of cloud locations, so a SQLite URL is always a local path.
    return Path(database)


def _make_engine(parsed: URL, url: str) -> Engine:
    """Create the engine for a parsed URL, or explain a missing driver.

    Parameters:
        parsed: The parsed connection URL.
        url: The URL as the caller wrote it, for error messages.

    Returns:
        The engine, not yet connected.

    Raises:
        ValueError: If the URL names a backend whose driver is not installed,
            distinguishing PostgreSQL -- whose driver SpinDoctor ships as an
            extra -- from a backend it does not support at all.
    """
    try:
        return sqlalchemy.create_engine(parsed)
    except ModuleNotFoundError as exc:
        if parsed.get_backend_name() == _POSTGRES_BACKEND:
            raise ValueError(
                f'{url}: the PostgreSQL driver is not installed. Install it with '
                f'"pip install rms-spindoctor[postgres]", or use a sqlite: URL.'
            ) from exc
        raise ValueError(
            f'{url}: the driver this URL names is not installed ({exc}), and SpinDoctor '
            f'ships no driver for that backend. {_SUPPORTED_URL_FORMS}'
        ) from exc


def _probe_sqlite_access(engine: Engine, url: str, *, create: bool) -> None:
    """Verify that a SQLite database can be locked, and written when it must be.

    Taking and releasing a write lock is the cheapest question that distinguishes
    a filesystem SQLite can be trusted on from one where concurrent writers
    silently corrupt the file.  Asking it at open turns a data-loss bug into a
    startup error.

    Read-only is a different answer, and the connection carries it: the journal
    mode :func:`_sqlite_on_connect` selects writes the database header, so its
    refusal is the read-only test, and the answer is recorded there.  The lock
    cannot be asked instead, because a rollback-journal database SQLite will
    never write still grants the reserved lock ``BEGIN IMMEDIATE`` asks for.

    Read-only is refused only when the caller means to write.  A consumer reads,
    and an archived or read-only-mounted index serves it, so the refusal an
    ingest deserves would be a false one here.

    Parameters:
        engine: The engine to probe.
        url: The URL as the caller wrote it, for error messages.
        create: Whether the caller intends to write the database.

    Raises:
        ValueError: If the write lock cannot be taken; if the database is
            read-only and the caller means to write it; or if it is read-only and
            cannot be read either.
    """
    read_only = False
    try:
        with engine.connect() as connection:
            read_only = bool(connection.info.get(_SQLITE_READ_ONLY_KEY))
            if not read_only:
                connection.exec_driver_sql('BEGIN IMMEDIATE')
                connection.exec_driver_sql('ROLLBACK')
    except sqlalchemy.exc.DBAPIError as exc:
        raise ValueError(
            f'{url}: could not take a SQLite write lock ({exc.orig}). A SQLite index '
            f'must live on a local filesystem that honors locking; use a '
            f'postgresql+psycopg: URL to share one index across machines.'
        ) from exc
    if not read_only:
        return
    if create:
        raise ValueError(
            f'{url}: this SQLite database is read-only, and ingest has to write it. '
            f'Ingest a writable copy; a consumer reads this one as it is.'
        )
    _require_sqlite_readable(engine, url)


def _require_sqlite_readable(engine: Engine, url: str) -> None:
    """Verify that a read-only SQLite database can at least be read.

    SQLite reads a write-ahead-logged database through a shared-memory index it
    creates beside the file, so such a database on a filesystem that permits no
    writes at all cannot be read either.  Saying that here beats letting the
    first query fail with a write error on what the caller asked to be a read.

    Parameters:
        engine: The engine to probe.
        url: The URL as the caller wrote it, for error messages.

    Raises:
        ValueError: If the database cannot be read.
    """
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql('SELECT count(*) FROM sqlite_master')
    except sqlalchemy.exc.DBAPIError as exc:
        raise ValueError(
            f'{url}: this SQLite database is read-only and cannot be read either '
            f'({exc.orig}). A write-ahead-logged database creates an index file beside '
            f'itself even to be read, so copy it somewhere writable.'
        ) from exc


def _create_schema(engine: Engine) -> None:
    """Create every missing table and stamp the database with its version.

    Parameters:
        engine: The engine to create the schema in.
    """
    METADATA.create_all(engine)
    with engine.begin() as connection:
        stamped = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).first()
        if stamped is None:
            connection.execute(
                SCHEMA_META.insert().values(
                    singleton=1,
                    schema_version=SCHEMA_VERSION,
                    created_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                )
            )


def _stamped_version(engine: Engine) -> int | None:
    """Return the schema version the database is stamped with.

    Parameters:
        engine: The engine to inspect.

    Returns:
        The stamped version, or None when the database carries no
        ``schema_meta`` row -- because the table is absent, or because it is
        empty after an interrupted creation.
    """
    if not sqlalchemy.inspect(engine).has_table(SCHEMA_META.name):
        return None
    with engine.connect() as connection:
        row = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).first()
    return None if row is None else int(row.schema_version)


def _verify_schema_version(stamped: int | None, url: str) -> None:
    """Verify that a stamped version is the one this code reads.

    Parameters:
        stamped: The version read from the database, or None when it carries no
            ``schema_meta`` row.
        url: The URL as the caller wrote it, for error messages.

    Raises:
        ValueError: If the database carries no ``schema_meta`` row, or one whose
            version differs from
            :data:`~spindoctor.results_index.schema.SCHEMA_VERSION`.
    """
    if stamped is None:
        raise ValueError(
            f'{url}: this is not a results index (it has no schema_meta row). '
            f'Run sd_stats_ingest first to build one.'
        )
    if stamped != SCHEMA_VERSION:
        raise ValueError(
            f'{url}: results index schema version {stamped} is not the '
            f'version {SCHEMA_VERSION} this code reads. There are no migrations: delete '
            f'the database and re-run sd_stats_ingest.'
        )


def _build_engine(url: str, *, create: bool) -> Engine:
    """Open the index, letting a driver's own exceptions escape untranslated.

    :func:`open_index` wraps this so that every escape becomes a ``ValueError``.
    The separation keeps the translation in one place rather than repeated around
    each call a driver can fail inside.

    Parameters:
        url: The connection URL.
        create: Whether to create missing tables and the version row.

    Returns:
        An open engine.

    Raises:
        ValueError: For the failures this module diagnoses itself.
    """
    parsed = sqlalchemy.engine.make_url(url)
    backend = parsed.get_backend_name()
    if backend == _SQLITE_BACKEND and not create:
        path = _sqlite_path(parsed)
        if path is not None and not path.exists():
            raise ValueError(
                f'{url}: there is no results index at {path}. Run sd_stats_ingest to build one.'
            )
    engine = _make_engine(parsed, url)
    try:
        if backend == _SQLITE_BACKEND:
            sqlalchemy.event.listen(engine, 'connect', _sqlite_on_connect)
            _probe_sqlite_access(engine, url, create=create)
        # The stamped version is checked before anything is written: creating
        # this version's tables inside a database stamped with another version
        # would leave a mixture no single version number describes.
        stamped = _stamped_version(engine)
        if stamped is not None:
            _verify_schema_version(stamped, url)
        if create:
            _create_schema(engine)
            stamped = _stamped_version(engine)
        _verify_schema_version(stamped, url)
    except Exception:
        engine.dispose()
        raise
    return engine


def open_index(url: str, *, create: bool = False) -> Engine:
    """Open the results index named by a connection URL.

    This is the only opener.  Every program that reads or writes the index goes
    through it, so the version gate cannot be bypassed.

    With ``create`` false -- every consumer -- a database that does not exist, or
    that carries no ``schema_meta`` row, is an error naming ``sd_stats_ingest``.
    A consumer pointed at a SQLite path that does not exist fails; it does not
    leave an empty database behind.  With ``create`` true -- the ingest programs
    -- missing tables are created and the version row is written.

    Either way a database stamped with a different schema version is refused,
    naming both versions, because the index carries no migrations and rebuilding
    it is always available and always correct.  Nothing is written to a database
    whose version is refused.

    Every failure is a ``ValueError`` naming the URL, including the ones a
    database driver raises: a consumer that wants to report the cause rather than
    crash catches one type, and the driver's own exception is kept as the
    ``__cause__``.

    Parameters:
        url: A ``sqlite:`` URL naming a local filesystem path, or a
            ``postgresql+psycopg:`` URL naming a server.
        create: Whether to create missing tables and the version row.

    Returns:
        An open engine.  SQLite engines have foreign keys, write-ahead logging
        and a busy timeout applied to every connection.

    Raises:
        ValueError: If the URL cannot be parsed, names a backend with no driver
            installed, or names a server that will not accept the connection; if
            a SQLite file's filesystem cannot honor write locking, or the file is
            read-only and ``create`` is true; if ``create`` is false and the
            database or its ``schema_meta`` row does not exist; or if the stamped
            schema version is not the one this code reads.
    """
    try:
        return _build_engine(url, create=create)
    except sqlalchemy.exc.NoSuchModuleError as exc:
        raise ValueError(
            f'{url}: there is no database driver for this URL scheme ({exc}). '
            f'{_SUPPORTED_URL_FORMS}'
        ) from exc
    except sqlalchemy.exc.SQLAlchemyError as exc:
        raise ValueError(
            f'{url}: could not open the results index ({type(exc).__name__}: {exc}).'
        ) from exc
