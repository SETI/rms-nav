"""Engine factory and version gate for the navigation results index.

:func:`open_index` is the only opener.  It selects the backend from the
connection URL, applies the SQLite settings the concurrency model depends on,
and refuses a database whose schema version is not the one this code reads.

Two URL forms are supported::

    sqlite:////data/nav-results/index.sqlite3
    postgresql+psycopg://user@host/spindoctor

A ``sqlite:`` URL names a **local filesystem path**.  It is the one path in this
system that is not a cloud-capable location: the C library opens it directly, so
a network filesystem that cannot honor its locking is refused at open rather than
corrupted later.  PostgreSQL is the option for sharing one index across machines,
and its driver ships as an optional extra.
"""

import datetime
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


def _sqlite_on_connect(dbapi_connection: Any, connection_record: Any) -> None:
    """Apply the per-connection SQLite settings the index depends on.

    Registered as a dialect connect-time event rather than issued as a query, so
    every connection the pool hands out carries them, including ones opened long
    after :func:`open_index` returned.

    Foreign keys are off by default in SQLite, and without them the cascade that
    makes an image's child rows disappear with it does nothing.  Write-ahead
    logging lets a reader and a writer work at the same time, and the busy
    timeout lets two writers queue instead of failing.

    Parameters:
        dbapi_connection: The freshly opened DBAPI connection.
        connection_record: The pool's bookkeeping record, which is not used.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute('PRAGMA foreign_keys = ON')
        cursor.execute(f'PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}')
        cursor.execute('PRAGMA journal_mode = WAL')
    finally:
        cursor.close()


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
        ValueError: If the URL names PostgreSQL and its driver is not installed.
    """
    try:
        return sqlalchemy.create_engine(parsed)
    except ModuleNotFoundError as exc:
        if parsed.get_backend_name() != _POSTGRES_BACKEND:
            raise
        raise ValueError(
            f'{url}: the PostgreSQL driver is not installed. Install it with '
            f'"pip install rms-spindoctor[postgres]", or use a sqlite: URL.'
        ) from exc


def _probe_lockability(engine: Engine, url: str) -> None:
    """Verify that the SQLite file's filesystem can honor a write lock.

    Taking and releasing a write lock is the cheapest question that distinguishes
    a filesystem SQLite can be trusted on from one where concurrent writers
    silently corrupt the file.  Asking it at open turns a data-loss bug into a
    startup error.

    Parameters:
        engine: The engine to probe.
        url: The URL as the caller wrote it, for error messages.

    Raises:
        ValueError: If the write lock cannot be taken.
    """
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql('BEGIN IMMEDIATE')
            connection.exec_driver_sql('ROLLBACK')
    except sqlalchemy.exc.DBAPIError as exc:
        raise ValueError(
            f'{url}: could not take a SQLite write lock ({exc.orig}). A SQLite index must '
            f'live on a local filesystem that honors locking; use a postgresql+psycopg: '
            f'URL to share one index across machines.'
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


def _verify_schema_version(engine: Engine, url: str) -> None:
    """Verify that the database is an index of the version this code reads.

    Parameters:
        engine: The engine to inspect.
        url: The URL as the caller wrote it, for error messages.

    Raises:
        ValueError: If the database carries no ``schema_meta`` row, or one whose
            version differs from
            :data:`~spindoctor.results_index.schema.SCHEMA_VERSION`.
    """
    stamped = None
    if sqlalchemy.inspect(engine).has_table(SCHEMA_META.name):
        with engine.connect() as connection:
            stamped = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).first()
    if stamped is None:
        raise ValueError(
            f'{url}: this is not a results index (it has no schema_meta row). '
            f'Run sd_stats_ingest first to build one.'
        )
    if stamped.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f'{url}: results index schema version {stamped.schema_version} is not the '
            f'version {SCHEMA_VERSION} this code reads. There are no migrations: delete '
            f'the database and re-run sd_stats_ingest.'
        )


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
    it is always available and always correct.

    Parameters:
        url: A ``sqlite:`` URL naming a local filesystem path, or a
            ``postgresql+psycopg:`` URL naming a server.
        create: Whether to create missing tables and the version row.

    Returns:
        An open engine.  SQLite engines have foreign keys, write-ahead logging
        and a busy timeout applied to every connection.

    Raises:
        ValueError: If a PostgreSQL URL's driver is not installed; if a SQLite
            file's filesystem cannot honor write locking; if ``create`` is false
            and the database or its ``schema_meta`` row does not exist; or if the
            stamped schema version is not the one this code reads.
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
            _probe_lockability(engine, url)
        if create:
            _create_schema(engine)
        _verify_schema_version(engine, url)
    except Exception:
        engine.dispose()
        raise
    return engine
