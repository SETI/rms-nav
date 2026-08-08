"""Engine factory and version gate for the navigation results index.

:func:`open_index` is the only opener.  It selects the backend from the
connection URL, applies the SQLite settings the concurrency model depends on,
and refuses a database whose schema version is not the one this code reads.
Every refusal is a ``ValueError`` naming the URL, including the ones a database
driver raises, so a caller that reports failures catches one type.  The URL is
named with its password masked: these messages are written to run logs and
handed to operators, and a database password belongs in neither.

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
import os
import re
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

_SQLITE_NOT_A_DATABASE = 'SQLITE_NOTADB'
"""Result code SQLite gives for a file that is not a database at all."""

_SQLITE_CANNOT_OPEN = 'SQLITE_CANTOPEN'
"""Result code SQLite gives for a path it cannot open, whatever the reason."""

_SQLITE_LOCK_REFUSED_PREFIXES = ('SQLITE_BUSY', 'SQLITE_IOERR')
"""Prefixes of the result codes that say the write lock itself was not granted.

These are the codes a filesystem that cannot honor SQLite locking produces, and
the only ones for which moving the index to a server is the remedy.
"""

_HIDDEN_PASSWORD = '***'
"""What a password is replaced by, matching what a parsed URL renders itself as."""

_PASSWORD_IN_URL = re.compile(r'(?P<credentials>\A(?:[^/@]*:)?/{1,2}[^/@]*:)[^@]*@')
"""The password of a URL string that could not be parsed, as text.

A password is matched only where a URL keeps one: at the very start of the
string, after an optional scheme and the one or two slashes that open the
authority section, and after the colon that follows the user name.  The password
itself runs to the ``@``, since a URL permits an unescaped ``/`` in one and a
pattern that stopped at a separator would leak such a password whole.  Anchoring
is what keeps a local path that merely happens to carry a colon and a later
``@`` from being read as credentials and mangled.
"""

_SUPPORTED_URL_FORMS = (
    'A results index is either a sqlite: URL naming a local path, or a '
    'postgresql+psycopg: URL naming a server.'
)


class _IndexOpenError(ValueError):
    """A failure this module has already diagnosed.

    :func:`open_index` turns whatever escapes the builder into a ``ValueError``
    naming the URL, and that catch-all cannot tell a message this module wrote
    from a bare ``ValueError`` a driver raised deep inside itself.  Raising this
    subclass from the failures diagnosed here keeps a complete message from
    being wrapped in a second one, while a caller still catches the
    ``ValueError`` the contract promises.
    """


def _masked_url(url: str) -> str:
    """Return a URL string with any password in it replaced.

    Used for a URL whose parsing is what failed, since an unparsed URL cannot
    render itself.  Everything else survives, because naming the URL is what
    tells a reader which of the resolution levels supplied the bad value.

    Parameters:
        url: The URL as the caller wrote it.

    Returns:
        The URL with its password, if any, masked.
    """
    return _PASSWORD_IN_URL.sub(rf'\g<credentials>{_HIDDEN_PASSWORD}@', url)


def _display_url(url: str) -> str:
    """Return the form of a URL that messages name it by.

    Parameters:
        url: The URL as the caller wrote it.

    Returns:
        The parsed URL rendered with its password hidden, or the text form
        masked by pattern when the URL is one the parser rejects.
    """
    try:
        return sqlalchemy.engine.make_url(url).render_as_string()
    except Exception:
        # Any failure at all: this runs while reporting another failure, and a
        # display string is never worth raising over.
        return _masked_url(url)


def _sqlite_error_name(exc: BaseException) -> str:
    """Return the SQLite result-code name a driver exception carries.

    Parameters:
        exc: The exception to read, either a driver exception or the wrapper
            SQLAlchemy raised around one.

    Returns:
        The result-code name, or an empty string when the exception carries
        none -- an exception from another driver, or one whose original is
        absent, which the caller diagnoses generically rather than crashing on.
    """
    original = getattr(exc, 'orig', exc)
    name = getattr(original, 'sqlite_errorname', '')
    return name if isinstance(name, str) else ''


def _is_read_only_error(error_name: str) -> bool:
    """Report whether a SQLite result code says the database will not be written.

    SQLite distinguishes a database it may never write -- a read-only file, or
    one whose directory it cannot create its side files in -- from a write it
    could not take at this moment, such as a busy lock or an I/O error.  Every
    result code in the first group shares one name prefix.

    Parameters:
        error_name: The result-code name to classify, possibly empty.

    Returns:
        True when the code says the database is read-only.
    """
    return error_name.startswith(_SQLITE_READONLY_PREFIX)


def _sqlite_on_connect(dbapi_connection: Any, connection_record: Any) -> None:
    """Apply the per-connection SQLite settings the index depends on.

    Registered as a dialect connect-time event rather than issued as a query, so
    every connection the pool hands out carries them, including ones opened long
    after :func:`open_index` returned.

    Foreign keys are off by default in SQLite, and without them the cascade that
    makes an image's child rows disappear with it does nothing.  Write-ahead
    logging lets a reader and a writer work at the same time, and the busy
    timeout lets two writers queue instead of failing.

    Selecting the journal mode writes the database header, so a database SQLite
    will never write refuses it.  That refusal is tolerated rather than raised,
    because such a database has no writers for a journal to protect and a
    consumer reading an archived copy is a deployment this index supports.
    Nothing is inferred from its absence: whether this caller needed a writable
    database is asked of the filesystem instead, in
    :func:`_require_writable_sqlite_database`.

    Parameters:
        dbapi_connection: The freshly opened DBAPI connection.
        connection_record: The pool's bookkeeping record, which these settings
            do not use.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute('PRAGMA foreign_keys = ON')
        cursor.execute(f'PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}')
        try:
            cursor.execute('PRAGMA journal_mode = WAL')
        except sqlite3.Error as exc:
            if not _is_read_only_error(_sqlite_error_name(exc)):
                raise
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
        url: The URL as messages name it.

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
            raise _IndexOpenError(
                f'{url}: the PostgreSQL driver is not installed. Install it with '
                f'"pip install rms-spindoctor[postgres]", or use a sqlite: URL.'
            ) from exc
        raise _IndexOpenError(
            f'{url}: the driver this URL names is not installed ({exc}), and SpinDoctor '
            f'ships no driver for that backend. {_SUPPORTED_URL_FORMS}'
        ) from exc


def _read_only_refusal(url: str) -> _IndexOpenError:
    """Return the refusal for a database ingest is never going to be able to write.

    Parameters:
        url: The URL as messages name it.

    Returns:
        The refusal to raise.
    """
    return _IndexOpenError(
        f'{url}: this SQLite database is read-only, and ingest has to write it. '
        f'Ingest a writable copy; a consumer reads this one as it is.'
    )


def _require_writable_sqlite_database(path: Path, url: str) -> None:
    """Verify that ingest can write the SQLite database a URL names.

    The question is put to the filesystem rather than to SQLite, because SQLite
    does not answer it at open.  A write-ahead-logged database -- the shape this
    opener always leaves behind, having selected that journal mode -- accepts
    both the journal-mode selection, which its header already records, and the
    write lock ``BEGIN IMMEDIATE`` takes, on a file it will never write.  It
    refuses only the first real write, in the middle of an ingest.

    The directory is asked about too: SQLite writes the write-ahead log and its
    shared-memory index beside the database, so a writable file in a directory
    that permits nothing is a database ingest still cannot write.

    Parameters:
        path: The database file's path.
        url: The URL as messages name it.

    Raises:
        ValueError: If the file exists and cannot be written, or if its
            directory exists and cannot be written.
    """
    if path.exists() and not os.access(path, os.W_OK):
        raise _read_only_refusal(url)
    directory = path.parent
    if directory.is_dir() and not os.access(directory, os.W_OK):
        raise _IndexOpenError(
            f'{url}: the directory {directory} is read-only, and ingest has to write the '
            f'write-ahead log beside the database. Ingest into a writable directory; a '
            f'consumer reads this one as it is.'
        )


def _cannot_open_message(exc: sqlalchemy.exc.DBAPIError, url: str, path: Path | None) -> str:
    """Return the message for a path SQLite could not open.

    One result code covers a directory that was never created, a path naming
    something that is not a file, and a file this user may not touch.  Each has
    a different remedy, and none of them is PostgreSQL.

    Parameters:
        exc: The driver exception, for what SQLite itself said.
        url: The URL as messages name it.
        path: The database file's path, or None for an in-memory database.

    Returns:
        The message.
    """
    if path is None:
        return f'{url}: SQLite could not open this database ({exc.orig}).'
    directory = path.parent
    if not directory.is_dir():
        return (
            f'{url}: the directory {directory} does not exist, so SQLite cannot open a '
            f'database in it. Create the directory first, or name a path inside one that '
            f'already exists.'
        )
    if path.exists() and not path.is_file():
        return (
            f'{url}: {path} is not a file, so it cannot be a SQLite database. Name the '
            f'database file itself.'
        )
    return (
        f'{url}: SQLite could not open {path} ({exc.orig}). Check that this user may read '
        f'and write both that file and its directory.'
    )


def _sqlite_probe_failure(
    exc: sqlalchemy.exc.DBAPIError, url: str, path: Path | None
) -> _IndexOpenError:
    """Return the refusal that fits what SQLite actually said.

    One exception type covers a file that is not a database, a path that cannot
    be opened at all, a lock a filesystem would not grant, and every other way a
    database can refuse a write.  Only the lock is a reason to move the index to
    a server, and prescribing that for the others sends an operator to rebuild a
    deployment over a directory they had not created yet, or over a disk that is
    merely full.  A code this classification does not know is reported as what
    SQLite said, with no remedy invented for it.

    Codes are matched by prefix, because SQLite refines several of them into
    extended forms -- ``SQLITE_IOERR_WRITE``, ``SQLITE_CANTOPEN_ISDIR`` -- that
    name the same cause more precisely.

    Parameters:
        exc: The driver exception the probe raised.
        url: The URL as messages name it.
        path: The database file's path, or None for an in-memory database.

    Returns:
        The refusal to raise.
    """
    error_name = _sqlite_error_name(exc)
    if error_name.startswith(_SQLITE_NOT_A_DATABASE):
        return _IndexOpenError(
            f'{url}: this file is not a SQLite database ({exc.orig}). Check the path: an '
            f'index is built by sd_stats_ingest, and this file holds something else.'
        )
    if error_name.startswith(_SQLITE_CANNOT_OPEN):
        return _IndexOpenError(_cannot_open_message(exc, url, path))
    if error_name == '' or error_name.startswith(_SQLITE_LOCK_REFUSED_PREFIXES):
        return _IndexOpenError(
            f'{url}: could not take a SQLite write lock ({exc.orig}). A SQLite index '
            f'must live on a local filesystem that honors locking; use a '
            f'postgresql+psycopg: URL to share one index across machines.'
        )
    return _IndexOpenError(
        f'{url}: SQLite refused {"this database" if path is None else path} '
        f'({error_name}: {exc.orig}).'
    )


def _probe_sqlite_access(engine: Engine, url: str, path: Path | None, *, create: bool) -> None:
    """Verify that a SQLite database can be locked, and read when that is all it is.

    Taking and releasing a write lock is the cheapest question that distinguishes
    a filesystem SQLite can be trusted on from one where concurrent writers
    silently corrupt the file.  Asking it at open turns a data-loss bug into a
    startup error.

    A read-only database answers this question in two ways depending on its
    journal mode, so neither answer is read as a verdict on whether the caller
    may write: that is settled from the filesystem before the engine is built.
    What matters here is that a read-only refusal is not reported as a locking
    failure, and that a database whose write-ahead logging makes it unreadable
    says so rather than failing a consumer's first query.  A refusal that
    reaches an ingest here is still reported as read-only, since a network
    filesystem answers the permission question from mode bits it does not
    itself enforce, and SQLite is the one that finds out.

    Parameters:
        engine: The engine to probe.
        url: The URL as messages name it.
        path: The database file's path, or None for an in-memory database.
        create: Whether the caller intends to write the database.

    Raises:
        ValueError: If the write lock cannot be taken, if the path is not a
            database this code can open, if the database is read-only and the
            caller means to write it, or if it is read-only and cannot be read
            either.
    """
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql('BEGIN IMMEDIATE')
            connection.exec_driver_sql('ROLLBACK')
    except sqlalchemy.exc.DBAPIError as exc:
        if not _is_read_only_error(_sqlite_error_name(exc)):
            raise _sqlite_probe_failure(exc, url, path) from exc
        if create:
            raise _read_only_refusal(url) from exc
        _require_sqlite_readable(engine, url)


def _require_sqlite_readable(engine: Engine, url: str) -> None:
    """Verify that a read-only SQLite database can at least be read.

    SQLite reads a write-ahead-logged database through a shared-memory index it
    creates beside the file, so such a database on a filesystem that permits no
    writes at all cannot be read either.  Saying that here beats letting the
    first query fail with a write error on what the caller asked to be a read.

    Parameters:
        engine: The engine to probe.
        url: The URL as messages name it.

    Raises:
        ValueError: If the database cannot be read.
    """
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql('SELECT count(*) FROM sqlite_master')
    except sqlalchemy.exc.DBAPIError as exc:
        raise _IndexOpenError(
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
        url: The URL as messages name it.

    Raises:
        ValueError: If the database carries no ``schema_meta`` row, or one whose
            version differs from
            :data:`~spindoctor.results_index.schema.SCHEMA_VERSION`.
    """
    if stamped is None:
        raise _IndexOpenError(
            f'{url}: this is not a results index (it has no schema_meta row). '
            f'Run sd_stats_ingest first to build one.'
        )
    if stamped != SCHEMA_VERSION:
        raise _IndexOpenError(
            f'{url}: results index schema version {stamped} is not the '
            f'version {SCHEMA_VERSION} this code reads. There are no migrations: delete '
            f'the database and re-run sd_stats_ingest.'
        )


def _sqlite_target(parsed: URL, url: str, *, create: bool) -> Path | None:
    """Return the local path a SQLite URL names, refusing one it cannot serve.

    Parameters:
        parsed: The parsed connection URL, whose backend is SQLite.
        url: The URL as messages name it.
        create: Whether the caller intends to write the database.

    Returns:
        The database file's path, or None for an in-memory database.

    Raises:
        ValueError: If the URL carries a query string; if the caller means to
            write a database the filesystem will not let it write; or if the
            caller means to read one that is not there.
    """
    if parsed.query:
        carried = ', '.join(sorted(parsed.query))
        raise _IndexOpenError(
            f'{url}: a SQLite index URL is a plain local path, and this one carries a '
            f'query string ({carried}). The driver would open a file named after the '
            f'query rather than the one named here, so name the file alone.'
        )
    path = _sqlite_path(parsed)
    if path is None:
        return None
    if create:
        _require_writable_sqlite_database(path, url)
    elif not path.exists():
        raise _IndexOpenError(
            f'{url}: there is no results index at {path}. Run sd_stats_ingest to build one.'
        )
    return path


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
    safe_url = parsed.render_as_string()
    backend = parsed.get_backend_name()
    path = _sqlite_target(parsed, safe_url, create=create) if backend == _SQLITE_BACKEND else None
    engine = _make_engine(parsed, safe_url)
    try:
        if backend == _SQLITE_BACKEND:
            sqlalchemy.event.listen(engine, 'connect', _sqlite_on_connect)
            _probe_sqlite_access(engine, safe_url, path, create=create)
        # The stamped version is checked before anything is written: creating
        # this version's tables inside a database stamped with another version
        # would leave a mixture no single version number describes.
        stamped = _stamped_version(engine)
        if stamped is not None:
            _verify_schema_version(stamped, safe_url)
        if create:
            _create_schema(engine)
            stamped = _stamped_version(engine)
        _verify_schema_version(stamped, safe_url)
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
    -- missing tables are created and the version row is written, and a database
    the filesystem will not let this user write is refused before anything is
    opened.

    Either way a database stamped with a different schema version is refused,
    naming both versions, because the index carries no migrations and rebuilding
    it is always available and always correct.  Nothing is written to a database
    whose version is refused.

    Every failure is a ``ValueError`` naming the URL, including the ones a
    database driver raises: a consumer that wants to report the cause rather than
    crash catches one type, and the driver's own exception is kept as the
    ``__cause__``.  The URL is named with its password masked.

    Parameters:
        url: A ``sqlite:`` URL naming a local filesystem path, or a
            ``postgresql+psycopg:`` URL naming a server.
        create: Whether to create missing tables and the version row.

    Returns:
        An open engine.  SQLite engines have foreign keys, write-ahead logging
        and a busy timeout applied to every connection.

    Raises:
        ValueError: If the URL cannot be parsed, carries a query string a SQLite
            index may not have, names a backend with no driver installed, or
            names a server that will not accept the connection; if SQLite
            refuses the file -- because its filesystem cannot honor write
            locking, or for any other cause it reports -- or the file cannot be
            written and ``create`` is true; if ``create`` is false and the
            database or its ``schema_meta`` row does not exist; or if the
            stamped schema version is not the one this code reads.
    """
    try:
        return _build_engine(url, create=create)
    except _IndexOpenError:
        raise
    except sqlalchemy.exc.NoSuchModuleError as exc:
        raise ValueError(
            f'{_display_url(url)}: there is no database driver for this URL scheme ({exc}). '
            f'{_SUPPORTED_URL_FORMS}'
        ) from exc
    except Exception as exc:
        # Everything else, not only SQLAlchemy's own exceptions: a dialect
        # reports a malformed port or an uncoercible connect argument as a bare
        # ValueError naming neither the URL nor the setting that supplied it.
        raise ValueError(
            f'{_display_url(url)}: could not open the results index ({type(exc).__name__}: {exc}).'
        ) from exc
