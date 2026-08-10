"""Shared fixtures and row factories for the results-index tests.

The row factories exist so a test that cares about one column does not have to
restate the NOT NULL columns around it, and so a column gaining a constraint
fails in one place rather than in every test.

The PostgreSQL fixtures give each test its own schema.  Two workers of a parallel
run, or two runs against the same server, otherwise share one set of tables and
see one another's rows.
"""

import builtins
import contextlib
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy.engine import Engine

from spindoctor.results_index import open_index

POSTGRES_URL_ENV_VAR = 'SPINDOCTOR_TEST_POSTGRES_URL'

ROOT_URL = 'file:///data/nav-results'

STUB = 'COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB'

AT_SIGN_USER = 'admin@pgsrv'
"""A user name carrying an at-sign, which is how a managed server names one.

``user@servername`` is the standard login form of a hosted PostgreSQL, and
SQLAlchemy's own parser accepts it. A rule that took the first at-sign as the
end of the credentials would find no password after it and leak the whole URL.
"""


EXPLODING_FACTORY_MESSAGE = 'the dialect exploded'
"""What the stand-in engine factory raises, standing for any escape from one."""


def sqlite_url_for(path: Path) -> str:
    """Return the SQLite URL naming a filesystem path.

    Parameters:
        path: The database file's path.

    Returns:
        The URL.
    """
    # as_posix rather than str: SQLAlchemy takes a URL, and a Windows path
    # separator in one is not a path separator.
    return f'sqlite:///{path.as_posix()}'


def without_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
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


def exploding_factory(*args: Any, **kwargs: Any) -> Engine:
    """Stand in for an engine factory that fails in a way nobody enumerated.

    A dialect coerces its own connect arguments and reports a bad one as a bare
    exception naming nothing, so the translation has to be a catch-all rather
    than a list of types.

    Parameters:
        args: Whatever the caller passed, all of it ignored.
        kwargs: Whatever the caller passed, all of it ignored.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError(EXPLODING_FACTORY_MESSAGE)


def image_row(**overrides: Any) -> dict[str, Any]:
    """Return a minimally valid ``images`` row, with overrides applied.

    Parameters:
        overrides: Column values replacing (or adding to) the defaults.

    Returns:
        A mapping ready to pass to an ``images`` insert.
    """
    row: dict[str, Any] = {
        'root_url': ROOT_URL,
        'results_path_stub': STUB,
        'image_name': 'N1294561202_1_CALIB.IMG',
        'instrument': 'COISS',
        'status': 'success',
        'n_techniques': 2,
    }
    row.update(overrides)
    return row


def technique_row(**overrides: Any) -> dict[str, Any]:
    """Return a minimally valid ``techniques`` row, with overrides applied.

    Parameters:
        overrides: Column values replacing (or adding to) the defaults.

    Returns:
        A mapping ready to pass to a ``techniques`` insert.
    """
    row: dict[str, Any] = {
        'root_url': ROOT_URL,
        'results_path_stub': STUB,
        'technique_name': 'star_field_from_catalog',
        'spurious': False,
        'at_edge': False,
    }
    row.update(overrides)
    return row


def feature_source_row(**overrides: Any) -> dict[str, Any]:
    """Return a minimally valid ``feature_sources`` row, with overrides applied.

    Parameters:
        overrides: Column values replacing (or adding to) the defaults.

    Returns:
        A mapping ready to pass to a ``feature_sources`` insert.
    """
    row: dict[str, Any] = {
        'root_url': ROOT_URL,
        'results_path_stub': STUB,
        'feature_type': 'STAR',
        'source_model': 'NavModelStars',
        'source_name': 'UCAC4',
        'n_features': 41,
        'n_gated': 3,
    }
    row.update(overrides)
    return row


@contextlib.contextmanager
def opened(url: str, *, create: bool = False) -> Iterator[Engine]:
    """Open an index and dispose of the engine afterwards.

    Parameters:
        url: The connection URL to open.
        create: Whether to create missing tables and the version row.

    Yields:
        The open engine.
    """
    engine = open_index(url, create=create)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def postgres_server_url() -> str:
    """Return the PostgreSQL URL the postgres tier runs against.

    Returns:
        The URL named by the environment.
    """
    url = os.environ.get(POSTGRES_URL_ENV_VAR)
    if url is None:
        pytest.skip(f'{POSTGRES_URL_ENV_VAR} is not set')
    return url


@pytest.fixture
def postgres_schema() -> str:
    """Return the name of the private schema this test's tables live in.

    Requested alongside ``postgres_url`` by a test that reads the server's
    catalog, where a query scoped by table name alone would answer from whatever
    schema happened to hold a table of that name -- another worker's, or a
    leftover in ``public``.

    Returns:
        A schema name no other test uses.
    """
    return f'ri_test_{uuid.uuid4().hex}'


@pytest.fixture
def postgres_decoy_schema() -> str:
    """Return the name of a second schema, behind this test's own on the path.

    A search path of one entry is the one shape in which every unqualified table
    name resolves to the same schema whatever the code does with it, so a
    fixture that pins one cannot see a query that resolves two names into two
    schemas.  Every postgres test therefore runs with a schema behind its own.

    Returns:
        A schema name no other test uses.
    """
    return f'ri_decoy_{uuid.uuid4().hex}'


DECOY_TABLE = 'customers'
"""What the decoy schema holds: a table of a name the index never uses.

Deliberately not one of the index's own names, so that the decoy makes the
search path longer than one entry without also making a bare ``images``
resolve into it.  A creating open binds the schema it builds in and does not
adopt a table of one of its names from anywhere else, and the drop reports
every table of those names its connection reaches; the tests that need either
of those build the collision themselves.
"""


@pytest.fixture
def postgres_url(
    postgres_server_url: str, postgres_schema: str, postgres_decoy_schema: str
) -> Iterator[str]:
    """Yield a PostgreSQL URL scoped to a schema of this test's own.

    The schemas are created before the test and dropped after it, so repeated
    runs and parallel workers never see one another's tables.  The index's own
    schema leads the search path, so that is where a creating open builds it;
    the decoy behind it is what makes the path more than one entry long.

    Parameters:
        postgres_server_url: The server URL the schemas are created on.
        postgres_schema: Name of the schema to create for the index.
        postgres_decoy_schema: Name of the schema to create behind it.

    Yields:
        A URL whose search path names the private schema and then the decoy.
    """
    schema = postgres_schema
    decoy = postgres_decoy_schema
    scoped = sqlalchemy.engine.make_url(postgres_server_url).update_query_dict(
        {'options': f'-csearch_path={schema},{decoy}'}
    )
    admin = sqlalchemy.create_engine(postgres_server_url)
    try:
        with admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
            connection.exec_driver_sql(f'CREATE SCHEMA "{decoy}"')
            connection.exec_driver_sql(f'CREATE TABLE "{decoy}".{DECOY_TABLE} (x INTEGER)')
        try:
            yield scoped.render_as_string(hide_password=False)
        finally:
            with admin.begin() as connection:
                connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
                connection.exec_driver_sql(f'DROP SCHEMA "{decoy}" CASCADE')
    finally:
        admin.dispose()


def url_scoped_to(server_url: str, *schemas: str) -> str:
    """Return the server URL with a search path naming these schemas in order.

    Parameters:
        server_url: The server URL, unscoped.
        schemas: The schemas to name, first one first.

    Returns:
        The scoped URL, with its password intact so it can be opened.
    """
    scoped = sqlalchemy.engine.make_url(server_url).update_query_dict(
        {'options': f'-csearch_path={",".join(schemas)}'}
    )
    return scoped.render_as_string(hide_password=False)
