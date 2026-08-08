"""Shared fixtures and row factories for the results-index tests.

The row factories exist so a test that cares about one column does not have to
restate the NOT NULL columns around it, and so a column gaining a constraint
fails in one place rather than in every test.

The PostgreSQL fixtures give each test its own schema.  Two workers of a parallel
run, or two runs against the same server, otherwise share one set of tables and
see one another's rows.
"""

import contextlib
import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy.engine import Engine

from spindoctor.results_index import open_index

POSTGRES_URL_ENV_VAR = 'SPINDOCTOR_TEST_POSTGRES_URL'

ROOT_URL = 'file:///data/nav-results'

STUB = 'COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB'


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
def postgres_url(postgres_server_url: str) -> Iterator[str]:
    """Yield a PostgreSQL URL scoped to a schema of this test's own.

    The schema is created before the test and dropped after it, so repeated runs
    and parallel workers never see one another's tables.

    Parameters:
        postgres_server_url: The server URL the schema is created on.

    Yields:
        A URL whose search path names the private schema.
    """
    schema = f'ri_test_{uuid.uuid4().hex}'
    scoped = sqlalchemy.engine.make_url(postgres_server_url).update_query_dict(
        {'options': f'-csearch_path={schema}'}
    )
    admin = sqlalchemy.create_engine(postgres_server_url)
    try:
        with admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        try:
            yield scoped.render_as_string(hide_password=False)
        finally:
            with admin.begin() as connection:
                connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
    finally:
        admin.dispose()
