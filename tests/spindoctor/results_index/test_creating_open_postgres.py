"""What a creating open builds on a real PostgreSQL server, and where.

Four things about the rule are properties of a server rather than of the schema
and cannot be asked of SQLite, whose database has one namespace.  That the
schema examined is the one the index resolves to and not the database around it,
since a server's other schemas belong to whoever made them.  That a table of one
of the index's own names in another schema the search path reaches is neither
built over nor built beside, which is the arrangement that would otherwise
spread one index across two schemas.  That a refusal names the URL with its
password hidden, since that URL goes into run logs.  And that a connection whose
search path names nothing that exists is told so rather than failing inside the
first ``CREATE TABLE``.
"""

import uuid
from collections.abc import Iterator

import pytest
import sqlalchemy
from tests.spindoctor.results_index.conftest import opened, url_scoped_to

from spindoctor.results_index import (
    IMAGES,
    SCHEMA_META,
    SCHEMA_VERSION,
    index_table_names,
    open_index,
)

pytestmark = pytest.mark.postgres

FOREIGN_TABLE = 'somebody_elses_table'
"""A table of the same schema that SpinDoctor did not create."""

COLLIDING_TABLE = IMAGES.name
"""A table SpinDoctor did not create, under a name SpinDoctor also uses."""

BOGUS_PASSWORD = 'hunter2-not-the-real-one'
"""A password a refusal must not repeat, distinctive enough to grep for."""


def _execute(url: str, *statements: str) -> None:
    """Run statements against a database as somebody other than the index.

    Parameters:
        url: The database URL.
        statements: The statements to run, in order.
    """
    engine = sqlalchemy.create_engine(url)
    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.exec_driver_sql(statement)
    finally:
        engine.dispose()


def _schema_tables(server_url: str, schema: str) -> list[str]:
    """Return every table of one schema, whoever created it.

    Parameters:
        server_url: URL of the server, unscoped.
        schema: The schema to list.

    Returns:
        The table names, sorted.
    """
    engine = sqlalchemy.create_engine(server_url)
    try:
        return sorted(sqlalchemy.inspect(engine).get_table_names(schema=schema))
    finally:
        engine.dispose()


def _rows_of(url: str, qualified: str) -> int:
    """Return how many rows a schema-qualified table holds.

    Parameters:
        url: The database URL.
        qualified: The table, named with its schema.

    Returns:
        The row count.
    """
    engine = sqlalchemy.create_engine(url)
    try:
        with engine.connect() as connection:
            counted = connection.exec_driver_sql(f'SELECT count(*) FROM {qualified}').scalar()
    finally:
        engine.dispose()
    return 0 if counted is None else int(counted)


def _refusal_of(url: str) -> str:
    """Return the message a creating open of a database is refused with.

    Parameters:
        url: The database URL.

    Returns:
        The refusal message.
    """
    with pytest.raises(ValueError) as excinfo:
        open_index(url, create=True)
    return str(excinfo.value)


@pytest.fixture
def tenant_schema(postgres_server_url: str) -> Iterator[str]:
    """Yield a schema standing for another tenant's, created empty.

    Parameters:
        postgres_server_url: The server URL the schema is created on.

    Yields:
        The schema's name.
    """
    schema = f'ri_tenant_{uuid.uuid4().hex}'
    admin = sqlalchemy.create_engine(postgres_server_url)
    try:
        with admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        try:
            yield schema
        finally:
            with admin.begin() as connection:
                connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
    finally:
        admin.dispose()


# ---------------------------------------------------------------------------
# The schema examined is the one the index resolves to
# ---------------------------------------------------------------------------


def test_an_empty_schema_of_a_populated_database_is_built_in(
    postgres_url: str, postgres_server_url: str, postgres_schema: str
) -> None:
    """A server holds schemas nobody here created, and they are not asked about.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
    """
    with opened(postgres_url, create=True):
        pass
    assert _schema_tables(postgres_server_url, postgres_schema) == sorted(index_table_names())


def test_a_foreign_table_in_that_schema_is_refused(postgres_url: str) -> None:
    """A results index owns the schema it lives in, whatever else the server holds.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _execute(postgres_url, f'CREATE TABLE {FOREIGN_TABLE} (x INTEGER)')
    assert 'does not own' in _refusal_of(postgres_url)


def test_a_stranger_s_table_of_one_of_our_names_is_refused(postgres_url: str) -> None:
    """The chain a wrong URL starts: adopt, fail, and be sent to the drop.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _execute(postgres_url, f'CREATE TABLE {COLLIDING_TABLE} (id int primary key, note text)')
    assert 'no schema_meta of SpinDoctor' in _refusal_of(postgres_url)


def test_a_stranger_s_table_keeps_its_rows(
    postgres_url: str, postgres_server_url: str, postgres_schema: str
) -> None:
    """Nothing is created and nothing is written, so their rows are still theirs.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
    """
    _execute(
        postgres_url,
        f'CREATE TABLE {COLLIDING_TABLE} (id int primary key, note text)',
        f"INSERT INTO {COLLIDING_TABLE} VALUES (1, 'theirs'), (2, 'also theirs')",
    )
    _refusal_of(postgres_url)
    assert _rows_of(postgres_server_url, f'"{postgres_schema}".{COLLIDING_TABLE}') == 2


def test_a_stranger_s_table_is_not_stamped_over(
    postgres_url: str, postgres_server_url: str, postgres_schema: str
) -> None:
    """A stamp beside it would make it this index's for every later reading.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
    """
    _execute(postgres_url, f'CREATE TABLE {COLLIDING_TABLE} (id int primary key, note text)')
    _refusal_of(postgres_url)
    assert _schema_tables(postgres_server_url, postgres_schema) == [COLLIDING_TABLE]


# ---------------------------------------------------------------------------
# A table of one of our names in another schema on the search path
# ---------------------------------------------------------------------------


def _tenant_holding_one_of_our_names(server_url: str, tenant_schema: str) -> None:
    """Put a table of one of the index's own names in another tenant's schema.

    Parameters:
        server_url: URL of the server, unscoped.
        tenant_schema: Schema standing for another tenant's.
    """
    _execute(
        server_url,
        f'CREATE TABLE "{tenant_schema}".{COLLIDING_TABLE} (id int primary key, payload text)',
        f'INSERT INTO "{tenant_schema}".{COLLIDING_TABLE} VALUES (1, \'not ours\')',
    )


def test_a_tenant_s_table_of_one_of_our_names_is_not_adopted(
    postgres_url: str, postgres_server_url: str, postgres_schema: str, tenant_schema: str
) -> None:
    """The index is built whole in one schema, not spread across the search path.

    A bare name resolves through the whole path, so a table of one of these
    names in a schema behind the index's own is one a creating open could find
    already there and build the rest of the index around.

    Parameters:
        postgres_url: URL whose fixture creates and removes the schema built in.
        postgres_server_url: URL of the server, unscoped.
        postgres_schema: Name of the schema the index is built in.
        tenant_schema: Schema standing for another tenant's.
    """
    _tenant_holding_one_of_our_names(postgres_server_url, tenant_schema)
    crossed = url_scoped_to(postgres_server_url, postgres_schema, tenant_schema)
    with opened(crossed, create=True):
        pass
    assert _schema_tables(postgres_server_url, postgres_schema) == sorted(index_table_names())


def test_a_tenant_s_table_keeps_its_rows_through_a_creating_open(
    postgres_url: str, postgres_server_url: str, postgres_schema: str, tenant_schema: str
) -> None:
    """Their table is not written into, which is what adoption would have done.

    Parameters:
        postgres_url: URL whose fixture creates and removes the schema built in.
        postgres_server_url: URL of the server, unscoped.
        postgres_schema: Name of the schema the index is built in.
        tenant_schema: Schema standing for another tenant's.
    """
    _tenant_holding_one_of_our_names(postgres_server_url, tenant_schema)
    crossed = url_scoped_to(postgres_server_url, postgres_schema, tenant_schema)
    with opened(crossed, create=True):
        pass
    assert _rows_of(postgres_server_url, f'"{tenant_schema}".{COLLIDING_TABLE}') == 1


def test_a_tenant_s_table_keeps_its_columns_through_a_creating_open(
    postgres_url: str, postgres_server_url: str, postgres_schema: str, tenant_schema: str
) -> None:
    """Nor is it altered into the shape the index wanted.

    Parameters:
        postgres_url: URL whose fixture creates and removes the schema built in.
        postgres_server_url: URL of the server, unscoped.
        postgres_schema: Name of the schema the index is built in.
        tenant_schema: Schema standing for another tenant's.
    """
    _tenant_holding_one_of_our_names(postgres_server_url, tenant_schema)
    crossed = url_scoped_to(postgres_server_url, postgres_schema, tenant_schema)
    with opened(crossed, create=True):
        pass
    engine = sqlalchemy.create_engine(postgres_server_url)
    try:
        columns = sqlalchemy.inspect(engine).get_columns(COLLIDING_TABLE, schema=tenant_schema)
    finally:
        engine.dispose()
    assert [column['name'] for column in columns] == ['id', 'payload']


def test_an_index_is_completed_in_the_schema_its_stamp_was_found_in(
    postgres_url: str, postgres_server_url: str, postgres_schema: str, tenant_schema: str
) -> None:
    """The stamp names the schema, so an ingest reached through a longer path finds it.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
        tenant_schema: Schema standing for another tenant's, empty and ahead of it.
    """
    with opened(postgres_url, create=True):
        pass
    reached = url_scoped_to(postgres_server_url, tenant_schema, postgres_schema)
    with opened(reached, create=True):
        pass
    assert _schema_tables(postgres_server_url, tenant_schema) == []


# ---------------------------------------------------------------------------
# What a refusal says
# ---------------------------------------------------------------------------


def test_a_refusal_names_the_schema_it_examined(postgres_url: str, postgres_schema: str) -> None:
    """Which schema a URL resolves to is not something the URL says.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_schema: Name of that schema.
    """
    _execute(postgres_url, f'CREATE TABLE {FOREIGN_TABLE} (x INTEGER)')
    assert f'schema {postgres_schema}' in _refusal_of(postgres_url)


def test_a_refusal_does_not_repeat_the_password(postgres_url: str) -> None:
    """These messages go to run logs, and a database password belongs in none.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _execute(postgres_url, f'CREATE TABLE {FOREIGN_TABLE} (x INTEGER)')
    with_password = sqlalchemy.engine.make_url(postgres_url).set(password=BOGUS_PASSWORD)
    refused = _refusal_of(with_password.render_as_string(hide_password=False))
    assert BOGUS_PASSWORD not in refused


def test_a_search_path_naming_no_schema_that_exists_is_refused(
    postgres_server_url: str,
) -> None:
    """There is nowhere to build, and saying so beats failing inside the first DDL.

    Parameters:
        postgres_server_url: URL of the server, unscoped.
    """
    nowhere = url_scoped_to(postgres_server_url, f'ri_absent_{uuid.uuid4().hex}')
    assert 'reaches no schema a table can be created in' in _refusal_of(nowhere)


# ---------------------------------------------------------------------------
# A stamp of another version
# ---------------------------------------------------------------------------


def test_a_stamped_schema_of_another_version_reaches_the_version_gate(
    postgres_url: str,
) -> None:
    """The database a version bump leaves behind is the index's own, and is told so.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _execute(
        postgres_url,
        f'CREATE TABLE {SCHEMA_META.name} '
        f'(singleton int primary key, schema_version int, created_utc text)',
        f'INSERT INTO {SCHEMA_META.name} VALUES (1, {SCHEMA_VERSION - 1}, now()::text)',
        f'CREATE TABLE {COLLIDING_TABLE} (root_url text, a_column_of_an_older_version text)',
    )
    assert f'schema version {SCHEMA_VERSION - 1} is not the version' in _refusal_of(postgres_url)


def test_a_foreign_table_beside_a_stamped_schema_is_refused_first(
    postgres_url: str,
) -> None:
    """A stamp says the tables of our names are ours, and nothing about the rest.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _execute(
        postgres_url,
        f'CREATE TABLE {SCHEMA_META.name} '
        f'(singleton int primary key, schema_version int, created_utc text)',
        f'INSERT INTO {SCHEMA_META.name} VALUES (1, {SCHEMA_VERSION}, now()::text)',
        f'CREATE TABLE {FOREIGN_TABLE} (x INTEGER)',
    )
    assert FOREIGN_TABLE in _refusal_of(postgres_url)
