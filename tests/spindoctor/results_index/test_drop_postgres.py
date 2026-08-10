"""Dropping a results index from a real PostgreSQL server.

Six things about the drop are properties of the server rather than of the
schema, and none of them can be asked of SQLite.  That the tables it names are
the only objects of the database it removes, with somebody else's left standing
beside them -- a shared server being the deployment the promise is for.  That a
table of one of these names which no stamp of ours stands over is left alone,
because a server holds databases nobody here created and ``images`` is not a
name anyone owns.  That a search path crossing schemas cannot make one drop span
two of them.  That an emptied schema and one nothing was ever built in are the
same database, which is what "indistinguishable from an index that never
existed" means where there is no file to tell them apart.  That a lock somebody
else holds ends the drop promptly instead of hanging it, in the reading as well
as in the drop.  And that when it does, the transaction takes every table back.

The tier is opt-in: it is excluded by the default marker filter and skips itself
when ``SPINDOCTOR_TEST_POSTGRES_URL`` is unset.  What must not regress lives in
the default tier as well; only what genuinely needs a server is here.
"""

import threading
import uuid
from collections.abc import Iterator

import pytest
import sqlalchemy
from tests.spindoctor.results_index.conftest import opened, url_scoped_to

from spindoctor.results_index import (
    IMAGES,
    SCHEMA_META,
    SCHEMA_VERSION,
    drop,
    drop_index_tables,
    index_contents,
    index_table_names,
    open_database,
    open_index,
)

pytestmark = pytest.mark.postgres

FOREIGN_TABLE = 'somebody_elses_table'
"""A table of the same schema that SpinDoctor did not create."""

COLLIDING_TABLE = IMAGES.name
"""A table SpinDoctor did not create, under a name SpinDoctor also uses."""

BRIEF_LOCK_WAIT_MS = 500
"""How long the contended drop waits, so that a held lock costs half a second.

The shipped bound is thirty seconds, which is the right wait for an operator and
the wrong one for a test; what is under test is that a bound applies at all, and
that the drop gives the table up rather than waiting on it forever.
"""


def _schema_tables(server_url: str, schema: str) -> list[str]:
    """Return every table of one schema, whoever created it.

    Scoped to the named schema rather than to a search path, because the
    catalog spans the server and another worker's schema holds tables of these
    same names.

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


def _add_foreign_table(url: str) -> None:
    """Create a table the index does not own, in the schema under test.

    Parameters:
        url: The scoped index URL.
    """
    _execute(url, f'CREATE TABLE {FOREIGN_TABLE} (x INTEGER)')


def _dropped(url: str) -> tuple[str, ...]:
    """Open a database, drop the index tables from it, and close it.

    Parameters:
        url: The database URL.

    Returns:
        The names of the tables that were dropped.
    """
    engine = open_database(url)
    try:
        return drop_index_tables(engine, index_contents(engine))
    finally:
        engine.dispose()


def _refusal_of(url: str) -> str:
    """Return the message a consumer's open of a database is refused with.

    Parameters:
        url: The database URL.

    Returns:
        The refusal message.
    """
    with pytest.raises(ValueError) as excinfo:
        open_index(url)
    return str(excinfo.value)


@pytest.fixture
def tenant_schema(postgres_server_url: str) -> Iterator[str]:
    """Yield a schema standing for another tenant's, holding an ``images`` of its own.

    Created after the fixtures the index is built in, and put in front of the
    index's schema on the search path only by the tests that ask for it, because
    a table of one of our names ahead of the index is a state a creating open
    would itself resolve wrongly.

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
# What the drop removes from a server
# ---------------------------------------------------------------------------


def test_the_drop_removes_every_index_table_from_the_server(
    postgres_url: str, postgres_server_url: str, postgres_schema: str
) -> None:
    """The metadata's DDL is accepted by a server on the way out as well as in.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
    """
    with opened(postgres_url, create=True):
        pass
    _dropped(postgres_url)
    assert _schema_tables(postgres_server_url, postgres_schema) == []


def test_a_table_the_index_does_not_own_survives_on_the_server(
    postgres_url: str, postgres_server_url: str, postgres_schema: str
) -> None:
    """A shared server is exactly what the "only our tables" promise is for.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
    """
    with opened(postgres_url, create=True):
        pass
    _add_foreign_table(postgres_url)
    _dropped(postgres_url)
    assert _schema_tables(postgres_server_url, postgres_schema) == [FOREIGN_TABLE]


def test_a_second_drop_removes_nothing_on_the_server(postgres_url: str) -> None:
    """Idempotent here too, where there is no file whose absence could say so.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True):
        pass
    _dropped(postgres_url)
    assert _dropped(postgres_url) == ()


def test_the_contents_name_the_schema_the_index_was_found_in(
    postgres_url: str, postgres_schema: str
) -> None:
    """Which is a real name on a server, not the one namespace a file has.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_schema: Name of that schema.
    """
    with opened(postgres_url, create=True):
        pass
    engine = open_database(postgres_url)
    try:
        assert index_contents(engine).schema == postgres_schema
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# A database holding no index of ours, under names it also uses
# ---------------------------------------------------------------------------


def test_a_stranger_s_table_of_one_of_our_names_is_not_dropped(
    postgres_url: str, postgres_server_url: str, postgres_schema: str
) -> None:
    """A database with no SpinDoctor index in it must lose nothing at all.

    A server holds databases nobody here created, and ``images`` is one of the
    commonest table names there are.  Nothing but our own stamp says a table of
    that name is ours.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
    """
    _execute(
        postgres_url,
        f'CREATE TABLE {COLLIDING_TABLE} (id serial primary key, caption text)',
        f"INSERT INTO {COLLIDING_TABLE} (caption) VALUES ('somebody elses cat'), ('their dog')",
        'CREATE TABLE customers_of_theirs (id serial primary key, name text)',
    )
    _dropped(postgres_url)
    assert _schema_tables(postgres_server_url, postgres_schema) == sorted(
        [COLLIDING_TABLE, 'customers_of_theirs']
    )


def test_a_stranger_s_rows_survive_a_drop_pointed_at_their_database(
    postgres_url: str, postgres_schema: str
) -> None:
    """The rows are the thing, and the account called them the index's.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_schema: Name of that schema.
    """
    _execute(
        postgres_url,
        f'CREATE TABLE {COLLIDING_TABLE} (id serial primary key, caption text)',
        f"INSERT INTO {COLLIDING_TABLE} (caption) VALUES ('somebody elses cat'), ('their dog')",
    )
    _dropped(postgres_url)
    assert _rows_of(postgres_url, f'"{postgres_schema}".{COLLIDING_TABLE}') == 2


def test_such_a_database_is_reported_as_holding_no_index(postgres_url: str) -> None:
    """So that the command refuses rather than reporting a drop of nothing.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _execute(postgres_url, f'CREATE TABLE {COLLIDING_TABLE} (id serial primary key)')
    engine = open_database(postgres_url)
    try:
        contents = index_contents(engine)
    finally:
        engine.dispose()
    assert contents.schema is None


def test_such_a_database_names_the_tables_it_could_not_account_for(postgres_url: str) -> None:
    """A refusal that named nothing would leave an operator no next step.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _execute(postgres_url, f'CREATE TABLE {COLLIDING_TABLE} (id serial primary key)')
    engine = open_database(postgres_url)
    try:
        contents = index_contents(engine)
    finally:
        engine.dispose()
    assert contents.unproven == (COLLIDING_TABLE,)


def _generated_password() -> str:
    """Return a password for a role this test creates and drops again.

    Generated per run rather than written down, for the same reason the role
    name is: a literal in a test file is a credential in the repository, and
    this one is handed to a real server.

    Returns:
        A password unique to this fixture's role.
    """
    return f'ri-pw-{uuid.uuid4().hex}'


@pytest.fixture
def restricted_url(
    postgres_url: str, postgres_server_url: str, postgres_schema: str
) -> Iterator[str]:
    """Yield a URL onto an index whose stamp column this account may not read.

    A schema version is read out of one column of one table, and a column is
    something an account can be allowed part of.  Reaching that state needs a
    second role, since the role that owns the tables is allowed everything by
    definition; a server that will not let this test make one is one the
    question cannot be put to, and the test says so rather than passing.

    Parameters:
        postgres_url: URL the index's tables are created through.
        postgres_server_url: URL of the server, unscoped.
        postgres_schema: Name of the schema the tables live in.

    Yields:
        The same scoped URL, connecting as the restricted role.
    """
    _execute(
        postgres_url,
        f'CREATE TABLE {SCHEMA_META.name} '
        f'(singleton int primary key, schema_version int, created_utc text)',
        f'INSERT INTO {SCHEMA_META.name} VALUES (1, {SCHEMA_VERSION}, now()::text)',
        f'CREATE TABLE {IMAGES.name} (root_url text)',
        f"INSERT INTO {IMAGES.name} VALUES ('file:///data/nav-results')",
    )
    role = f'ri_reader_{uuid.uuid4().hex[:16]}'
    password = _generated_password()
    admin = sqlalchemy.create_engine(postgres_server_url, isolation_level='AUTOCOMMIT')
    try:
        try:
            with admin.connect() as connection:
                connection.exec_driver_sql(f'CREATE ROLE "{role}" LOGIN PASSWORD \'{password}\'')
        except sqlalchemy.exc.SQLAlchemyError:
            pytest.skip('this account may not create a role on this server')
        try:
            with admin.connect() as connection:
                connection.exec_driver_sql(f'GRANT USAGE ON SCHEMA "{postgres_schema}" TO "{role}"')
                connection.exec_driver_sql(
                    f'GRANT SELECT ON "{postgres_schema}".{IMAGES.name} TO "{role}"'
                )
                connection.exec_driver_sql(
                    f'GRANT SELECT (singleton) ON "{postgres_schema}".{SCHEMA_META.name} '
                    f'TO "{role}"'
                )
            yield (
                sqlalchemy.engine.make_url(postgres_url)
                .set(username=role, password=password)
                .render_as_string(hide_password=False)
            )
        finally:
            with admin.connect() as connection:
                connection.exec_driver_sql(f'DROP OWNED BY "{role}"')
                connection.exec_driver_sql(f'DROP ROLE "{role}"')
    finally:
        admin.dispose()


def test_a_stamp_this_account_may_not_read_is_reported_as_no_stamp(restricted_url: str) -> None:
    """A version that will not come out of a database is a version it does not have.

    The read is taken inside a savepoint, because a statement PostgreSQL refuses
    ends the transaction around it, and every later reading of this account
    would then fail with the first failure's shadow instead of answering.

    Parameters:
        restricted_url: URL onto an index whose stamp column is unreadable.
    """
    engine = open_database(restricted_url)
    try:
        assert index_contents(engine).schema_version is None
    finally:
        engine.dispose()


def test_a_stamp_this_account_may_not_read_leaves_the_rest_of_the_account(
    restricted_url: str,
) -> None:
    """The stamp is one line of the account, and the tables and their rows are the rest.

    Parameters:
        restricted_url: URL onto an index whose stamp column is unreadable.
    """
    engine = open_database(restricted_url)
    try:
        contents = index_contents(engine)
    finally:
        engine.dispose()
    assert [(table.name, table.rows) for table in contents.tables] == [
        (SCHEMA_META.name, 1),
        (IMAGES.name, 1),
    ]


# ---------------------------------------------------------------------------
# A search path that crosses schemas
# ---------------------------------------------------------------------------


def _index_behind_a_tenant(
    postgres_url: str, postgres_server_url: str, postgres_schema: str, tenant_schema: str
) -> str:
    """Build the index, then put another tenant's ``images`` in front of it.

    The index is built normally, and the search path is changed afterwards, so
    that a bare ``images`` reaches somebody else's table while a bare
    ``schema_meta`` still reaches the index's own.

    Parameters:
        postgres_url: URL the index is built through.
        postgres_server_url: URL of the server, unscoped.
        postgres_schema: Schema the index was built in.
        tenant_schema: Schema standing for another tenant's.

    Returns:
        A URL whose search path names the tenant's schema and then the index's.
    """
    with opened(postgres_url, create=True):
        pass
    _execute(
        postgres_server_url,
        f'CREATE TABLE "{tenant_schema}".{COLLIDING_TABLE} (id int primary key, payload text)',
        f'INSERT INTO "{tenant_schema}".{COLLIDING_TABLE} VALUES '
        f"(1, 'not ours'), (2, 'also not ours')",
    )
    return url_scoped_to(postgres_server_url, tenant_schema, postgres_schema)


def test_a_search_path_crossing_schemas_does_not_reach_the_other_one(
    postgres_url: str, postgres_server_url: str, postgres_schema: str, tenant_schema: str
) -> None:
    """Six bare names resolved one at a time is how one drop spans two schemas.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
        tenant_schema: Schema standing for another tenant's.
    """
    crossed = _index_behind_a_tenant(
        postgres_url, postgres_server_url, postgres_schema, tenant_schema
    )
    _dropped(crossed)
    assert _schema_tables(postgres_server_url, tenant_schema) == [COLLIDING_TABLE]


def test_the_other_schema_keeps_its_rows(
    postgres_url: str, postgres_server_url: str, postgres_schema: str, tenant_schema: str
) -> None:
    """And the account did not call them ours on the way past.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
        tenant_schema: Schema standing for another tenant's.
    """
    crossed = _index_behind_a_tenant(
        postgres_url, postgres_server_url, postgres_schema, tenant_schema
    )
    _dropped(crossed)
    assert _rows_of(postgres_server_url, f'"{tenant_schema}".{COLLIDING_TABLE}') == 2


def test_the_index_s_own_schema_is_emptied_whole(
    postgres_url: str, postgres_server_url: str, postgres_schema: str, tenant_schema: str
) -> None:
    """The other half of the same failure: our own ``images`` was left standing.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
        tenant_schema: Schema standing for another tenant's.
    """
    crossed = _index_behind_a_tenant(
        postgres_url, postgres_server_url, postgres_schema, tenant_schema
    )
    _dropped(crossed)
    assert _schema_tables(postgres_server_url, postgres_schema) == []


def test_the_rows_counted_are_the_rows_of_the_index_s_own_schema(
    postgres_url: str, postgres_server_url: str, postgres_schema: str, tenant_schema: str
) -> None:
    """An account reading another schema's rows is an account of the wrong table.

    The index here holds no image rows and the tenant's table holds two, so the
    count is the whole of the difference between the two readings.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
        tenant_schema: Schema standing for another tenant's.
    """
    crossed = _index_behind_a_tenant(
        postgres_url, postgres_server_url, postgres_schema, tenant_schema
    )
    engine = open_database(crossed)
    try:
        counted = {table.name: table.rows for table in index_contents(engine).tables}
    finally:
        engine.dispose()
    assert counted[IMAGES.name] == 0


def test_the_schema_the_drop_binds_to_is_the_one_holding_the_stamp(
    postgres_url: str, postgres_server_url: str, postgres_schema: str, tenant_schema: str
) -> None:
    """Named, so that a reading and the drop that follows cannot disagree.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
        tenant_schema: Schema standing for another tenant's.
    """
    crossed = _index_behind_a_tenant(
        postgres_url, postgres_server_url, postgres_schema, tenant_schema
    )
    engine = open_database(crossed)
    try:
        assert index_contents(engine).schema == postgres_schema
    finally:
        engine.dispose()


def test_an_index_in_front_on_the_path_is_the_one_that_goes(
    postgres_url: str, postgres_server_url: str, postgres_schema: str, tenant_schema: str
) -> None:
    """Two whole indexes on one path: the one this URL reaches is the one dropped.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
        tenant_schema: Schema standing for another tenant's.
    """
    with opened(postgres_url, create=True):
        pass
    with opened(url_scoped_to(postgres_server_url, tenant_schema), create=True):
        pass
    _dropped(url_scoped_to(postgres_server_url, tenant_schema, postgres_schema))
    assert _schema_tables(postgres_server_url, postgres_schema) == sorted(index_table_names())


def test_the_index_behind_it_on_the_path_is_the_one_left(
    postgres_url: str, postgres_server_url: str, postgres_schema: str, tenant_schema: str
) -> None:
    """The same drop, read from the other side.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
        tenant_schema: Schema standing for another tenant's.
    """
    with opened(postgres_url, create=True):
        pass
    with opened(url_scoped_to(postgres_server_url, tenant_schema), create=True):
        pass
    _dropped(url_scoped_to(postgres_server_url, tenant_schema, postgres_schema))
    assert _schema_tables(postgres_server_url, tenant_schema) == []


# ---------------------------------------------------------------------------
# A dropped index and one that never existed
# ---------------------------------------------------------------------------


def test_an_emptied_schema_is_refused_exactly_as_an_untouched_one_is(
    postgres_url: str,
) -> None:
    """The whole of "indistinguishable from an index that never existed".

    One URL is asked twice: before anything is built in it, and after an index
    has been built and dropped.  With no file to tell the two states apart, a
    consumer's refusal is the entire observable difference between them, and the
    two messages are compared character for character.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    never_existed = _refusal_of(postgres_url)
    with opened(postgres_url, create=True):
        pass
    _dropped(postgres_url)
    assert _refusal_of(postgres_url) == never_existed


def test_an_emptied_schema_takes_an_ingest_again(postgres_url: str) -> None:
    """Left usable rather than left broken, on the backend with no file to delete.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    _dropped(postgres_url)
    with opened(postgres_url, create=True) as engine, engine.connect() as connection:
        stamped = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).scalar()
    assert stamped == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# A lock somebody else holds
# ---------------------------------------------------------------------------


BOUNDED_WAIT_S = 20.0
"""How long a test waits for a contended drop before calling the bound broken.

Forty times the shortened bound above.  A drop that is still waiting after this
has no bound at all, and saying so is what keeps a regression in the bound from
arriving as a suite that never finishes.
"""


def _bounded(url: str, work: str) -> BaseException:
    """Hold ``images`` against a reader, run one step of a drop, and return what stopped it.

    The step runs on a thread of its own so that the wait can be bounded by this
    test rather than by the drop: what is under test is that the step gives the
    table up, and one with no bound would otherwise take the whole suite down
    with it.

    Parameters:
        url: The index URL, whose ``images`` table is the one held.
        work: Which step to run, ``'read'`` for the reading that precedes the
            question or ``'drop'`` for the drop itself.

    Returns:
        The database failure the step ended with.
    """
    failure: list[BaseException] = []

    def attempt() -> None:
        try:
            if work == 'drop':
                _dropped(url)
            else:
                engine = open_database(url)
                try:
                    index_contents(engine)
                finally:
                    engine.dispose()
        except BaseException as exc:
            failure.append(exc)

    holder = sqlalchemy.create_engine(url)
    try:
        with holder.begin() as held:
            # A plain read is enough to hold a drop off; an exclusive lock is
            # what holds a read off too, and is what a VACUUM FULL, an ALTER
            # TABLE or a second drop takes.
            if work == 'drop':
                held.execute(sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES))
            else:
                held.exec_driver_sql(f'LOCK TABLE {IMAGES.name} IN ACCESS EXCLUSIVE MODE')
            worker = threading.Thread(target=attempt, daemon=True)
            worker.start()
            worker.join(BOUNDED_WAIT_S)
            waiting_still = worker.is_alive()
    finally:
        holder.dispose()
    if waiting_still:
        pytest.fail(f'the {work} was still waiting for the lock after {BOUNDED_WAIT_S} seconds')
    if not failure:
        pytest.fail(f'the {work} finished although another session held one of its tables')
    return failure[0]


def test_a_lock_another_session_holds_ends_the_drop(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bounded rather than refused in advance, because no backend can be asked.

    A session that has merely read a table holds ``DROP TABLE`` off for as long
    as its transaction stays open, and without a bound the drop waits on that
    silently and forever.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        monkeypatch: Fixture the shipped wait is shortened through.
    """
    monkeypatch.setattr(drop, 'DROP_LOCK_TIMEOUT_MS', BRIEF_LOCK_WAIT_MS)
    with opened(postgres_url, create=True):
        pass
    assert 'lock timeout' in str(_bounded(postgres_url, 'drop'))


def test_a_lock_another_session_holds_ends_the_reading_too(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reading counts rows, and counting rows takes a lock of its own.

    A bound that began after the confirmation would leave the command hanging
    before anybody had been asked anything, which is the hang hardest of all to
    account for: nothing has been printed and nothing is being waited on that
    the operator can see.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        monkeypatch: Fixture the shipped wait is shortened through.
    """
    monkeypatch.setattr(drop, 'DROP_LOCK_TIMEOUT_MS', BRIEF_LOCK_WAIT_MS)
    with opened(postgres_url, create=True):
        pass
    assert 'lock timeout' in str(_bounded(postgres_url, 'read'))


def test_the_lock_bound_lasts_no_longer_than_the_drop(postgres_url: str) -> None:
    """``SET LOCAL``, so the bound is this transaction's and no later statement's.

    A session-wide setting would work as well for the drop and would then ride
    the pooled connection into whatever ran on it next, silently bounding
    statements nobody bounded.  What says the bound was local is that the
    setting reads the same after the drop's transaction has ended as it did
    before it began, whatever this server's own setting happens to be.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True):
        pass
    engine = open_database(postgres_url)
    try:
        with engine.connect() as connection:
            before = connection.exec_driver_sql('SHOW lock_timeout').scalar()
        drop_index_tables(engine, index_contents(engine))
        with engine.connect() as connection:
            after = connection.exec_driver_sql('SHOW lock_timeout').scalar()
    finally:
        engine.dispose()
    assert after == before


def test_a_drop_that_could_not_finish_leaves_every_table(
    postgres_url: str,
    postgres_server_url: str,
    postgres_schema: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refused rather than half-completed: the transaction takes them all back.

    The stamp is dropped first, so a drop that gave up on a later table without
    a transaction around it would have left the index unstamped.  Here it is
    exactly as it was.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
        monkeypatch: Fixture the shipped wait is shortened through.
    """
    monkeypatch.setattr(drop, 'DROP_LOCK_TIMEOUT_MS', BRIEF_LOCK_WAIT_MS)
    with opened(postgres_url, create=True):
        pass
    _bounded(postgres_url, 'drop')
    assert _schema_tables(postgres_server_url, postgres_schema) == sorted(index_table_names())


def test_a_drop_that_could_not_finish_leaves_the_index_openable(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which is what "no partially-dropped state" is for: a consumer still reads it.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        monkeypatch: Fixture the shipped wait is shortened through.
    """
    monkeypatch.setattr(drop, 'DROP_LOCK_TIMEOUT_MS', BRIEF_LOCK_WAIT_MS)
    with opened(postgres_url, create=True):
        pass
    _bounded(postgres_url, 'drop')
    with opened(postgres_url) as engine, engine.connect() as connection:
        stamped = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).scalar()
    assert stamped == SCHEMA_VERSION


def test_the_contents_read_from_the_server_count_the_rows(postgres_url: str) -> None:
    """Counted through the table name alone, which a strict backend still types.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION))
    engine = open_database(postgres_url)
    try:
        contents = index_contents(engine)
    finally:
        engine.dispose()
    assert contents.rows == 1


# ---------------------------------------------------------------------------
# A stamp the server's own typing will not read as a version
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'stamp',
    ["'v6-beta'", 'NULL'],
    ids=['a-version-that-is-text', 'a-version-that-is-null'],
)
def test_a_stamp_that_is_not_a_version_is_reported_as_none(stamp: str, postgres_url: str) -> None:
    """A strict backend hands back the text, and int() is what refuses it.

    Parameters:
        stamp: The value the stamp column is given.
        postgres_url: URL of an empty schema of this test's own.
    """
    _execute(
        postgres_url,
        f'CREATE TABLE {SCHEMA_META.name} '
        f'(singleton int primary key, schema_version text, created_utc text)',
        f'INSERT INTO {SCHEMA_META.name} VALUES (1, {stamp}, NULL)',
        f'CREATE TABLE {COLLIDING_TABLE} (id int)',
    )
    engine = open_database(postgres_url)
    try:
        contents = index_contents(engine)
    finally:
        engine.dispose()
    assert contents.schema_version is None


@pytest.mark.parametrize(
    'stamp',
    ["'v6-beta'", 'NULL'],
    ids=['a-version-that-is-text', 'a-version-that-is-null'],
)
def test_a_stamp_that_is_not_a_version_does_not_stop_the_drop(
    stamp: str, postgres_url: str, postgres_server_url: str, postgres_schema: str
) -> None:
    """It is a malformed index, which is one of the states the drop is for.

    Parameters:
        stamp: The value the stamp column is given.
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
    """
    _execute(
        postgres_url,
        f'CREATE TABLE {SCHEMA_META.name} '
        f'(singleton int primary key, schema_version text, created_utc text)',
        f'INSERT INTO {SCHEMA_META.name} VALUES (1, {stamp}, NULL)',
        f'CREATE TABLE {COLLIDING_TABLE} (id int)',
    )
    _dropped(postgres_url)
    assert _schema_tables(postgres_server_url, postgres_schema) == []


def test_a_table_of_ours_in_another_schema_is_not_counted_as_present_here(
    postgres_url: str, postgres_server_url: str, postgres_schema: str, tenant_schema: str
) -> None:
    """Presence is asked of the bound schema, not of whatever the path reaches.

    The index here is missing one of its tables and the tenant's schema holds a
    table of that name, so a presence check resolved through the search path
    answers yes about a table that is not in the index's schema at all.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
        tenant_schema: Schema standing for another tenant's.
    """
    with opened(postgres_url, create=True):
        pass
    _execute(
        postgres_server_url,
        f'DROP TABLE "{postgres_schema}".failed_files',
        f'CREATE TABLE "{tenant_schema}".failed_files (root_url text)',
    )
    crossed = url_scoped_to(postgres_server_url, tenant_schema, postgres_schema)
    assert 'failed_files' not in _dropped(crossed)


def test_such_a_table_survives_in_the_schema_it_belongs_to(
    postgres_url: str, postgres_server_url: str, postgres_schema: str, tenant_schema: str
) -> None:
    """And the drop that skipped it still removed the five that were there.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
        tenant_schema: Schema standing for another tenant's.
    """
    with opened(postgres_url, create=True):
        pass
    _execute(
        postgres_server_url,
        f'DROP TABLE "{postgres_schema}".failed_files',
        f'CREATE TABLE "{tenant_schema}".failed_files (root_url text)',
    )
    crossed = url_scoped_to(postgres_server_url, tenant_schema, postgres_schema)
    _dropped(crossed)
    assert _schema_tables(postgres_server_url, postgres_schema) == []
