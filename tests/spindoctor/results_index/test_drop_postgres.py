"""Dropping a results index from a real PostgreSQL server.

Four things about the drop are properties of the server rather than of the
schema, and none of them can be asked of SQLite.  That the tables it names are
the only objects of the database it removes, with somebody else's left standing
beside them -- a shared server being the deployment the promise is for.  That an
emptied schema and one nothing was ever built in are the same database, which is
what "indistinguishable from an index that never existed" means where there is
no file to tell them apart.  That a lock somebody else holds ends the drop
promptly instead of hanging it.  And that when it does, the transaction takes
every table back, which is a guarantee PostgreSQL gives and the SQLite driver
does not.

The tier is opt-in: it is excluded by the default marker filter and skips itself
when ``SPINDOCTOR_TEST_POSTGRES_URL`` is unset.  What must not regress lives in
the default tier as well; only what genuinely needs a server is here.
"""

import threading

import pytest
import sqlalchemy
from tests.spindoctor.results_index.conftest import opened

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


def _add_foreign_table(url: str) -> None:
    """Create a table the index does not own, in the schema under test.

    Parameters:
        url: The scoped index URL.
    """
    engine = sqlalchemy.create_engine(url)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(f'CREATE TABLE {FOREIGN_TABLE} (x INTEGER)')
    finally:
        engine.dispose()


def _dropped(url: str) -> tuple[str, ...]:
    """Open a database, drop the index tables from it, and close it.

    Parameters:
        url: The database URL.

    Returns:
        The names of the tables that were dropped.
    """
    engine = open_database(url)
    try:
        return drop_index_tables(engine)
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


BOUNDED_WAIT_S = 20.0
"""How long a test waits for a contended drop before calling the bound broken.

Forty times the shortened bound above.  A drop that is still waiting after this
has no bound at all, and saying so is what keeps a regression in the bound from
arriving as a suite that never finishes.
"""


def _drop_while_a_lock_is_held(url: str) -> BaseException:
    """Hold a lock on one table, run a drop against it, and return what stopped it.

    The drop runs on a thread of its own so that the wait can be bounded by this
    test rather than by the drop: what is under test is that the drop gives the
    table up, and a drop with no bound would otherwise take the whole suite down
    with it.

    Parameters:
        url: The index URL, whose ``images`` table is the one held.

    Returns:
        The database failure the drop ended with.
    """
    failure: list[BaseException] = []

    def attempt() -> None:
        try:
            _dropped(url)
        except BaseException as exc:
            failure.append(exc)

    holder = sqlalchemy.create_engine(url)
    try:
        with holder.begin() as held:
            held.execute(sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES))
            worker = threading.Thread(target=attempt, daemon=True)
            worker.start()
            worker.join(BOUNDED_WAIT_S)
            waiting_still = worker.is_alive()
    finally:
        holder.dispose()
    if waiting_still:
        pytest.fail(f'the drop was still waiting for the lock after {BOUNDED_WAIT_S} seconds')
    if not failure:
        pytest.fail('the drop finished although another session held one of its tables')
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
    assert 'lock timeout' in str(_drop_while_a_lock_is_held(postgres_url))


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
    _drop_while_a_lock_is_held(postgres_url)
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
    _drop_while_a_lock_is_held(postgres_url)
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
