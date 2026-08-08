"""Results-index tests that run against a real PostgreSQL server.

SQLite accepts almost anything: it stores a boolean as an integer, a double as
whatever the value looked like, and a foreign key as a comment unless asked
otherwise. A schema that behaves on SQLite therefore proves very little about
the backend an index shared across machines actually runs on. These re-ask the
questions that the type discipline exists to answer, against a server that
enforces them.

The tier is opt-in: it is excluded by the default marker filter and skips itself
when ``SPINDOCTOR_TEST_POSTGRES_URL`` is unset, so a checkout with no server
still runs a green suite.

Each test gets a schema of its own from the ``postgres_url`` fixture, so a
repeated run, or two workers of a parallel run, never share a table.
"""

import pytest
import sqlalchemy
from tests.spindoctor.results_index.conftest import STUB, image_row, opened, technique_row

from spindoctor.results_index import IMAGES, SCHEMA_META, SCHEMA_VERSION, TECHNIQUES, open_index

pytestmark = pytest.mark.postgres

OTHER_STUB = 'COISS_2002/data/1295221349_1296000000/N1294561202_1_CALIB'

FIFTEEN_DIGIT_OFFSET = -1234.56789012345

BOGUS_PASSWORD = 'sup3rs3cr3t'
"""A password distinctive enough that finding it anywhere is proof of a leak."""


def _password_of(url: str) -> str:
    """Return the password a URL carries.

    Parameters:
        url: The URL to read.

    Returns:
        The password, or an empty string when the URL carries none.
    """
    return sqlalchemy.engine.make_url(url).password or ''


def _with_password(url: str, password: str) -> str:
    """Return a URL carrying a different password.

    Parameters:
        url: The URL to rewrite.
        password: The password to put in it.

    Returns:
        The rewritten URL, with the password in plain text as a caller writes it.
    """
    rewritten = sqlalchemy.engine.make_url(url).set(password=password)
    return rewritten.render_as_string(hide_password=False)


def _refusal_without_the_password(url: str, message: str) -> str:
    """Open a URL, require the refusal it raises, and return that message.

    Parameters:
        url: The URL to open.
        message: Pattern the refusal message must match.

    Returns:
        The refusal message.
    """
    if not _password_of(url):
        pytest.skip('the configured server URL carries no password to mask')
    with pytest.raises(ValueError, match=message) as excinfo:
        open_index(url)
    return str(excinfo.value)


def test_creating_the_schema_creates_every_table(postgres_url: str) -> None:
    """The metadata emits DDL a server accepts, not just DDL SQLite accepts.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        found = sorted(sqlalchemy.inspect(engine).get_table_names())
    assert found == ['feature_sources', 'images', 'ingest_runs', 'schema_meta', 'techniques']


def test_creating_the_schema_stamps_the_version(postgres_url: str) -> None:
    """The stamp is what the version gate reads on every later open.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine, engine.connect() as connection:
        stamped = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).scalar()
    assert stamped == SCHEMA_VERSION


def test_a_consumer_refuses_a_schema_that_was_never_ingested(postgres_url: str) -> None:
    """An empty database is not an index, and absence is not "nothing navigated".

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with pytest.raises(ValueError, match='not a results index'):
        open_index(postgres_url)


def test_a_database_stamped_with_another_version_is_refused(postgres_url: str) -> None:
    """The gate is the whole migration strategy, so it has to hold here too.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    with pytest.raises(ValueError, match=f'schema version {SCHEMA_VERSION + 1}'):
        open_index(postgres_url)


def test_the_version_message_says_to_delete_and_re_ingest(postgres_url: str) -> None:
    """There are no migrations, so the instruction is the whole remedy.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    with pytest.raises(ValueError, match='delete the database and re-run sd_stats_ingest'):
        open_index(postgres_url)


def test_the_not_an_index_refusal_does_not_repeat_the_password(postgres_url: str) -> None:
    """A server URL is the one that carries a password, and this gate names it.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    message = _refusal_without_the_password(postgres_url, 'not a results index')
    assert f':{_password_of(postgres_url)}@' not in message


def test_the_version_refusal_does_not_repeat_the_password(postgres_url: str) -> None:
    """Every route names the URL, so every route has to mask it.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    message = _refusal_without_the_password(postgres_url, 'is not the version')
    assert f':{_password_of(postgres_url)}@' not in message


def test_a_masked_refusal_still_names_the_server(postgres_url: str) -> None:
    """Masking must not cost the identification the message exists for.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    message = _refusal_without_the_password(postgres_url, 'not a results index')
    assert str(sqlalchemy.engine.make_url(postgres_url).host) in message


def test_a_rejected_password_is_not_repeated_in_the_refusal(postgres_server_url: str) -> None:
    """The failure most likely to carry a password is the one that is about it.

    Parameters:
        postgres_server_url: URL of the server the tier runs against.
    """
    with pytest.raises(ValueError, match='could not open the results index') as excinfo:
        open_index(_with_password(postgres_server_url, BOGUS_PASSWORD))
    assert BOGUS_PASSWORD not in str(excinfo.value)


def test_the_offset_round_trips_bit_for_bit(postgres_url: str) -> None:
    """Double precision on the server, not the single precision Float may give.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row(offset_dv=FIFTEEN_DIGIT_OFFSET))
        with engine.connect() as connection:
            stored = connection.execute(sqlalchemy.select(IMAGES.c.offset_dv)).scalar()
    assert repr(stored) == repr(FIFTEEN_DIGIT_OFFSET)


def test_a_boolean_column_round_trips_true(postgres_url: str) -> None:
    """A native boolean, not an integer flag that happens to read back as one.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(TECHNIQUES.insert(), technique_row(spurious=True))
        with engine.connect() as connection:
            stored = connection.execute(sqlalchemy.select(TECHNIQUES.c.spurious)).scalar()
    assert stored is True


def test_comparing_a_boolean_column_to_an_integer_is_an_error(postgres_url: str) -> None:
    """This is why the integer spellings had to go, demonstrated rather than asserted.

    ``spurious = 0`` is ordinary SQLite and a type error here, so a query that
    kept the SQLite spelling would fail the moment an operator moved the index to
    a server.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:  # noqa: SIM117
        with pytest.raises(sqlalchemy.exc.ProgrammingError, match='boolean'):
            with engine.connect() as connection:
                connection.execute(sqlalchemy.text('SELECT * FROM techniques WHERE spurious = 0'))


def test_the_same_basename_in_two_volumes_produces_two_rows(postgres_url: str) -> None:
    """The composite key is a real constraint on the server, not a convention.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(IMAGES.insert(), image_row(results_path_stub=OTHER_STUB))
        with engine.connect() as connection:
            stubs = connection.execute(
                sqlalchemy.select(IMAGES.c.results_path_stub).order_by(IMAGES.c.results_path_stub)
            ).scalars()
            found = list(stubs)
    assert found == [STUB, OTHER_STUB]


def test_deleting_an_image_deletes_its_technique_rows(postgres_url: str) -> None:
    """The cascade is enforced natively here, with no pragma to remember.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(TECHNIQUES.insert(), technique_row())
            connection.execute(IMAGES.delete().where(IMAGES.c.results_path_stub == STUB))
        with engine.connect() as connection:
            remaining = connection.execute(
                sqlalchemy.select(sqlalchemy.func.count()).select_from(TECHNIQUES)
            ).scalar()
    assert remaining == 0


def test_a_json_column_round_trips_a_list(postgres_url: str) -> None:
    """The JSON columns are jsonb here, which the direct-SQL examples rely on.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row(excluded_from_consensus=['ring_edge']))
        with engine.connect() as connection:
            stored = connection.execute(
                sqlalchemy.select(IMAGES.c.excluded_from_consensus)
            ).scalar()
    assert stored == ['ring_edge']


def test_a_json_column_is_jsonb(postgres_url: str) -> None:
    """A plain ``json`` column would reject ``jsonb_array_elements_text``.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine, engine.connect() as connection:
        found = connection.execute(
            sqlalchemy.text(
                'SELECT data_type FROM information_schema.columns '
                'WHERE table_name = :table AND column_name = :column'
            ),
            {'table': 'images', 'column': 'excluded_from_consensus'},
        ).scalar()
    assert found == 'jsonb'


def test_a_nanosecond_mtime_round_trips(postgres_url: str) -> None:
    """A 32-bit integer column would refuse this value outright.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    mtime_ns = 1755000000123456789
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row(mtime_ns=mtime_ns))
        with engine.connect() as connection:
            stored = connection.execute(sqlalchemy.select(IMAGES.c.mtime_ns)).scalar()
    assert stored == mtime_ns
