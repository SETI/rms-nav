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
repeated run, or two workers of a parallel run, never share a table.  A test
that reads the server's catalog takes ``postgres_schema`` as well and filters on
it: the catalog spans every schema on the server, so a lookup by table name
alone answers from whichever schema happens to hold a table of that name.
"""

import psycopg
import pytest
import sqlalchemy
from tests.spindoctor.results_index.conftest import (
    STUB,
    feature_source_row,
    image_row,
    opened,
    technique_row,
)

from spindoctor.results_index import (
    FAILED_FILES,
    FEATURE_SOURCES,
    IMAGES,
    INGEST_RUNS,
    SCHEMA_META,
    SCHEMA_VERSION,
    TECHNIQUES,
    open_index,
)
from spindoctor.results_index.selection import read_result_stubs

pytestmark = pytest.mark.postgres

OTHER_STUB = 'COISS_2002/data/1295221349_1296000000/N1294561202_1_CALIB'

FIFTEEN_DIGIT_OFFSET = -1234.56789012345

BOGUS_PASSWORD = 'sup3rs3cr3t'
"""A password distinctive enough that finding it anywhere is proof of a leak."""

UNDEFINED_FUNCTION = '42883'
"""SQLSTATE for an operator the server has no definition of.

Stable across every server locale, which the message text is not.
"""


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


def _refusal_of(url: str, message: str) -> str:
    """Open a URL, require the refusal it raises, and return that message.

    Parameters:
        url: The URL to open.
        message: Pattern the refusal message must match.

    Returns:
        The refusal message.
    """
    with pytest.raises(ValueError, match=message) as excinfo:
        open_index(url)
    return str(excinfo.value)


def _extra_password_appearances(message: str, url: str) -> list[str]:
    """Return the appearances of a URL's password a refusal does not account for.

    The bare password is searched for, rather than the ``:password@`` form alone,
    because a leak is a leak whatever shape it arrives in.  It is counted rather
    than merely looked for, because the password a server is configured with is
    also, on an ordinary local server, the role name and the database name, and
    those the message names legitimately: what proves masking is that the
    password appears no more often than the password-hiding rendering of the
    same URL accounts for.

    Parameters:
        message: The refusal message to read.
        url: The URL the refusal names.

    Returns:
        One entry per unaccounted appearance, empty when nothing leaked.
    """
    password = _password_of(url)
    masked = sqlalchemy.engine.make_url(url).render_as_string()
    extra = message.count(password) - masked.count(password)
    return [password] * max(extra, 0)


def _skip_without_a_password(url: str) -> None:
    """Skip the calling test when the configured URL carries no password.

    Parameters:
        url: The URL the test drives.
    """
    if not _password_of(url):
        pytest.skip('the configured server URL carries no password to mask')


def test_creating_the_schema_creates_every_table(postgres_url: str, postgres_schema: str) -> None:
    """The metadata emits DDL a server accepts, not just DDL SQLite accepts.

    The listing is scoped to this test's own schema rather than to whatever the
    connection's search path resolves to, so a table another worker left in
    ``public`` cannot answer for one this test was supposed to create.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_schema: Name of that schema.
    """
    with opened(postgres_url, create=True) as engine:
        found = sorted(sqlalchemy.inspect(engine).get_table_names(schema=postgres_schema))
    assert found == [
        'failed_files',
        'feature_sources',
        'images',
        'ingest_runs',
        'schema_meta',
        'techniques',
    ]


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
    _skip_without_a_password(postgres_url)
    message = _refusal_of(postgres_url, 'not a results index')
    assert _extra_password_appearances(message, postgres_url) == []


def test_the_version_refusal_does_not_repeat_the_password(postgres_url: str) -> None:
    """Every route names the URL, so every route has to mask it.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _skip_without_a_password(postgres_url)
    with opened(postgres_url, create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    message = _refusal_of(postgres_url, 'is not the version')
    assert _extra_password_appearances(message, postgres_url) == []


def test_a_masked_refusal_still_names_the_server(postgres_url: str) -> None:
    """Masking must not cost the identification the message exists for.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    message = _refusal_of(postgres_url, 'not a results index')
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

    The SQLSTATE is what is asserted, not the message: a server translates its
    messages according to ``lc_messages``, so the word "boolean" is absent from
    the text on a server configured for another language, and the test would then
    fail for a reason that has nothing to do with the type discipline.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:  # noqa: SIM117
        with pytest.raises(sqlalchemy.exc.ProgrammingError) as excinfo:
            with engine.connect() as connection:
                connection.execute(sqlalchemy.text('SELECT * FROM techniques WHERE spurious = 0'))
    original = excinfo.value.orig
    # SQLAlchemy types the wrapped exception as any exception at all, so the
    # driver's own class is stated before its result code is read.
    assert isinstance(original, psycopg.Error)
    # 42883 is "undefined function", which is how the server reports that no
    # boolean-to-integer equality operator exists.
    assert original.sqlstate == UNDEFINED_FUNCTION


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


def test_a_second_row_for_one_technique_of_one_image_is_refused(postgres_url: str) -> None:
    """The uniqueness is a server constraint here, not a SQLite convention.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(TECHNIQUES.insert(), technique_row())
        with pytest.raises(sqlalchemy.exc.IntegrityError, match='uq_techniques_image_technique'):  # noqa: SIM117
            with engine.begin() as connection:
                connection.execute(TECHNIQUES.insert(), technique_row(confidence=0.5))


def test_a_second_row_for_one_feature_source_of_one_image_is_refused(postgres_url: str) -> None:
    """And the wider tuple the feature inventory is aggregated by.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(FEATURE_SOURCES.insert(), feature_source_row())
        with pytest.raises(sqlalchemy.exc.IntegrityError, match='uq_feature_sources_image_source'):  # noqa: SIM117
            with engine.begin() as connection:
                connection.execute(FEATURE_SOURCES.insert(), feature_source_row(n_features=7))


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


def test_a_json_column_is_jsonb(postgres_url: str, postgres_schema: str) -> None:
    """A plain ``json`` column would reject ``jsonb_array_elements_text``.

    The catalog is server-wide, so the lookup is scoped to this test's own
    schema: filtered by table and column alone it would read whichever
    ``images`` row the catalog returned first, which under parallel workers is
    somebody else's.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_schema: Name of that schema.
    """
    with opened(postgres_url, create=True) as engine, engine.connect() as connection:
        found = connection.execute(
            sqlalchemy.text(
                'SELECT data_type FROM information_schema.columns '
                'WHERE table_schema = :schema AND table_name = :table '
                'AND column_name = :column'
            ),
            {'schema': postgres_schema, 'table': 'images', 'column': 'excluded_from_consensus'},
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


SELECTION_ROOT = '/data/nav-results'
"""The results root the selection filters are asked about."""

SELECTION_OTHER_ROOT = '/data/other-nav-results'
"""A second ingested root, holding a row for the same stub."""

REFUSED_STUB = 'COISS_2001/data/1294561143_1295221348/N1294561203_1_CALIB'
"""A file the ingest refused, which is still a file that exists."""

SELECTION_VOLUME = 'COISS_2001'
"""The volume the selection reads."""

SELECTION_INGESTED = '2026-08-08T00:00:00+00:00'
"""When the pass over the root under test finished."""

SELECTION_OTHER_INGESTED = '2026-08-09T00:00:00+00:00'
"""When the pass over the other root finished, which is later and is not this one."""


def _seed_selection_rows(url: str) -> None:
    """Create the index and write the rows the selection filters read.

    The row under test records a fatal error and no ``status_error`` at all,
    which is the value SQL comparison handles differently from every other; the
    other root's row for the same stub records the SPICE error the filters tell
    apart, so a query that dropped the root would answer with it.

    The two roots' run rows differ the same way.  The other root is passed over
    second, so its run is the newest in the index, and it is the only one that
    records a missed directory: what the pass over this root recorded about
    itself is therefore visibly its own.

    Parameters:
        url: The index to create and write into.
    """
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(
            IMAGES.insert(),
            [
                image_row(
                    root_url=SELECTION_ROOT,
                    results_path_stub=STUB,
                    volume=SELECTION_VOLUME,
                    status='error',
                    status_error=None,
                    has_summary_png=True,
                ),
                image_row(
                    root_url=SELECTION_OTHER_ROOT,
                    results_path_stub=STUB,
                    volume=SELECTION_VOLUME,
                    status='error',
                    status_error='missing_spice_data',
                    has_summary_png=True,
                ),
            ],
        )
        connection.execute(
            FAILED_FILES.insert(),
            [
                {
                    'root_url': SELECTION_ROOT,
                    'results_path_stub': REFUSED_STUB,
                    'reason': 'not a current-schema navigation document',
                    'volume': SELECTION_VOLUME,
                    'has_summary_png': True,
                    'mtime_ns': 1,
                    'size_bytes': 2,
                }
            ],
        )
        connection.execute(
            INGEST_RUNS.insert(),
            [
                {
                    'root_url': root_url,
                    'started_utc': stamp,
                    'finished_utc': stamp,
                    'directories_missed': missed,
                    'schema_version': SCHEMA_VERSION,
                }
                for root_url, stamp, missed in (
                    (SELECTION_ROOT, SELECTION_INGESTED, None),
                    (SELECTION_OTHER_ROOT, SELECTION_OTHER_INGESTED, 4),
                )
            ],
        )


def test_the_selection_reads_a_document_and_a_refusal_on_postgresql(postgres_url: str) -> None:
    """The two tables are read as one, on the backend that types their columns.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    stubs = read_result_stubs(postgres_url, SELECTION_ROOT, [SELECTION_VOLUME])
    assert stubs.with_metadata == frozenset({STUB, REFUSED_STUB})


def test_the_summary_flag_survives_the_union_on_postgresql(postgres_url: str) -> None:
    """A boolean column of each table is one column of one type in the union.

    PostgreSQL refuses a union whose columns disagree, and refuses an integer
    where a boolean belongs; SQLite accepts both.  Both rows carry a summary,
    one on an image and one on a refusal, so the union is exercised from both
    sides.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    stubs = read_result_stubs(postgres_url, SELECTION_ROOT, [SELECTION_VOLUME])
    assert stubs.with_summary_png == frozenset({STUB, REFUSED_STUB})


def test_a_fatal_error_with_no_cause_is_not_a_spice_error_on_postgresql(
    postgres_url: str,
) -> None:
    """NULL is not equal to anything and not unequal to anything either.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    stubs = read_result_stubs(
        postgres_url, SELECTION_ROOT, [SELECTION_VOLUME], has_offset_nonspice_error=True
    )
    assert stubs.matching_error == frozenset({STUB})


def test_the_error_filter_answers_for_one_root_on_postgresql(postgres_url: str) -> None:
    """The other root's row for this stub is a SPICE error, and is not read.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    stubs = read_result_stubs(
        postgres_url, SELECTION_ROOT, [SELECTION_VOLUME], has_offset_spice_error=True
    )
    assert stubs.matching_error == frozenset()


def test_the_missed_count_answers_for_one_root_on_postgresql(postgres_url: str) -> None:
    """The run table is keyed by root as well, and the newest run in it is another's.

    This root's run records no count at all, which on a strictly typed backend
    is a NULL integer rather than a zero, and the other root's -- the newer of
    the two -- records four.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    stubs = read_result_stubs(postgres_url, SELECTION_ROOT, [SELECTION_VOLUME])
    assert stubs.directories_missed == 0


def test_the_snapshot_time_answers_for_one_root_on_postgresql(postgres_url: str) -> None:
    """How old this answer is, on the backend that returns the stamp as it typed it.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    stubs = read_result_stubs(postgres_url, SELECTION_ROOT, [SELECTION_VOLUME])
    assert stubs.ingested_utc == SELECTION_INGESTED
