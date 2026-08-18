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

import contextlib
from collections.abc import Iterator

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

from spindoctor.cli.ck.inputs import read_whole_mission
from spindoctor.nav_records import NavRecord, RecordSource, Selection, UnreadableFile
from spindoctor.results_index import (
    FAILED_FILES,
    FEATURE_SOURCES,
    IMAGES,
    INGEST_RUNS,
    SCHEMA_META,
    SCHEMA_VERSION,
    TECHNIQUES,
    open_index,
    open_record_source,
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
    with pytest.raises(ValueError, match='empty the database with sd_stats_ingest --drop-index'):
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

OTHER_ROOT_REFUSED_STUB = 'COISS_2001/data/1294561143_1295221348/N1294561205_1_CALIB'
"""A file the other root refused, which the root under test holds nothing for.

The refusals are keyed by root and stub together exactly as the images are, and
this stub is the one that says so: it belongs to no image row anywhere, so a
refusals arm reading the stub without its root shows it up as a document of the
root under test.
"""

NAVIGATED_STUB = 'COISS_2001/data/1294561143_1295221348/N1294561204_1_CALIB'
"""A document recording an outcome that is not a fatal error."""

SELECTION_SUBTREE = 'COISS_2001'
"""The subtree the selection reads."""

SELECTION_INGESTED = '2026-08-08T00:00:00+00:00'
"""When the pass over the root under test finished."""

SELECTION_OTHER_INGESTED = '2026-08-09T00:00:00+00:00'
"""When the pass over the other root finished, which is later and is not this one."""


def _seed_selection_rows(url: str) -> None:
    """Create the index and write the rows the selection filters read.

    The row under test records a fatal error and no ``status_error`` at all,
    which is the value SQL comparison handles differently from every other; the
    other root's row for the same stub records the SPICE error the filters tell
    apart, so a query that dropped the root would answer with it.  A second row
    of the root under test records a run that finished, so that the filter for
    a document recording no fatal error has something to select and is not
    satisfied by answering nothing.

    Both roots refuse a file, and the other root's refusal names a stub no
    image row anywhere carries, so the refusals arm is held to its root by the
    same evidence the images arm is: read without its root it adds a stub to
    what the root under test holds.

    The two roots' run rows differ the same way.  The other root is passed over
    second, so its run is the newest in the index and its finish time is not
    this root's: what the pass over this root recorded about itself is
    therefore visibly its own.

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
                    subtree=SELECTION_SUBTREE,
                    status='error',
                    status_error=None,
                ),
                image_row(
                    root_url=SELECTION_ROOT,
                    results_path_stub=NAVIGATED_STUB,
                    subtree=SELECTION_SUBTREE,
                    status='failure',
                    status_error=None,
                ),
                image_row(
                    root_url=SELECTION_OTHER_ROOT,
                    results_path_stub=STUB,
                    subtree=SELECTION_SUBTREE,
                    status='error',
                    status_error='missing_spice_data',
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
                    'subtree': SELECTION_SUBTREE,
                    'mtime_ns': 1,
                    'size_bytes': 2,
                },
                {
                    'root_url': SELECTION_OTHER_ROOT,
                    'results_path_stub': OTHER_ROOT_REFUSED_STUB,
                    'reason': 'not a current-schema navigation document',
                    'subtree': SELECTION_SUBTREE,
                    'mtime_ns': 3,
                    'size_bytes': 4,
                },
            ],
        )
        connection.execute(
            INGEST_RUNS.insert(),
            [
                {
                    'root_url': root_url,
                    'started_utc': stamp,
                    'finished_utc': stamp,
                    'schema_version': SCHEMA_VERSION,
                }
                for root_url, stamp in (
                    (SELECTION_ROOT, SELECTION_INGESTED),
                    (SELECTION_OTHER_ROOT, SELECTION_OTHER_INGESTED),
                )
            ],
        )


def test_the_selection_reads_a_document_and_a_refusal_on_postgresql(postgres_url: str) -> None:
    """The two tables are read as one, on the backend that types their columns.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    stubs = read_result_stubs(postgres_url, SELECTION_ROOT, [SELECTION_SUBTREE])
    assert stubs.with_metadata == frozenset({STUB, NAVIGATED_STUB, REFUSED_STUB})


def test_another_roots_refusal_is_not_this_roots_document_on_postgresql(
    postgres_url: str,
) -> None:
    """The refusals arm is keyed by root and stub together, as the images arm is.

    The other root's refusal names a file the root under test holds nothing
    for, so an arm reading the stub alone hands the enumeration a stub whose
    document is under a different root -- and the enumeration would then read
    it as an image of this root that has already been navigated.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    stubs = read_result_stubs(postgres_url, SELECTION_ROOT, [SELECTION_SUBTREE])
    assert OTHER_ROOT_REFUSED_STUB not in stubs.with_metadata


def test_the_error_flag_survives_the_union_on_postgresql(postgres_url: str) -> None:
    """The two arms' second column is one column of one type in the union.

    PostgreSQL refuses a union whose columns disagree; SQLite reconciles them.
    The image arm computes a predicate over the row and the refusal arm writes
    a literal false, so the union is exercised from both sides, and the answer
    is what says the predicate reached the row rather than the literal.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    stubs = read_result_stubs(
        postgres_url, SELECTION_ROOT, [SELECTION_SUBTREE], has_offset_error=True
    )
    assert stubs.matching_error == frozenset({STUB})


def test_the_negative_error_filter_selects_on_postgresql(postgres_url: str) -> None:
    """The inequality is a predicate of the same type as the equality beside it.

    The image arm's second column is computed either way, so this is what says
    a filter phrased in the negative unions with the refusal arm's literal on
    the backend that types both -- and that the refusal, which records no
    status at all, is on the same side of the answer as the fatal error.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    stubs = read_result_stubs(
        postgres_url, SELECTION_ROOT, [SELECTION_SUBTREE], has_no_offset_error=True
    )
    assert stubs.matching_error == frozenset({NAVIGATED_STUB})


def test_a_fatal_error_with_no_cause_is_not_a_spice_error_on_postgresql(
    postgres_url: str,
) -> None:
    """NULL is not equal to anything and not unequal to anything either.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    stubs = read_result_stubs(
        postgres_url, SELECTION_ROOT, [SELECTION_SUBTREE], has_offset_nonspice_error=True
    )
    assert stubs.matching_error == frozenset({STUB})


def test_the_error_filter_answers_for_one_root_on_postgresql(postgres_url: str) -> None:
    """The other root's row for this stub is a SPICE error, and is not read.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    stubs = read_result_stubs(
        postgres_url, SELECTION_ROOT, [SELECTION_SUBTREE], has_offset_spice_error=True
    )
    assert stubs.matching_error == frozenset()


def test_the_snapshot_time_answers_for_one_root_on_postgresql(postgres_url: str) -> None:
    """How old this answer is, on the backend that returns the stamp as it typed it.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    stubs = read_result_stubs(postgres_url, SELECTION_ROOT, [SELECTION_SUBTREE])
    assert stubs.ingested_utc == SELECTION_INGESTED


def _seeded_without_the_refusals(url: str) -> None:
    """Seed the selection rows and then take the refusals table away.

    An index whose account was granted the rows it reports on and not the
    bookkeeping beside them is the case the refusal names, and dropping the
    table is how that account's view of it is reproduced.

    Parameters:
        url: The index to create, write into, and take a table from.
    """
    _seed_selection_rows(url)
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(sqlalchemy.text(f'DROP TABLE {FAILED_FILES.name}'))


def test_a_table_this_account_cannot_read_is_reported_on_postgresql(postgres_url: str) -> None:
    """A missing relation is a different exception class here, and is still translated.

    An index whose account was granted the rows it reports on and not the
    bookkeeping beside them is the case the refusal names, and it is a
    PostgreSQL case: the class SQLite raises for the same query is another one,
    so a seam that caught only what SQLite raises would let this one out.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seeded_without_the_refusals(postgres_url)
    with pytest.raises(ValueError, match='could not be read'):
        read_result_stubs(postgres_url, SELECTION_ROOT, [SELECTION_SUBTREE])


def test_a_failing_query_raises_no_database_exception_on_postgresql(postgres_url: str) -> None:
    """The consumer that never imports the database layer cannot name its types.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seeded_without_the_refusals(postgres_url)
    with pytest.raises(ValueError) as excinfo:
        read_result_stubs(postgres_url, SELECTION_ROOT, [SELECTION_SUBTREE])
    assert not isinstance(excinfo.value, sqlalchemy.exc.SQLAlchemyError)


def test_a_failing_query_is_reported_without_its_sql_on_postgresql(postgres_url: str) -> None:
    """The advice is what an operator reads, and a statement dump buries it.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seeded_without_the_refusals(postgres_url)
    with pytest.raises(ValueError) as excinfo:
        read_result_stubs(postgres_url, SELECTION_ROOT, [SELECTION_SUBTREE])
    assert 'SELECT' not in str(excinfo.value)


# ---------------------------------------------------------------------------
# The record source
# ---------------------------------------------------------------------------

RECORD_ROOT = '/data/record-nav-results'
"""The results root the record source below is opened over."""

RECORD_OTHER_ROOT = '/data/record-other-results'
"""A second ingested root, holding the same stubs with other values."""

RECORD_SUBTREE = 'COISS_2001'
"""The subtree the first two images live under."""

RECORD_OTHER_SUBTREE = 'COISS_2002'
"""The subtree the third lives under, so a subtree restriction has work to do."""

FIRST_RECORD_STUB = f'{RECORD_SUBTREE}/data/N1294561202_1_CALIB'
"""The image every per-image test below reads."""

SECOND_RECORD_STUB = f'{RECORD_SUBTREE}/data/N1294561203_1_CALIB'
"""A second image of the same subtree, exposed later."""

THIRD_RECORD_STUB = f'{RECORD_OTHER_SUBTREE}/data/N1294561204_1_CALIB'
"""An image of the other subtree."""

RECORD_REFUSED_STUB = f'{RECORD_SUBTREE}/data/junk'
"""A file this root's ingest refused, which is still a file that exists."""

ONLY_OTHER_ROOT_STUB = f'{RECORD_SUBTREE}/data/N1294561299_1_CALIB'
"""An image only the other root holds, which this root must never answer for."""

OTHER_REFUSED_STUB = f'{RECORD_OTHER_SUBTREE}/data/other_junk'
"""A file the other root's ingest refused, somewhere else and under another name."""

RECORD_OFFSET = (1.5, -2.5)
"""What this root's images record."""

OTHER_RECORD_OFFSET = (9.5, -8.5)
"""What the other root's rows for the same stubs record."""

FIRST_MIDTIME_ET = 100.0
"""The exposure midtime the first image records."""

SECOND_MIDTIME_ET = 300.0
"""The exposure midtime the second records, well after the first."""

RECORD_COLUMNS = (IMAGES.c.status, IMAGES.c.instrument, IMAGES.c.offset_dv, IMAGES.c.offset_du)
"""A consumer's columns, standing in for any consumer's."""


def _record_rows(root_url: str, offset: tuple[float, float]) -> list[dict[str, object]]:
    """Return one root's image rows, written in reverse of their path order.

    Reversed on purpose.  No statement the source issues asks for an order, so
    what a server hands back is its own; on a freshly written table that is the
    order the rows went in, and a run that needs path order has to impose it
    rather than inherit it.

    Parameters:
        root_url: The root these rows belong to.
        offset: The offset each of them records, which is what tells one root's
            rows from the other's.

    Returns:
        The rows, ready to insert.
    """
    return [
        image_row(
            root_url=root_url,
            results_path_stub=stub,
            subtree=stub.split('/')[0],
            instrument='coiss',
            offset_dv=offset[0],
            offset_du=offset[1],
            midtime_et=midtime,
            source_file=f'{root_url}/{stub}_metadata.json',
            mtime_ns=1_700_000_000_000_000_000,
            size_bytes=512,
        )
        for stub, midtime in (
            (THIRD_RECORD_STUB, SECOND_MIDTIME_ET),
            (SECOND_RECORD_STUB, SECOND_MIDTIME_ET),
            (FIRST_RECORD_STUB, FIRST_MIDTIME_ET),
        )
    ]


def _seed_record_rows(url: str) -> None:
    """Create the index and write the rows the record source reads.

    Two roots, holding the same three stubs with different offsets, each with a
    refused file of its own at a stub the other root has no file at.  A query
    that dropped the root half of the key would therefore answer with the wrong
    offsets, or fail an image this root has nothing to say about.

    Parameters:
        url: The index to create and write into.
    """
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(
            IMAGES.insert(),
            [
                *_record_rows(RECORD_ROOT, RECORD_OFFSET),
                *_record_rows(RECORD_OTHER_ROOT, OTHER_RECORD_OFFSET),
                # A stub only the other root holds, so a query that bound a key
                # and dropped the root is caught whichever order the server
                # returned its rows in.
                image_row(
                    root_url=RECORD_OTHER_ROOT,
                    results_path_stub=ONLY_OTHER_ROOT_STUB,
                    subtree=RECORD_SUBTREE,
                    instrument='coiss',
                    offset_dv=OTHER_RECORD_OFFSET[0],
                    offset_du=OTHER_RECORD_OFFSET[1],
                    midtime_et=SECOND_MIDTIME_ET,
                    source_file=f'{RECORD_OTHER_ROOT}/{ONLY_OTHER_ROOT_STUB}_metadata.json',
                    mtime_ns=1_700_000_000_000_000_000,
                    size_bytes=512,
                ),
            ],
        )
        connection.execute(
            FAILED_FILES.insert(),
            [
                {
                    'root_url': root_url,
                    'results_path_stub': stub,
                    'reason': 'not a current-schema navigation document',
                    'subtree': stub.split('/')[0],
                    'mtime_ns': 1_700_000_000_000_000_000,
                    'size_bytes': 64,
                }
                for root_url, stub in (
                    (RECORD_ROOT, RECORD_REFUSED_STUB),
                    (RECORD_OTHER_ROOT, OTHER_REFUSED_STUB),
                )
            ],
        )
        connection.execute(
            INGEST_RUNS.insert(),
            [
                {
                    'root_url': root_url,
                    'started_utc': SELECTION_INGESTED,
                    'finished_utc': SELECTION_INGESTED,
                    'schema_version': SCHEMA_VERSION,
                }
                for root_url in (RECORD_ROOT, RECORD_OTHER_ROOT)
            ],
        )


@contextlib.contextmanager
def _record_source(url: str, *roots: str) -> Iterator[RecordSource]:
    """Open a record source over some of the seeded roots, and close it after.

    Parameters:
        url: The index URL.
        roots: The roots the source is to hold.

    Yields:
        The source.
    """
    with open_record_source(list(roots), results_db_url=url, columns=RECORD_COLUMNS) as source:
        yield source


def test_the_listing_unions_both_tables_on_postgresql(postgres_url: str) -> None:
    """A server types the columns of a union, and one arm records no path at all.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = sorted(entry.stub for entry in source.listing(Selection()))
    assert found == sorted(
        [FIRST_RECORD_STUB, SECOND_RECORD_STUB, THIRD_RECORD_STUB, RECORD_REFUSED_STUB]
    )


def test_the_listing_answers_for_one_root_on_postgresql(postgres_url: str) -> None:
    """The other root holds the same stubs and a refusal of its own.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = [entry.path.as_posix() for entry in source.listing(Selection())]
    assert [path for path in found if not path.startswith(RECORD_ROOT)] == []


def test_the_listing_narrows_to_a_subtree_on_postgresql(postgres_url: str) -> None:
    """Both arms of the union carry the restriction.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = sorted(entry.stub for entry in source.listing(Selection(subtrees=('COISS_2002',))))
    assert found == [THIRD_RECORD_STUB]


def test_the_stream_reads_one_roots_values_on_postgresql(postgres_url: str) -> None:
    """The other root records another offset for every one of these stubs.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = {
            tuple(entry.metadata['offset'])
            for entry in source.records(Selection(instrument='coiss'))
            if isinstance(entry, NavRecord)
        }
    assert found == {RECORD_OFFSET}


def test_the_stream_bounds_time_on_postgresql(postgres_url: str) -> None:
    """A double comparison, against the epoch the document recorded.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = sorted(
            entry.stub
            for entry in source.records(Selection(stop_et=FIRST_MIDTIME_ET))
            if isinstance(entry, NavRecord)
        )
    assert found == [FIRST_RECORD_STUB]


def test_the_stream_reports_a_refusal_on_postgresql(postgres_url: str) -> None:
    """The refused file of this root, and not the one the other root refused.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = [
            entry.stub for entry in source.records(Selection()) if isinstance(entry, UnreadableFile)
        ]
    assert found == [RECORD_REFUSED_STUB]


def test_named_stubs_come_back_in_the_order_named_on_postgresql(postgres_url: str) -> None:
    """Bound as a list of keys, and put back into the order a caller named them.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    named = (SECOND_RECORD_STUB, FIRST_RECORD_STUB, THIRD_RECORD_STUB)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = [entry.stub for entry in source.records(Selection(stubs=named))]
    assert found == list(named)


def test_a_named_stub_reads_this_roots_row_on_postgresql(postgres_url: str) -> None:
    """The other root holds the same key, recording another offset.

    Read against the root whose rows are written *first*.  The batch read builds
    what it found with a dictionary update, so a query that dropped the root
    half of the key would be answered by whichever row came back last, and
    asking for the root that was written last would be answered correctly by the
    defect.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = [
            tuple(entry.metadata['offset'])
            for entry in source.records(Selection(stubs=(FIRST_RECORD_STUB,)))
            if isinstance(entry, NavRecord)
        ]
    assert found == [RECORD_OFFSET]


def test_one_image_is_read_for_one_root_on_postgresql(postgres_url: str) -> None:
    """The per-image lookup joins both tables onto a key that carries the root.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_OTHER_ROOT) as source:
        found = source.record(FIRST_RECORD_STUB)
    assert tuple(found.metadata['offset']) == OTHER_RECORD_OFFSET


def test_the_other_roots_refusal_is_not_this_ones_on_postgresql(postgres_url: str) -> None:
    """A refusal lookup blind to the root would fail an image this root has no file for.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with (
        _record_source(postgres_url, RECORD_ROOT) as source,
        pytest.raises(FileNotFoundError, match=OTHER_REFUSED_STUB),
    ):
        source.record(OTHER_REFUSED_STUB)


def test_a_run_puts_the_records_in_path_order_on_postgresql(postgres_url: str) -> None:
    """No statement sorts, so the order a run works in is the one it imposes itself.

    A server sorts text under its own collation, which is why nothing here asks
    it to; the rows are written in reverse of their path order, so a run that
    took the order it was handed would be caught.  Asserted through the function
    the kernel writer collects its mission with, because the ordering is that
    run's guarantee rather than the source's.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        records, _unreadable = read_whole_mission(source.records(Selection(instrument='coiss')))
    paths = [record.path.as_posix() for record in records]
    assert paths == sorted(paths)


def test_a_stream_over_a_table_this_account_cannot_read_is_reported_on_postgresql(
    postgres_url: str,
) -> None:
    """A missing relation is a different exception class here, and is still translated.

    The refusal reaches the caller out of the stream rather than out of the
    call that built it, which is where a failure of a lazily executed query
    arrives.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with opened(postgres_url) as engine, engine.begin() as connection:
        connection.execute(sqlalchemy.text(f'DROP TABLE {FAILED_FILES.name}'))
    with (
        _record_source(postgres_url, RECORD_ROOT) as source,
        pytest.raises(ValueError, match='could not be read'),
    ):
        list(source.records(Selection()))


def test_a_named_stub_only_the_other_root_holds_yields_nothing_on_postgresql(
    postgres_url: str,
) -> None:
    """Naming a key does not stop it being a key under one root.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = list(source.records(Selection(stubs=(ONLY_OTHER_ROOT_STUB,))))
    assert found == []


def test_a_named_stub_only_the_other_root_refused_yields_nothing_on_postgresql(
    postgres_url: str,
) -> None:
    """The refusal half of a named-stub read carries its own root term.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = list(source.records(Selection(stubs=(OTHER_REFUSED_STUB,))))
    assert found == []
