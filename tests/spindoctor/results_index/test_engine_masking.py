"""Tests that a results-index refusal names its URL without its password.

These messages are written to run logs and pasted into bug reports, so a
database password may not survive into one. Everything else about the URL has
to, because naming the URL is what tells a reader which of the three resolution
levels supplied the bad value.

Two routes produce the name. A URL SQLAlchemy can parse renders itself with the
password hidden. A URL it cannot parse -- a stray space from a value copied
across two lines, a hyphen in the scheme, a missing scheme -- cannot render
itself at all, and is masked structurally instead. The structural rule is
covered here twice: directly, as a table of URLs and exactly what masking each
must produce, and again through the real opener, because the gap between the two
is where a leak survived three review rounds.
"""

import dataclasses

import pytest
import sqlalchemy
from tests.spindoctor.results_index.conftest import (
    EXPLODING_FACTORY_MESSAGE,
    exploding_factory,
    without_module,
)

from spindoctor.results_index import engine as engine_module
from spindoctor.results_index import open_index

PASSWORD = 'sup3rs3cr3t'
"""A password distinctive enough that finding it anywhere is proof of a leak."""

SLASHED_PASSWORD = 'aB3/xY9z'
"""A password carrying a slash, which a URL permits unescaped.

The route that masks structurally is the one that runs on a URL nothing could
parse, so nothing there knows where the password ends except the ``@``. A rule
that stopped at a path separator would leave this one in the message whole.
"""

AT_SIGN_USER = 'admin@pgsrv'
"""A user name carrying an at-sign, which is how a managed server names one.

``user@servername`` is the standard login form of a hosted PostgreSQL, and
SQLAlchemy's own parser accepts it. A rule that took the first at-sign as the
end of the credentials would find no password after it and leak the whole URL.
"""


@dataclasses.dataclass(frozen=True)
class _Route:
    """One way a URL is refused, and what its refusal has to say.

    Attributes:
        url: The URL to open, carrying a password.
        message: Pattern the refusal message must match.
        identifies: Text of the URL the message must keep, so a reader can tell
            which of the resolution levels supplied the value.
        cause: Exception type the refusal keeps as its ``__cause__``.
        password: The password this route's URL carries, which its refusal may
            not repeat.
        hidden_module: Module to make unimportable first, or None.
        needs_psycopg: Whether the route reaches the PostgreSQL driver itself.
        breaks_the_factory: Whether the engine factory is replaced by one that
            raises, standing for a failure inside a dialect that no enumeration
            of exception types would have caught.
    """

    url: str
    message: str
    identifies: str
    cause: type[BaseException]
    password: str = PASSWORD
    hidden_module: str | None = None
    needs_psycopg: bool = False
    breaks_the_factory: bool = False


REFUSAL_ROUTES = [
    pytest.param(
        _Route(
            url=f'postgresql+psycopg://user:{PASSWORD}@localhost:5432/spindoctor',
            message=r'rms-spindoctor\[postgres\]',
            identifies='localhost:5432',
            cause=ModuleNotFoundError,
            hidden_module='psycopg',
        ),
        id='driver-not-installed',
    ),
    pytest.param(
        _Route(
            url=f'mysql+mysqldb://user:{PASSWORD}@localhost/spindoctor',
            message='MySQLdb',
            identifies='mysql+mysqldb',
            cause=ModuleNotFoundError,
            hidden_module='MySQLdb',
        ),
        id='unsupported-backend',
    ),
    pytest.param(
        _Route(
            url=f'frobnicate://user:{PASSWORD}@localhost/spindoctor',
            message='no database driver for this URL scheme',
            identifies='frobnicate',
            cause=sqlalchemy.exc.NoSuchModuleError,
        ),
        id='unknown-scheme',
    ),
    pytest.param(
        _Route(
            url=f'postgresql+psycopg://user:{PASSWORD}@localhost:notaport/spindoctor',
            message='could not open the results index',
            identifies='localhost:notaport',
            cause=ValueError,
        ),
        id='unparseable-port',
    ),
    pytest.param(
        _Route(
            # A stray space in the scheme, which is what a URL copied across two
            # lines of a configuration file arrives as. Nothing parses it, so
            # nothing can render it either: this is the route on which the
            # password is masked structurally, and the only one on which that
            # rule is what stands between the password and the run log.
            url=f'postgresql psycopg://svc:{SLASHED_PASSWORD}@db.example/spindoctor',
            message='could not open the results index',
            identifies='db.example',
            cause=sqlalchemy.exc.ArgumentError,
            password=SLASHED_PASSWORD,
        ),
        id='unparseable-url',
    ),
    pytest.param(
        _Route(
            # The managed-server login form, whose at-sign in the user name puts
            # a second at-sign in the URL, together with a port nothing can
            # parse. Both routes have to reach the password that lies between
            # them.
            url=f'postgresql+psycopg://{AT_SIGN_USER}:{PASSWORD}@host:notaport/spindoctor',
            message='could not open the results index',
            identifies='host:notaport',
            cause=ValueError,
        ),
        id='an-at-sign-in-the-user-name',
    ),
    pytest.param(
        _Route(
            # The same login form on a URL that does not parse at all, which is
            # the route with no rendering to fall back on.
            url=f'postgresql psycopg://{AT_SIGN_USER}:{PASSWORD}@host:5432/spindoctor',
            message='could not open the results index',
            identifies='host:5432',
            cause=sqlalchemy.exc.ArgumentError,
        ),
        id='an-at-sign-in-the-user-name-of-an-unparseable-url',
    ),
    pytest.param(
        _Route(
            url=f'postgresql+psycopg://spindoctor:{PASSWORD}@127.0.0.1:1/spindoctor',
            message='could not open the results index',
            identifies='127.0.0.1:1',
            cause=sqlalchemy.exc.OperationalError,
            needs_psycopg=True,
        ),
        id='server-refuses-the-connection',
    ),
    pytest.param(
        _Route(
            url=f'postgresql+psycopg://user:{PASSWORD}@db.example:5432/spindoctor',
            message=EXPLODING_FACTORY_MESSAGE,
            identifies='db.example:5432',
            cause=RuntimeError,
            breaks_the_factory=True,
        ),
        id='failure-inside-the-engine-factory',
    ),
]
"""Every route by which a URL carrying a password reaches a refusal."""


def _refusal_of(route: _Route, monkeypatch: pytest.MonkeyPatch) -> ValueError:
    """Open a route's URL and return the refusal it raised.

    Parameters:
        route: The route to drive.
        monkeypatch: Fixture the import hook is installed through.

    Returns:
        The refusal, for assertions on what it says.
    """
    if route.needs_psycopg:
        pytest.importorskip('psycopg')
    if route.hidden_module is not None:
        without_module(monkeypatch, route.hidden_module)
    if route.breaks_the_factory:
        monkeypatch.setattr(sqlalchemy, 'create_engine', exploding_factory)
    with pytest.raises(ValueError, match=route.message) as excinfo:
        open_index(route.url)
    return excinfo.value


@pytest.mark.parametrize('route', REFUSAL_ROUTES)
def test_every_refusal_route_masks_its_password_and_keeps_everything_else(
    route: _Route, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three things are true of every refusal, and one open shows all three.

    Translating the type must not throw away what the driver actually said, so
    the driver's exception stays the cause. The password reaches neither a run
    log nor an operator. And masking must not cost the identification the
    message exists for: a program resolves its URL from a command line, a
    configuration file or the environment, and the URL is what says which of
    them supplied this one.

    Parameters:
        route: The refusal route under test.
        monkeypatch: Fixture the import hook is installed through.
    """
    refusal = _refusal_of(route, monkeypatch)
    assert isinstance(refusal.__cause__, route.cause)
    assert route.password not in str(refusal)
    assert route.identifies in str(refusal)


@dataclasses.dataclass(frozen=True)
class _MaskingCase:
    """One URL the structural rule is asked about, and what it must return.

    Attributes:
        name: Identifier the case is reported under.
        url: The URL as a caller wrote it.
        expected: Exactly what masking it must produce, which states both
            directions at once -- that no password survives, and that nothing
            which is not a password was touched.
        secret: Text that is a password in this URL and must therefore be gone
            from the result, or None when the URL carries no password at all.
    """

    name: str
    url: str
    expected: str
    secret: str | None = None


MASKING_CASES = [
    _MaskingCase('a-password-carrying-a-slash', '//user:pa/ss@h', '//user:***@h', 'pa/ss'),
    _MaskingCase('no-user-name', '//:pw@h', '//:***@h', 'pw'),
    _MaskingCase('one-slash-and-no-scheme', '/user:pw@h', '/user:***@h', 'pw'),
    _MaskingCase(
        'an-at-sign-in-the-user-name',
        f'//{AT_SIGN_USER}:{PASSWORD}@host:5432/db',
        f'//{AT_SIGN_USER}:***@host:5432/db',
        PASSWORD,
    ),
    _MaskingCase(
        'an-at-sign-in-the-user-name-and-an-unparseable-port',
        f'//{AT_SIGN_USER}:{PASSWORD}@host:notaport/db',
        f'//{AT_SIGN_USER}:***@host:notaport/db',
        PASSWORD,
    ),
    _MaskingCase(
        'a-leading-space',
        f' postgresql+psycopg://{AT_SIGN_USER}:{PASSWORD}@host:5432/db',
        f' postgresql+psycopg://{AT_SIGN_USER}:***@host:5432/db',
        PASSWORD,
    ),
    _MaskingCase(
        'a-hyphen-in-the-scheme',
        f'postgresql-psycopg://{AT_SIGN_USER}:{PASSWORD}@host:5432/db',
        f'postgresql-psycopg://{AT_SIGN_USER}:***@host:5432/db',
        PASSWORD,
    ),
    _MaskingCase(
        'a-url-copied-across-two-lines',
        f'postgresql psycopg://{AT_SIGN_USER}:{PASSWORD}@host:5432/db',
        f'postgresql psycopg://{AT_SIGN_USER}:***@host:5432/db',
        PASSWORD,
    ),
    _MaskingCase(
        'a-slashed-password-on-an-unparseable-url',
        f'postgresql psycopg://svc:{SLASHED_PASSWORD}@db.example/spindoctor',
        'postgresql psycopg://svc:***@db.example/spindoctor',
        SLASHED_PASSWORD,
    ),
    _MaskingCase(
        'a-slashed-password-and-an-unparseable-port',
        f'postgresql+psycopg://svc:{SLASHED_PASSWORD}@db.example:5432x/spindoctor',
        'postgresql+psycopg://svc:***@db.example:5432x/spindoctor',
        SLASHED_PASSWORD,
    ),
    _MaskingCase(
        'one-slash-after-the-scheme',
        'postgresql+psycopg:/svc:aB3xY9z@db.example/spindoctor',
        'postgresql+psycopg:/svc:***@db.example/spindoctor',
        'aB3xY9z',
    ),
    _MaskingCase(
        'a-port-and-an-at-sign-in-the-database-name',
        'postgresql psycopg://host:5432/my@db',
        'postgresql psycopg://host:5432/my@db',
    ),
    _MaskingCase(
        'a-user-name-and-no-password',
        'postgresql+psycopg://user@host/spindoctor',
        'postgresql+psycopg://user@host/spindoctor',
    ),
    _MaskingCase(
        'a-local-path-carrying-a-colon',
        'sqlite:////data/a:b/index.sqlite3',
        'sqlite:////data/a:b/index.sqlite3',
    ),
    _MaskingCase(
        'a-local-path-carrying-a-colon-and-an-at-sign',
        'sqlite:////data/a:b/i@dex.sqlite3',
        'sqlite:////data/a:b/i@dex.sqlite3',
    ),
    _MaskingCase(
        'a-local-path-carrying-a-drive-letter',
        'sqlite:///C:/data/index.sqlite3',
        'sqlite:///C:/data/index.sqlite3',
    ),
    _MaskingCase('a-scheme-and-nothing-else', 'postgresql+psycopg:', 'postgresql+psycopg:'),
    _MaskingCase('the-empty-string', '', ''),
]
"""URLs the structural rule masks, and URLs it must leave exactly as they are."""


MASKING_PARAMS = [pytest.param(case, id=case.name) for case in MASKING_CASES]

CREDENTIAL_PARAMS = [
    pytest.param(case.url, case.expected, case.secret, id=case.name)
    for case in MASKING_CASES
    if case.secret is not None
]
"""The subset carrying a password, which the opener itself is driven with.

Every one of these is a URL SQLAlchemy cannot parse, which is what puts the
structural rule on the path an operator's message actually takes.
"""


@pytest.mark.parametrize('case', MASKING_PARAMS)
def test_the_rule_masks_a_password_and_nothing_else(case: _MaskingCase) -> None:
    """The structural rule is the only defense where the URL did not parse.

    It has to reach a password whatever the password contains -- a URL permits an
    unescaped slash in one -- and whatever the user name contains, since the
    managed-server login form puts an at-sign in it. It has to leave alone a
    local path that merely happens to carry a colon and a later at-sign, and a
    server URL whose database name carries one, since mangling either costs the
    identification these messages exist for.

    Parameters:
        case: The URL under test and exactly what masking it must produce.
    """
    assert engine_module._masked_url(case.url) == case.expected


@pytest.mark.parametrize(('url', 'expected', 'secret'), CREDENTIAL_PARAMS)
def test_the_opener_names_a_url_it_could_not_parse_by_its_masked_form(
    url: str, expected: str, secret: str
) -> None:
    """The rule is only worth anything where the opener actually reaches it.

    Asserting on the helper alone leaves the opener free to name the URL by some
    other route, which is exactly how a leak survived three reviews of the
    helper. Both directions are asserted on the one refusal: the password is
    gone, and what is left is the whole URL, which is what would otherwise leave
    the reader nothing to correct.

    Parameters:
        url: The URL under test, carrying a password.
        expected: The masked form the refusal must name it by.
        secret: The password that must not survive into the refusal.
    """
    with pytest.raises(ValueError) as excinfo:
        open_index(url)
    assert secret not in str(excinfo.value)
    assert expected in str(excinfo.value)
