"""Tests that a results-index refusal names its URL without its credentials.

These messages are written to run logs and pasted into bug reports, so a
database password may not survive into one. Everything else about the URL has
to, because naming the URL is what tells a reader which of the three resolution
levels supplied the bad value.

The rule is asked about a corpus rather than a list of remembered shapes. Every
combination of scheme, of the slashes that follow it, of credentials, and of
what a password may contain is built here and asserted in both directions: the
secret is gone, and the result is exactly the URL with its credentials replaced
and nothing else touched. A list of shapes somebody thought of is what let the
first several leaks through, each in a shape nobody had thought of yet.

Each dimension is **covered rather than sampled**, and the two are not the same
thing. A corpus that carried no slash, one and two stopped one value short of
the three-slash spelling, and the rule leaked every URL written that way; a
corpus that varied one special character of a password at a time could not
reach a password carrying an at-sign *and* a slash, and the rule leaked that
one's tail. So the slash count runs from none to :data:`MAXIMUM_SLASHES`, and
the passwords carry every ordered pair of the characters that mean something to
a URL rather than one character each. Tests below assert that the dimensions
are still crossed, because a corpus quietly narrowed is a corpus that proves
less than it says.

The corpus is driven through the real opener as well, on the URLs that reach a
refusal without opening a socket, because a rule nothing calls protects nothing.
"""

import dataclasses
import itertools

import pytest
import sqlalchemy
from tests.spindoctor.results_index.conftest import (
    EXPLODING_FACTORY_MESSAGE,
    exploding_factory,
    without_module,
)

from spindoctor.results_index import masked_url, open_index

PASSWORD = 'sup3rs3cr3t'
"""A password distinctive enough that finding it anywhere is proof of a leak."""

HIDDEN = '***'
"""What a credential is replaced by, matching what a parsed URL renders itself as."""

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
            # The password as a query parameter, which authenticates exactly as
            # the authority form does. A parsed URL renders this one verbatim,
            # so the route that names a URL by its parsed rendering leaked it in
            # full.
            url=f'postgresql+psycopg://user@localhost:5432/spindoctor?password={PASSWORD}',
            message=r'rms-spindoctor\[postgres\]',
            identifies='localhost:5432',
            cause=ModuleNotFoundError,
            hidden_module='psycopg',
        ),
        id='a-password-query-parameter',
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


# ---------------------------------------------------------------------------
# The corpus: every combination of the parts a URL is built from
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Part:
    """One value of one dimension of the corpus.

    Attributes:
        name: Identifier the value is reported under.
        text: The text it contributes to the URL.
    """

    name: str
    text: str


SCHEMES = [
    _Part('no-scheme', ''),
    _Part('a-bare-scheme', 'postgresql:'),
    _Part('a-driver-scheme', 'postgresql+psycopg:'),
]
"""A scheme with a ``+driver`` suffix, one without, and none at all."""

MAXIMUM_SLASHES = 4
"""How many slashes after the scheme the corpus goes up to.

The dimension is covered from none to this many rather than sampled. Two is
the spelling a URL is defined with; one is what a hand-edited setting arrives
as; none is the spelling on which the text before the colon reads equally as a
scheme and as a user name; three is the spelling that omits the host to name a
local socket, and the one a ``sqlite:///path`` habituates. Stopping the
dimension one value short of three is what let the fifth leak through, so it is
carried two values past the last one anybody has a use for.
"""

SLASHES = [_Part(f'{count}-slashes', '/' * count) for count in range(MAXIMUM_SLASHES + 1)]
"""How many slashes follow the scheme, from none to :data:`MAXIMUM_SLASHES`."""

TAILS = [
    _Part('a-bare-host', 'db.example'),
    _Part('a-host-and-a-path', 'db.example/spindoctor'),
    _Part('a-port-and-a-path', 'db.example:5432/spindoctor'),
]
"""What follows the credentials.

The bare host is the shape with no slash anywhere after the authority begins,
so a rule that ends the credentials at a slash has none to find; the port is a
second colon after the password's own.
"""

URL_SIGNIFICANT_CHARACTERS = ('/', '@', ':', '#', '?')
"""Every character a password may carry that also means something to a URL.

A slash reads as the end of the authority, an at-sign as the end of the
credentials, a colon as the start of the password, a question mark as the start
of a query, and a hash as the start of a fragment. Each of them is what a rule
reading the URL by eye stops at too early.
"""

CHARACTER_NAMES = {
    '/': 'a-slash',
    '@': 'an-at-sign',
    ':': 'a-colon',
    '#': 'a-hash',
    '?': 'a-question-mark',
}
"""What each of those characters is called in a case identifier."""


def _carrying(*characters: str) -> str:
    """Return a password carrying the given characters, in the given order.

    Parameters:
        characters: The characters to embed, in the order they must appear.

    Returns:
        A password distinctive enough that finding any of it is a leak.
    """
    text = 'pw'
    for index, character in enumerate(characters):
        text = f'{text}{character}s{index}'
    return text


def _password_parts() -> list[_Part]:
    """Build the password shapes, covering the characters and their order.

    One password per character is not enough, and that is the whole lesson of
    this rule's history: the leak that survived a corpus varying one character
    at a time needed an at-sign *and* a slash, in that order. So every ordered
    pair is here, including a character paired with itself, because the order
    of two occurrences is what decides where a rule reading by eye stops.

    Returns:
        The parts, in a stable order.
    """
    parts = [
        _Part('plain', PASSWORD),
        _Part('digits-only', '86753090'),
        _Part('empty', ''),
        # The two shapes an operator actually types, spelled as they are typed
        # rather than as the generator spells them.
        _Part('an-at-sign-then-a-slash-as-typed', 'p@ss/word'),
        _Part('a-colon-then-an-at-sign-as-typed', 'pw:with@both'),
    ]
    parts += [
        _Part(f'carrying-{CHARACTER_NAMES[character]}', _carrying(character))
        for character in URL_SIGNIFICANT_CHARACTERS
    ]
    parts += [
        _Part(
            f'carrying-{CHARACTER_NAMES[first]}-then-{CHARACTER_NAMES[second]}',
            _carrying(first, second),
        )
        for first, second in itertools.product(URL_SIGNIFICANT_CHARACTERS, repeat=2)
    ]
    parts += [
        _Part('carrying-all-of-them', _carrying(*URL_SIGNIFICANT_CHARACTERS)),
        _Part('carrying-all-of-them-reversed', _carrying(*reversed(URL_SIGNIFICANT_CHARACTERS))),
    ]
    return parts


PASSWORDS = _password_parts()
"""Passwords with no URL character, with one each, and with every ordered pair."""


@dataclasses.dataclass(frozen=True)
class _Userinfo:
    """One way the credentials of a URL are written.

    Attributes:
        name: Identifier the value is reported under.
        text: The userinfo as written, including its trailing ``@``, or empty.
        password: The password it carries, or None when it carries none.
    """

    name: str
    text: str
    password: str | None


USERINFOS = [
    _Userinfo('no-credentials', '', None),
    _Userinfo('a-user-name-alone', 'svc@', None),
    *[_Userinfo(f'a-password-{part.name}', f'svc:{part.text}@', part.text) for part in PASSWORDS],
]
"""Credentials absent, a user name with no password, and each password shape."""


def _expected_userinfo(scheme: _Part, slashes: _Part, userinfo: _Userinfo) -> str:
    """Return what masking must leave of one URL's credentials.

    Parameters:
        scheme: The scheme the URL was built with.
        slashes: The slashes that follow it.
        userinfo: The credentials the URL was built with.

    Returns:
        The masked userinfo, including its trailing ``@``, or empty.
    """
    if not userinfo.text:
        return ''
    if scheme.text and not slashes.text:
        # With no slash to mark the end of the scheme, the scheme's colon reads
        # equally as the one that introduces a password, and it is read that
        # way: the other reading leaves the password of a URL written with no
        # scheme at all visible in full. The user name goes with it, which is
        # one word of a message about a URL no driver would have accepted.
        return f'{HIDDEN}@'
    if userinfo.password is None:
        return userinfo.text
    return f'svc:{HIDDEN}@'


@dataclasses.dataclass(frozen=True)
class _Case:
    """One URL the rule is asked about, and exactly what it must return.

    Attributes:
        name: Identifier the case is reported under.
        url: The URL as a caller wrote it.
        expected: Exactly what masking it must produce, which states both
            directions at once -- that no credential survives, and that nothing
            which is not one was touched.
        secret: Text that is a credential in this URL and must therefore be gone
            from the result, or None when the URL carries none.
    """

    name: str
    url: str
    expected: str
    secret: str | None = None


def _corpus() -> list[_Case]:
    """Build one case per combination of the parts a URL is made of.

    Returns:
        The cases, in a stable order.
    """
    cases: list[_Case] = []
    for scheme, slashes, userinfo, tail in itertools.product(SCHEMES, SLASHES, USERINFOS, TAILS):
        masked = _expected_userinfo(scheme, slashes, userinfo)
        cases.append(
            _Case(
                name=f'{scheme.name}-{slashes.name}-{userinfo.name}-{tail.name}',
                url=f'{scheme.text}{slashes.text}{userinfo.text}{tail.text}',
                expected=f'{scheme.text}{slashes.text}{masked}{tail.text}',
                secret=userinfo.password or None,
            )
        )
    return cases


CORPUS = _corpus()
"""Every combination of scheme, slashes, credentials and tail."""


QUERY_CASES = [
    _Case(
        'a-password-query-parameter',
        f'postgresql+psycopg://svc@db.example/spindoctor?password={PASSWORD}',
        'postgresql+psycopg://svc@db.example/spindoctor?password=***',
        PASSWORD,
    ),
    _Case(
        'a-password-query-parameter-in-capitals',
        f'postgresql+psycopg://svc@db.example/spindoctor?PASSWORD={PASSWORD}',
        'postgresql+psycopg://svc@db.example/spindoctor?PASSWORD=***',
        PASSWORD,
    ),
    _Case(
        'a-key-passphrase-beside-an-ordinary-setting',
        f'postgresql+psycopg://svc@db.example/spindoctor?sslpassword={PASSWORD}&connect_timeout=3',
        'postgresql+psycopg://svc@db.example/spindoctor?sslpassword=***&connect_timeout=3',
        PASSWORD,
    ),
    _Case(
        'a-client-secret',
        f'postgresql+psycopg://svc@db.example/spindoctor?client_secret={PASSWORD}',
        'postgresql+psycopg://svc@db.example/spindoctor?client_secret=***',
        PASSWORD,
    ),
    _Case(
        'an-access-token',
        f'postgresql+psycopg://svc@db.example/spindoctor?access_token={PASSWORD}',
        'postgresql+psycopg://svc@db.example/spindoctor?access_token=***',
        PASSWORD,
    ),
    _Case(
        'the-short-spellings',
        f'postgresql+psycopg://svc@db.example/spindoctor?pwd={PASSWORD}&passwd={PASSWORD}',
        'postgresql+psycopg://svc@db.example/spindoctor?pwd=***&passwd=***',
        PASSWORD,
    ),
    _Case(
        'a-password-in-the-authority-and-in-the-query',
        f'postgresql+psycopg://svc:{SLASHED_PASSWORD}@db.example/spindoctor?password={PASSWORD}',
        'postgresql+psycopg://svc:***@db.example/spindoctor?password=***',
        PASSWORD,
    ),
    _Case(
        'a-password-followed-by-a-fragment',
        f'postgresql+psycopg://svc@db.example/spindoctor?password={PASSWORD}#note',
        'postgresql+psycopg://svc@db.example/spindoctor?password=***#note',
        PASSWORD,
    ),
    _Case(
        # libpq separates parameters with ';' as readily as with '&', so both
        # separators are covered rather than the commoner one alone.
        'a-semicolon-separated-password',
        f'postgresql+psycopg://svc@db.example/spindoctor?a=1;password={PASSWORD}',
        'postgresql+psycopg://svc@db.example/spindoctor?a=1;password=***',
        PASSWORD,
    ),
    _Case(
        'a-semicolon-separated-password-before-an-ordinary-setting',
        f'postgresql+psycopg://svc@db.example/sd?password={PASSWORD};connect_timeout=3',
        'postgresql+psycopg://svc@db.example/sd?password=***;connect_timeout=3',
        PASSWORD,
    ),
    _Case(
        'both-separators-in-one-query',
        f'postgresql+psycopg://svc@db.example/sd?a=1;password={PASSWORD}&pwd={PASSWORD};b=2',
        'postgresql+psycopg://svc@db.example/sd?a=1;password=***&pwd=***;b=2',
        PASSWORD,
    ),
    _Case(
        'an-ordinary-setting',
        'postgresql+psycopg://svc@db.example/spindoctor?connect_timeout=3',
        'postgresql+psycopg://svc@db.example/spindoctor?connect_timeout=3',
    ),
    _Case(
        'ordinary-settings-separated-by-semicolons',
        'postgresql+psycopg://svc@db.example/sd?connect_timeout=3;sslmode=require',
        'postgresql+psycopg://svc@db.example/sd?connect_timeout=3;sslmode=require',
    ),
    _Case(
        'a-search-path-option',
        'postgresql+psycopg://svc@db.example/spindoctor?options=-csearch_path%3Dstats',
        'postgresql+psycopg://svc@db.example/spindoctor?options=-csearch_path%3Dstats',
    ),
    _Case(
        'a-parameter-with-no-value',
        'postgresql+psycopg://svc@db.example/spindoctor?sslmode',
        'postgresql+psycopg://svc@db.example/spindoctor?sslmode',
    ),
]
"""Credentials carried as query parameters, and settings that are not credentials."""


NEGATIVE_CASES = [
    _Case(
        'a-local-path-carrying-a-colon',
        'sqlite:////data/a:b/index.sqlite3',
        'sqlite:////data/a:b/index.sqlite3',
    ),
    _Case(
        'a-local-path-carrying-a-colon-and-an-at-sign',
        'sqlite:////data/a:b/i@dex.sqlite3',
        'sqlite:////data/a:b/i@dex.sqlite3',
    ),
    _Case(
        'a-local-path-carrying-a-space',
        'sqlite:////data/nav results/index.sqlite3',
        'sqlite:////data/nav results/index.sqlite3',
    ),
    _Case(
        'a-local-path-carrying-a-question-mark',
        'sqlite:////data/a?b/index.sqlite3',
        'sqlite:////data/a?b/index.sqlite3',
    ),
    _Case(
        'a-local-path-carrying-a-drive-letter',
        'sqlite:///C:/data/index.sqlite3',
        'sqlite:///C:/data/index.sqlite3',
    ),
    _Case(
        'a-cloud-results-root', 'gs://rms-nav/nav-offset-results', 'gs://rms-nav/nav-offset-results'
    ),
    _Case(
        'a-web-results-root',
        'https://storage.example/nav-offset-results',
        'https://storage.example/nav-offset-results',
    ),
    _Case(
        'a-web-results-root-with-a-port',
        'https://storage.example:8443/nav-offset-results',
        'https://storage.example:8443/nav-offset-results',
    ),
    _Case('a-local-results-root', '/data/nav-offset-results', '/data/nav-offset-results'),
    _Case('a-scheme-and-nothing-else', 'postgresql+psycopg:', 'postgresql+psycopg:'),
    _Case('the-empty-string', '', ''),
]
"""Strings that carry no credential, which masking must return exactly as they are."""


AMBIGUOUS_CASES = [
    _Case(
        'a-port-and-an-at-sign-in-the-database-name',
        'postgresql psycopg://host:5432/my@db',
        'postgresql psycopg://host:***@db',
    ),
    _Case(
        'a-digit-prefixed-slashed-password',
        'postgresql+psycopg:/svc:123/xY9z@db.example/spindoctor',
        'postgresql+psycopg:/svc:***@db.example/spindoctor',
        '123/xY9z',
    ),
]
"""The one shape that reads two ways, in both of its readings.

``host:5432/path@name`` is equally a host with a port and a path, and a user
name with a password that carries a slash. It is read as credentials, which is
how the URL parser reads it too: the alternative leaves ``123/secret`` visible
in full, and a mangled host in a message about an unusable URL is the cheaper
mistake.
"""


LAST_AT_SIGN_CASES = [
    _Case(
        # The password carries an at-sign and then a hash. Ending the
        # credentials at the last at-sign *before* the hash -- reading the hash
        # as the start of a fragment -- stops inside the password and leaves
        # 's0#s1' in the message, so the bound is the last at-sign of the
        # string instead.
        'a-password-carrying-an-at-sign-before-a-hash',
        'postgresql+psycopg://svc:pw@s0#s1@db.example/sd',
        'postgresql+psycopg://svc:***@db.example/sd',
        'pw@s0#s1',
    ),
    _Case(
        'a-password-carrying-an-at-sign-before-a-slash',
        'postgresql+psycopg://svc:p@ss/word@db.example/sd',
        'postgresql+psycopg://svc:***@db.example/sd',
        'p@ss/word',
    ),
    _Case(
        # What the last-at-sign bound costs, stated so the cost is a decision
        # rather than a surprise: a fragment carrying an at-sign is swallowed
        # whole. A connection URL has no use for a fragment, so this is a
        # mangled message about a URL no driver would have accepted, which is
        # the trade section 2.4 rule 3 takes everywhere else too.
        'a-fragment-carrying-an-at-sign',
        'postgresql psycopg://svc:pw@db.example/sd#note@host',
        'postgresql psycopg://svc:***@host',
        'pw',
    ),
]
"""Where the credentials end when more than one at-sign is a candidate."""


AWKWARD_SPELLINGS = [
    _Case(
        'a-leading-space',
        f' postgresql+psycopg://{AT_SIGN_USER}:{PASSWORD}@host:5432/db',
        f' postgresql+psycopg://{AT_SIGN_USER}:{HIDDEN}@host:5432/db',
        PASSWORD,
    ),
    _Case(
        'a-hyphen-in-the-scheme',
        f'postgresql-psycopg://{AT_SIGN_USER}:{PASSWORD}@host:5432/db',
        f'postgresql-psycopg://{AT_SIGN_USER}:{HIDDEN}@host:5432/db',
        PASSWORD,
    ),
    _Case(
        'a-url-copied-across-two-lines',
        f'postgresql psycopg://{AT_SIGN_USER}:{PASSWORD}@host:5432/db',
        f'postgresql psycopg://{AT_SIGN_USER}:{HIDDEN}@host:5432/db',
        PASSWORD,
    ),
    _Case(
        'an-at-sign-in-the-user-name-and-an-unparseable-port',
        f'//{AT_SIGN_USER}:{PASSWORD}@host:notaport/db',
        f'//{AT_SIGN_USER}:{HIDDEN}@host:notaport/db',
        PASSWORD,
    ),
    _Case('no-user-name', '//:pw@h', f'//:{HIDDEN}@h', 'pw'),
]
"""Spellings a setting arrives in that no parser accepts, and the login form.

The at-sign in ``admin@pgsrv`` is a managed server's own login form, so the
at-sign that ends the credentials is the last one rather than the first.
"""


ALL_CASES = (
    CORPUS + QUERY_CASES + NEGATIVE_CASES + AMBIGUOUS_CASES + LAST_AT_SIGN_CASES + AWKWARD_SPELLINGS
)

CASE_PARAMS = [pytest.param(case, id=case.name) for case in ALL_CASES]

SECRET_PARAMS = [
    pytest.param(case, id=case.name) for case in ALL_CASES if case.secret and case.secret.strip()
]
"""The subset carrying a secret worth naming, for the direction stated on its own."""


def test_the_corpus_is_the_whole_product_of_its_dimensions() -> None:
    """A corpus that samples a dimension proves less than it appears to.

    Every leak this rule has had lived in a value of some dimension the corpus
    did not reach, so the product being complete is itself worth asserting: a
    later edit that drops a dimension's values to shorten the run fails here
    rather than quietly narrowing what the rest of the file proves.
    """
    dimensions = len(SCHEMES) * len(SLASHES) * len(USERINFOS) * len(TAILS)
    assert len(CORPUS) == dimensions


def test_the_corpus_covers_every_slash_count_from_none() -> None:
    """Three slashes is a real spelling, and stopping at two leaked every one."""
    counts = {part.text.count('/') for part in SLASHES}
    assert counts == set(range(MAXIMUM_SLASHES + 1))


@pytest.mark.parametrize(
    ('first', 'second'), list(itertools.product(URL_SIGNIFICANT_CHARACTERS, repeat=2))
)
def test_the_corpus_holds_a_password_carrying_one_character_then_another(
    first: str, second: str
) -> None:
    """Order decides where a rule reading the URL by eye stops.

    A password carrying an at-sign and then a slash is not the same test as one
    carrying a slash and then an at-sign: the rule that ended the credentials at
    the at-sign before the first slash passed the second and leaked the first.

    Parameters:
        first: The character that must appear first.
        second: The character that must appear after it.
    """
    carried = [
        part.text
        for part in PASSWORDS
        if first in part.text and second in part.text[part.text.index(first) + 1 :]
    ]
    assert carried


@pytest.mark.parametrize('case', CASE_PARAMS)
def test_the_rule_replaces_the_credentials_and_nothing_else(case: _Case) -> None:
    """The structural rule is the only defense where the URL did not parse.

    Asserting the whole result rather than the absence of the secret states both
    directions at once: a rule that returned ``***`` for everything would hide
    every password and be useless, and a rule that returned its argument would
    leave every one visible.

    Parameters:
        case: The URL under test and exactly what masking it must produce.
    """
    assert masked_url(case.url) == case.expected


@pytest.mark.parametrize('case', SECRET_PARAMS)
def test_no_secret_survives_the_rule(case: _Case) -> None:
    """Said on its own, because it is the property a leak breaks.

    Parameters:
        case: The URL under test and the secret it carries.
    """
    assert case.secret is not None
    assert case.secret not in masked_url(case.url)


UNPARSEABLE_CREDENTIAL_CASES = [
    case
    for case in ALL_CASES
    if case.secret
    and case.url.startswith(('postgresql psycopg:', 'postgresql-psycopg:', ' ', '//'))
]
"""Credential-bearing URLs the opener refuses without opening a socket.

Every one of these is a URL SQLAlchemy cannot parse, which is what puts the
structural rule on the path an operator's message actually takes, and what keeps
the test off the network.
"""

OPENER_PARAMS = [pytest.param(case, id=case.name) for case in UNPARSEABLE_CREDENTIAL_CASES]


@pytest.mark.parametrize('case', OPENER_PARAMS)
def test_the_opener_names_a_url_it_could_not_parse_by_its_masked_form(case: _Case) -> None:
    """The rule is only worth anything where the opener actually reaches it.

    Asserting on the helper alone leaves the opener free to name the URL by some
    other route, which is exactly how a leak survived three reviews of the
    helper. Both directions are asserted on the one refusal: the password is
    gone, and what is left is the whole URL, which is what would otherwise leave
    the reader nothing to correct.

    Parameters:
        case: The URL under test, carrying a password.
    """
    assert case.secret is not None
    with pytest.raises(ValueError) as excinfo:
        open_index(case.url)
    assert case.secret not in str(excinfo.value)
    assert case.expected in str(excinfo.value)


PARSEABLE_URLS = [
    f'postgresql+psycopg://user:{PASSWORD}@host:5432/spindoctor',
    f'postgresql+psycopg://admin%40pgsrv:{PASSWORD}@host/spindoctor',
    'postgresql+psycopg://user@host/spindoctor',
    'postgresql+psycopg://host:5432/spindoctor',
    'postgresql+psycopg://host:5432/path@name',
    'postgresql+psycopg://user:123/xY9z@host/spindoctor',
]
"""URLs the parser accepts and carries no query, which have a second opinion.

The parser is not an authority on a credential carried as a query parameter --
it renders one verbatim -- so the comparison is drawn only over the part of a
URL it does hide.
"""


@pytest.mark.parametrize('url', PARSEABLE_URLS)
def test_the_rule_hides_what_the_parser_hides(url: str) -> None:
    """The structural rule and the parser agree about where a password is.

    Nothing else can check the rule's reading of an authority against anything.
    Running it over URLs the parser does accept gives the one comparison
    available: for every shape both can read, including the ambiguous one, the
    same characters have to disappear. A rule that read a password as a port
    would disagree here first.

    Parameters:
        url: A URL the parser accepts.
    """
    parsed = sqlalchemy.engine.make_url(url).render_as_string()
    assert masked_url(url) == parsed
