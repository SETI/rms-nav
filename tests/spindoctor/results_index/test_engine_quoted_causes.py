"""Tests that a refusal quoting its own cause quotes no credential with it.

A refusal names the URL it was given, with the credentials replaced, and beside
it what the failure underneath said.  That second half is where a password
comes back: a URL grammar stops the password at the first at-sign, so a password
carrying one has its tail read as a host and a port, and the port is quoted back
in cleartext by the failure to read it as a number.  A driver reached with
arguments it will not accept quotes them back in full for the same reason.

What is asserted is that no run of the password survives, rather than that the
password does not: the surviving part is a slice of the string rather than a
field of it, and which slice a parser stops on is a property of the password.
The other direction is asserted beside it, because a message cleaned of its
cause would be safe and useless.

The masking of the URL itself, over the corpus of every shape one can be
written in, is in ``test_masking``.
"""

import dataclasses
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy.engine import URL, Engine
from tests.spindoctor.results_index.conftest import AT_SIGN_USER

from spindoctor.results_index import open_index

SHORTEST_HIDDEN_RUN = 3
"""Shortest run of a password the rule undertakes to hide, mirroring the module.

Stated here rather than imported, so that widening the rule's own constant has
to be argued for against a number written down independently.
"""

ECHOED_CONNECT_FAILURE = 'the dialect rejected the connect arguments for'
"""What the quoting stand-in factory says before repeating the URL it was given."""

UNREADABLE_PORT_FAILURE = 'invalid literal for int()'
"""What a URL parser says when a password's tail lands where the port belongs.

The diagnosis a reader of a failed run needs, and the half of the message that
carries no credential.  It is named here so that each case can assert the
detail it expects to keep: the wrapper's own text survives whether or not the
cause does, so asserting that alone would pass against a message cleaned of
everything it was quoting.
"""

TAIL_LEAKING_PASSWORD = 'se@cr:etlongsecretpassword'
"""A password whose tail a URL parser reports as the port it could not read.

The password group of a URL grammar stops at the first at-sign, so everything
after the one inside this password is read as a host and a port; the port is
then quoted back in the failure to convert it to a number, and it is a run of
the password in cleartext.
"""

EVERY_CHARACTER_PASSWORD = 'aB3@xY9:zQ7/wE1?rT5'
"""A password carrying every character that means something to a URL.

The at-sign and the colon put its tail where the parser quotes it back; the
slash and the question mark decide how much of it the parser reads as one field,
and a rule tested on a password carrying one of them at a time never sees this.
"""

SLASHED_ECHO_PASSWORD = 'aB3xY9zQ7wE1'
"""A password on a URL that parses, so the failure comes from the driver instead.

Reaching a driver is the other half of the surface: what it says about its
connect arguments is not a parse error and is not shaped like one, and a rule
that only cleaned parse errors would leave this route wide open.
"""


def _echoing_factory(parsed: URL, *args: Any, **kwargs: Any) -> Engine:
    """Stand in for a driver that quotes the URL it was handed back at the caller.

    A driver reports an argument it will not accept by naming the arguments it
    was given, and its rendering of them hides nothing: a parsed URL renders a
    ``?password=`` query parameter verbatim.

    Parameters:
        parsed: The parsed connection URL, which is quoted back in full.
        args: Whatever the caller passed, all of it ignored.
        kwargs: Whatever the caller passed, all of it ignored.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError(f'{ECHOED_CONNECT_FAILURE} {parsed.render_as_string(hide_password=False)}')


@dataclasses.dataclass(frozen=True)
class _Quoted:
    """One refusal that quotes a failure of its own back, and the secret at risk.

    Attributes:
        name: Identifier for the case.
        url: The URL to open.
        secret: The credential that URL carries, no run of which may survive.
        detail: The case's own diagnosis, carrying no credential, which the
            cleaning must leave behind.  Asserted per case rather than once for
            all of them, because what a reader needs differs by route: which
            field a parser stopped on, or what a driver would not accept.
        reaches_the_driver: Whether the URL parses, so that the failure comes
            from the engine factory rather than from the parser.
    """

    name: str
    url: str
    secret: str
    detail: str
    reaches_the_driver: bool = False


QUOTED_CASES = [
    _Quoted(
        'a-password-whose-tail-reads-as-a-port',
        f'postgresql+psycopg://user:{TAIL_LEAKING_PASSWORD}@dbhost/spindoctor',
        TAIL_LEAKING_PASSWORD,
        UNREADABLE_PORT_FAILURE,
    ),
    _Quoted(
        'a-password-carrying-every-url-character',
        f'postgresql+psycopg://user:{EVERY_CHARACTER_PASSWORD}@dbhost/spindoctor',
        EVERY_CHARACTER_PASSWORD,
        UNREADABLE_PORT_FAILURE,
    ),
    _Quoted(
        'an-at-sign-in-the-user-name',
        f'postgresql+psycopg://{AT_SIGN_USER}:{TAIL_LEAKING_PASSWORD}@dbhost/spindoctor',
        TAIL_LEAKING_PASSWORD,
        UNREADABLE_PORT_FAILURE,
    ),
    _Quoted(
        'a-password-the-driver-quotes-back',
        f'postgresql+psycopg://user:{SLASHED_ECHO_PASSWORD}@dbhost/spindoctor',
        SLASHED_ECHO_PASSWORD,
        ECHOED_CONNECT_FAILURE,
        reaches_the_driver=True,
    ),
    _Quoted(
        'a-password-query-parameter-the-driver-quotes-back',
        f'postgresql+psycopg://user@dbhost/spindoctor?password={SLASHED_ECHO_PASSWORD}',
        SLASHED_ECHO_PASSWORD,
        ECHOED_CONNECT_FAILURE,
        reaches_the_driver=True,
    ),
]
"""Refusals whose own quoted cause is where a password can still be read."""

QUOTED_PARAMS = [pytest.param(case, id=case.name) for case in QUOTED_CASES]


def _runs_of(secret: str, length: int) -> list[str]:
    """Return every run of the given length that a secret contains.

    Parameters:
        secret: The credential.
        length: How long each run is.

    Returns:
        The runs, in the order they begin, or an empty list for a secret shorter
        than one run.
    """
    return [secret[start : start + length] for start in range(len(secret) - length + 1)]


def _quoted_refusal(case: _Quoted, monkeypatch: pytest.MonkeyPatch) -> ValueError:
    """Open a case's URL and return the refusal it raised.

    Parameters:
        case: The case to drive.
        monkeypatch: Fixture the engine factory is replaced through.

    Returns:
        The refusal, for assertions on what it says.
    """
    if case.reaches_the_driver:
        monkeypatch.setattr(sqlalchemy, 'create_engine', _echoing_factory)
    with pytest.raises(ValueError) as excinfo:
        open_index(case.url)
    return excinfo.value


@pytest.mark.parametrize('case', QUOTED_PARAMS)
def test_no_run_of_a_password_survives_what_a_refusal_quotes_back(
    case: _Quoted, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Masking the URL is not enough: the cause beside it quotes the URL too.

    A refusal names the URL it was given and, beside it, what the failure
    underneath said. That second half is a slice of the string rather than a
    field of it, so asking whether the whole password survives answers nothing:
    what survives is a run of it. Every run is asked about, because which run a
    parser stops on is a property of the password rather than of the rule.

    Parameters:
        case: The refusal under test.
        monkeypatch: Fixture the engine factory is replaced through.
    """
    message = str(_quoted_refusal(case, monkeypatch))
    surviving = [run for run in _runs_of(case.secret, SHORTEST_HIDDEN_RUN) if run in message]
    assert surviving == []


@pytest.mark.parametrize('case', QUOTED_PARAMS)
def test_a_refusal_that_quotes_a_cause_still_says_what_failed(
    case: _Quoted, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping the cause would hide the password and the diagnosis with it.

    The quoted cause is the only thing that tells an unreadable port from a
    server that refused the connection, so it has to survive the cleaning: a
    refusal that said nothing but the URL would be safe and useless.  The
    case's own diagnosis is asserted beside the wrapper's text, because the
    wrapper says the same thing whether the cause survived or was cleaned away
    with the credential.

    Parameters:
        case: The refusal under test.
        monkeypatch: Fixture the engine factory is replaced through.
    """
    message = str(_quoted_refusal(case, monkeypatch))
    assert 'could not open the results index' in message
    assert case.detail in message


def test_a_quoted_cause_keeps_what_is_not_a_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleaning a message is worth nothing if it takes the message with it.

    The host, the database and what the driver was complaining about are all
    outside the credentials and are what a reader corrects, so they survive
    whole.
    """
    monkeypatch.setattr(sqlalchemy, 'create_engine', _echoing_factory)
    with pytest.raises(ValueError) as excinfo:
        open_index(f'postgresql+psycopg://user:{SLASHED_ECHO_PASSWORD}@dbhost/spindoctor')
    assert ECHOED_CONNECT_FAILURE in str(excinfo.value)
