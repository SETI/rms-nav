"""The rule that decides which part of a connection URL is a credential.

A connection URL reaches a person in several places -- a refusal naming the
index that could not be opened, a run log recording the command line a program
was given -- and one part of it is a password.  The rule that hides it is stated
here, once, so that everything putting a URL in front of a person hides the same
characters: the opener in :mod:`spindoctor.results_index.engine`, the refusals
of :mod:`spindoctor.results_index.roots`, and
:func:`spindoctor.support.command_line.masked_command_line`, which decides which
words of a command line are URLs and applies this rule to each of them.

It is a structural rule over the string rather than a parse.  A parsed URL
renders itself with its password hidden but renders a ``?password=`` query
parameter verbatim, and that parameter authenticates exactly as the authority
form does, so adopting the parser for the URLs it accepts would adopt its blind
spot with it.  The URLs a parser refuses are also the ones a refusal quotes back
most often, since being unreadable is what the refusal is about.

Two questions are asked of a URL here.  :func:`masked_url` renders the URL
itself with its credentials replaced.  :func:`without_credentials` cleans a
message that quoted a piece of that URL back, which is a different problem: a
driver reports the fragment it stopped on rather than a field of the URL, so
what has to go is every run of the credential rather than a span of the string.
"""

import re

__all__ = ['masked_url', 'without_credentials']

_HIDDEN_PASSWORD = '***'
"""What a password is replaced by, matching what a parsed URL renders itself as."""

_SQLITE_SCHEME = 'sqlite'
"""Scheme of the one URL form that names a local path instead of a server.

Such a URL has no credentials at all, and a path is free to carry the colons,
at-signs and question marks that would otherwise read as some.
"""

_CREDENTIAL_QUERY_MARKERS = ('password', 'passwd', 'pwd', 'secret', 'token', 'credential')
"""Text that makes a query parameter's value a credential rather than a setting.

A connection URL may carry its password as a query parameter instead of in the
authority -- ``postgresql+psycopg://user@host/db?password=...`` authenticates
exactly as the authority form does -- and a driver accepts any parameter its
library knows, so the name is matched by what it contains rather than against a
fixed list.  Over-hiding a setting whose name says credential costs a word of a
message; under-hiding one puts a working password in a run log.
"""

_QUERY_SEPARATOR = re.compile(r'([;&])')
"""What separates one query parameter from the next, kept by the split.

A connection URL's query is separated by ``&`` or by ``;``: libpq accepts
either, and so does every driver built on it.  The separator is captured so
that a masked query is rebuilt with the separators it was written with.
"""

_SHORTEST_HIDDEN_RUN = 3
"""Shortest run of a credential that is hidden where something quotes one back.

A message quoting a URL back rarely quotes the whole of it.  A driver that could
not read a URL reports the fragment it stopped on, and that fragment is a slice
of the string rather than a field of it: SQLAlchemy reads
``user:se@cr:etpassword@host`` as a host and a port, and says it could not read
``etpassword@host`` as a number.  Only the credential itself says which slices
are its own, so every run of one that appears in a quoted message is replaced.

Three characters is where that stops being worth doing.  Shorter runs collide
with ordinary words often enough to turn a diagnosis into a row of markers, and
a secret disclosed three characters at a time, with neither the order of the
runs nor the gaps between them, is not disclosed.  A credential shorter than
that is hidden whole, since there is nothing to be lost by mangling a message
about a two-character password.
"""


def _scheme_base(url: str) -> str:
    """Return the backend a URL's scheme names, whatever driver it asks for.

    Parameters:
        url: The URL as the caller wrote it.

    Returns:
        The scheme with any ``+driver`` suffix and surrounding space removed and
        lower cased, or an empty string when the string carries no scheme.
    """
    scheme, separator, _remainder = url.partition(':')
    if not separator:
        return ''
    return scheme.strip().split('+', 1)[0].lower()


def _authority_start(url: str) -> int:
    """Return the index at which a URL's authority section begins.

    A scheme is only recognized as one when a ``/`` follows its ``:``.  Without
    that slash the text before the colon reads equally as a scheme and as a user
    name -- ``postgresql:svc:pw@host/db`` and ``svc:pw@host/db`` are the same
    shape -- and reading it as a scheme is what leaves the password of the
    second one visible.  It is therefore read as a user name in both, which
    hides the first one's user name along with its password.  That is one word
    of a message about a URL no driver would have accepted anyway; the other
    reading loses a working password.

    Every slash of the run is consumed, however many there are.  Two is the
    spelling a URL is defined with and one is what a hand-edited setting
    arrives as, but three is an ordinary spelling too -- ``postgresql:///db``
    omits the host to name a local socket, and ``sqlite:///path`` habituates
    the form -- and a rule that stopped counting at two would leave the
    authority start on a slash, which reads as a path beginning before any
    password and returns the URL whole.

    Parameters:
        url: The URL as the caller wrote it.

    Returns:
        The index just past the scheme and the slashes that open the authority.
    """
    colon = url.find(':')
    start = 0
    if colon >= 0 and url[colon + 1 : colon + 2] == '/':
        prefix = url[:colon]
        # A '/' or an '@' before the colon puts the colon inside the authority
        # rather than after a scheme, whatever follows it.
        if '/' not in prefix and '@' not in prefix:
            start = colon + 1
    end = start
    while url[end : end + 1] == '/':
        end += 1
    return end


def _password_span(url: str) -> tuple[int, int] | None:
    """Return the half-open range of characters holding a URL's password.

    The rule is the one a URL's own grammar states.  The user name runs from the
    start of the authority to the first ``:``; only a ``:`` introduces a
    password, and that password runs to the ``@`` that ends the credentials.

    Which ``@`` ends them is the whole question.  It is the **last** one in the
    string.  A user name is free to carry one -- ``user@servername`` is the
    login form of a managed server -- and so is a password: ``p@ssword``,
    ``p@ss/word`` and ``pw:with@both`` are all things an operator types.  Every
    narrower choice stops inside a password that carries the character it stops
    at.  Ending at the ``@`` before the first ``/`` leaves the tail of
    ``p@ss/word`` in the message; ending at the last ``@`` before a ``#``
    leaves the tail of ``pw@part#rest``.  The last ``@`` is the only bound that
    cannot stop early, because the span it produces contains every other
    candidate span.

    What that costs is over-masking a URL whose real credentials end sooner and
    whose tail happens to carry an ``@`` -- a fragment such as
    ``...?password=x#note@host`` is masked to its last character.  A connection
    URL has no use for a fragment, so that is a mangled message about a URL no
    driver would have accepted, against a working password in a run log.

    One shape is genuinely ambiguous: ``host:5432/path@name`` reads equally as a
    host with a port and a path, or as a user name with a password that carries
    a slash.  It is read as credentials, which is how the URL parser itself
    reads it -- for the spelling of that shape a parser accepts, this rule and
    ``render_as_string()`` hide the same characters.  Reading it as a port
    instead would leave a password beginning with digits, ``123/secret``, in
    every message; the cost of the reading taken is a mangled host and database
    name in a message about a URL that was already unusable.

    Parameters:
        url: The URL as the caller wrote it.

    Returns:
        The first and last-plus-one index of the password, or None when the URL
        carries no password to hide.
    """
    start = _authority_start(url)
    colon = url.find(':', start)
    if colon < 0:
        return None
    slash = url.find('/', start)
    if 0 <= slash < colon:
        # The authority ended before any colon, so what follows is a path.
        return None
    at = url.rfind('@', colon)
    if at < 0:
        return None
    return colon + 1, at


def _names_a_credential(name: str) -> bool:
    """Whether a query parameter's name says its value is a credential.

    Parameters:
        name: The parameter's name, as written.

    Returns:
        True when the name carries one of :data:`_CREDENTIAL_QUERY_MARKERS`.
    """
    return any(marker in name.strip().lower() for marker in _CREDENTIAL_QUERY_MARKERS)


def _masked_parameter(parameter: str) -> str:
    """Return one query parameter with its value replaced if it is a credential.

    Parameters:
        parameter: The parameter as written, ``name=value`` or a bare name.

    Returns:
        The parameter, with its value replaced when its name says credential.
    """
    name, separator, _value = parameter.partition('=')
    if not separator:
        return parameter
    if not _names_a_credential(name):
        return parameter
    return f'{name}={_HIDDEN_PASSWORD}'


def _query_span(url: str) -> tuple[int, int] | None:
    """Return the half-open range of characters holding a URL's query string.

    Parameters:
        url: The URL as the caller wrote it.

    Returns:
        The first and last-plus-one index of the query, without its leading
        ``?`` and without any fragment after it, or None when the URL carries no
        query at all.
    """
    start = url.find('?')
    if start < 0:
        return None
    end = url.find('#', start)
    return start + 1, len(url) if end < 0 else end


def _query_pieces(url: str) -> list[str]:
    """Split a URL's query into its parameters and the separators between them.

    Parameters:
        url: The URL as the caller wrote it, which must carry a query.

    Returns:
        The parameters at even positions and the separators at odd ones.  A
        query is separated by ``&`` or by ``;``, both of which libpq and the
        drivers accept, and splitting on one alone leaves a parameter written
        with the other unexamined; the separators are kept so that a masked
        query is rebuilt with the ones it was written with.
    """
    span = _query_span(url)
    if span is None:
        return []
    first, past_last = span
    return _QUERY_SEPARATOR.split(url[first:past_last])


def _masked_query(url: str) -> str:
    """Return a URL with the value of every credential query parameter replaced.

    Run after the authority has been masked, so that a ``?`` inside a password
    has already gone with the password and cannot be mistaken for the start of a
    query.

    Parameters:
        url: The URL, with its authority already masked.

    Returns:
        The URL with any credential-bearing parameter hidden.
    """
    span = _query_span(url)
    if span is None:
        return url
    first, past_last = span
    pieces = _query_pieces(url)
    masked = [
        piece if index % 2 else _masked_parameter(piece) for index, piece in enumerate(pieces)
    ]
    if masked == pieces:
        return url
    return f'{url[:first]}{"".join(masked)}{url[past_last:]}'


def masked_url(url: str) -> str:
    """Return a URL string with every credential in it replaced.

    Anything that puts a connection URL in front of a person -- a refusal whose
    parsing is what failed, a run log recording the command line it was given --
    calls this, so that one structural rule decides what a credential is.  It is
    the only rule: a parsed URL renders itself with its password hidden, but it
    renders a ``?password=`` query parameter verbatim, so adopting the parser
    for the URLs it accepts would adopt its blind spot with it.

    Everything outside a credential survives, because naming the URL is what
    tells a reader which of the resolution levels supplied the value.

    A ``sqlite:`` URL is returned exactly as it came.  It names a local
    filesystem path, which has no credentials at all, and a path is free to carry
    the colons, at-signs and question marks that would otherwise read as some.

    A results root is not a connection URL and is never passed here.  It has no
    credentials to hide, and a root is the one string an operator reads a run
    log to correct, so mangling one costs more than it protects.

    Parameters:
        url: The URL as the caller wrote it.

    Returns:
        The URL with its credentials, if any, masked.
    """
    if _scheme_base(url) == _SQLITE_SCHEME:
        return url
    span = _password_span(url)
    if span is not None:
        first, past_last = span
        url = f'{url[:first]}{_HIDDEN_PASSWORD}{url[past_last:]}'
    return _masked_query(url)


def _credentials(url: str) -> list[str]:
    """Return every credential a URL carries, exactly as it is written.

    The same structural rule :func:`masked_url` masks by, read as values rather
    than as spans, so that what a message quotes back is measured against the
    same idea of a credential the URL itself is.

    Parameters:
        url: The URL as the caller wrote it.

    Returns:
        The password from the authority and the value of every credential-
        bearing query parameter, skipping the empty ones.  A ``sqlite:`` URL is
        a local filesystem path and carries none.
    """
    if _scheme_base(url) == _SQLITE_SCHEME:
        return []
    found: list[str] = []
    span = _password_span(url)
    if span is not None:
        first, past_last = span
        found.append(url[first:past_last])
    for index, piece in enumerate(_query_pieces(url)):
        name, separator, value = piece.partition('=')
        if not index % 2 and separator and _names_a_credential(name):
            found.append(value)
    return [secret for secret in found if secret]


def _without_runs_of(text: str, secret: str) -> str:
    """Return text with every run of one secret in it replaced.

    Parameters:
        text: The text to clean, which is not a URL and cannot be masked as one.
        secret: The credential whose runs are to go.

    Returns:
        The text, with every run of :data:`_SHORTEST_HIDDEN_RUN` or more
        characters that also appears in the secret replaced, and a secret
        shorter than that replaced wherever it appears whole.  Scanned from the
        left, taking the longest run at each position: a run left over when a
        shorter one has been replaced is examined again at the position it
        resumes from, so no run of the secret survives in part.
    """
    kept: list[str] = []
    index = 0
    while index < len(text):
        length = 0
        while index + length < len(text) and text[index : index + length + 1] in secret:
            length += 1
        if length and length >= min(_SHORTEST_HIDDEN_RUN, len(secret)):
            kept.append(_HIDDEN_PASSWORD)
            index += length
        else:
            kept.append(text[index])
            index += 1
    return ''.join(kept)


def without_credentials(text: str, url: str) -> str:
    """Return a message quoting a URL back with the credentials of that URL gone.

    Masking the URL a refusal names is not enough on its own.  A refusal also
    quotes what the failure underneath it said, and a driver that could not read
    a URL says so by quoting the piece of it that stopped it -- which, for a
    password carrying an ``@`` and a ``:``, is a run of the password in
    cleartext.  Such a message travels: it is written to run logs, returned in a
    cloud task's result, and collected into event logs an operator concatenates
    and hands on.

    Parameters:
        text: What the underlying failure said.
        url: The URL it was raised about, as the caller wrote it.

    Returns:
        The text with every run of every credential of that URL replaced.
    """
    for secret in _credentials(url):
        text = _without_runs_of(text, secret)
    return text
