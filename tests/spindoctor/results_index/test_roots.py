"""Tests for the one spelling of a results root, and for the refusal of an unknown one.

Every row of the index names the root it was ingested from and every consumer
filters on the root it was itself pointed at, so the two only meet if both spell
it the same way. A root reaches a program as a command-line value, a
configuration key or an environment variable, and those three routinely differ
by a trailing slash or by being relative to the working directory.

These are assertions about the functions' contracts rather than about any one
backend's behavior: that two spellings of one root produce one string, that the
filesystem root -- the one root whose separator is its whole name -- survives
intact, and that the refusal of a root nobody ingested names its index without
its password and its roots exactly as they were given.
"""

from pathlib import Path

import pytest
from filecache import FCPath

from spindoctor.results_index import normalize_root_url, open_index, require_ingested_roots

PASSWORD = 'sup3rs3cr3t'
"""A password distinctive enough that finding it anywhere is proof of a leak."""

SERVER_URL = f'postgresql+psycopg://svc:{PASSWORD}@db.example/spindoctor'
"""An index URL carrying a password, as a consumer's own resolution produces it."""


def test_a_trailing_separator_does_not_make_two_roots() -> None:
    """One program writes the trailing slash and another does not."""
    assert normalize_root_url('/data/nav-results/') == normalize_root_url('/data/nav-results')


def test_a_repeated_separator_does_not_make_two_roots() -> None:
    """A root pasted together from two settings arrives with the join doubled."""
    assert normalize_root_url('/data//nav-results') == '/data/nav-results'


def test_a_relative_root_becomes_an_absolute_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root named relatively on one run and absolutely on the next is one root."""
    monkeypatch.chdir(tmp_path)
    assert normalize_root_url('results') == (tmp_path / 'results').as_posix()


def test_the_filesystem_root_keeps_its_separator() -> None:
    """It is the one root whose trailing separator is the whole of its name."""
    assert normalize_root_url('/') == '/'


def test_a_cloud_root_keeps_its_scheme() -> None:
    """A results root is any URL the file layer accepts, not only a local path."""
    assert normalize_root_url('gs://rms-nav-results/coiss/') == 'gs://rms-nav-results/coiss'


def test_an_fcpath_normalizes_the_same_way_as_its_text() -> None:
    """A caller holding the path object must not get a second spelling of it."""
    assert normalize_root_url(FCPath('/data/nav-results/')) == normalize_root_url(
        '/data/nav-results'
    )


def _refusal_of_an_unknown_root(tmp_path: Path) -> str:
    """Ask an empty index for a root nobody ingested and return the refusal text.

    The index URL handed in is not the one the engine was opened with: the point
    is what the refusal does with the URL its caller names, and a consumer's
    resolved URL is a server URL carrying a password far more often than the
    local file a test can build.

    Parameters:
        tmp_path: Directory the index file lives under.

    Returns:
        The refusal message.
    """
    engine = open_index(f'sqlite:///{(tmp_path / "index.sqlite3").as_posix()}', create=True)
    try:
        with (
            engine.connect() as connection,
            pytest.raises(ValueError, match='no completed ingest') as excinfo,
        ):
            require_ingested_roots(connection, ['/data/nav-results'], url=SERVER_URL)
    finally:
        engine.dispose()
    return str(excinfo.value)


def test_the_refusal_masks_its_index_and_leaves_its_roots_alone(tmp_path: Path) -> None:
    """Three things are true of the refusal, and one call shows all three.

    It is printed to a terminal and written to run logs, so the index password
    may not survive into it; it is masked inside this function rather than by
    each consumer, because a consumer that forgets is a leak in a program nobody
    thought to check. Which of the three resolution levels supplied the URL is
    half of what the message is for, so the rest of the URL has to survive. And
    the root is printed exactly as it was given, because a results root has
    nothing to hide and is the string the reader has to correct.
    """
    refusal = _refusal_of_an_unknown_root(tmp_path)
    assert PASSWORD not in refusal
    assert 'postgresql+psycopg://svc:***@db.example/spindoctor' in refusal
    assert '/data/nav-results' in refusal


def test_the_refusal_leaves_a_credential_shaped_root_alone(tmp_path: Path) -> None:
    """Masking a root would corrupt the one string the message exists to deliver."""
    engine = open_index(f'sqlite:///{(tmp_path / "index.sqlite3").as_posix()}', create=True)
    try:
        with (
            engine.connect() as connection,
            pytest.raises(ValueError, match='no completed ingest') as excinfo,
        ):
            require_ingested_roots(connection, ['//store:8443/nav@results'], url=SERVER_URL)
    finally:
        engine.dispose()
    assert '//store:8443/nav@results' in str(excinfo.value)
