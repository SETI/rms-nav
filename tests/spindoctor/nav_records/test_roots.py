"""Tests for the one spelling of a results root, and the one root behind two of them.

Everything that keys anything on a root only meets what it is looking for if
both sides spell the root the same way. A root reaches a program as a
command-line value, a configuration key or an environment variable, and those
three routinely differ by a trailing slash or by being relative to the working
directory.

These are assertions about the functions' contracts rather than about any one
backend's behavior: that any number of spellings of one root produce one string,
that the filesystem root -- the one root whose separator is its whole name --
survives intact, and that a list naming one root twice means the tree once.

The spelling is absolute *and* resolved, which is what lets everything
downstream stop thinking about paths: once a root is canonical from the moment
it is named, joining a key onto it has one answer and no reader needs a rule of
its own about what that join produced.
"""

from pathlib import Path

import pytest
from filecache import FCPath

from spindoctor.nav_records import distinct_roots, normalize_root_url


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
    assert normalize_root_url('results') == (tmp_path.resolve() / 'results').as_posix()


def test_the_filesystem_root_keeps_its_separator() -> None:
    """It is the one root whose trailing separator is the whole of its name."""
    assert normalize_root_url('/') == '/'


def test_a_cloud_root_keeps_its_scheme() -> None:
    """A results root is any URL the file layer accepts, not only a local path."""
    assert normalize_root_url('gs://rms-nav-results/coiss/') == 'gs://rms-nav-results/coiss'


def test_a_root_reached_through_a_link_is_the_root_it_points_at(tmp_path: Path) -> None:
    """One operator names the link and another names the directory: one root.

    The two spellings would otherwise key two sets of rows over one tree, and a
    walk under one of them would report every document under the other as one
    that had left the tree.
    """
    real = tmp_path / 'results-2026'
    real.mkdir()
    (tmp_path / 'latest').symlink_to(real)
    assert normalize_root_url(tmp_path / 'latest') == real.resolve().as_posix()


def test_a_root_naming_a_parent_directory_is_the_directory_it_names() -> None:
    """A spelling with ``..`` in it is one spelling of the place it lands."""
    assert normalize_root_url('/data/elsewhere/../nav-results') == '/data/nav-results'


def test_a_home_relative_root_is_the_directory_it_expands_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configuration file is where a root gets written with a tilde in it.

    Parameters:
        monkeypatch: Fixture the home directory is set through.
    """
    monkeypatch.setenv('HOME', '/home/operator')
    assert normalize_root_url('~/nav-results') == '/home/operator/nav-results'


def test_a_cloud_root_is_not_resolved_against_the_local_filesystem() -> None:
    """Resolution is a no-op on a URL, which is what makes the rule unconditional.

    A remote location has no links and no relative form, so the same call
    answers for both kinds of root without a branch -- and without a round trip.
    """
    assert normalize_root_url('s3://rms-nav-results/coiss') == 's3://rms-nav-results/coiss'


def test_an_fcpath_normalizes_the_same_way_as_its_text() -> None:
    """A caller holding the path object must not get a second spelling of it."""
    assert normalize_root_url(FCPath('/data/nav-results/')) == normalize_root_url(
        '/data/nav-results'
    )


def test_a_root_spelled_as_nothing_at_all_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty spelling renders as the working directory, and is not a root.

    ``--nav-results-root "$ROOT"`` with the variable unset hands a program an
    empty word, and a program that resolved it would walk whatever directory it
    happens to be in, write those documents under a root nobody named, and
    report a pass that completed.
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='not a location'):
        normalize_root_url('')


def test_a_root_carrying_a_null_byte_is_refused() -> None:
    """It renders perfectly well, and fails at the first call that reaches disk.

    Refused where the root is spelled, it is charged to the root; left to
    render, it becomes an exception out of a directory listing, naming the
    listing rather than the word that caused it.
    """
    with pytest.raises(ValueError, match='null byte'):
        normalize_root_url('/data/nav\x00results')


def test_two_spellings_of_one_root_are_one_root() -> None:
    """A command line naming both means the tree once, not twice."""
    assert distinct_roots(['/data/nav-results', '/data/nav-results/']) == ['/data/nav-results']


def test_the_roots_keep_the_order_they_were_named_in() -> None:
    """A run reports on its roots in the order it was given them."""
    assert distinct_roots(['/data/second', '/data/first']) == ['/data/second', '/data/first']


def test_the_first_spelling_of_a_repeated_root_is_the_one_kept() -> None:
    """De-duplication must not reorder what survives it."""
    assert distinct_roots(['/data/a', '/data/b', '/data/a/']) == ['/data/a', '/data/b']


def test_naming_no_root_at_all_yields_no_root() -> None:
    """An empty list is not an error, and is not the working directory either."""
    assert distinct_roots([]) == []


def test_a_root_that_is_not_a_location_is_refused_here_too() -> None:
    """Every mode of a pass reads its roots through this, so the refusal is here."""
    with pytest.raises(ValueError, match='not a location'):
        distinct_roots(['/data/nav-results', ''])


def test_a_root_handed_over_as_a_path_is_the_root_it_names(tmp_path: Path) -> None:
    """A caller holding paths hands them over as they are.

    Rendering one itself would put a second spelling rule beside the one this
    module exists to be, and a path rendered by the platform is not the POSIX
    rendering every key is stored and compared as.
    """
    assert distinct_roots([tmp_path / 'results']) == [(tmp_path.resolve() / 'results').as_posix()]


def test_two_kinds_of_path_naming_one_root_are_one_root(tmp_path: Path) -> None:
    """One program holds a path, another an FCPath, and a third the URL: one root."""
    root = tmp_path / 'results'
    assert distinct_roots([root, FCPath(root.as_posix()), root.as_posix()]) == [
        (tmp_path.resolve() / 'results').as_posix()
    ]
