"""Tests for the one spelling of a results root.

Every row of the index names the root it was ingested from and every consumer
filters on the root it was itself pointed at, so the two only meet if both spell
it the same way. A root reaches a program as a command-line value, a
configuration key or an environment variable, and those three routinely differ
by a trailing slash or by being relative to the working directory.

These are assertions about the function's contract rather than about any one
backend's behavior: what it promises is that two spellings of one root produce
one string, and that the filesystem root -- the one root whose separator is its
whole name -- survives intact.
"""

from pathlib import Path

import pytest
from filecache import FCPath

from spindoctor.results_index import normalize_root_url


def test_a_trailing_separator_does_not_make_two_roots() -> None:
    """One program writes the trailing slash and another does not."""
    assert normalize_root_url('/data/nav-results/') == normalize_root_url('/data/nav-results')


def test_a_repeated_separator_does_not_make_two_roots() -> None:
    """A root pasted together from two settings arrives with the join doubled."""
    assert normalize_root_url('/data//nav-results') == '/data/nav-results'


def test_a_relative_root_becomes_an_absolute_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root named relatively on one run and absolutely on the next is one root.

    Parameters:
        tmp_path: A directory to resolve the relative name against.
        monkeypatch: Fixture the working directory is changed through.
    """
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
