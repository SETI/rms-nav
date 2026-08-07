"""Hermetic tests for ``spindoctor.cli.ck.metakernel``.

The meta-kernel's whole job is load order, so the tests that matter furnish it
and read back the order SPICE ended up with rather than reading the text.  The
rest pin what a path may look like, because a path a text kernel cannot express
is truncated rather than refused.
"""

from pathlib import Path

import cspyce
import numpy as np
import pytest
from filecache import FCPath

from spindoctor.cli.ck.metakernel import build_meta_kernel_lines, write_meta_kernel

_CK_FRAME_ID = -82000


def _write_ck(path: Path) -> None:
    """Write a one-segment C-kernel that furnishes without any other kernel.

    Parameters:
        path: The file to create.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ticks = np.array([100.0, 200.0, 300.0])
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (3, 1))
    handle = cspyce.ckopn(str(path), path.name, 0)
    try:
        cspyce.ckw03(
            handle,
            100.0,
            300.0,
            _CK_FRAME_ID,
            'J2000',
            False,
            'seg',
            ticks,
            quats,
            np.zeros((3, 3)),
            1,
            [100.0],
        )
    finally:
        cspyce.ckcls(handle)


def _furnished_ck_order(meta: Path) -> list[str]:
    """Furnish a meta-kernel and report the C-kernels it loaded, in order.

    Parameters:
        meta: The meta-kernel to furnish.

    Returns:
        The paths of the C-kernels in the pool, in load order.
    """
    cspyce.furnsh(str(meta))
    try:
        return [str(cspyce.kdata(at, 'CK')[0]) for at in range(int(cspyce.ktotal('CK')))]
    finally:
        cspyce.unload(str(meta))


def test_the_meta_kernel_furnishes_originals_before_corrections(tmp_path: Path) -> None:
    """SPICE gives precedence to the C-kernel furnished last, so it is the correction."""
    original = tmp_path / 'orig.bc'
    correction = tmp_path / 'orig_nav.bc'
    _write_ck(original)
    _write_ck(correction)
    meta = FCPath(str(tmp_path / 'set.tm'))
    write_meta_kernel(meta, originals=[original], corrections=[correction])
    assert _furnished_ck_order(tmp_path / 'set.tm') == [str(original), str(correction)]


def test_every_correction_comes_after_every_original(tmp_path: Path) -> None:
    """Not merely each correction after its own original."""
    originals = [tmp_path / 'a.bc', tmp_path / 'b.bc']
    corrections = [tmp_path / 'a_nav.bc', tmp_path / 'b_nav.bc']
    for path in (*originals, *corrections):
        _write_ck(path)
    meta = FCPath(str(tmp_path / 'set.tm'))
    write_meta_kernel(meta, originals=originals, corrections=corrections)
    loaded = _furnished_ck_order(tmp_path / 'set.tm')
    assert loaded == [str(path) for path in (*originals, *corrections)]


def test_a_long_path_furnishes_unchanged(tmp_path: Path) -> None:
    """A path over 80 characters is truncated by SPICE unless it is continued."""
    deep = tmp_path / ('d' * 60) / ('e' * 60)
    original = deep / 'orig.bc'
    _write_ck(original)
    assert len(str(original)) > 80
    meta = FCPath(str(tmp_path / 'long.tm'))
    write_meta_kernel(meta, originals=[original], corrections=[])
    assert _furnished_ck_order(tmp_path / 'long.tm') == [str(original)]


def test_a_meta_kernel_naming_nothing_is_refused() -> None:
    """It would furnish nothing and report no error when furnished."""
    with pytest.raises(ValueError, match='names no kernels'):
        build_meta_kernel_lines([], [])


def test_a_path_ending_in_the_continuation_character_is_refused() -> None:
    """Its last character is indistinguishable from the marker that joins it on."""
    with pytest.raises(ValueError, match='continuation'):
        build_meta_kernel_lines(['/kernels/orig.bc+'], [])


def test_an_empty_path_is_refused() -> None:
    """An empty string in the list would be furnished as the working directory."""
    with pytest.raises(ValueError, match='empty path'):
        build_meta_kernel_lines([''], [])


def test_a_path_holding_a_quote_is_refused() -> None:
    """A quote ends the string value early and the rest becomes syntax."""
    with pytest.raises(ValueError, match='holds a quote'):
        build_meta_kernel_lines(["/kernels/o'rig.bc"], [])


def test_a_path_holding_a_newline_is_refused() -> None:
    """A newline splits one value into two the parser cannot rejoin."""
    with pytest.raises(ValueError, match='non-printing'):
        build_meta_kernel_lines(['/kernels/orig\n.bc'], [])


def test_the_meta_kernel_declares_itself_a_meta_kernel(tmp_path: Path) -> None:
    """The first line is the type marker SPICE identifies the file by."""
    assert build_meta_kernel_lines([tmp_path / 'a.bc'], [])[0] == 'KPL/MK'


def test_the_meta_kernel_says_why_the_originals_are_there() -> None:
    """A reader must not conclude the corrections replace them."""
    lines = build_meta_kernel_lines(['/kernels/orig.bc'], ['/kernels/orig_nav.bc'])
    assert any('rather than replaced' in line for line in lines)


def test_a_path_of_exactly_the_string_limit_furnishes(tmp_path: Path) -> None:
    """The boundary between one string and two is not a truncation point."""
    padding = 80 - len(str(tmp_path / 'x.bc'))
    original = tmp_path / ('x' * max(padding, 0) + 'x.bc')
    _write_ck(original)
    meta = FCPath(str(tmp_path / 'edge.tm'))
    write_meta_kernel(meta, originals=[original], corrections=[])
    assert _furnished_ck_order(tmp_path / 'edge.tm') == [str(original)]


def test_a_path_ending_in_a_blank_is_refused() -> None:
    """SPICE trims it, and then looks for a name one character shorter.

    Measured through a real furnsh: a 119-character path whose last character
    is a space loads a 118-character name, with no error of any kind.
    """
    with pytest.raises(ValueError, match='ends in a blank'):
        build_meta_kernel_lines(['/kernels/orig.bc '], [])


def _path_with_a_blank_at(tmp_path: Path, index: int, tail: str) -> Path:
    """Return a path under ``tmp_path`` whose character at ``index`` is a space.

    Parameters:
        tmp_path: The directory the path lies under.
        index: The absolute character position the space must land on.
        tail: What follows the space.

    Returns:
        The path.
    """
    prefix = f'{tmp_path}/'
    assert len(prefix) <= index
    return tmp_path / ('x' * (index - len(prefix)) + ' ' + tail)


def test_no_piece_ends_in_a_blank(tmp_path: Path) -> None:
    """A join is walked back off a blank, so a name cannot lose one at a cut."""
    spaced = _path_with_a_blank_at(tmp_path, 78, 'z' * 40 + '.bc')
    assert str(spaced)[78] == ' '
    lines = build_meta_kernel_lines([spaced], [])
    quoted = [line.strip() for line in lines if line.strip().startswith("'")]
    for piece in quoted:
        assert not piece.rstrip("'+").endswith(' ')


def test_a_path_with_a_blank_at_a_join_furnishes_unchanged(tmp_path: Path) -> None:
    """And the path SPICE ends up with is the one that was asked for."""
    directory = _path_with_a_blank_at(tmp_path, 78, 'z' * 30)
    original = directory / 'orig.bc'
    assert str(original)[78] == ' '
    _write_ck(original)
    meta = FCPath(str(tmp_path / 'blank.tm'))
    write_meta_kernel(meta, originals=[original], corrections=[])
    assert _furnished_ck_order(tmp_path / 'blank.tm') == [str(original)]


def test_a_run_of_blanks_longer_than_a_piece_is_refused() -> None:
    """There is no cut left to walk back to, so it is named rather than cut."""
    with pytest.raises(ValueError, match='run of more than'):
        build_meta_kernel_lines(['/kernels/' + ' ' * 100 + 'orig.bc'], [])
