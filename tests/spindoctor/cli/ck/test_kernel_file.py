"""Hermetic tests for ``spindoctor.cli.ck.kernel_file``.

The file writer's two jobs are to put the segments and the comments into one
file in the order SPICE allows, and to reserve the comment room before the file
is opened.  The reservation is what these tests work hardest on, because SPICE
does not report a reservation that was too small: it silently rewrites the file
instead, moving every data record to make room.  What is measurable is that the
first data record does not move, and that is asserted directly.

The third job is judging a whole run's destinations before any of them is
written, and its tests work hardest on the paths that answer ``False`` to
:meth:`~pathlib.Path.exists` while something is nevertheless there.  Those are
pinned as measurements of their own, because they are the reason the check is
not written as ``exists``.
"""

import os
from pathlib import Path

import cspyce
import numpy as np
import pytest
from tests.spindoctor.cli.ck.conftest import (
    CASSINI_CK_FRAME_ID,
    ET0,
    TICKS_PER_SECOND,
    KernelPool,
)

from spindoctor.cli.ck.comments import (
    COMMENT_MAX_LINE_CHARS,
    read_comment_area,
    reserved_comment_chars,
)
from spindoctor.cli.ck.kernel_file import (
    check_ck_file,
    check_output_paths,
    first_data_record,
    write_ck_file,
)
from spindoctor.cli.ck.segment import CkSegment

# Enough comment to need more than one 1024-character record, so that an
# under-reserved file is measurably different from an adequately reserved one.
_COMMENT_LINES = tuple(f'{"comment line":<70} {at:03d}' for at in range(40))


def _segment(segid: str = 'image', *, offset_s: float = 0.0) -> CkSegment:
    """Build a three-record segment that needs no kernel to construct.

    Parameters:
        segid: Segment identifier.
        offset_s: Seconds to shift the segment's window by, so that two
            segments in one file cover disjoint epochs.

    Returns:
        The segment.
    """
    ticks = (np.array([ET0, ET0 + 1.0, ET0 + 2.0]) + offset_s) * TICKS_PER_SECOND
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (3, 1))
    return CkSegment(
        ck_frame_id=CASSINI_CK_FRAME_ID, segid=segid, sclkdp=ticks, quats=quats, avvs=None
    )


def test_the_file_carries_its_segment(tmp_path: Path) -> None:
    """The written file describes the object the segment named."""
    path = tmp_path / 'orig_nav.bc'
    write_ck_file(path, [_segment()], _COMMENT_LINES)
    assert [int(value) for value in cspyce.ckobj(str(path))] == [CASSINI_CK_FRAME_ID]


def test_the_file_carries_every_segment(tmp_path: Path) -> None:
    """Two images in one file are two segments in it."""
    path = tmp_path / 'orig_nav.bc'
    write_ck_file(path, [_segment('a'), _segment('b', offset_s=100.0)], _COMMENT_LINES)
    windows = cspyce.ckcov(str(path), CASSINI_CK_FRAME_ID, False, 'SEGMENT', 0.0, 'SCLK')
    assert len(windows) == 4


def test_the_file_carries_its_comments(tmp_path: Path) -> None:
    """The comment area reads back exactly what it was given."""
    path = tmp_path / 'orig_nav.bc'
    write_ck_file(path, [_segment()], _COMMENT_LINES)
    assert read_comment_area(path) == _COMMENT_LINES


def _write_with_reservation(path: Path, ncomch: int) -> tuple[int, int]:
    """Write a commented C-kernel with a chosen reservation, measuring the move.

    Parameters:
        path: The file to create.
        ncomch: Characters to reserve for comments.

    Returns:
        The first data record before and after the comments were added.
    """
    handle = int(cspyce.ckopn(str(path), path.name, ncomch))
    try:
        segment = _segment()
        cspyce.ckw03(
            handle,
            segment.begtim,
            segment.endtim,
            segment.ck_frame_id,
            'J2000',
            False,
            segment.segid,
            np.asarray(segment.sclkdp),
            np.asarray(segment.quats),
            np.zeros((segment.record_count, 3)),
            1,
            [segment.begtim],
        )
    finally:
        cspyce.ckcls(handle)
    before = first_data_record(path)
    daf = int(cspyce.dafopw(str(path)))
    try:
        cspyce.dafac(daf, list(_COMMENT_LINES))
    finally:
        cspyce.dafcls(daf)
    return before, first_data_record(path)


def test_an_adequate_reservation_leaves_the_data_where_it_was(tmp_path: Path) -> None:
    """The reservation the writer computes is large enough for its own comments."""
    before, after = _write_with_reservation(
        tmp_path / 'ok_nav.bc', reserved_comment_chars(_COMMENT_LINES)
    )
    assert before == after


def test_an_absent_reservation_moves_the_data(tmp_path: Path) -> None:
    """Which is the failure the reservation exists to avoid, and it is silent."""
    before, after = _write_with_reservation(tmp_path / 'none_nav.bc', 0)
    assert before != after


def test_a_half_reservation_moves_the_data(tmp_path: Path) -> None:
    """A reservation that is merely too small fails the same way as none at all."""
    before, after = _write_with_reservation(
        tmp_path / 'half_nav.bc', reserved_comment_chars(_COMMENT_LINES) // 4
    )
    assert before != after


def test_the_written_file_reserves_what_the_writer_computed(tmp_path: Path) -> None:
    """The file the writer produces is the one whose data did not move."""
    written = tmp_path / 'written_nav.bc'
    write_ck_file(written, [_segment()], _COMMENT_LINES)
    reference = tmp_path / 'reference_nav.bc'
    before, _after = _write_with_reservation(reference, reserved_comment_chars(_COMMENT_LINES))
    assert first_data_record(written) == before


def test_a_file_with_no_segments_is_refused(tmp_path: Path) -> None:
    """SPICE refuses to close one, and it would claim a correction it lacks."""
    path = tmp_path / 'empty_nav.bc'
    with pytest.raises(ValueError, match='no segments'):
        write_ck_file(path, [], _COMMENT_LINES)


def test_a_file_with_no_segments_is_not_created(tmp_path: Path) -> None:
    """The refusal happens before anything is opened, so nothing is left behind."""
    path = tmp_path / 'empty_nav.bc'
    with pytest.raises(ValueError, match='no segments'):
        write_ck_file(path, [], _COMMENT_LINES)
    assert not path.exists()


def test_a_file_with_no_comments_is_refused(tmp_path: Path) -> None:
    """A corrected kernel that says nothing about itself is not written."""
    path = tmp_path / 'silent_nav.bc'
    with pytest.raises(ValueError, match='no comment lines'):
        write_ck_file(path, [_segment()], [])


@pytest.mark.parametrize(
    ('line', 'message'),
    [
        ('x' * (COMMENT_MAX_LINE_CHARS + 1), 'longer than the'),
        ('trailing space ', 'ends in whitespace'),
        ('embedded\ttab', 'non-printing character'),
    ],
    ids=['too-long', 'trailing-whitespace', 'embedded-tab'],
)
def test_an_unstorable_comment_line_is_refused_before_the_file_is_opened(
    tmp_path: Path, line: str, message: str
) -> None:
    """A line SPICE cannot store stops the write, and leaves nothing behind.

    The comment area is written after ``ckcls`` has already succeeded, so a
    line judged there would leave a complete, furnishable kernel with the right
    coverage and an empty comment area -- indistinguishable from a good product
    except for the provenance record -- and the guard against overwriting an
    existing kernel would then block the obvious re-run.

    Parameters:
        line: A comment line SPICE cannot store and read back unchanged.
        message: Text the refusal must name.
    """
    path = tmp_path / 'orig_nav.bc'
    with pytest.raises(ValueError, match=message):
        write_ck_file(path, [_segment()], [*_COMMENT_LINES, line])
    assert not path.exists()


def test_a_name_too_long_to_be_the_internal_name_is_refused(tmp_path: Path) -> None:
    """A truncated internal name identifies a different file."""
    path = tmp_path / ('n' * 60 + '_nav.bc')
    with pytest.raises(ValueError, match='internal name'):
        write_ck_file(path, [_segment()], _COMMENT_LINES)


def test_writing_over_an_existing_file_is_refused(tmp_path: Path) -> None:
    """Regeneration replaces a corrected kernel; it never appends to one.

    Said in words rather than left to SPICE, which reports only an operating
    system status number for it.
    """
    path = tmp_path / 'orig_nav.bc'
    write_ck_file(path, [_segment()], _COMMENT_LINES)
    with pytest.raises(ValueError, match='already exists'):
        write_ck_file(path, [_segment()], _COMMENT_LINES)


def test_a_refused_overwrite_leaves_the_existing_file_alone(tmp_path: Path) -> None:
    """The refusal happens before anything is opened, so nothing is truncated."""
    path = tmp_path / 'orig_nav.bc'
    write_ck_file(path, [_segment()], _COMMENT_LINES)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        write_ck_file(path, [_segment()], _COMMENT_LINES)
    assert path.read_bytes() == before


def test_the_kernel_reads_with_no_writer_code_present(pool: KernelPool) -> None:
    """A plain furnish and a plain lookup answer inside the written window."""
    path = pool.root / 'plain_nav.bc'
    write_ck_file(path, [_segment()], _COMMENT_LINES)
    pool.furnish(path)
    cmat, _clkout = cspyce.ckgp(CASSINI_CK_FRAME_ID, (ET0 + 1.0) * TICKS_PER_SECOND, 0.0, 'J2000')
    assert np.allclose(np.asarray(cmat), np.eye(3))


def test_a_name_exactly_as_long_as_the_field_is_accepted(tmp_path: Path) -> None:
    """The bound is the last name SPICE stores whole, not the first it truncates."""
    path = tmp_path / ('n' * 53 + '_nav.bc')
    assert len(path.name) == 60
    write_ck_file(path, [_segment()], _COMMENT_LINES)
    assert path.exists()


def test_a_name_one_character_longer_is_refused(tmp_path: Path) -> None:
    """And the first it truncates is one character beyond that."""
    path = tmp_path / ('n' * 54 + '_nav.bc')
    assert len(path.name) == 61
    with pytest.raises(ValueError, match='internal name'):
        write_ck_file(path, [_segment()], _COMMENT_LINES)


def test_writing_through_a_symbolic_link_is_refused(tmp_path: Path) -> None:
    """The corrected kernel would land wherever the link points, not here."""
    target = tmp_path / 'elsewhere.bc'
    write_ck_file(target, [_segment()], _COMMENT_LINES)
    link = tmp_path / 'orig_nav.bc'
    link.symlink_to(target)
    with pytest.raises(ValueError, match='is a symbolic link'):
        write_ck_file(link, [_segment()], _COMMENT_LINES)


def test_writing_through_a_dangling_symbolic_link_is_refused(tmp_path: Path) -> None:
    """``Path.exists`` follows the link and reports the absent target as absent."""
    link = tmp_path / 'orig_nav.bc'
    link.symlink_to(tmp_path / 'never_written.bc')
    with pytest.raises(ValueError, match='is a symbolic link'):
        write_ck_file(link, [_segment()], _COMMENT_LINES)


def test_a_refused_dangling_link_creates_no_target(tmp_path: Path) -> None:
    """Which is the harm: the write would create a file outside the directory."""
    target = tmp_path / 'never_written.bc'
    link = tmp_path / 'orig_nav.bc'
    link.symlink_to(target)
    with pytest.raises(ValueError, match='is a symbolic link'):
        write_ck_file(link, [_segment()], _COMMENT_LINES)
    assert not target.exists()


# ---------------------------------------------------------------------------
# Judging a whole run's destinations before any of them is written
# ---------------------------------------------------------------------------


def test_a_clean_set_of_paths_is_accepted(tmp_path: Path) -> None:
    """Nothing known in advance stops any of these three writes."""
    check_output_paths([tmp_path / f'orig_{at}_nav.bc' for at in range(3)])


def test_an_empty_set_of_paths_is_accepted() -> None:
    """A run that corrects nothing writes no kernel and has nothing to refuse."""
    check_output_paths([])


def _occupied(tmp_path: Path, make: str) -> Path:
    """Occupy one path in the way named, and return it.

    Parameters:
        tmp_path: The directory to work in.
        make: What to put at the path.

    Returns:
        The occupied path.
    """
    path = tmp_path / 'orig_b_nav.bc'
    if make == 'file':
        path.write_bytes(b'not a kernel')
    elif make == 'directory':
        path.mkdir()
    elif make == 'dangling-link':
        path.symlink_to(tmp_path / 'no_such_file.bc')
    elif make == 'link-to-file':
        target = tmp_path / 'target.bc'
        target.write_bytes(b'not a kernel')
        path.symlink_to(target)
    elif make == 'link-to-itself':
        path.symlink_to(path)
    elif make == 'fifo':
        os.mkfifo(path)
    else:
        raise AssertionError(f'unknown occupant {make!r}')
    return path


@pytest.mark.parametrize(
    ('make', 'message'),
    [
        ('file', 'already exists'),
        ('directory', 'already exists'),
        ('fifo', 'already exists'),
        ('dangling-link', 'is a symbolic link'),
        ('link-to-file', 'is a symbolic link'),
        ('link-to-itself', 'is a symbolic link'),
    ],
    ids=['file', 'directory', 'fifo', 'dangling-link', 'link-to-file', 'link-to-itself'],
)
def test_an_occupied_path_is_refused(tmp_path: Path, make: str, message: str) -> None:
    """Occupied is occupied however it got that way, and two of these fool ``exists``.

    A dangling link and a link to itself both report ``False`` from
    :meth:`~pathlib.Path.exists`, the first because the target is absent and
    the second because resolving it loops.  A write through either would create
    a file, in the first case outside the output directory entirely.

    Parameters:
        make: What to put at the path.
        message: Text the refusal must name.
    """
    path = _occupied(tmp_path, make)
    with pytest.raises(ValueError, match=message):
        check_output_paths([tmp_path / 'orig_a_nav.bc', path])


@pytest.mark.parametrize(
    'make', ['dangling-link', 'link-to-itself'], ids=['dangling-link', 'link-to-itself']
)
def test_the_path_that_fools_exists_is_the_one_refused(tmp_path: Path, make: str) -> None:
    """Pinned as a measurement, since it is why the check is not ``Path.exists``.

    Parameters:
        make: What to put at the path.
    """
    assert not _occupied(tmp_path, make).exists()


def test_a_refusal_names_the_path_that_failed(tmp_path: Path) -> None:
    """An operator has to know which file to move, not that one of them is there."""
    occupied = _occupied(tmp_path, 'file')
    with pytest.raises(ValueError, match=r'orig_b_nav\.bc'):
        check_output_paths([tmp_path / 'orig_a_nav.bc', occupied])


def test_a_refusal_names_every_path_that_failed(tmp_path: Path) -> None:
    """Otherwise the set is cleared one rerun at a time."""
    first = tmp_path / 'orig_a_nav.bc'
    first.write_bytes(b'not a kernel')
    second = tmp_path / 'orig_b_nav.bc'
    second.write_bytes(b'not a kernel')
    with pytest.raises(ValueError) as refusal:
        check_output_paths([first, second])
    assert 'orig_a_nav.bc' in str(refusal.value)
    assert 'orig_b_nav.bc' in str(refusal.value)


def test_a_path_named_twice_is_refused(tmp_path: Path) -> None:
    """No per-file check can see it: the second write replaces the first file."""
    path = tmp_path / 'orig_nav.bc'
    with pytest.raises(ValueError, match='named twice'):
        check_output_paths([path, path])


def test_a_path_named_twice_is_reported_once(tmp_path: Path) -> None:
    """The repeat is one fault, not one per repetition of an otherwise fine path."""
    path = tmp_path / 'orig_nav.bc'
    with pytest.raises(ValueError) as refusal:
        check_output_paths([path, path])
    assert str(refusal.value).count('named twice') == 1


def test_one_file_reached_two_ways_through_dot_dot_is_refused(tmp_path: Path) -> None:
    """Comparing the spellings would miss it, and the second write would win."""
    (tmp_path / 'inner').mkdir()
    with pytest.raises(ValueError, match='named twice'):
        check_output_paths([tmp_path / 'orig_nav.bc', tmp_path / 'inner' / '..' / 'orig_nav.bc'])


def test_one_file_reached_through_a_linked_directory_is_refused(tmp_path: Path) -> None:
    """A link at the directory is a second spelling too, and only a resolve sees it."""
    (tmp_path / 'real').mkdir()
    (tmp_path / 'linked').symlink_to(tmp_path / 'real')
    with pytest.raises(ValueError, match='named twice'):
        check_output_paths([tmp_path / 'real' / 'orig_nav.bc', tmp_path / 'linked' / 'orig_nav.bc'])


def test_two_files_under_one_linked_directory_are_accepted(tmp_path: Path) -> None:
    """Resolving the directory must not merge paths that name different files."""
    (tmp_path / 'real').mkdir()
    (tmp_path / 'linked').symlink_to(tmp_path / 'real')
    check_output_paths([tmp_path / 'real' / 'orig_a_nav.bc', tmp_path / 'linked' / 'orig_b_nav.bc'])


@pytest.mark.parametrize('links', [1, 2], ids=['self-loop', 'two-link-loop'])
def test_a_looping_directory_link_is_refused_by_name(tmp_path: Path, links: int) -> None:
    """Resolving it raises, and that must not become the whole call's answer.

    An operator has to be told which output path failed and what about it, not
    handed the exception the duplicate check happened to hit while deciding
    whether two paths were the same file.

    Parameters:
        links: How many links the loop is made of.
    """
    if links == 1:
        (tmp_path / 'loop').symlink_to(tmp_path / 'loop')
    else:
        (tmp_path / 'loop').symlink_to(tmp_path / 'other')
        (tmp_path / 'other').symlink_to(tmp_path / 'loop')
    with pytest.raises(ValueError, match='is not a directory'):
        check_output_paths([tmp_path / 'loop' / 'orig_nav.bc'])


def test_a_link_at_the_output_path_is_not_resolved_away(tmp_path: Path) -> None:
    """Only the directory is resolved; resolving the basename would follow the link.

    A link whose target is a free path would then look like a free path, and
    the write through it would be exactly the harm the occupancy check refuses.
    """
    link = tmp_path / 'orig_nav.bc'
    link.symlink_to(tmp_path / 'free.bc')
    with pytest.raises(ValueError, match='is a symbolic link'):
        check_output_paths([link])


def test_a_name_exactly_as_long_as_the_field_passes_the_set_check(tmp_path: Path) -> None:
    """The same bound the file writer applies, and the same last accepted name."""
    path = tmp_path / ('n' * 53 + '_nav.bc')
    assert len(path.name) == 60
    check_output_paths([path])


def test_a_name_one_character_longer_fails_the_set_check(tmp_path: Path) -> None:
    """So a set holding one is refused before the shorter-named files are written."""
    path = tmp_path / ('n' * 54 + '_nav.bc')
    assert len(path.name) == 61
    with pytest.raises(ValueError, match='internal name'):
        check_output_paths([tmp_path / 'orig_a_nav.bc', path])


def test_a_missing_directory_is_refused(tmp_path: Path) -> None:
    """``ckopn`` reports an operating system status number and no reason at all."""
    with pytest.raises(ValueError, match='is not a directory'):
        check_output_paths([tmp_path / 'absent' / 'orig_nav.bc'])


def test_a_file_where_the_directory_should_be_is_refused(tmp_path: Path) -> None:
    """A path whose parent is a regular file names no file that can be created."""
    blocker = tmp_path / 'blocker'
    blocker.write_bytes(b'not a directory')
    with pytest.raises(ValueError, match='is not a directory'):
        check_output_paths([blocker / 'orig_nav.bc'])


@pytest.mark.skipif(os.geteuid() == 0, reason='the superuser writes into a mode 0o500 directory')
def test_an_unwritable_directory_is_refused(tmp_path: Path) -> None:
    """The whole set fails, rather than the first file failing inside ``ckopn``."""
    directory = tmp_path / 'readonly'
    directory.mkdir(mode=0o500)
    try:
        with pytest.raises(ValueError, match='cannot be written to'):
            check_output_paths([directory / 'orig_nav.bc'])
    finally:
        directory.chmod(0o700)


@pytest.mark.parametrize(
    'name', ['orig\x00nav.bc', 'orig\nnav.bc', 'orig\tnav.bc'], ids=['null', 'newline', 'tab']
)
def test_a_path_holding_a_non_printing_character_is_refused(tmp_path: Path, name: str) -> None:
    """SPICE is handed the name as a C string and the meta-kernel writes it as text.

    A null truncates the first and no non-printing character survives the
    second, so such a path reaches the consumer as a file nobody asked for.

    Parameters:
        name: A basename holding one character that cannot be written down.
    """
    with pytest.raises(ValueError, match='non-printing character'):
        check_output_paths([tmp_path / name])


def test_a_null_in_a_path_is_not_reported_as_an_absent_file(tmp_path: Path) -> None:
    """Pinned as a measurement: ``lexists`` answers False rather than raising."""
    assert not os.path.lexists(tmp_path / 'orig\x00nav.bc')


@pytest.mark.parametrize('spelling', ['/', '.', ''], ids=['root', 'here', 'empty'])
def test_a_path_naming_no_file_is_refused(spelling: str) -> None:
    """Every one of these has an empty basename, so none of them is a file to write.

    Parameters:
        spelling: A path that names a directory or nothing at all.
    """
    with pytest.raises(ValueError, match='names no file'):
        check_output_paths([Path(spelling)])


def test_a_relative_path_is_judged_against_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It is what the write would do, so it is what the check has to do.

    Parameters:
        tmp_path: Made the working directory for the duration.
        monkeypatch: Used to change and restore the working directory.
    """
    (tmp_path / 'orig_nav.bc').write_bytes(b'not a kernel')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='already exists'):
        check_output_paths([Path('orig_nav.bc')])


def test_the_set_check_and_the_file_check_refuse_the_same_path(tmp_path: Path) -> None:
    """The per-file refusal is the last line of defense, not a weaker one."""
    link = tmp_path / 'orig_nav.bc'
    link.symlink_to(tmp_path / 'no_such_file.bc')
    with pytest.raises(ValueError, match='is a symbolic link'):
        check_output_paths([link])
    with pytest.raises(ValueError, match='is a symbolic link'):
        check_ck_file(link, [_segment()], _COMMENT_LINES)


def test_a_failed_segment_write_reports_its_own_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not the one closing a segment-less C-kernel raises on the way out.

    ``write_segment`` is replaced rather than provoked: SPICE refuses a record
    set this module's own dataclass refuses first, so there is no segment that
    reaches ``ckw03`` and fails there.  What is under test is which of the two
    errors the caller is left holding.
    """

    def _refuse(handle: int, segment: CkSegment) -> None:
        """Fail the way a rejected record set would."""
        raise ValueError('the record set was rejected')

    monkeypatch.setattr('spindoctor.cli.ck.kernel_file.write_segment', _refuse)
    with pytest.raises(ValueError, match='the record set was rejected'):
        write_ck_file(tmp_path / 'broken_nav.bc', [_segment()], _COMMENT_LINES)


def _fail_the_write(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Make the next segment write fail, and record the handle it opened.

    Parameters:
        monkeypatch: Used to replace the two collaborators.

    Returns:
        The list the opened handle is appended to.
    """
    opened: list[int] = []
    real_ckopn = cspyce.ckopn

    def _record(fname: str, ifname: str, ncomch: int) -> int:
        """Open the file as usual and remember the handle."""
        handle = int(real_ckopn(fname, ifname, ncomch))
        opened.append(handle)
        return handle

    def _refuse(handle: int, segment: CkSegment) -> None:
        """Fail the way a rejected record set would."""
        raise ValueError('the record set was rejected')

    monkeypatch.setattr(cspyce, 'ckopn', _record)
    monkeypatch.setattr('spindoctor.cli.ck.kernel_file.write_segment', _refuse)
    return opened


def test_a_failed_segment_write_still_closes_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leaked DAF handle breaks an unrelated open later, naming neither.

    Asserted on the handle itself rather than on a symptom: SPICE answers
    ``dafhsf`` for a handle it still has open and refuses it once the file is
    closed.  Measured for the same reason: ``ckcls`` on a segment-less
    C-kernel raises *and leaves the file open*, so the obvious cleanup does
    not do this.
    """
    opened = _fail_the_write(monkeypatch)
    with pytest.raises(ValueError, match='the record set was rejected'):
        write_ck_file(tmp_path / 'broken_nav.bc', [_segment()], _COMMENT_LINES)
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match='DAFNOSUCHHANDLE'):
        cspyce.dafhsf(opened[0])


def test_a_failed_segment_write_leaves_no_file_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the obvious next step -- run again -- fails on the leftover."""
    _fail_the_write(monkeypatch)
    with pytest.raises(ValueError, match='the record set was rejected'):
        write_ck_file(tmp_path / 'broken_nav.bc', [_segment()], _COMMENT_LINES)
    monkeypatch.undo()
    assert not (tmp_path / 'broken_nav.bc').exists()


def test_a_run_again_after_a_failed_write_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which is the point of removing it."""
    _fail_the_write(monkeypatch)
    with pytest.raises(ValueError, match='the record set was rejected'):
        write_ck_file(tmp_path / 'broken_nav.bc', [_segment()], _COMMENT_LINES)
    monkeypatch.undo()
    write_ck_file(tmp_path / 'broken_nav.bc', [_segment()], _COMMENT_LINES)
    assert (tmp_path / 'broken_nav.bc').exists()
