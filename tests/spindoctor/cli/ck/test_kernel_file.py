"""Hermetic tests for ``spindoctor.cli.ck.kernel_file``.

The file writer's two jobs are to put the segments and the comments into one
file in the order SPICE allows, and to reserve the comment room before the file
is opened.  The reservation is what these tests work hardest on, because SPICE
does not report a reservation that was too small: it silently rewrites the file
instead, moving every data record to make room.  What is measurable is that the
first data record does not move, and that is asserted directly.
"""

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
from spindoctor.cli.ck.kernel_file import first_data_record, write_ck_file
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
