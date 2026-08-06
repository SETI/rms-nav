"""Hermetic tests for ``spindoctor.cli.ck.comments``.

These pin what a corrected kernel says about itself, that it says it through a
comment area SPICE reads back unchanged, and that a line SPICE would store but
not return is refused before any file is opened.
"""

from pathlib import Path

import cspyce
import numpy as np
import pytest
from tests.spindoctor.cli.ck.conftest import (
    CASSINI_CK_FRAME_ID,
    KernelPool,
)

from spindoctor.cli.ck.comments import (
    COMMENT_MAX_LINE_CHARS,
    CommentArea,
    build_comment_lines,
    read_comment_area,
    reserved_comment_chars,
    write_comment_area,
)
from spindoctor.cli.ck.report import ImageFacts

_VERSION = '0.5.2'
_CONFIG_HASH = 'c0ffee1234567890'
_BASELINE = '03236_04002ra.bc'
_SCLK = 'cas00172.tsc'


def _facts(**overrides: object) -> ImageFacts:
    """Build one image's reported facts, with fields replaced.

    Parameters:
        overrides: Fields to replace on the default fully-populated image.

    Returns:
        The facts.
    """
    defaults: dict[str, object] = {
        'image_name': 'N1484573295_1_CALIB',
        'utc': '2005-06-01T12:00:00.000',
        'et': 1.7e8,
        'sclk': '1/1484573295.118',
        'offset_dv': -3.25,
        'offset_du': 1.125,
        'sigma_dv': 0.0625,
        'sigma_du': 0.03125,
        'confidence': 0.8125,
        'confidence_rank': 'high',
        'status': 'success',
        'status_reason': 'ensemble_agreement',
    }
    defaults.update(overrides)
    return ImageFacts(**defaults)  # type: ignore[arg-type]  # a table of literals


def _area(**overrides: object) -> CommentArea:
    """Build a comment area, with fields replaced.

    Parameters:
        overrides: Fields to replace on the default one-image area.

    Returns:
        The comment area.
    """
    defaults: dict[str, object] = {
        'generator_version': _VERSION,
        'configuration_hash': _CONFIG_HASH,
        'baseline_basenames': (_BASELINE,),
        'sclk_basename': _SCLK,
        'images': (_facts(),),
    }
    defaults.update(overrides)
    return CommentArea(**defaults)  # type: ignore[arg-type]  # a table of literals


def _written_ck(path: Path, ncomch: int) -> None:
    """Write a one-segment C-kernel with a given comment reservation.

    Parameters:
        path: The file to create.
        ncomch: Characters to reserve for comments.
    """
    ticks = np.array([100.0, 200.0, 300.0])
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (3, 1))
    handle = cspyce.ckopn(str(path), path.name, ncomch)
    try:
        cspyce.ckw03(
            handle,
            100.0,
            300.0,
            CASSINI_CK_FRAME_ID,
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


# ---------------------------------------------------------------------------
# What the comment area says
# ---------------------------------------------------------------------------


def test_the_comment_names_the_generator_version() -> None:
    """A kernel says what wrote it."""
    assert any(_VERSION in line for line in build_comment_lines(_area()))


def test_the_comment_names_the_configuration_hash() -> None:
    """And under what configuration."""
    assert any(_CONFIG_HASH in line for line in build_comment_lines(_area()))


def test_the_comment_names_the_baseline_kernel() -> None:
    """And which original it corrects, which is what must stay furnished."""
    assert any(_BASELINE in line for line in build_comment_lines(_area()))


def test_the_comment_names_the_clock_kernel() -> None:
    """And which clock its time tags are encoded against."""
    assert any(_SCLK in line for line in build_comment_lines(_area()))


def test_the_comment_names_every_baseline_it_was_given() -> None:
    """A file mirroring more than one original names them all."""
    lines = build_comment_lines(_area(baseline_basenames=(_BASELINE, '03236_04002rb.bc')))
    assert any('03236_04002rb.bc' in line for line in lines)


def test_the_comment_says_the_original_is_still_needed() -> None:
    """The overlay rule is in the file, not only in the documentation."""
    assert any('overlay' in line for line in build_comment_lines(_area()))


def test_one_line_per_image() -> None:
    """Two images make two image lines, in the order given."""
    lines = build_comment_lines(
        _area(images=(_facts(image_name='A_CALIB'), _facts(image_name='B_CALIB')))
    )
    named = [line for line in lines if line.startswith(('A_CALIB', 'B_CALIB'))]
    assert len(named) == 2
    assert named[0].startswith('A_CALIB')


@pytest.mark.parametrize(
    'expected',
    [
        'N1484573295_1_CALIB',
        '2005-06-01T12:00:00.000',
        '-3.2500',
        '1.1250',
        '0.0625',
        '0.0312',
        '0.8125',
        'high',
        'success',
        'ensemble_agreement',
    ],
    ids=[
        'name',
        'utc',
        'offset-dv',
        'offset-du',
        'sigma-dv',
        'sigma-du',
        'confidence',
        'rank',
        'status',
        'status-reason',
    ],
)
def test_an_image_line_carries_its_measurement(expected: str) -> None:
    """Name, time, offset, sigma, confidence, rank, status and reason."""
    assert expected in build_comment_lines(_area())[-1]


def test_an_unmeasured_value_is_marked_rather_than_blank() -> None:
    """A gap in a column of numbers is legible rather than empty."""
    line = build_comment_lines(_area(images=(_facts(offset_dv=None, offset_du=None),)))[-1]
    assert ' - ' in line


def test_the_comment_counts_the_images() -> None:
    """The count is a cross-check against the segments the file holds."""
    lines = build_comment_lines(_area(images=(_facts(image_name='A'), _facts(image_name='B'))))
    assert any(line.startswith('Images corrected:') and line.endswith('2') for line in lines)


# ---------------------------------------------------------------------------
# What a comment area may not hold
# ---------------------------------------------------------------------------


def test_an_area_describing_no_images_is_refused() -> None:
    """A corrected kernel holding no segments is never written."""
    with pytest.raises(ValueError, match='names no images'):
        _area(images=())


def test_an_area_naming_no_baseline_is_refused() -> None:
    """A correction is meaningless without the original it corrects."""
    with pytest.raises(ValueError, match='names the original'):
        _area(baseline_basenames=())


@pytest.mark.parametrize(
    'field',
    ['generator_version', 'configuration_hash', 'sclk_basename'],
    ids=['version', 'hash', 'clock'],
)
def test_an_empty_identity_field_is_refused(field: str) -> None:
    """An empty field reads as a fact rather than as a missing one."""
    with pytest.raises(ValueError, match=f'{field} is empty'):
        _area(**{field: ''})


def test_an_empty_baseline_basename_is_refused() -> None:
    """So does an empty name inside the list."""
    with pytest.raises(ValueError, match='basename is empty'):
        _area(baseline_basenames=('',))


def test_a_line_longer_than_spice_can_read_back_is_refused() -> None:
    """SPICE stores it and then cannot return the comment area at all."""
    with pytest.raises(ValueError, match='longer than the 255'):
        build_comment_lines(_area(images=(_facts(status_reason='r' * 300),)))


def test_a_line_at_the_readable_limit_is_accepted() -> None:
    """The bound is the last length SPICE returns, not the first it refuses."""
    measured = len(build_comment_lines(_area(images=(_facts(status_reason='r'),)))[-1])
    padded = _facts(status_reason='r' * (COMMENT_MAX_LINE_CHARS - measured + 1))
    assert len(build_comment_lines(_area(images=(padded,)))[-1]) == COMMENT_MAX_LINE_CHARS


def test_a_line_one_character_over_the_limit_is_refused() -> None:
    """And the first length it refuses is one character beyond that."""
    measured = len(build_comment_lines(_area(images=(_facts(status_reason='r'),)))[-1])
    padded = _facts(status_reason='r' * (COMMENT_MAX_LINE_CHARS - measured + 2))
    with pytest.raises(ValueError, match='longer than the 255'):
        build_comment_lines(_area(images=(padded,)))


def test_a_tab_in_a_field_is_refused() -> None:
    """``dafac`` refuses a non-printing character after the segments are written."""
    with pytest.raises(ValueError, match='non-printing'):
        build_comment_lines(_area(images=(_facts(status_reason='a\tb'),)))


def test_a_line_ending_in_whitespace_is_refused() -> None:
    """Trailing whitespace is silently dropped by the read back."""
    with pytest.raises(ValueError, match='ends in whitespace'):
        write_comment_area(Path('unused.bc'), ['trailing  '])


# ---------------------------------------------------------------------------
# Writing and reading it back
# ---------------------------------------------------------------------------


def test_the_comment_area_reads_back_exactly(pool: KernelPool) -> None:
    """What was written is what ``dafopr`` / ``dafec`` return."""
    path = pool.root / 'commented.bc'
    lines = build_comment_lines(_area())
    _written_ck(path, reserved_comment_chars(lines))
    write_comment_area(path, lines)
    assert read_comment_area(path) == lines


def test_a_kernel_with_no_comments_reads_back_empty(pool: KernelPool) -> None:
    """Reading a file that carries none is not an error."""
    path = pool.root / 'bare.bc'
    _written_ck(path, 0)
    assert read_comment_area(path) == ()


def test_writing_no_comment_lines_is_refused(pool: KernelPool) -> None:
    """SPICE refuses an empty comment, so it is refused before the file opens."""
    path = pool.root / 'empty.bc'
    _written_ck(path, 1000)
    with pytest.raises(ValueError, match='no comment lines'):
        write_comment_area(path, [])


def test_the_reservation_covers_the_text_it_was_measured_from() -> None:
    """SPICE counts a line as its characters plus a terminator."""
    lines = ('a' * 10, 'b' * 20)
    assert reserved_comment_chars(lines) > sum(len(line) + 1 for line in lines)


def test_the_reservation_of_nothing_is_still_positive() -> None:
    """An empty comment still reserves the slack a later append would need."""
    assert reserved_comment_chars(()) > 0


def test_a_blank_comment_line_survives_the_round_trip(pool: KernelPool) -> None:
    """The layout uses blank lines, so they have to come back as blank lines."""
    path = pool.root / 'blanks.bc'
    lines = ('first', '', 'third')
    _written_ck(path, reserved_comment_chars(lines))
    write_comment_area(path, lines)
    assert read_comment_area(path) == lines
