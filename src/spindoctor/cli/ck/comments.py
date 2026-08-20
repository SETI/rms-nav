"""The comment area a corrected C-kernel carries about itself.

A C-kernel that has been furnished is anonymous: it answers attitude queries
and says nothing about where the attitude came from.  SPICE gives every binary
kernel a comment area for exactly that, and this module fills one -- what wrote
the file, against what configuration, which original kernel it corrects, which
spacecraft clock its time tags are encoded against, and one line per image
naming what that image measured and how well.

The measurement lines carry the same facts the report carries, so a reader who
has the kernel but not the report is no worse informed about the images inside
it.  They are laid out in fixed columns rather than terse, because a comment
area is read by people.

Two mechanical facts govern what may be written, both measured against the
installed toolkit rather than assumed:

- A comment line longer than 255 characters is stored by ``dafac`` without
  complaint and then cannot be read back at all: ``dafec`` reads into a
  255-character buffer and raises ``SPICE(COMMENTTOOLONG)`` on the first line
  that overflows it, which makes the whole comment area unreadable rather than
  the one line lossy.
- Trailing whitespace does not survive the round trip, and a non-printing
  character is refused outright by ``dafac`` -- after the segments have already
  been written, so the file is left with a comment area it was meant to have
  and does not.

Both are refused here, before a file is opened, so that a file this package
writes reads back exactly what it was given.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cspyce

from spindoctor.cli.ck.report import ImageReportFacts

# The longest comment line ``dafec`` can read back.  A longer line is stored
# without complaint and then makes the entire comment area unreadable.
COMMENT_MAX_LINE_CHARS = 255

# How much room to reserve beyond the comment text itself.  The reservation is
# made when the file is opened and the text is known exactly by then, so the
# slack is not there to cover an estimate: it is there so that a later append
# -- a second ``dafac`` adding provenance, say -- still fits without SPICE
# having to shift every data record in the file to make room.  A quarter again
# of the text, plus one whole comment record, which is the granularity SPICE
# reserves in anyway.
_COMMENT_SLACK_DIVISOR = 4
_COMMENT_SLACK_CHARS = 1024

# The fixed-column layout of one image line.  The name column fits the longest
# name a segment identifier can hold, the time column an ISO calendar time with
# milliseconds, and the rest are wide enough for the values the pipeline
# records; a value that overflows its column widens the line rather than being
# truncated, and the line-length guard catches the pathological case.
_NAME_WIDTH = 40
_UTC_WIDTH = 23
_NUMBER_WIDTH = 10
_RANK_WIDTH = 12
_STATUS_WIDTH = 12

# What a cell holds when the metadata recorded no value for it.  Spelled rather
# than left blank so that a column of numbers with one gap in it is legible.
_ABSENT = '-'

# What marks a free-text field cut to fit its line.
_ELISION = '...'

_HEADER_TITLE = 'SpinDoctor corrected-pointing C-kernel'

# Said in the file itself because a furnished kernel is the only thing many
# consumers will ever see of this system, and a corrected kernel that is read
# as a replacement for the original gives no pointing at all outside the
# exposures it corrects.
_SCOPE_NOTE = (
    'Each segment corrects one exposure and covers only that exposure.  Outside those',
    'windows the original kernel named above still supplies the pointing, so it must',
    'remain furnished: this file is an overlay, not a replacement.  Pointing is claimed',
    'at the record epochs; between them the segment interpolates.',
)


@dataclass(frozen=True)
class CommentArea:
    """What a corrected C-kernel says about itself.

    Parameters:
        generator_version: Version of the software that wrote the file.
        configuration_hash: Digest of the configuration it ran under.
        baseline_basenames: Basenames of the original C-kernels whose pointing
            the segments correct.  A corrected file mirrors exactly one
            original, so this normally holds one name.
        sclk_basename: Basename of the spacecraft clock kernel the time tags
            are encoded against, which is the one navigation used.
        images: The images whose segments the file carries, in the order they
            should be listed.

    Raises:
        ValueError: if any field is empty, or if no image is named -- a
            corrected file with no segments is never written, so a comment area
            describing one is a defect rather than an empty case.
    """

    generator_version: str
    configuration_hash: str
    baseline_basenames: tuple[str, ...]
    sclk_basename: str
    images: tuple[ImageReportFacts, ...]

    def __post_init__(self) -> None:
        """Refuse a description that describes nothing."""
        for label, value in (
            ('generator_version', self.generator_version),
            ('configuration_hash', self.configuration_hash),
            ('sclk_basename', self.sclk_basename),
        ):
            if len(value) == 0:
                raise ValueError(f'{label} is empty; the comment area would name no {label}')
        if len(self.baseline_basenames) == 0:
            raise ValueError(
                'baseline_basenames is empty; a corrected kernel names the original it corrects'
            )
        for basename in self.baseline_basenames:
            if len(basename) == 0:
                raise ValueError('a baseline kernel basename is empty')
        if len(self.images) == 0:
            raise ValueError(
                'the comment area names no images; a corrected kernel holding no segments is '
                'never written, so describing one would describe a file that does not exist'
            )


def build_comment_lines(area: CommentArea) -> tuple[str, ...]:
    """Render one corrected kernel's comment area.

    Parameters:
        area: What the file should say about itself.

    Returns:
        The lines, ready for :func:`write_comment_area`.

    Raises:
        ValueError: for anything :func:`check_comment_lines` refuses -- a
            rendered line longer than :data:`COMMENT_MAX_LINE_CHARS`, one
            ending in whitespace, or one holding a character outside printable
            ASCII.  A free-text reason is elided to fit, so only a field that
            overflows its own column can reach the length refusal.
    """
    lines: list[str] = [
        _HEADER_TITLE,
        '',
        f'Generator version:    {area.generator_version}',
        f'Configuration hash:   {area.configuration_hash}',
        f'Baseline kernel(s):   {", ".join(area.baseline_basenames)}',
        f'Spacecraft clock:     {area.sclk_basename}',
        f'Images corrected:     {len(area.images)}',
        '',
        *_SCOPE_NOTE,
        '',
        _image_header(),
    ]
    lines.extend(_image_line(facts) for facts in area.images)
    for line in lines:
        _check_line(line)
    return tuple(lines)


def check_comment_lines(lines: Sequence[str]) -> None:
    """Refuse a comment area SPICE cannot store and read back unchanged.

    This is the whole of what may be rejected about a comment area, gathered
    into one call so that a caller can make the judgment before it opens a
    file.  Doing it afterwards leaves a complete, furnishable kernel with an
    empty comment area behind: the segments are written, ``ckcls`` succeeds,
    and only then does the comment write refuse -- so the file is
    indistinguishable from a good product except for the provenance record the
    comment area exists to carry.

    Parameters:
        lines: The comment lines to check.

    Raises:
        ValueError: if a line is longer than :data:`COMMENT_MAX_LINE_CHARS`, if
            it ends in whitespace, or if it holds a character outside printable
            ASCII.
    """
    for line in lines:
        _check_line(line)


def _check_line(line: str) -> None:
    """Refuse a comment line SPICE cannot store and read back unchanged.

    Parameters:
        line: The line to check.

    Raises:
        ValueError: if it is longer than :data:`COMMENT_MAX_LINE_CHARS`, if it
            ends in whitespace, which the read back silently drops, or if it
            holds a character outside printable ASCII, which ``dafac`` refuses
            once the segments have already been written.
    """
    if len(line) > COMMENT_MAX_LINE_CHARS:
        raise ValueError(
            f'comment line is {len(line)} characters, longer than the '
            f'{COMMENT_MAX_LINE_CHARS} SPICE can read back, which would make the whole comment '
            f'area unreadable: {line!r}'
        )
    if line != line.rstrip():
        raise ValueError(f'comment line ends in whitespace, which is not read back: {line!r}')
    for character in line:
        # ``isprintable`` alone would pass accented letters and smart quotes,
        # which are printable to Python and refused by ``dafac`` all the same:
        # SPICE accepts only printable ASCII in a comment area.
        if not (character.isascii() and character.isprintable()):
            raise ValueError(
                f'comment line holds the character {character!r}, which is outside the '
                f'printable ASCII SPICE accepts and which it refuses once the segments are '
                f'written: {line!r}'
            )


def _image_header() -> str:
    """Return the header naming the columns of the per-image lines.

    Returns:
        The header line.
    """
    return (
        f'{"image_name":<{_NAME_WIDTH}} {"utc":<{_UTC_WIDTH}} '
        f'{"offset_dv":>{_NUMBER_WIDTH}} {"offset_du":>{_NUMBER_WIDTH}} '
        f'{"sigma_dv":>{_NUMBER_WIDTH}} {"sigma_du":>{_NUMBER_WIDTH}} '
        f'{"confidence":>{_NUMBER_WIDTH}} {"rank":<{_RANK_WIDTH}} '
        f'{"status":<{_STATUS_WIDTH}} status_reason'
    ).rstrip()


def _image_line(facts: ImageReportFacts) -> str:
    """Return one image's line of the comment area.

    Parameters:
        facts: What the image's metadata says about it.

    Returns:
        The line, in the same columns as :func:`_image_header`.  The free-text
        reason is elided to what the line can hold: it is the one field whose
        length the pipeline does not bound, a comment area is read rather than
        parsed, and the report carries the full value -- so a verbose reason
        must not refuse the whole file.
    """
    prefix = (
        f'{facts.image_name:<{_NAME_WIDTH}} {_text(facts.utc):<{_UTC_WIDTH}} '
        f'{_number(facts.offset_dv):>{_NUMBER_WIDTH}} '
        f'{_number(facts.offset_du):>{_NUMBER_WIDTH}} '
        f'{_number(facts.sigma_dv):>{_NUMBER_WIDTH}} '
        f'{_number(facts.sigma_du):>{_NUMBER_WIDTH}} '
        f'{_number(facts.confidence):>{_NUMBER_WIDTH}} '
        f'{_text(facts.confidence_rank):<{_RANK_WIDTH}} '
        f'{_text(facts.status):<{_STATUS_WIDTH}} '
    )
    reason = _elided(_text(facts.status_reason), COMMENT_MAX_LINE_CHARS - len(prefix))
    return f'{prefix}{reason}'.rstrip()


def _elided(text: str, room: int) -> str:
    """Shorten free text to what its line has room for.

    Parameters:
        text: The text to place.
        room: How many characters the line can still hold.

    Returns:
        The text unchanged when it fits, or cut to ``room`` with a trailing
        :data:`_ELISION` marking the cut.  When ``room`` cannot even hold the
        marker -- a fixed column overflowed its width, which only malformed
        metadata produces -- the text is returned unchanged so the line-length
        refusal reports the defect instead of this function hiding it.
    """
    if len(text) <= room or room < len(_ELISION):
        return text
    return text[: room - len(_ELISION)] + _ELISION


def _number(value: float | None) -> str:
    """Render one measured number for the comment area.

    Parameters:
        value: The value, or ``None`` when the metadata recorded none.

    Returns:
        The value to four decimals, or a marker for a value that was never
        measured.  Four decimals is the pipeline's own pixel resolution, and a
        comment area is read rather than parsed, so it is rendered for reading;
        the report carries the unrounded value.
    """
    if value is None:
        return _ABSENT
    return f'{value:.4f}'


def _text(value: str | None) -> str:
    """Render one text field for the comment area.

    Parameters:
        value: The value, or ``None`` when the metadata recorded none.

    Returns:
        The value, or a marker for a field that was never recorded.
    """
    if value is None or len(value) == 0:
        return _ABSENT
    return value


def reserved_comment_chars(lines: Sequence[str]) -> int:
    """Return how many comment characters a file should be opened with.

    SPICE counts a comment line as its characters plus the terminator, and
    reserves whole 1024-character records; a file opened with too few has to be
    rewritten -- every data record shifted -- when the comments are added, so
    the reservation is deliberately larger than the text.

    Parameters:
        lines: The comment lines the file will carry.

    Returns:
        The number of characters to reserve, which is at least the text's own
        length even for an empty comment.
    """
    chars = sum(len(line) + 1 for line in lines)
    return chars + chars // _COMMENT_SLACK_DIVISOR + _COMMENT_SLACK_CHARS


def write_comment_area(path: Path, lines: Sequence[str]) -> None:
    """Add a comment area to a C-kernel that has already been written.

    The file is reopened for writing as a plain DAF, which is what a C-kernel
    is, since ``ckopn`` reserves the comment area but does not fill it.

    Parameters:
        path: The kernel to add the comments to.
        lines: The comment lines.

    Raises:
        ValueError: if no line is given, which SPICE refuses, or if a line is
            one SPICE cannot store and read back unchanged.
        OSError: if the file cannot be opened for writing.
    """
    if len(lines) == 0:
        raise ValueError(f'no comment lines to write to {path}; SPICE refuses an empty comment')
    check_comment_lines(lines)
    handle = int(cspyce.dafopw(str(path)))
    try:
        cspyce.dafac(handle, list(lines))
    finally:
        cspyce.dafcls(handle)


def read_comment_area(path: Path) -> tuple[str, ...]:
    """Read back the comment area of a C-kernel.

    Parameters:
        path: The kernel to read.

    Returns:
        Its comment lines, empty when it carries none.

    Raises:
        OSError: if the file cannot be opened for reading.
        MemoryError: if it holds a line longer than SPICE can read back, which
            no file this package writes does.
    """
    handle = int(cspyce.dafopr(str(path)))
    try:
        return tuple(str(line) for line in cspyce.dafec(handle))
    finally:
        cspyce.dafcls(handle)
