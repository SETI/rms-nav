"""The meta-kernel that loads a corrected file set in the right order.

A corrected C-kernel is an overlay: it covers only the exposures that were
navigated, and everywhere else the original it mirrors is what answers.  SPICE
resolves the overlap by load order -- the C-kernel furnished last wins where
two cover the same object and epoch -- so a user who furnishes the corrections
before the originals gets the originals' pointing everywhere and no error to
say so.

That ordering is written down here rather than left as something to know.  The
meta-kernel names every original first and every correction after, so one
``furnsh`` of it gives corrected pointing inside the navigated exposures and the
original pointing outside them.

One mechanical constraint shapes the file.  A string in a SPICE text kernel
holds at most 80 characters, and a path longer than that is silently truncated
to 80 rather than refused -- the resulting error names a file nobody asked for.
SPICE's continuation mechanism is used instead: a value ending in ``+`` is
joined to the one after it, so a long path is written as several strings.  A
path that itself ends in ``+`` cannot be written that way at all, since its last
character is indistinguishable from the marker, and it is refused by name.

Blanks need the same care, and get less of it from SPICE.  A trailing blank is
trimmed from a text kernel string, so a path whose last character is a blank
comes back one character shorter than it went in -- measured through a real
``furnsh``: a 119-character path whose last character is a space loads a
118-character name, with no error.  Such a path is refused.  A blank *inside* a
path survives on the toolkit installed here, at a join as anywhere else, but
the joins are chosen to fall on non-blanks anyway: it costs nothing, the
holdings tree holds names with spaces in them, and a name that lost a character
in the middle would reach the consumer as a file it never asked for.
"""

from collections.abc import Sequence
from pathlib import Path

from filecache import FCPath

# The longest string a SPICE text kernel holds.  A longer one is truncated to
# this length when the kernel is parsed, without any complaint.
_MAX_STRING_CHARS = 80

# The character that joins one string value to the next.  It occupies one of
# the 80, so a continued piece of a path is one shorter.
_CONTINUATION = '+'
_MAX_PIECE_CHARS = _MAX_STRING_CHARS - len(_CONTINUATION)

_HEADER = (
    'KPL/MK',
    '',
    'SpinDoctor corrected-pointing meta-kernel.',
    '',
    'The original kernels are listed first and the corrected kernels after them,',
    'so that a correction takes precedence over the original it mirrors wherever',
    'the two overlap.  Each correction covers only the exposures that were',
    'navigated; outside those windows the originals are what answer, which is why',
    'they are furnished here rather than replaced.',
    '',
)


def build_meta_kernel_lines(
    originals: Sequence[str | Path | FCPath], corrections: Sequence[str | Path | FCPath]
) -> tuple[str, ...]:
    """Render the meta-kernel that furnishes a corrected file set.

    Parameters:
        originals: Paths of the original kernels, in the order to furnish them.
        corrections: Paths of the corrected kernels, in the order to furnish
            them.  Every one is listed after every original.

    Returns:
        The lines of the meta-kernel.

    Raises:
        ValueError: if no kernel is named at all, if a path is empty, if a path
            ends in the continuation character, or if a path holds a quote or a
            non-printing character, none of which a text kernel can express
            unambiguously.
    """
    paths = [str(path) for path in (*originals, *corrections)]
    if len(paths) == 0:
        raise ValueError('a meta-kernel that names no kernels furnishes nothing')
    values: list[str] = []
    for path in paths:
        values.extend(_quoted_pieces(path))
    body = [
        '\\begindata',
        '',
        'KERNELS_TO_LOAD = (',
        *(f'   {value}' for value in values),
        ')',
        '',
        '\\begintext',
        '',
    ]
    return (*_HEADER, *body)


def _quoted_pieces(path: str) -> list[str]:
    """Split one path into the quoted text-kernel strings that spell it.

    Parameters:
        path: The path to write.

    Returns:
        One quoted string per piece, each at most 80 characters of content,
        every piece but the last carrying the continuation marker.  No piece
        ends in a blank: SPICE trims a trailing blank from a string value, and
        a piece that ended in one would lose a character out of the middle of
        the joined path.

    Raises:
        ValueError: if the path is empty, ends in the continuation character
            or in a blank, or holds a quote or a non-printing character.
    """
    if len(path) == 0:
        raise ValueError('a meta-kernel cannot name an empty path')
    if path.endswith(_CONTINUATION):
        raise ValueError(
            f'{path!r} ends in {_CONTINUATION!r}, which a meta-kernel reads as a continuation '
            f'onto the next kernel rather than as part of this one'
        )
    if path != path.rstrip():
        raise ValueError(
            f'{path!r} ends in a blank, which SPICE trims from a text kernel string; the name it '
            f'would then look for is one character shorter than the one asked for'
        )
    if "'" in path:
        raise ValueError(f'{path!r} holds a quote, which a text kernel string cannot carry')
    for character in path:
        if not character.isprintable():
            raise ValueError(f'{path!r} holds the non-printing character {character!r}')
    pieces = _split_on_non_blanks(path)
    return [
        f"'{piece}{_CONTINUATION if at < len(pieces) - 1 else ''}'"
        for at, piece in enumerate(pieces)
    ]


def _split_on_non_blanks(path: str) -> list[str]:
    """Split a path into pieces none of which ends in a blank.

    A path is cut at the widest piece the string limit allows, and then the cut
    is walked back over any blank it landed on, so that a directory or file
    name holding a space cannot lose it at a join.  The holdings tree does hold
    such names, and the loss would be silent: the consumer is told a file it
    never asked for is missing.

    Parameters:
        path: The path to split, already known to end in a non-blank.

    Returns:
        The pieces, in order, each at most ``_MAX_PIECE_CHARS`` long.

    Raises:
        ValueError: if the path holds a run of blanks longer than a piece,
            which leaves no cut to walk back to.
    """
    pieces: list[str] = []
    rest = path
    while len(rest) > _MAX_PIECE_CHARS:
        width = _MAX_PIECE_CHARS
        while width > 0 and rest[width - 1].isspace():
            width -= 1
        if width == 0:
            raise ValueError(
                f'{path!r} holds a run of more than {_MAX_PIECE_CHARS} blanks, which cannot be '
                f'written as text kernel strings without one of them falling at a join'
            )
        pieces.append(rest[:width])
        rest = rest[width:]
    pieces.append(rest)
    return pieces


def write_meta_kernel(
    path: FCPath,
    *,
    originals: Sequence[str | Path | FCPath],
    corrections: Sequence[str | Path | FCPath],
) -> None:
    """Write the meta-kernel that furnishes a corrected file set.

    Parameters:
        path: The meta-kernel to write, local or remote.
        originals: Paths of the original kernels.
        corrections: Paths of the corrected kernels.

    Raises:
        ValueError: if no kernel is named, or if a path cannot be expressed in
            a text kernel.
    """
    path.write_text('\n'.join(build_meta_kernel_lines(originals, corrections)) + '\n')
