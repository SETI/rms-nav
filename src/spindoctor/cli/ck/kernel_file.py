"""Writing corrected C-kernels: their segments and the comments about them.

Two things are judged here, and the difference between them is the point of the
module.  :func:`check_ck_file` judges one file: everything about it that can be
decided without opening anything.  :func:`check_output_paths` judges a whole
run's worth of destinations at once, so that a run whose third file cannot be
written writes none of the three rather than two and a refusal.  A partial set
is worse than no set: the meta-kernel and the report that would say what is in
it are written last and never get written, and the refusal to overwrite an
existing corrected kernel then blocks the obvious rerun.

Neither is atomicity, and nothing here claims it.  What the two together
establish is that nothing *knowable in advance* stops the writing part way
through.  What is not knowable in advance stays possible: the disk filling up,
a permission or a path changing between the check and the write, and a record
set ``ckw03`` refuses once the file is open.

``ckopn`` reserves the comment area and ``ckw03`` writes segments, but nothing
in the CK interface fills the comments; that is done through the plain DAF
interface afterwards, on the file the CK interface has already closed.  This
module runs the two halves in the one order that works, and reserves the room
the comments need before the file is opened.

Reserving matters.  A comment that does not fit what was reserved is still
stored -- SPICE extends the area by shifting every data record in the file --
so the failure is a rewritten file rather than a lost comment, and it is
invisible unless the file is measured before and after.  It is measurable: the
address of the first data record does not move when the reservation was enough
and does when it was not.

A write that fails part way through is closed with ``dafcls`` rather than
``ckcls``, and the half-written file is removed.  ``ckcls`` on a C-kernel that
received no segment raises in its own right and leaves the file open, so using
it on the failure path would replace the error worth reading with one that is
not *and* leak the handle; ``dafcls`` closes the file and has nothing to say
about segments.

That covers the segments, and it is why everything else is judged before the
file is opened.  The comment area is the case that makes the ordering matter:
it is written after ``ckcls`` has already succeeded, so a line SPICE cannot
store, refused there, would leave behind a kernel that furnishes cleanly and
covers exactly the right exposures with an empty comment area -- a file
indistinguishable from a good product except for the provenance record the
comment area exists to carry, and one the guard against overwriting an existing
kernel then stops the operator from simply regenerating.
"""

import os
from collections.abc import Sequence
from pathlib import Path

import cspyce

from spindoctor.cli.ck.comments import (
    check_comment_lines,
    reserved_comment_chars,
    write_comment_area,
)
from spindoctor.cli.ck.segment import CkSegment, write_segment

# The longest internal file name SPICE stores in a DAF file record.
_IFNAME_MAX_CHARS = 60


def check_output_paths(paths: Sequence[Path]) -> None:
    """Refuse a run's corrected kernels before the first of them is written.

    Every destination the run intends to write is judged together, and every
    one that fails is named with its reason in a single refusal, so that an
    operator learns about all of them at once rather than one rerun at a time.
    The point of judging the set rather than each file as it is reached is that
    a refusal on the third file has already left the first two behind: a
    partial set, with no meta-kernel and no report to say what is in it, that
    the refusal to overwrite an existing corrected kernel then blocks a rerun
    on.

    What this establishes, for each path:

    * Nothing already occupies it, a symbolic link with no target included.
    * Its basename fits the field SPICE stores as a file's internal name.
    * Its directory exists and its permission bits allow writing into it.
    * It can be spelled: it names a file, and holds no character that a path
      cannot carry through SPICE or through the meta-kernel that will name it.
    * No other path in the set is spelled the same way.

    What it does not establish, and cannot: that the write will succeed.  Space
    on the device, a path or a permission changed after this returns, a
    directory reachable only through a symbolic link that moves, and everything
    SPICE decides once a file is open are all outside what any check made
    beforehand can see.

    Parameters:
        paths: The corrected kernels the run intends to write, in any order.
            An empty sequence is accepted and judged clean; a run that writes
            no corrected kernel has nothing here to refuse.

    Raises:
        ValueError: naming every path that cannot be written and why.  Nothing
            has been written when it is raised, and the caller is expected to
            write nothing.
    """
    refusals: list[str] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            # A set-level refusal, and the one no per-file check can make: the
            # second write of a repeated path would replace the first file's
            # segments with the second's, silently.
            refusals.append(
                f'{path} is named twice; the second would replace the first rather than join it'
            )
            continue
        seen.add(path)
        refusals.extend(_path_refusals(path))
    if len(refusals) > 0:
        raise ValueError(
            f'no corrected kernel is written, because the run cannot write all of them: '
            f'{"; ".join(refusals)}'
        )


def check_ck_file(path: Path, segments: Sequence[CkSegment], comment_lines: Sequence[str]) -> None:
    """Refuse one corrected C-kernel without opening anything.

    This is everything :func:`write_ck_file` decides before it calls ``ckopn``,
    factored out so that a caller writing several files can decide all of it
    for all of them first.  ``write_ck_file`` calls it too, so the two cannot
    drift apart and a direct caller is judged by the same rules.

    Parameters:
        path: The file that would be created.
        segments: The segments that would be written.
        comment_lines: The comment area that would be attached.

    Raises:
        ValueError: if no segment is given -- SPICE refuses to close a C-kernel
            holding none, and the half-written file would be left behind -- if
            no comment line is given, if a comment line is one SPICE cannot
            store and read back unchanged, if anything already occupies the
            path, or if the file's own name is too long to be its internal
            name.
    """
    if len(segments) == 0:
        raise ValueError(
            f'no segments to write to {path.name}; a corrected kernel carrying none claims a '
            f'correction it does not hold, and SPICE refuses to close it'
        )
    if len(comment_lines) == 0:
        raise ValueError(f'no comment lines to write to {path.name}')
    # Before ``ckopn``, because the comment area is written after ``ckcls``:
    # a line SPICE cannot store, judged there, would leave behind a complete
    # and furnishable kernel whose comment area is empty -- and the guard on an
    # existing file below would then block the obvious next step of fixing the
    # line and running again.
    check_comment_lines(comment_lines)
    for refusal in (_occupied_refusal(path), _internal_name_refusal(path.name)):
        if refusal is not None:
            raise ValueError(refusal)


def write_ck_file(path: Path, segments: Sequence[CkSegment], comment_lines: Sequence[str]) -> None:
    """Write one corrected C-kernel, comments and all.

    Parameters:
        path: The file to create.  Nothing may occupy it: SPICE refuses to open
            an existing file for creation, and a corrected kernel is
            regenerated by replacing it rather than by appending to it.
        segments: The segments to write, in order.  At least one is required.
        comment_lines: The comment area to attach.  At least one line is
            required.

    Raises:
        ValueError: for anything :func:`check_ck_file` refuses, all of it
            judged before the file is opened.  Also if SPICE refuses a segment
            once the file is open, which it does for a segment identifier
            holding a non-printing character or a quaternion of magnitude zero.
        RuntimeError: if SPICE cannot create the file, for example because the
            directory does not exist or cannot be written to.
        OSError: if SPICE refuses a segment write for an operating-system
            reason once the file is open.

    Whatever the failure, no file is left behind: the up-front refusals happen
    before ``ckopn``, and a failure after it closes the file and removes it.
    """
    check_ck_file(path, segments, comment_lines)
    handle = int(cspyce.ckopn(str(path), path.name, reserved_comment_chars(comment_lines)))
    try:
        for segment in segments:
            write_segment(handle, segment)
    except Exception:
        # Closed through the plain DAF interface rather than ``ckcls``, and
        # measured rather than assumed: ``ckcls`` on a C-kernel that received
        # no segment raises SPICE(NOSEGMENTSFOUND) *and leaves the file open*,
        # so using it here would both replace the failure worth reading with
        # one that is not and leak the handle anyway.  ``dafcls`` closes it and
        # has nothing to say about segments.
        cspyce.dafcls(handle)
        # The file ``ckopn`` created is removed with it.  It holds no usable
        # segment, and leaving it behind would make the obvious next step --
        # fix the cause and run again -- fail on the refusal to write over an
        # existing corrected kernel.
        path.unlink(missing_ok=True)
        raise
    cspyce.ckcls(handle)
    write_comment_area(path, comment_lines)


def _path_refusals(path: Path) -> list[str]:
    """Return every reason one corrected kernel cannot be written to a path.

    Parameters:
        path: The file that would be created.

    Returns:
        One message per reason, empty when nothing known in advance stops the
        write.
    """
    spelling = _spelling_refusal(path)
    if spelling is not None:
        # The only thing worth saying about a path the operating system cannot
        # be asked about at all.  ``os.path.lexists`` answers False for a name
        # holding a null byte, which would read here as "nothing is there".
        return [spelling]
    return [
        refusal
        for refusal in (
            _occupied_refusal(path),
            _internal_name_refusal(path.name),
            _directory_refusal(path.parent),
        )
        if refusal is not None
    ]


def _spelling_refusal(path: Path) -> str | None:
    """Return why a path cannot be spelled as a file to write, or ``None``.

    Parameters:
        path: The path to judge.

    Returns:
        The reason, or ``None`` if the path names a file and every character of
        it survives the two places it has to be written down: the file name
        SPICE is given, which is a C string and ends at a null, and the
        meta-kernel, whose text strings carry printing characters only.
    """
    if len(path.name) == 0:
        return f'{path} names no file to write'
    text = str(path)
    for character in text:
        if not character.isprintable():
            return (
                f'{text!r} holds the non-printing character {character!r}; SPICE is given a file '
                f'name as a C string and the meta-kernel names it as text, and neither carries one'
            )
    return None


def _occupied_refusal(path: Path) -> str | None:
    """Return why something already occupies a path, or ``None``.

    Parameters:
        path: The path to judge.

    Returns:
        The reason, or ``None`` if nothing is there.  A symbolic link is
        refused whether or not it has a target: one with a target would be
        written through, putting the corrected kernel wherever the link points
        instead of in the output directory, and one without a target is the
        case :meth:`~pathlib.Path.exists` alone misses, since it follows the
        link and reports the absent target rather than the link itself.
    """
    if path.is_symlink():
        return (
            f'{path} is a symbolic link; a corrected kernel is written as a file of its own, and '
            f'writing through the link would put it wherever the link points'
        )
    if path.exists():
        # Said here rather than left to SPICE, which reports an operating
        # system status number and no reason at all.  A corrected kernel is
        # regenerated by replacing it, so the operator has to say which of the
        # two files they meant to keep.
        return (
            f'{path} already exists; a corrected kernel is regenerated by replacing it, so remove '
            f'or move the existing file rather than writing beside it'
        )
    return None


def _internal_name_refusal(name: str) -> str | None:
    """Return why a basename cannot be a file's internal name, or ``None``.

    Parameters:
        name: The basename of the file being written, which is what identifies
            it once it has been copied somewhere else.

    Returns:
        The reason, or ``None`` if the name fits the field SPICE stores it in.
        A truncated internal name would identify a different file.
    """
    if len(name) > _IFNAME_MAX_CHARS:
        return (
            f'{name!r} is longer than the {_IFNAME_MAX_CHARS} characters SPICE stores as a '
            f"file's internal name"
        )
    return None


def _directory_refusal(directory: Path) -> str | None:
    """Return why a directory cannot take a corrected kernel, or ``None``.

    Parameters:
        directory: The directory the file would be created in.

    Returns:
        The reason, or ``None`` if the directory exists and its permission bits
        allow a file to be created in it.  Those bits are what is checked, and
        all that is: a directory that passes can still refuse the write later,
        because the device fills up or because the permissions change in
        between.
    """
    if not directory.is_dir():
        return f'{directory} is not a directory, so no corrected kernel can be created in it'
    if not os.access(
        directory, os.W_OK | os.X_OK, effective_ids=os.access in os.supports_effective_ids
    ):
        return f'{directory} cannot be written to'
    return None


def first_data_record(path: Path) -> int:
    """Return the record number of a DAF file's first data record.

    This is what the comment area's reservation buys: the comments sit between
    the file record and the data, so a file whose reservation was large enough
    has the same first data record before and after its comments are added, and
    a file whose reservation was too small does not -- SPICE moved the data.

    Parameters:
        path: The file to measure.

    Returns:
        The record number of the first descriptor record.

    Raises:
        OSError: if the file cannot be opened for reading.
    """
    handle = int(cspyce.dafopr(str(path)))
    try:
        _nd, _ni, _ifname, fward, _bward, _free = cspyce.dafrfr(handle)
        return int(fward)
    finally:
        cspyce.dafcls(handle)
