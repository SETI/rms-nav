"""The pre-index of candidate C-kernels a corrected segment can be built on.

An image's metadata records which SPICE kernels were furnished when it
navigated, but it records sorted basenames only: no load order, no directory,
and in a batch run the list accumulates every kernel earlier images needed, so
it is a superset of the ones this image actually used.  Identifying the
baseline kernel an image navigated against therefore takes two steps, and this
module is the first of them: scan every C-kernel under the kernel directories
once per run, recording which objects each file describes and over what epochs,
so that the second step -- furnishing candidates one at a time and keeping the
ones that reproduce the recorded attitude -- has only a handful of files to try
per image instead of the whole holdings.

The scan uses ``ckobj`` and ``ckcov`` at segment level in TDB.  Both read a
file by name, so no C-kernel has to be furnished to be indexed, but the
leapseconds kernel and the spacecraft clock kernel of every object encountered
must be, since that is what converts a coverage window from clock ticks into
TDB.  An object whose clock is not furnished contributes no coverage and is
recorded as unreadable instead, because a real kernel can name an object no
clock kernel describes and no run should be stopped by an object none of its
images asks about; an image that does ask about one is refused by the
assignment step.  Both calls also need a real local file, so a kernel tree that
lives remotely is fetched as it is scanned; for a tree that is already local
nothing is copied.

Coverage is read with a tolerance for the objects whose attitude was navigated
through a tolerance-snapped pointing lookup.  Such a lookup answers with a
record up to its tolerance away from the epoch asked for, so an exposure just
outside a segment's window is still served by that segment, and a coverage
filter that did not allow for it would drop the only candidate that reproduces.
The filter is only a filter: reproduction, not coverage, decides.
"""

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import cast

import cspyce
from filecache import FCPath

from spindoctor.cli.ck.segment import FROZEN_ATTITUDE_CK_IDS

# The extensions a binary C-kernel is stored under in the holdings.  Text
# kernels are never C-kernels, and the label files sitting beside the binaries
# carry their own extensions, so an extension test is enough to enumerate them.
CK_SUFFIXES = frozenset({'.bc', '.ck'})

# What a corrected file's name carries beyond the original's, before the
# extension.  It lives here because it is a fact about kernel filenames that
# the scan needs: a corrected kernel written back beside its original must not
# be indexed as a candidate for the next run.
OUTPUT_NAME_MARKER = '_nav'

# ``ckcov`` reads coverage at segment level -- the window each segment's
# descriptor advertises -- and reports it in TDB seconds, matching the epochs
# the metadata records.
_COVERAGE_LEVEL = 'SEGMENT'
_COVERAGE_TIME_SYSTEM = 'TDB'
_COVERAGE_NEEDS_ANGULAR_VELOCITY = False

# The widest tolerance, in spacecraft clock ticks, that a snapped pointing
# lookup can be made at: the value oops registers its fallback Voyager frame
# with.  It is declared once and read twice, because the two uses have to
# agree.  The reproduction step asks for pointing this far from an exposure
# midtime, and the scan below widens a frozen-attitude object's coverage by
# the same amount so that every image that lookup can serve survives the
# coverage filter and reaches the reproduction step at all.  Widening by less
# than the lookup reaches drops the only candidate that reproduces and reports
# the image as having no baseline.  The two agree to within half a tick at the
# extreme edge, because the filter measures from the exposure midtime while
# the lookup measures from that midtime rounded to a whole tick; an image that
# far from any pointing record is refused rather than corrected, which is the
# safe direction.
SNAPPED_LOOKUP_TOL_TICKS = 80000.0

# Coverage tolerance for an object whose attitude is read by evaluating a
# frame chain at the epoch itself, which reaches exactly as far as it is asked.
_COVERAGE_TOL_TICKS = 0.0


class KernelClass(Enum):
    """How the producer of a C-kernel described its pointing.

    The holdings keep reconstructed, gapfill and predicted kernels in separate
    directories with overlapping coverage, so an image can navigate against a
    baseline that several files reproduce.  This is the first key of the
    tie-break that then picks one of them.
    """

    RECONSTRUCTED = 'reconstructed'
    GAPFILL = 'gapfill'
    PREDICTED = 'predicted'
    UNCLASSIFIED = 'unclassified'

    @property
    def preference_rank(self) -> int:
        """Rank in the tie-break, lowest first.

        Returns:
            The class's position in the preference order: reconstructed
            pointing before gapfill before predicted, and a directory naming no
            class last of all.
        """
        return _CLASS_PREFERENCE.index(self)


# Reconstructed pointing is measured, gapfill fills what reconstruction did not
# cover, and predicted pointing is a plan.  A directory whose name says nothing
# about which of those it holds is preferred last, so a kernel that does say is
# always chosen over one that does not.  That ranks the Cassini cruise
# directories below predicted even though they hold reconstructed pointing,
# which is inert only because their epochs do not overlap the directories that
# do name a class; the ranking decides which output file carries a segment and
# never which attitude it carries, so it is a filing question either way.
_CLASS_PREFERENCE: tuple[KernelClass, ...] = (
    KernelClass.RECONSTRUCTED,
    KernelClass.GAPFILL,
    KernelClass.PREDICTED,
    KernelClass.UNCLASSIFIED,
)

# The token each class is recognized by in a directory name, lowercased.
_CLASS_TOKENS: tuple[tuple[str, KernelClass], ...] = (
    ('reconstructed', KernelClass.RECONSTRUCTED),
    ('gapfill', KernelClass.GAPFILL),
    ('predicted', KernelClass.PREDICTED),
)


@dataclass(frozen=True)
class CoverageInterval:
    """One span of epochs over which a C-kernel file describes one object.

    Parameters:
        ck_frame_id: SPICE id of the object covered.
        start_et: First covered epoch, TDB seconds past J2000.
        stop_et: Last covered epoch, TDB seconds past J2000.
    """

    ck_frame_id: int
    start_et: float
    stop_et: float

    def __post_init__(self) -> None:
        """Refuse a window that is not a window.

        Raises:
            ValueError: if either endpoint is not finite, or if the interval
                ends before it starts.  An infinite endpoint would report every
                epoch as covered, and a NaN one would report none, in both
                cases without any comparison ever failing.
        """
        for label, value in (('start_et', self.start_et), ('stop_et', self.stop_et)):
            if not math.isfinite(value):
                raise ValueError(
                    f'coverage {label} for object {self.ck_frame_id} is not finite: {value!r}'
                )
        if self.stop_et < self.start_et:
            raise ValueError(
                f'coverage for object {self.ck_frame_id} ends at {self.stop_et!r} before it '
                f'starts at {self.start_et!r}'
            )

    def contains(self, ck_frame_id: int, et: float) -> bool:
        """Report whether this interval covers one object at one epoch.

        Parameters:
            ck_frame_id: SPICE id of the object asked about.
            et: TDB seconds past J2000.

        Returns:
            True when the object matches and the epoch lies within the
            interval, endpoints included.  False for a non-finite epoch: both
            endpoints are finite, refused at construction otherwise, so an
            infinity falls outside the window and a NaN answers every
            comparison with False.
        """
        if ck_frame_id != self.ck_frame_id:
            return False
        return bool(self.start_et <= et <= self.stop_et)


@dataclass(frozen=True)
class CkFile:
    """One C-kernel file the index found, and what it covers.

    Parameters:
        path: Path to the file, local or remote.
        kernel_class: How its producer described its pointing, taken from the
            name of the directory it was found in.
        coverage: One interval per object per segment window, in the order
            SPICE reported them.
        unreadable_objects: The objects the file describes whose coverage
            could not be expressed in TDB, because no furnished kernel defines
            the spacecraft clock their time tags are encoded against.  The file
            offers no coverage for them, so it is never a candidate for one,
            and an image that needs one is refused by the assignment step
            rather than being told its baseline has drifted.
    """

    path: FCPath
    kernel_class: KernelClass
    coverage: tuple[CoverageInterval, ...]
    unreadable_objects: tuple[int, ...] = ()

    @property
    def basename(self) -> str:
        """The file's basename, which is what image metadata records."""
        return self.path.name

    def covers(self, ck_frame_id: int, et: float) -> bool:
        """Report whether this file describes one object at one epoch.

        Parameters:
            ck_frame_id: SPICE id of the object asked about.
            et: TDB seconds past J2000.

        Returns:
            True when any of the file's coverage intervals contains the epoch
            for that object.
        """
        return any(interval.contains(ck_frame_id, et) for interval in self.coverage)


@dataclass(frozen=True)
class CkIndex:
    """Every C-kernel found under a run's kernel directories.

    Parameters:
        files: The indexed files.  A basename may appear more than once, since
            two directories may hold files of the same name; each such file is
            an independent candidate.

    Raises:
        ValueError: if two entries name the same path, which would offer the
            same file twice as a candidate.
    """

    files: tuple[CkFile, ...]
    _by_basename: dict[str, tuple[CkFile, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Group the files by basename, which is how metadata names them."""
        paths = [ck_file.path.as_posix() for ck_file in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError('the index holds the same path more than once')
        grouped: dict[str, list[CkFile]] = {}
        for ck_file in self.files:
            grouped.setdefault(ck_file.basename, []).append(ck_file)
        object.__setattr__(
            self, '_by_basename', {name: tuple(found) for name, found in grouped.items()}
        )

    @property
    def unreadable_objects(self) -> frozenset[int]:
        """The objects whose coverage no indexed file could express in TDB.

        An object appears here when the spacecraft clock its time tags are
        encoded against is not furnished, so its coverage window cannot be
        converted.  Nothing in the index offers coverage for such an object,
        which makes it invisible to :meth:`candidates`; the assignment step
        reads this so an image that needs one is refused for the reason it
        actually has.
        """
        return frozenset(
            ck_frame_id for ck_file in self.files for ck_frame_id in ck_file.unreadable_objects
        )

    def candidates(
        self, *, basenames: Collection[str], ck_frame_id: int, et: float
    ) -> tuple[CkFile, ...]:
        """Return the indexed files that could have supplied one image's baseline.

        A file qualifies when the image's metadata names its basename, when it
        describes the CK object the image's correction targets, and when its
        coverage includes the exposure midtime.  A basename found in more than
        one directory contributes each of those files, since only reproduction
        can tell them apart.

        Parameters:
            basenames: The kernel basenames the image's metadata recorded.
                Repeats are ignored; a basename the index does not hold
                contributes nothing.
            ck_frame_id: SPICE id of the object the correction targets.
            et: The exposure midtime, TDB seconds past J2000.

        Returns:
            The qualifying files, ordered by preference: kernel class first,
            then the greatest basename, then the greatest path, so that the
            first entry is the one the tie-break selects.
        """
        found: list[CkFile] = []
        for basename in set(basenames):
            if basename not in self._by_basename:
                continue
            found.extend(
                ck_file
                for ck_file in self._by_basename[basename]
                if ck_file.covers(ck_frame_id, et)
            )
        return tuple(sorted(found, key=preference_key, reverse=True))


def preference_key(ck_file: CkFile) -> tuple[int, str, str]:
    """Return the sort key that orders candidates by preference, greatest first.

    Several kernels can reproduce one image's baseline, because the holdings
    carry reconstructed, gapfill and predicted sets with overlapping coverage.
    They agree on the attitude by construction -- that is what reproducing it
    means -- so the choice among them decides only which output file carries
    the image's segment, and all it has to be is deterministic.

    Parameters:
        ck_file: The candidate to key.

    Returns:
        A key whose greatest value is the preferred candidate: the negated
        class rank first, so reconstructed beats gapfill beats predicted, then
        the basename, so the lexicographically greatest name wins, then the
        full path, so two directories holding the same basename in the same
        class still resolve the same way on every run.
    """
    return (
        -ck_file.kernel_class.preference_rank,
        ck_file.basename,
        ck_file.path.as_posix(),
    )


def kernel_class_for_directory(directory: str | Path | FCPath) -> KernelClass:
    """Classify a kernel directory from its name.

    Parameters:
        directory: The directory holding the kernels.

    Returns:
        The class its name declares, or ``UNCLASSIFIED`` when the name names
        none.

    Raises:
        ValueError: if the path has no final component to read, or if its name
            declares more than one class, which no preference order can resolve
            and which would otherwise be settled silently by the order the
            names happen to be tested in.
    """
    root = FCPath(directory)
    name = root.name
    if len(name) == 0 or name in ('.', '..'):
        raise ValueError(
            f'kernel directory {root.as_posix()!r} has no name to classify; name the directory '
            f'itself rather than a path ending in a separator or a dot'
        )
    lowered = name.lower()
    matched = [kernel_class for token, kernel_class in _CLASS_TOKENS if token in lowered]
    if len(matched) > 1:
        raise ValueError(
            f'kernel directory {name!r} names more than one kernel class: '
            f'{[kernel_class.value for kernel_class in matched]}'
        )
    if len(matched) == 0:
        return KernelClass.UNCLASSIFIED
    return matched[0]


def build_ck_index(roots: Sequence[str | Path | FCPath]) -> CkIndex:
    """Scan the kernel directories of a run and index every C-kernel in them.

    Each directory is listed without recursion, so a subdirectory of kernels is
    indexed only when it is named as a root of its own.  The caller must have
    furnished the leapseconds kernel and the spacecraft clock kernel of every
    mission whose kernels are scanned, since coverage is reported in TDB and
    converting a clock tick to TDB needs both.

    A corrected kernel is never indexed.  Writing the corrections back beside
    the originals is the natural thing to do, and a corrected kernel reproduces
    its own baseline exactly wherever the correction was the identity, so
    indexing one would offer a correction as the baseline for the next run.

    Parameters:
        roots: The directories to scan.  Each is classified by its own name, so
            the reconstructed, gapfill and predicted directories of a mission
            are listed separately.

    Returns:
        The index.

    Raises:
        ValueError: if no directory is given, if one names the same directory
            as another once symbolic links and ``..`` are resolved, if one does
            not exist or is not a directory, if a directory name declares more
            than one kernel class, or if no C-kernel is found under any of them
            -- an empty index would report every image as having no reproducing
            baseline, which is indistinguishable from a genuine baseline drift.
        OSError: if a file with a C-kernel extension cannot be read as one.
    """
    if len(roots) == 0:
        raise ValueError('no kernel directory to scan; the index would hold nothing')
    directories = [FCPath(root) for root in roots]
    # Resolved for the duplicate test only.  The class comes from the name as
    # given, since resolving a symbolic link can replace the very component
    # that names the class.
    resolved = [root.resolve().as_posix() for root in directories]
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            f'a kernel directory is named more than once: '
            f'{[root.as_posix() for root in directories]}'
        )
    files: list[CkFile] = []
    for root in directories:
        kernel_class = kernel_class_for_directory(root)
        files.extend(_index_one_file(path, kernel_class) for path in _kernel_files(root))
    if len(files) == 0:
        raise ValueError(
            f'no C-kernel found under {[root.as_posix() for root in directories]}; every image '
            f'would be reported as having no reproducing baseline'
        )
    return CkIndex(files=tuple(files))


def _kernel_files(root: FCPath) -> list[FCPath]:
    """List the C-kernels of one directory, in a stable order.

    The directory is listed rather than probed first: for a remote tree an
    existence check is a round trip of its own, and the listing answers the
    same question as a side effect.

    Parameters:
        root: The directory to list.

    Returns:
        The C-kernels it holds, excluding corrected kernels, ordered by path.

    Raises:
        ValueError: if the directory does not exist or is not a directory.
    """
    try:
        entries = sorted(root.iterdir(), key=lambda entry: entry.as_posix())
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError(
            f'kernel directory {root.as_posix()!r} does not exist or is not a directory'
        ) from exc
    return [
        entry
        for entry in entries
        if entry.suffix.lower() in CK_SUFFIXES
        and not entry.stem.endswith(OUTPUT_NAME_MARKER)
        and entry.is_file()
    ]


def _index_one_file(path: FCPath, kernel_class: KernelClass) -> CkFile:
    """Read one C-kernel's objects and their coverage.

    Parameters:
        path: The file to read.
        kernel_class: The class its directory declares.

    Returns:
        The indexed file, carrying the coverage of every object whose clock is
        furnished and the ids of the objects whose clock is not.

    Raises:
        OSError: if the file cannot be read as a C-kernel.
    """
    # SPICE reads a C-kernel by name from the local filesystem, so a remote one
    # is fetched first.  This is a no-op for a kernel that is already local.
    local = str(cast(Path, path.retrieve()))
    coverage: list[CoverageInterval] = []
    unreadable: list[int] = []
    for ck_frame_id in sorted(int(value) for value in cspyce.ckobj(local)):
        tolerance = (
            SNAPPED_LOOKUP_TOL_TICKS
            if ck_frame_id in FROZEN_ATTITUDE_CK_IDS
            else _COVERAGE_TOL_TICKS
        )
        try:
            window = [
                float(value)
                for value in cspyce.ckcov(
                    local,
                    ck_frame_id,
                    _COVERAGE_NEEDS_ANGULAR_VELOCITY,
                    _COVERAGE_LEVEL,
                    tolerance,
                    _COVERAGE_TIME_SYSTEM,
                )
            ]
        except LookupError:
            # A real kernel can describe an object whose spacecraft clock no
            # kernel defines: a merged New Horizons pointing file names object
            # -1 beside the spacecraft, and the clock id SPICE computes for it
            # is 0, which no SCLK kernel supplies.  Converting that object's
            # coverage to TDB is impossible, and refusing the whole scan over
            # it would make every New Horizons directory unindexable for the
            # sake of an object no image will ever ask about.  The object is
            # recorded instead, and an image that does ask about one is refused
            # before any candidate is tried.
            unreadable.append(ck_frame_id)
            continue
        coverage.extend(
            CoverageInterval(ck_frame_id=ck_frame_id, start_et=window[at], stop_et=window[at + 1])
            for at in range(0, len(window), 2)
        )
    return CkFile(
        path=path,
        kernel_class=kernel_class,
        coverage=tuple(coverage),
        unreadable_objects=tuple(unreadable),
    )
