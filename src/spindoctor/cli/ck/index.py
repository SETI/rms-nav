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
TDB.

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

import cspyce

from spindoctor.cli.ck.segment import FROZEN_ATTITUDE_CK_IDS

# The extensions a binary C-kernel is stored under in the holdings.  Text
# kernels are never C-kernels, and the label files sitting beside the binaries
# carry their own extensions, so an extension test is enough to enumerate them.
CK_SUFFIXES = frozenset({'.bc', '.ck'})

# ``ckcov`` reads coverage at segment level -- the window each segment's
# descriptor advertises -- and reports it in TDB seconds, matching the epochs
# the metadata records.
_COVERAGE_LEVEL = 'SEGMENT'
_COVERAGE_TIME_SYSTEM = 'TDB'
_COVERAGE_NEEDS_ANGULAR_VELOCITY = False

# Coverage tolerance in spacecraft clock ticks.  Zero for an object whose
# attitude is read by evaluating a frame chain at the epoch itself; for a
# frozen-attitude object it is the widest tolerance the navigated lookup could
# have used, so that an image the baseline serves only through that tolerance
# still reaches the reproduction step.
_COVERAGE_TOL_TICKS = 0.0
SNAPPED_COVERAGE_TOL_TICKS = 80000.0


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
# always chosen over one that does not.
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
            interval, endpoints included.  False for a non-finite epoch: NaN
            answers every comparison with False and an infinity would sit
            inside any window reaching to the same infinity, so neither is
            left to the comparisons to decide.
        """
        if ck_frame_id != self.ck_frame_id:
            return False
        # Deliberately redundant with the finite endpoints enforced above,
        # which already leave every comparison against a non-finite epoch
        # False: this states the method's own contract, so it holds however
        # the interval was built rather than only because it was built here.
        if not math.isfinite(et):
            return False
        return bool(self.start_et <= et <= self.stop_et)


@dataclass(frozen=True)
class CkFile:
    """One C-kernel file the index found, and what it covers.

    Parameters:
        path: Path to the file.
        kernel_class: How its producer described its pointing, taken from the
            name of the directory it was found in.
        coverage: One interval per object per segment window, in the order
            SPICE reported them.
    """

    path: Path
    kernel_class: KernelClass
    coverage: tuple[CoverageInterval, ...]

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
        paths = [ck_file.path for ck_file in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError('the index holds the same path more than once')
        grouped: dict[str, list[CkFile]] = {}
        for ck_file in self.files:
            grouped.setdefault(ck_file.basename, []).append(ck_file)
        object.__setattr__(
            self, '_by_basename', {name: tuple(found) for name, found in grouped.items()}
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
    return (-ck_file.kernel_class.preference_rank, ck_file.basename, str(ck_file.path))


def kernel_class_for_directory(directory: Path) -> KernelClass:
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
    name = directory.name
    if len(name) == 0:
        raise ValueError(
            f'kernel directory {str(directory)!r} has no name to classify; name the directory '
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


def build_ck_index(roots: Sequence[Path]) -> CkIndex:
    """Scan the kernel directories of a run and index every C-kernel in them.

    Each directory is listed without recursion, so a subdirectory of kernels is
    indexed only when it is named as a root of its own.  The caller must have
    furnished the leapseconds kernel and the spacecraft clock kernel of every
    mission whose kernels are scanned, since coverage is reported in TDB and
    converting a clock tick to TDB needs both.

    Parameters:
        roots: The directories to scan.  Each is classified by its own name, so
            the reconstructed, gapfill and predicted directories of a mission
            are listed separately.

    Returns:
        The index.

    Raises:
        ValueError: if no directory is given, if one is named twice, if one
            does not exist or is not a directory, if a directory name declares
            more than one kernel class, or if no C-kernel is found under any of
            them -- an empty index would report every image as having no
            reproducing baseline, which is indistinguishable from a genuine
            baseline drift.
        OSError: if a file with a C-kernel extension cannot be read as one.
    """
    if len(roots) == 0:
        raise ValueError('no kernel directory to scan; the index would hold nothing')
    if len(set(roots)) != len(roots):
        raise ValueError(f'a kernel directory is named more than once: {[str(r) for r in roots]}')
    files: list[CkFile] = []
    for root in roots:
        kernel_class = kernel_class_for_directory(root)
        if not root.is_dir():
            raise ValueError(f'kernel directory {str(root)!r} does not exist or is not a directory')
        for path in sorted(root.iterdir()):
            if path.suffix.lower() in CK_SUFFIXES and path.is_file():
                files.append(_index_one_file(path, kernel_class))
    if len(files) == 0:
        raise ValueError(
            f'no C-kernel found under {[str(root) for root in roots]}; every image would be '
            f'reported as having no reproducing baseline'
        )
    return CkIndex(files=tuple(files))


def _index_one_file(path: Path, kernel_class: KernelClass) -> CkFile:
    """Read one C-kernel's objects and their coverage.

    Parameters:
        path: The file to read.
        kernel_class: The class its directory declares.

    Returns:
        The indexed file.

    Raises:
        OSError: if the file cannot be read as a C-kernel.
        KeyError: if the spacecraft clock of an object the file describes is
            not furnished, so its coverage cannot be expressed in TDB.
    """
    coverage: list[CoverageInterval] = []
    for ck_frame_id in sorted(int(value) for value in cspyce.ckobj(str(path))):
        tolerance = (
            SNAPPED_COVERAGE_TOL_TICKS
            if ck_frame_id in FROZEN_ATTITUDE_CK_IDS
            else _COVERAGE_TOL_TICKS
        )
        window = [
            float(value)
            for value in cspyce.ckcov(
                str(path),
                ck_frame_id,
                _COVERAGE_NEEDS_ANGULAR_VELOCITY,
                _COVERAGE_LEVEL,
                tolerance,
                _COVERAGE_TIME_SYSTEM,
            )
        ]
        coverage.extend(
            CoverageInterval(ck_frame_id=ck_frame_id, start_et=window[at], stop_et=window[at + 1])
            for at in range(0, len(window), 2)
        )
    return CkFile(path=path, kernel_class=kernel_class, coverage=tuple(coverage))
