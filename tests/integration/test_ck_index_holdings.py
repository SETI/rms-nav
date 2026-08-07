"""Real-holdings integration tests for ``spindoctor.cli.ck.index``.

Three things about the pre-index cannot be settled by kernels the suite writes
itself.  The first is that it survives a real kernel directory, which holds
subdirectories, comment files and label files beside the binaries.  The second
is the claim the assignment rules rest on: that Voyager's decades-spanning bus
kernels describe a different object from the scan platform the ISS pointing is
read from, so the object filter excludes them without any epoch reasoning.  The
third is that a real kernel can name an object whose spacecraft clock no kernel
defines, which the New Horizons holdings do and which the scan has to survive.

The third is the classification itself.  A kernel's class comes from its own
basename, and whether the patterns cover every name the holdings actually hold
is a question only the holdings can answer -- a hand-written sample proves
nothing about the 1200 names nobody typed out.  So every C-kernel of every
mission is classified here and the totals are asserted per directory, which is
also the one place the directory names appear: they are the grouping the counts
are stated against, not an input to the answer.

Everything else about the index is exercised hermetically.
"""

import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_RESOURCES = os.environ.get('OOPS_RESOURCES', '')
_SPICE_ROOT = Path(_RESOURCES) / 'SPICE'
_VOYAGER_CK_DIR = _SPICE_ROOT / 'Voyager' / 'CK'
_NEW_HORIZONS_ROOT = _SPICE_ROOT / 'New-Horizons'
_NEW_HORIZONS_CK_DIR = _NEW_HORIZONS_ROOT / 'CK-reconstructed'

if len(_RESOURCES) == 0 or not _VOYAGER_CK_DIR.is_dir():
    pytest.skip(
        'OOPS_RESOURCES does not name a local SPICE tree; skipping C-kernel index holdings tests',
        allow_module_level=True,
    )

import cspyce  # noqa: E402  (guarded import)

from spindoctor.cli.ck.index import (  # noqa: E402  (guarded import)
    CK_SUFFIXES,
    CkIndex,
    KernelClass,
    build_ck_index,
    kernel_class_for_basename,
)

# The Voyager 1 scan platform, which ISS pointing is read from, and the bus,
# which the long-baseline kernels describe.
_PLATFORM_ID = -31100
_BUS_ID = -31000

_PLATFORM_KERNEL = 'vg1_sat_version1_type1_iss_sedr.bc'
_BUS_KERNEL = 'vgr1_super.bc'

# The New Horizons spacecraft, and the object one merged pointing file names
# beside it whose clock no kernel defines.
_SPACECRAFT_ID = -98000
_CLOCKLESS_OBJECT_ID = -1
_CLOCKLESS_KERNEL = 'nh_scispi_2015_recon.bc'
# Every C-kernel directory of every mission the pipeline navigates, with the
# class every kernel in it must be found to declare.  The Cassini cruise and
# Jupiter directories name no class and their kernels do, which is the whole
# reason the class is read from the basename; New Horizons marks only the pair
# of kernels that exist in both forms; Voyager and Galileo mark nothing at all.
_CLASSIFIED_DIRS: tuple[tuple[str, dict[KernelClass, int]], ...] = (
    ('Cassini/CK-reconstructed', {KernelClass.RECONSTRUCTED: 998}),
    ('Cassini/CK-cruise', {KernelClass.RECONSTRUCTED: 31}),
    ('Cassini/CK-jup', {KernelClass.RECONSTRUCTED: 64}),
    ('Cassini/CK-gapfill', {KernelClass.GAPFILL: 15}),
    ('Cassini/CK-predicted', {KernelClass.PREDICTED: 104}),
    ('Cassini/CK-predicted-v02', {KernelClass.PREDICTED: 104}),
    (
        'New-Horizons/CK-reconstructed',
        {KernelClass.RECONSTRUCTED: 1, KernelClass.UNCLASSIFIED: 29},
    ),
    ('New-Horizons/CK-predicted', {KernelClass.PREDICTED: 1, KernelClass.UNCLASSIFIED: 1}),
    ('Voyager/CK', {KernelClass.UNCLASSIFIED: 10}),
    ('Galileo/CK', {KernelClass.UNCLASSIFIED: 52}),
)


def _holdings_basenames(directory: str) -> list[str]:
    """List the C-kernel basenames one holdings directory holds.

    Parameters:
        directory: The directory's path below the SPICE root.

    Returns:
        The basenames, sorted, of every file with a C-kernel extension.
    """
    root = _SPICE_ROOT / directory
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_file() and entry.suffix.lower() in CK_SUFFIXES
    )


@pytest.fixture
def voyager_clocks() -> Iterator[None]:
    """Furnish the leapseconds and Voyager clock kernels, then unload them.

    Coverage is reported in TDB, and converting a spacecraft clock tick to TDB
    needs both.  The pool is process-global, so exactly what was furnished here
    is unloaded again rather than clearing everything.

    Yields:
        Nothing; the kernels are furnished for the body of the test.
    """
    kernels = [_SPICE_ROOT / 'leapseconds.ker']
    kernels.extend(sorted((_SPICE_ROOT / 'Voyager' / 'SCLK').glob('*.tsc')))
    for kernel in kernels:
        cspyce.furnsh(str(kernel))
    try:
        yield
    finally:
        for kernel in reversed(kernels):
            cspyce.unload(str(kernel))


@pytest.fixture
def voyager_index(voyager_clocks: None) -> CkIndex:
    """Index the real Voyager C-kernel directory.

    Parameters:
        voyager_clocks: The furnished clock kernels the coverage read needs.

    Returns:
        The index of every C-kernel in the directory.
    """
    return build_ck_index([_VOYAGER_CK_DIR])


def test_the_index_reads_a_real_kernel_directory(voyager_index: CkIndex) -> None:
    """A directory holding comment files, labels and a subdirectory indexes cleanly."""
    basenames = {ck_file.basename for ck_file in voyager_index.files}
    assert _PLATFORM_KERNEL in basenames
    assert _BUS_KERNEL in basenames
    assert all(name.lower().endswith(('.bc', '.ck')) for name in basenames)


def test_the_bus_kernels_describe_another_object(voyager_index: CkIndex) -> None:
    """The long-baseline Voyager kernels cover the bus, not the scan platform."""
    by_name = {ck_file.basename: ck_file for ck_file in voyager_index.files}
    assert {interval.ck_frame_id for interval in by_name[_BUS_KERNEL].coverage} == {_BUS_ID}
    assert {interval.ck_frame_id for interval in by_name[_PLATFORM_KERNEL].coverage} == {
        _PLATFORM_ID
    }


def test_a_platform_epoch_selects_no_bus_kernel(voyager_index: CkIndex) -> None:
    """An ISS image's candidates exclude the bus kernels by object alone.

    The image's own kernel list names every file in the directory, so only the
    object filter can exclude the bus kernel -- which is the reason the
    assignment rules need no Voyager-specific file selection.
    """
    by_name = {ck_file.basename: ck_file for ck_file in voyager_index.files}
    platform = by_name[_PLATFORM_KERNEL].coverage[0]
    midtime = (platform.start_et + platform.stop_et) / 2.0
    candidates = voyager_index.candidates(
        basenames=list(by_name), ck_frame_id=_PLATFORM_ID, et=midtime
    )
    assert _BUS_KERNEL not in {ck_file.basename for ck_file in candidates}
    assert _PLATFORM_KERNEL in {ck_file.basename for ck_file in candidates}


@pytest.fixture
def new_horizons_clocks() -> Iterator[None]:
    """Furnish the leapseconds and New Horizons clock kernels, then unload them.

    Yields:
        Nothing; the kernels are furnished for the body of the test.
    """
    kernels = [_SPICE_ROOT / 'leapseconds.ker']
    kernels.extend(sorted((_NEW_HORIZONS_ROOT / 'SCLK').glob('*.tsc')))
    for kernel in kernels:
        cspyce.furnsh(str(kernel))
    try:
        yield
    finally:
        for kernel in reversed(kernels):
            cspyce.unload(str(kernel))


@pytest.fixture
def new_horizons_index(new_horizons_clocks: None) -> CkIndex:
    """Index the real New Horizons reconstructed C-kernel directory.

    Parameters:
        new_horizons_clocks: The furnished clock kernels the coverage read
            needs.

    Returns:
        The index of every C-kernel in the directory.
    """
    return build_ck_index([_NEW_HORIZONS_CK_DIR])


def test_a_real_directory_holding_an_object_with_no_clock_indexes(
    new_horizons_index: CkIndex,
) -> None:
    """The New Horizons holdings scan through, object -1 and all.

    One merged pointing file names object -1 beside the spacecraft, and the
    clock id SPICE computes for it is 0, which no SCLK kernel defines.  Reading
    that object's coverage in TDB is impossible, and refusing the scan over it
    would leave the whole mission unindexable.
    """
    by_name = {ck_file.basename: ck_file for ck_file in new_horizons_index.files}
    assert by_name[_CLOCKLESS_KERNEL].unreadable_objects == (_CLOCKLESS_OBJECT_ID,)


def test_that_file_still_covers_the_spacecraft(new_horizons_index: CkIndex) -> None:
    """The object the images actually correct keeps its coverage."""
    by_name = {ck_file.basename: ck_file for ck_file in new_horizons_index.files}
    covered = {interval.ck_frame_id for interval in by_name[_CLOCKLESS_KERNEL].coverage}
    assert covered == {_SPACECRAFT_ID}


def test_the_spacecraft_is_not_reported_unreadable(new_horizons_index: CkIndex) -> None:
    """Nothing an image needs is among the objects the index could not read."""
    assert _SPACECRAFT_ID not in new_horizons_index.unreadable_objects


@pytest.mark.parametrize(
    ('directory', 'expected'),
    _CLASSIFIED_DIRS,
    ids=[directory for directory, _ in _CLASSIFIED_DIRS],
)
def test_every_kernel_in_a_holdings_directory_classifies(
    directory: str, expected: dict[KernelClass, int]
) -> None:
    """Every C-kernel a real directory holds declares the class it should.

    Parameters:
        directory: The directory's path below the SPICE root.
        expected: How many of its kernels each class must account for.
    """
    counts = Counter(kernel_class_for_basename(name) for name in _holdings_basenames(directory))
    assert dict(counts) == expected


def test_the_cassini_holdings_classify_every_kernel_but_none_as_unclassified() -> None:
    """Cassini names a class on all 1316 of its kernels, in six directories.

    Two of those directories name no class at all, so this is the measurement
    the change was made for: reading the directory leaves 95 reconstructed
    kernels ranked below predicted, and reading the basename leaves none.
    """
    cassini = [
        name
        for directory, _ in _CLASSIFIED_DIRS
        if directory.startswith('Cassini/')
        for name in _holdings_basenames(directory)
    ]
    counts = Counter(kernel_class_for_basename(name) for name in cassini)
    assert dict(counts) == {
        KernelClass.RECONSTRUCTED: 1093,
        KernelClass.GAPFILL: 15,
        KernelClass.PREDICTED: 208,
    }


@pytest.mark.parametrize('directory', ['Voyager/CK', 'Galileo/CK'], ids=['voyager', 'galileo'])
def test_a_mission_encoding_no_class_gets_none_invented_for_it(directory: str) -> None:
    """Voyager and Galileo hold one kind of C-kernel and name it nowhere.

    Every one of their kernels must be unclassified, so the class rank ties for
    every candidate and the tie-break falls through to the basename.

    Parameters:
        directory: The mission's C-kernel directory below the SPICE root.
    """
    classes = {kernel_class_for_basename(name) for name in _holdings_basenames(directory)}
    assert classes == {KernelClass.UNCLASSIFIED}


def test_no_holdings_basename_declares_two_classes() -> None:
    """The shipped patterns are mutually exclusive over every name in the holdings.

    A name matching two patterns of different classes is refused, so a rule set
    that overlapped anywhere in the holdings would abort the scan.  Classifying
    every name without a refusal is the assertion.
    """
    classified = [
        kernel_class_for_basename(name)
        for directory, _ in _CLASSIFIED_DIRS
        for name in _holdings_basenames(directory)
    ]
    assert len(classified) == 1410
