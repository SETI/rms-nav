"""Real-holdings integration tests for ``spindoctor.cli.ck.index``.

Three things about the pre-index cannot be settled by kernels the suite writes
itself.  The first is that it survives a real kernel directory, which holds
subdirectories, comment files and label files beside the binaries.  The second
is the claim the assignment rules rest on: that Voyager's decades-spanning bus
kernels describe a different object from the scan platform the ISS pointing is
read from, so the object filter excludes them without any epoch reasoning.  The
third is that a real kernel can name an object whose spacecraft clock no kernel
defines, which the New Horizons holdings do and which the scan has to survive.

Everything else about the index is exercised hermetically.
"""

import os
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

from spindoctor.cli.ck.index import CkIndex, build_ck_index  # noqa: E402  (guarded import)

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
