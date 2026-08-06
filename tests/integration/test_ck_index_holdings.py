"""Real-holdings integration tests for ``spindoctor.cli.ck.index``.

Two things about the pre-index cannot be settled by kernels the suite writes
itself.  The first is that it survives a real kernel directory, which holds
subdirectories, comment files and label files beside the binaries.  The second
is the claim the assignment rules rest on: that Voyager's decades-spanning bus
kernels describe a different object from the scan platform the ISS pointing is
read from, so the object filter excludes them without any epoch reasoning.

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
