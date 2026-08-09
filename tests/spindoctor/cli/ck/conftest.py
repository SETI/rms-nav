"""Fixtures managing the SPICE kernel pool for the C-kernel writer tests.

The SPICE kernel pool is process-global.  A test that had loaded a real image
through oops leaves real mission kernels furnished, and they define the same
objects the hermetic kernels in ``ck_helpers`` do -- a real Cassini frame
kernel names -82000, a real clock kernel names -82 -- so what these tests
measure would depend on which file ran first in the worker.  Every test in
this package therefore runs against an emptied pool, and the pool it borrowed
is furnished again afterwards.  The helpers that write the hermetic kernels
themselves live in ``ck_helpers``, which is a plain module so that any test
file can import from it.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.kernel_pool import isolated_kernel_pool
from tests.spindoctor.cli.ck.ck_helpers import KernelPool, write_support_kernels


@pytest.fixture(scope='module', autouse=True)
def empty_kernel_pool() -> Iterator[None]:
    """Run the tests in each module here against a pool holding nothing else.

    Autouse rather than requested, because the tests that read the pool itself
    -- what a meta-kernel furnished, which files define a frame -- take no
    fixture at all and are exactly the ones a foreign kernel misleads.

    Module-scoped rather than per test, because emptying and restoring a pool
    that oops has filled costs about a second and no foreign kernel can appear
    between two tests of one module: nothing else is running.  Every test here
    already unloads what it furnished, and one that stopped would now be
    visible to the next test rather than hidden by a reset.

    Yields:
        Nothing; the module's tests run against an empty pool.
    """
    with isolated_kernel_pool():
        yield


@pytest.fixture
def pool(tmp_path: Path, empty_kernel_pool: None) -> Iterator[KernelPool]:
    """Furnish the hermetic LSK, SCLK and FK, and unload them afterwards.

    Parameters:
        empty_kernel_pool: Requested by name so the pool is emptied before
            these kernels go into it, whatever order pytest would otherwise
            pick.

    Yields:
        The record of what was furnished, for a test that furnishes more.
    """
    kernels = KernelPool(tmp_path)
    for path in write_support_kernels(tmp_path):
        kernels.furnish(path)
    try:
        yield kernels
    finally:
        kernels.unload_all()
