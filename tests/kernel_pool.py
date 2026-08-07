"""Running a test against an empty SPICE kernel pool, and putting the pool back.

The SPICE kernel pool is process-global and nothing in this project unloads
what oops furnishes when it loads an observation, so a test that reads the pool
or that depends on which kernel answers for an object is at the mercy of what
ran before it in the same worker.  That is not a hypothetical ordering: the
project's own test command is ``pytest -n auto --dist=loadfile``, which assigns
files to workers dynamically, and the tests that furnish real mission kernels
are gated by a ``skipif`` rather than by the ``integration`` marker, so they run
in a plain ``pytest`` wherever the holdings are present.

The isolation here is a loan rather than a reset.  ``kclear`` would be simpler
and is wrong: oops keeps its own record of what it has furnished and its own
frame objects that evaluate against the pool, so a test that clears the pool
leaves every later test in the worker holding frames whose kernels are gone,
with no error naming the cause.  What happens instead is that the pool is
emptied, the test runs, and exactly what was found is furnished again in the
order it was found -- so oops's record of the pool is true again by the time any
oops code runs.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import cspyce


def furnished_kernels() -> tuple[str, ...]:
    """Return the kernels a caller furnished directly, in load order.

    A kernel a meta-kernel pulled in is deliberately not reported: furnishing
    it again beside its meta-kernel would load it twice.  Restoring the
    meta-kernel restores its contents with it.

    Returns:
        The paths, as they were given to ``furnsh``.
    """
    direct: list[str] = []
    for at in range(int(cspyce.ktotal('ALL'))):
        path, _filtyp, source, _handle = cspyce.kdata(at, 'ALL')
        if len(str(source)) == 0:
            direct.append(str(path))
    return tuple(direct)


@contextmanager
def isolated_kernel_pool() -> Iterator[None]:
    """Empty the SPICE kernel pool for the body, then furnish back what was found.

    Yields:
        Nothing; the body runs with an empty pool.

    Raises:
        RuntimeError: if the pool cannot be emptied, which would leave the body
            running against kernels it did not furnish.  What was found is put
            back before this is raised.
    """
    borrowed = furnished_kernels()
    _empty_the_pool()
    left = furnished_kernels()
    if len(left) > 0:
        _refurnish(borrowed)
        raise RuntimeError(
            f'the SPICE kernel pool still holds {list(left)} after unloading everything it '
            f'reported; this test needs a pool it furnished itself'
        )
    try:
        yield
    finally:
        _empty_the_pool()
        _refurnish(borrowed)


def _empty_the_pool() -> None:
    """Unload every directly furnished kernel, most recent first."""
    for path in reversed(furnished_kernels()):
        cspyce.unload(path)


def _refurnish(paths: tuple[str, ...]) -> None:
    """Furnish kernels again in the order they were originally furnished.

    Parameters:
        paths: The paths to furnish.
    """
    for path in paths:
        cspyce.furnsh(path)
