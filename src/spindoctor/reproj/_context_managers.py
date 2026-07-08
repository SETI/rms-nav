"""Context managers for exception-safe mutation of shared state during reprojection.

These managers guarantee that oops precision configuration is always restored,
even if the wrapped code raises an exception. They are not thread-safe: concurrent
calls to reproject() in separate threads will interfere with each other through
the shared oops global state.
"""

import contextlib
from collections.abc import Iterator

import oops


@contextlib.contextmanager
def _reduced_oops_precision(*, dlt: int = 1) -> Iterator[None]:
    """Reduce oops light-travel precision settings, restoring originals on exit.

    Parameters:
        dlt: The dlt_precision value to use during the block. Defaults to 1,
            which is fast enough for reprojection purposes.

    Yields:
        None; reduced precision is active for the duration of the block.
    """
    old_path = oops.config.PATH_PHOTONS.dlt_precision
    old_surf = oops.config.SURFACE_PHOTONS.dlt_precision
    oops.config.PATH_PHOTONS.dlt_precision = dlt
    oops.config.SURFACE_PHOTONS.dlt_precision = dlt
    try:
        yield
    finally:
        oops.config.PATH_PHOTONS.dlt_precision = old_path
        oops.config.SURFACE_PHOTONS.dlt_precision = old_surf
