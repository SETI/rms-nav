"""Loaning one kernel to the process-global SPICE pool for the length of a block.

Several judgments this package makes -- which frame kernel defines a frame,
which clock kernel defines a clock, which candidate C-kernel reproduces a
baseline -- are answered by furnishing one kernel, asking, and unloading it
again, so that the answer is a property of that kernel rather than of whatever
else the pool happens to hold.  The loan lives here, once, because every module
that borrows the pool must return it the same way: a copy that drifted -- one
that forgot to unload on the failure path, say -- would leak its kernel into
every later question asked of the pool.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import cspyce
from filecache import FCPath


@contextmanager
def furnished(path: FCPath) -> Iterator[None]:
    """Furnish one kernel for the duration of a block and unload it after.

    SPICE furnishes a kernel by name from the local filesystem, so a remote one
    is fetched first; that is a no-op for a kernel that is already local.  The
    same local name is unloaded, since SPICE knows the kernel by the name it
    was given.

    Parameters:
        path: The kernel to furnish, local or remote.

    Yields:
        Nothing; the kernel is furnished for the body of the block.
    """
    local = str(cast(Path, path.retrieve()))
    cspyce.furnsh(local)
    try:
        yield
    finally:
        cspyce.unload(local)
