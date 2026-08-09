"""Choosing the spacecraft clock kernel a navigated image's time tags need.

A segment's time tags are encoded with ``sce2c``, which reads whichever
spacecraft clock kernel happens to be furnished.  Encoding against a different
version of that kernel from the one navigation used shifts every time tag by
however much the two disagree, and nothing reports it: the segment is written,
the file loads, and the pointing is served a little early or a little late
forever after.

So the kernel is not chosen by name or by version number, and it is not simply
the newest.  An image's provenance records the basenames of every kernel that
was furnished when it navigated -- a superset, sorted, with no directories and
no load order -- and the one to use is the one among them that defines the
clock this image's CK object is timed against.  Each candidate is furnished on
its own and asked whether it defines that clock; exactly one must, and both
none and several are refused rather than resolved by picking.

The probe needs the pool to hold no definition of the clock already, since a
kernel furnished earlier would answer for every candidate alike and make them
all look right.  That is checked rather than assumed.
"""

from collections.abc import Mapping

import cspyce
from filecache import FCPath

from spindoctor.cli.ck.pool import furnished

# The extension a text spacecraft clock kernel carries in the holdings.
SCLK_SUFFIX = '.tsc'

# The pool variables a spacecraft clock kernel defines for the clock it
# describes.  Either is enough: every kernel in the holdings declares both, and
# asking for two independent names means a kernel that omits the optional data
# type declaration is still recognized.
_CLOCK_VARIABLES = ('SCLK_DATA_TYPE_{n}', 'SCLK_PARTITION_START_{n}')


def clock_is_defined(sclk_id: int) -> bool:
    """Report whether the furnished kernels define one spacecraft clock.

    Parameters:
        sclk_id: The spacecraft clock id, for example -82 for Cassini.

    Returns:
        True when some furnished kernel declares that clock.
    """
    for template in _CLOCK_VARIABLES:
        try:
            cspyce.dtpool(template.format(n=abs(sclk_id)))
        except KeyError:
            continue
        return True
    return False


def select_sclk_kernel(candidates: Mapping[str, FCPath], sclk_id: int) -> str:
    """Return the basename of the clock kernel an image's time tags need.

    Parameters:
        candidates: The clock kernels named by the image's provenance that the
            run's kernel directories resolve, keyed by basename.
        sclk_id: The spacecraft clock the image's CK object is timed against.

    Returns:
        The basename of the one candidate that defines that clock.

    Raises:
        ValueError: if the clock is already defined by a furnished kernel,
            which would make every candidate look right; if no candidate
            defines it, which means the run's kernel directories do not hold
            the kernel navigation used; or if more than one does, which is a
            choice this step refuses to make for the operator, since the two
            versions disagree about the very thing being encoded.
        OSError: if a candidate cannot be furnished.
    """
    if clock_is_defined(sclk_id):
        raise ValueError(
            f'spacecraft clock {sclk_id} is already defined by a furnished kernel; the kernel an '
            f'image navigated against cannot be identified while another answers for it'
        )
    defining = sorted(
        basename for basename, path in candidates.items() if _defines_clock(path, sclk_id)
    )
    if len(defining) == 0:
        raise ValueError(
            f'none of the kernels recorded for this image defines spacecraft clock {sclk_id}: '
            f'{sorted(candidates)}; the clock kernel navigation used is not among the run'
            f"'s kernel directories"
        )
    if len(defining) > 1:
        raise ValueError(
            f'{len(defining)} kernels recorded for this image define spacecraft clock '
            f'{sclk_id}: {defining}; which one encoded the navigated time tags cannot be told '
            f'from the record, and encoding against the wrong one shifts every time tag silently'
        )
    return defining[0]


def _defines_clock(path: FCPath, sclk_id: int) -> bool:
    """Report whether one kernel defines one spacecraft clock.

    Parameters:
        path: The kernel to test, local or remote.
        sclk_id: The spacecraft clock id.

    Returns:
        True when furnishing it defines that clock.

    Raises:
        OSError: if the kernel cannot be furnished.
    """
    with furnished(path):
        return clock_is_defined(sclk_id)
