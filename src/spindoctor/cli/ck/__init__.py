"""C-kernel generation from the corrected pointing navigation recorded.

Navigation measures where a camera was actually pointing and records that
measurement as a C-matrix in the image's metadata.  This package turns those
recorded matrices into SPICE C-kernels, so the same measurement loads into any
SPICE-aware tool with a ``furnsh`` instead of having to be applied as a pixel
offset by hand.

The package consumes the recorded C-matrix and performs no offset-to-rotation
conversion of its own.  It imports ``cspyce`` and **nothing** from oops, and
nothing from ``spindoctor.support`` either, since the module that computes the
C-matrix lives there and imports oops: one convenience import would pull the
whole geometry stack into a program that only writes kernels.  That guarantee
is checked on ``sys.modules`` in a fresh interpreter rather than by reading the
source.

The one table it shares with the attitude computation -- which spacecraft clock
each CK object's time tags are encoded against -- lives in
``spindoctor.spice_ids``, a constants module importing only the standard
library.  That table is the check against ``ckmeta`` computing a clock id
rather than validating one, so a second copy of it here would be a silent way
for the check to rot on one side while it kept passing on the other.

One global to respect: ``cspyce.use_errors()`` / ``cspyce.use_flags()`` is
process-wide and shared with oops.  This package assumes the exceptions regime
(``use_errors``, the package default) and never flips it.
"""

from spindoctor.cli.ck.pointing import ImagePointing
from spindoctor.cli.ck.segment import (
    CkSegment,
    build_segment,
    resolve_sclk_id,
    write_segment,
)

__all__ = [
    'CkSegment',
    'ImagePointing',
    'build_segment',
    'resolve_sclk_id',
    'write_segment',
]
