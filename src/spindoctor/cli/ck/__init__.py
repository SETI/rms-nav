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

This package is part of the ``spindoctor.cli`` tree, which holds the command
line programs and the helpers only they use; the importable library API is the
rest of ``spindoctor``.  The ``__all__`` below is therefore the surface the
kernel-generating program imports from its own package -- what the modules
here offer each other and the driver, gathered in one place so that a reader
can see the shape of the writer without opening five files.  It is not
published library API: nothing outside ``spindoctor.cli`` imports it, it
carries no stability promise, and it has no page in the API reference, which
covers the library packages.  Every other ``spindoctor.cli`` subpackage stands
the same way.

One global to respect: ``cspyce.use_errors()`` / ``cspyce.use_flags()`` is
process-wide and shared with oops.  This package assumes the exceptions regime
(``use_errors``, the package default) and never flips it.
"""

from spindoctor.cli.ck.assignment import (
    Assignment,
    OutputGroup,
    assign_images,
    attitudes_reproduce,
    baseline_attitudes,
    group_for_output,
    output_basename,
    reproduces_baseline,
    rotation_angle_rad,
)
from spindoctor.cli.ck.images import ImageEntry, OmissionReason, botsim_losers
from spindoctor.cli.ck.index import (
    CkFile,
    CkIndex,
    CoverageInterval,
    KernelClass,
    build_ck_index,
    kernel_class_for_directory,
)
from spindoctor.cli.ck.pointing import ImagePointing
from spindoctor.cli.ck.segment import (
    CkSegment,
    build_segment,
    resolve_sclk_id,
    write_segment,
)

# The package's own surface, not published API: see the module docstring.
__all__ = [
    'Assignment',
    'CkFile',
    'CkIndex',
    'CkSegment',
    'CoverageInterval',
    'ImageEntry',
    'ImagePointing',
    'KernelClass',
    'OmissionReason',
    'OutputGroup',
    'assign_images',
    'attitudes_reproduce',
    'baseline_attitudes',
    'botsim_losers',
    'build_ck_index',
    'build_segment',
    'group_for_output',
    'kernel_class_for_directory',
    'output_basename',
    'reproduces_baseline',
    'resolve_sclk_id',
    'rotation_angle_rad',
    'write_segment',
]
