"""SPICE ids shared by the attitude computation and the C-kernel writer.

The one fact recorded here is which spacecraft clock a C-kernel object's time
tags are encoded against.  ``cspyce.ckmeta`` computes that mapping rather than
validating it -- it answers -999 for the nonexistent object -999999 without
raising -- so a wrong CK object id yields a plausible clock id, a successful
encoding, and silently wrong time tags on every record.  Every consumer
therefore checks what ``ckmeta`` resolves against the mapping below instead of
trusting it, and every consumer reads this one mapping: a second copy of these
numbers is a way for one side's check to rot while the other side's keeps
passing.

The clock is not derivable from the CK object by arithmetic.  Python's
``-31100 // 1000`` is -32, which is the other Voyager, so each object states
its clock explicitly.

This module holds constants and imports only the standard library, which is
what lets the kernel writer read it: the writer must pull in neither oops nor
anything from ``spindoctor.support``, and that guarantee is asserted on
``sys.modules`` in a fresh interpreter.
"""

from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    'CK_OBJECT_SCLK_ID',
]

# Read-only so that no consumer can edit the table the other consumer checks
# against.
CK_OBJECT_SCLK_ID: Mapping[int, int] = MappingProxyType(
    {
        -82000: -82,  # Cassini bus
        -77001: -77,  # Galileo scan platform
        -98000: -98,  # New Horizons spacecraft
        -31100: -31,  # Voyager 1 scan platform
        -32100: -32,  # Voyager 2 scan platform
    }
)
"""The spacecraft clock each supported CK object's time tags are encoded against."""
