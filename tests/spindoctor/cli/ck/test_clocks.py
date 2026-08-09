"""Hermetic tests for ``spindoctor.cli.ck.clocks``.

Which clock kernel encoded a navigated image's time tags is not a question of
naming, so the tests write two kernels that describe the same clock differently
and check that the selection refuses to choose between them, and that it finds
the one that describes the clock among several that do not.

The pool fixture furnishes a clock kernel of its own, so these tests build
their own pool: the selection deliberately refuses to run while the clock it is
looking for is already defined, and that refusal is itself one of the cases.
"""

import re
from collections.abc import Iterator
from pathlib import Path

import cspyce
import pytest
from filecache import FCPath

from spindoctor.cli.ck.clocks import clock_is_defined, select_sclk_kernel

# Two clocks: the one the tests select for, and one they do not, so a kernel
# that defines the wrong clock is a real candidate rather than an empty file.
_CLOCK_ID = -82
_OTHER_CLOCK_ID = -31

_SCLK_TEMPLATE = """KPL/SCLK

A minimal spacecraft clock kernel for one clock, written by the SpinDoctor
C-kernel writer tests.  The tick rate is a parameter so that two kernels for
the same clock can disagree about it.

\\begindata
SCLK_KERNEL_ID           = ( @2000-01-01/00:00:00 )

SCLK_DATA_TYPE_{n}        = ( 1 )
SCLK01_TIME_SYSTEM_{n}    = ( 1 )
SCLK01_N_FIELDS_{n}       = ( 2 )
SCLK01_MODULI_{n}         = ( 4294967296 {ticks} )
SCLK01_OFFSETS_{n}        = ( 0 0 )
SCLK01_OUTPUT_DELIM_{n}   = ( 1 )
SCLK_PARTITION_START_{n}  = ( 0.0000000000000E+00 )
SCLK_PARTITION_END_{n}    = ( 1.0995116277750E+12 )
SCLK01_COEFFICIENTS_{n}   = ( 0.0000000000000E+00 0.0000000000000E+00 1.0000000000000E+00 )
\\begintext
"""

_LSK_TEXT = """KPL/LSK

\\begindata
DELTET/DELTA_T_A = 32.184
DELTET/K         = 1.657D-3
DELTET/EB        = 1.671D-2
DELTET/M         = ( 6.239996D0 1.99096871D-7 )
DELTET/DELTA_AT  = ( 10, @1972-JAN-1
                     32, @1999-JAN-1
                     37, @2017-JAN-1 )
\\begintext
"""


@pytest.fixture
def clock_root(tmp_path: Path) -> Iterator[Path]:
    """Yield a directory holding candidate kernels, with nothing furnished.

    The selection refuses to run while the clock is already defined, so this
    fixture deliberately furnishes only the leapseconds kernel, and unloads it
    afterwards rather than clearing the pool an unrelated test may share.
    """
    lsk = tmp_path / 'test.tls'
    lsk.write_text(_LSK_TEXT)
    cspyce.furnsh(str(lsk))
    try:
        yield tmp_path
    finally:
        cspyce.unload(str(lsk))


def _sclk(root: Path, name: str, *, clock_id: int = _CLOCK_ID, ticks: int = 256) -> FCPath:
    """Write one clock kernel and return its path.

    Parameters:
        root: Directory to write into.
        name: Basename of the kernel.
        clock_id: The clock it describes.
        ticks: The tick rate it declares, so two kernels can disagree.

    Returns:
        The kernel's path.
    """
    path = root / name
    path.write_text(_SCLK_TEMPLATE.format(n=abs(clock_id), ticks=ticks))
    return FCPath(str(path))


def test_the_one_kernel_defining_the_clock_is_selected(clock_root: Path) -> None:
    """A kernel for another spacecraft is not a candidate for this one."""
    candidates = {
        'right.tsc': _sclk(clock_root, 'right.tsc'),
        'wrong.tsc': _sclk(clock_root, 'wrong.tsc', clock_id=_OTHER_CLOCK_ID),
    }
    assert select_sclk_kernel(candidates, _CLOCK_ID) == 'right.tsc'


def test_no_kernel_defining_the_clock_is_refused(clock_root: Path) -> None:
    """The kernel navigation used is not among the run's directories."""
    candidates = {'wrong.tsc': _sclk(clock_root, 'wrong.tsc', clock_id=_OTHER_CLOCK_ID)}
    with pytest.raises(ValueError, match='none of the kernels'):
        select_sclk_kernel(candidates, _CLOCK_ID)


@pytest.mark.usefixtures('clock_root')
def test_an_empty_candidate_set_is_refused() -> None:
    """Naming no clock kernel at all is the same failure as naming wrong ones."""
    with pytest.raises(ValueError, match='none of the kernels'):
        select_sclk_kernel({}, _CLOCK_ID)


def test_two_kernels_defining_the_clock_are_refused(clock_root: Path) -> None:
    """Picking one would encode every time tag against a guess."""
    candidates = {
        'old.tsc': _sclk(clock_root, 'old.tsc', ticks=256),
        'new.tsc': _sclk(clock_root, 'new.tsc', ticks=128),
    }
    with pytest.raises(ValueError, match='2 kernels'):
        select_sclk_kernel(candidates, _CLOCK_ID)


def test_the_refusal_names_both_kernels(clock_root: Path) -> None:
    """So the operator can see which two versions the record cannot separate."""
    candidates = {
        'old.tsc': _sclk(clock_root, 'old.tsc', ticks=256),
        'new.tsc': _sclk(clock_root, 'new.tsc', ticks=128),
    }
    with pytest.raises(ValueError, match=re.escape("'new.tsc', 'old.tsc'")):
        select_sclk_kernel(candidates, _CLOCK_ID)


def test_a_clock_already_defined_is_refused(clock_root: Path) -> None:
    """A kernel furnished earlier answers for every candidate alike."""
    already = _sclk(clock_root, 'already.tsc')
    cspyce.furnsh(str(already))
    try:
        with pytest.raises(ValueError, match='already defined'):
            select_sclk_kernel({'right.tsc': _sclk(clock_root, 'right.tsc')}, _CLOCK_ID)
    finally:
        cspyce.unload(str(already))


def test_the_probe_leaves_the_pool_as_it_found_it(clock_root: Path) -> None:
    """Every candidate is unloaded again, whether or not it was chosen."""
    candidates = {'right.tsc': _sclk(clock_root, 'right.tsc')}
    select_sclk_kernel(candidates, _CLOCK_ID)
    assert not clock_is_defined(_CLOCK_ID)


def test_a_candidate_that_does_not_exist_is_refused(clock_root: Path) -> None:
    """A basename the run's directories claimed to resolve and cannot."""
    with pytest.raises(FileNotFoundError, match=re.escape('gone.tsc')):
        select_sclk_kernel({'gone.tsc': FCPath(str(clock_root / 'gone.tsc'))}, _CLOCK_ID)


def test_a_file_that_defines_no_clock_is_not_a_candidate(clock_root: Path) -> None:
    """SPICE reads a file with no assignments in it without complaint.

    A clock kernel is therefore recognized by the clock it defines and never by
    its name, since a file that happens to be named like one contributes
    nothing and is reported as one of the kernels that did not define it.
    """
    path = clock_root / 'broken.tsc'
    path.write_text('this is not a SPICE text kernel\n')
    with pytest.raises(ValueError, match='none of the kernels'):
        select_sclk_kernel({'broken.tsc': FCPath(str(path))}, _CLOCK_ID)


@pytest.mark.usefixtures('clock_root')
def test_a_clock_no_kernel_defines_reads_as_undefined() -> None:
    """The probe answers False rather than raising for an unknown clock."""
    assert clock_is_defined(-12345) is False
