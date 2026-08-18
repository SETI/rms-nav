"""The C-kernel writer's import guarantee.

The writer exists so that a navigated attitude can be turned into a kernel
without the geometry stack, and one convenience import of the module that
computes a C-matrix from a navigated offset would drag oops in transitively.
That is what the guarantee is about, so oops is what is asserted: naming a whole
package as a proxy for it would forbid modules that import no oops at all --
which is what the values a record carries, and the document a record is written
to, are -- and a rule that forbids the harmless is one that gets worked around
rather than kept.  Reading the source cannot prove it either way, since the
offending import could be two modules deep, so the check runs a fresh
interpreter and looks at what actually loaded.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import spindoctor

# An import that hangs -- a module blocking on a network resource, say -- would
# otherwise stall the run with no failure to read.
_PROBE_TIMEOUT_S = 120.0

_PROBE = """
import json
import sys

import spindoctor.cli.ck

forbidden = sorted(
    name for name in sys.modules if name == 'oops' or name.startswith('oops.')
)
print(
    json.dumps(
        {
            'forbidden': forbidden,
            'loaded': sorted(sys.modules),
            'package': spindoctor.cli.ck.__file__,
        }
    )
)
"""


@pytest.fixture(scope='module')
def probed_modules() -> dict[str, list[str]]:
    """Import the writer package in a fresh interpreter and report sys.modules.

    The interpreter is a real subprocess, which is the whole point: the
    guarantee is about what an import does from nothing, and this process has
    already imported oops for other tests.  What is reused across the two tests
    below is only the answer that one subprocess gave.

    Returns:
        A dict with the forbidden modules that loaded, every module that loaded,
        and the file the probe imported the package from.
    """
    # The package under test is named absolutely, and by where this process
    # imported it from.  The suite runs every test from a directory of its own,
    # so a relative PYTHONPATH resolves against that directory instead and the
    # probe imports whatever copy of SpinDoctor happens to be installed -- which
    # is a probe that answers for somebody else's code and passes whatever this
    # branch does to the writer's imports.
    package_root = Path(spindoctor.__file__).resolve().parent.parent
    environment = dict(os.environ, PYTHONPATH=str(package_root))
    completed = subprocess.run(
        [sys.executable, '-c', _PROBE],
        capture_output=True,
        text=True,
        check=False,
        timeout=_PROBE_TIMEOUT_S,
        env=environment,
    )
    # check=False so a failing probe reports its own stderr instead of a bare
    # CalledProcessError that hides why the interpreter refused the import.
    assert completed.returncode == 0, completed.stderr
    result: dict[str, list[str]] = json.loads(completed.stdout)
    return result


def test_writer_package_loads_no_oops(probed_modules: dict[str, list[str]]) -> None:
    """Importing the writer pulls in no oops, transitively or otherwise."""
    assert probed_modules['forbidden'] == []


def test_the_probe_would_notice_the_geometry_stack(
    probed_modules: dict[str, list[str]],
) -> None:
    """The module whose import would drag oops in is one the writer could reach.

    Without this the assertion above could pass because nothing in the writer
    imports anything at all from the library, rather than because what it does
    import is oops-free.  ``spindoctor.support.nav_record`` is in the writer's
    imports and lives in the same package as the C-matrix computation, so the
    probe is looking at a real opportunity to fail.
    """
    assert 'spindoctor.support.nav_record' in probed_modules['loaded']


def test_writer_package_really_imported(probed_modules: dict[str, list[str]]) -> None:
    """The probe proves its own point only if the writer actually loaded."""
    assert 'spindoctor.cli.ck.segment' in probed_modules['loaded']
    assert 'cspyce' in probed_modules['loaded']


def test_the_probe_imported_the_package_under_test(
    probed_modules: dict[str, list[str]],
) -> None:
    """The probe answers for this checkout rather than an installed copy.

    Asserted rather than assumed because it was not true: the suite runs each
    test from a directory of its own, so the relative PYTHONPATH the probe used
    to inherit resolved against that directory and the subprocess imported
    whichever SpinDoctor was installed.  A guarantee tested against other code
    holds whatever this code does.
    """
    expected = Path(spindoctor.__file__).resolve().parent / 'cli' / 'ck' / '__init__.py'
    assert Path(str(probed_modules['package'])).resolve() == expected
