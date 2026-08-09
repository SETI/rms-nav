"""The C-kernel writer's import guarantee.

The writer exists so that a navigated attitude can be turned into a kernel
without the geometry stack, and one convenience import from
``spindoctor.support`` -- where the module that computes the C-matrix lives --
would drag oops in transitively.  Reading the source cannot prove that, since
the offending import could be two modules deep, so the check runs a fresh
interpreter and looks at what actually loaded.
"""

import json
import subprocess
import sys

import pytest

# An import that hangs -- a module blocking on a network resource, say -- would
# otherwise stall the run with no failure to read.
_PROBE_TIMEOUT_S = 120.0

_PROBE = """
import json
import sys

import spindoctor.cli.ck  # noqa: F401

forbidden = sorted(
    name
    for name in sys.modules
    if name == 'oops' or name.startswith('oops.') or name.startswith('spindoctor.support')
)
print(json.dumps({'forbidden': forbidden, 'loaded': sorted(sys.modules)}))
"""


@pytest.fixture(scope='module')
def probed_modules() -> dict[str, list[str]]:
    """Import the writer package in a fresh interpreter and report sys.modules.

    The interpreter is a real subprocess, which is the whole point: the
    guarantee is about what an import does from nothing, and this process has
    already imported oops for other tests.  What is reused across the two tests
    below is only the answer that one subprocess gave.

    Returns:
        A dict with the forbidden modules that loaded and every module that
        loaded.
    """
    completed = subprocess.run(
        [sys.executable, '-c', _PROBE],
        capture_output=True,
        text=True,
        check=False,
        timeout=_PROBE_TIMEOUT_S,
    )
    # check=False so a failing probe reports its own stderr instead of a bare
    # CalledProcessError that hides why the interpreter refused the import.
    assert completed.returncode == 0, completed.stderr
    result: dict[str, list[str]] = json.loads(completed.stdout)
    return result


def test_writer_package_loads_no_oops_and_no_support(
    probed_modules: dict[str, list[str]],
) -> None:
    """Importing the writer pulls in neither oops nor spindoctor.support."""
    assert probed_modules['forbidden'] == []


def test_writer_package_really_imported(probed_modules: dict[str, list[str]]) -> None:
    """The probe proves its own point only if the writer actually loaded."""
    assert 'spindoctor.cli.ck.segment' in probed_modules['loaded']
    assert 'cspyce' in probed_modules['loaded']
