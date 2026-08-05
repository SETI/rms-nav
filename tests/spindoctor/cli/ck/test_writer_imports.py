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


def _probe_modules() -> dict[str, list[str]]:
    """Import the writer package in a fresh interpreter and report sys.modules.

    Returns:
        A dict with the forbidden modules that loaded and every module that
        loaded.
    """
    completed = subprocess.run(
        [sys.executable, '-c', _PROBE],
        capture_output=True,
        text=True,
        check=True,
    )
    result: dict[str, list[str]] = json.loads(completed.stdout)
    return result


def test_writer_package_loads_no_oops_and_no_support() -> None:
    """Importing the writer pulls in neither oops nor spindoctor.support."""
    assert _probe_modules()['forbidden'] == []


def test_writer_package_really_imported() -> None:
    """The probe proves its own point only if the writer actually loaded."""
    loaded = _probe_modules()['loaded']
    assert 'spindoctor.cli.ck.segment' in loaded
    assert 'cspyce' in loaded
