"""What importing the record seam must not drag in.

This package is the half of the seam that reads documents, and reading a
document needs no database at all.  A checkout with no index anywhere, and a
machine with no server to reach, still has to be able to import it and read
every record a results root holds, so a database layer imported here would make
the storage that needs none depend on the one that does.

Reading the source cannot prove that, since the offending import could be two
modules deep, so the check runs a fresh interpreter and looks at what actually
loaded.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.conftest import child_interpreter_environment

import spindoctor

_PROBE_TIMEOUT_S = 120.0
"""An import that hangs would otherwise stall the run with no failure to read."""

_PROBE = """
import json
import sys

import spindoctor.nav_records

print(
    json.dumps(
        {
            'forbidden': sorted(
                name for name in sys.modules if name.split('.')[0] == 'sqlalchemy'
            ),
            'loaded': sorted(sys.modules),
            'package': spindoctor.nav_records.__file__,
        }
    )
)
"""


@dataclass(frozen=True)
class _Probed:
    """What a fresh import of the package loaded, and where it loaded it from.

    Parameters:
        forbidden: The database modules that loaded.
        loaded: Every module that loaded.
        package: The file the probe imported the package from.
    """

    forbidden: list[str]
    loaded: list[str]
    package: str


@pytest.fixture(scope='module')
def probed_modules() -> _Probed:
    """Import the package in a fresh interpreter and report what loaded.

    The interpreter is a real subprocess, which is the whole point: the
    guarantee is about what an import does from nothing, and this process has
    already imported half the tree for other tests.  Its environment names this
    checkout, so the probe answers for this code rather than an installed copy.

    Returns:
        What the probe reported.
    """
    completed = subprocess.run(
        [sys.executable, '-c', _PROBE],
        capture_output=True,
        text=True,
        check=False,
        timeout=_PROBE_TIMEOUT_S,
        env=child_interpreter_environment(),
    )
    # check=False so a failing probe reports its own stderr instead of a bare
    # CalledProcessError that hides why the interpreter refused the import.
    assert completed.returncode == 0, completed.stderr
    return _Probed(**json.loads(completed.stdout))


def test_the_record_seam_loads_no_database_layer(probed_modules: _Probed) -> None:
    """Every navigation run reaches this package, and most of them name no index."""
    assert probed_modules.forbidden == []


def test_the_probe_really_imported_the_walk(probed_modules: _Probed) -> None:
    """The assertion above proves its point only if the source itself loaded.

    Without this it could pass because the package imports almost nothing, which
    is a guarantee about an empty package rather than about this one.
    """
    assert 'spindoctor.nav_records.tree' in probed_modules.loaded


def test_the_probe_imported_the_package_under_test(probed_modules: _Probed) -> None:
    """The probe answers for this checkout rather than an installed copy.

    Asserted rather than assumed because it was not true elsewhere: the suite
    runs each test from a directory of its own, so a relative ``PYTHONPATH``
    resolves against that directory and the subprocess imports whichever
    SpinDoctor is installed.  A guarantee tested against other code holds
    whatever this code does.
    """
    expected = Path(spindoctor.__file__).resolve().parent / 'nav_records' / '__init__.py'
    assert Path(probed_modules.package).resolve() == expected
