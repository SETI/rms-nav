"""Guard that every UI module the API reference documents is importable with Qt mocked.

Contract under test (docs/api_reference/api_ui.rst plus the
``autodoc_mock_imports`` list in docs/conf.py): the documentation build has no
PyQt6 installed, so autodoc imports each ``automodule`` target with ``PyQt6``
and ``matplotlib`` replaced by Sphinx's mock objects.  A mock does not
implement ``|``, so a module-level or class-body expression such as
``QWidget | None`` raises ``TypeError`` at import and autodoc reports the
module as unimportable.  Under ``sphinx-build -W`` that warning is an error and
the documentation build fails.

The failure is invisible on a workstation that has PyQt6 installed, because
anything importing real PyQt6 first -- a matplotlib Qt backend, for instance --
puts it in ``sys.modules`` where the mock finder never sees it.  So each module
here is imported in a fresh subprocess, which is the only way to reproduce the
documentation builder's environment from a session that may already hold the
real package.

Deferring annotation evaluation with ``from __future__ import annotations``
keeps these modules importable; this test fails if one loses that import or
gains a Qt expression evaluated at import time.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_SRC_DIR = _REPO_ROOT / 'src'
_API_UI_RST = _REPO_ROOT / 'docs' / 'api_reference' / 'api_ui.rst'
_AUTOMODULE_RE = re.compile(r'^\.\.\s+automodule::\s+(\S+)\s*$', re.MULTILINE)
_MOCKED = ('PyQt6', 'matplotlib')


def _documented_ui_modules() -> list[str]:
    """Return every module name the UI API reference chapter documents.

    Returns:
        The dotted module names named by ``automodule`` directives, in file order.
    """
    return _AUTOMODULE_RE.findall(_API_UI_RST.read_text())


def test_api_ui_chapter_documents_modules() -> None:
    """The chapter names at least one module, so an empty parse cannot pass silently."""
    assert _documented_ui_modules()


@pytest.mark.parametrize('module_name', _documented_ui_modules())
def test_documented_ui_module_imports_with_qt_mocked(module_name: str) -> None:
    """Importing the module under Sphinx's mocks succeeds, as the docs build requires.

    Parameters:
        module_name: Dotted name of a module the UI API reference documents.
    """
    program = textwrap.dedent(f"""
        import importlib
        from sphinx.ext.autodoc.mock import mock

        with mock({list(_MOCKED)!r}):
            importlib.import_module({module_name!r})
    """)
    # This tree's src is prepended so the subprocess exercises the checkout under
    # test rather than whatever copy an editable install happens to point at.
    env = dict(os.environ)
    existing = env.get('PYTHONPATH')
    env['PYTHONPATH'] = f'{_SRC_DIR}{os.pathsep}{existing}' if existing else str(_SRC_DIR)
    completed = subprocess.run(
        [sys.executable, '-c', program],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, (
        f'{module_name} is documented but fails to import with Qt mocked, so '
        f'sphinx-build -W will fail on it:\n{completed.stderr}'
    )
