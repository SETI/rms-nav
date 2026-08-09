"""Tests that every per-image backend floors its section at the image level.

A handler is built at the most verbose level any module could ask for, so that
raising one component actually writes.  The discrimination is then done by the
level the section is opened at.  A backend that opens its image section without
one leaves the handler as the only filter, and the handler is deliberately the
permissive half of the pair -- so the image level stops governing that backend
entirely, silently.

This is checked over the backends together rather than one at a time, because
the failure is invisible in the backend that has it: its logs simply contain
more than they should, which reads as a verbose run rather than as a defect.
"""

import ast
from pathlib import Path

import pdslogger
import pytest
from filecache import FCPath

from spindoctor.config import IMAGE_LOGGER
from spindoctor.config.logging_config import (
    LogLevels,
    LogSinks,
    build_image_log_handlers,
    set_log_levels,
)

_STAMP = '2026-07-29T12-00-00'
_SRC = FCPath(Path(__file__).resolve().parents[3]) / 'src' / 'spindoctor'

# Every site that opens a per-image section, and the backend it opens it for.
_IMAGE_SECTION_SITES = [
    ('navigate_image_files.py', 'nav'),
    ('cli/sd_offset.py', 'nav'),
    ('cli/backplanes/backplanes.py', 'backplanes'),
    ('cli/sd_mosaic.py', 'reproj'),
    ('cli/sd_mosaic_cloud_tasks.py', 'reproj'),
    ('cli/sd_create_ck.py', 'ck'),
]


def test_the_image_level_is_a_floor_not_a_suggestion(tmp_path: Path) -> None:
    """A record below the image level does not reach the log.

    The case that matters is an image level with a module raised above it:
    raising the module opens the handler, and only the section floor is left to
    suppress everything else.  With the levels equal, the handler hides a
    missing floor completely.
    """
    levels = LogLevels(image='WARNING', modules={'obs': 'DEBUG'})
    set_log_levels(levels)
    handlers, path = build_image_log_handlers(
        'nav', 'vol/N1', LogSinks(log_root=FCPath(tmp_path)), levels, timestamp=_STAMP
    )
    try:
        with IMAGE_LOGGER.open('IMAGE', handler=handlers, level=levels.image_section_level()):
            IMAGE_LOGGER.info('BELOW-THE-IMAGE-LEVEL')
    finally:
        for handler in handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
    assert path is not None
    with path.open('r') as stream:
        assert 'BELOW-THE-IMAGE-LEVEL' not in stream.read()


def test_a_raised_module_still_writes_below_it(tmp_path: Path) -> None:
    """Flooring the section does not silence the module that was raised."""
    levels = LogLevels(image='WARNING', modules={'obs': 'DEBUG'})
    set_log_levels(levels)
    handlers, path = build_image_log_handlers(
        'nav', 'vol/N2', LogSinks(log_root=FCPath(tmp_path)), levels, timestamp=_STAMP
    )
    try:
        with (
            IMAGE_LOGGER.open('IMAGE', handler=handlers, level=levels.image_section_level()),
            IMAGE_LOGGER.open('OBS', level=levels.section_level_for('obs')),
        ):
            IMAGE_LOGGER.debug('RAISED-MODULE-RECORD')
    finally:
        for handler in handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
    assert path is not None
    with path.open('r') as stream:
        assert 'RAISED-MODULE-RECORD' in stream.read()


@pytest.mark.parametrize(('relative', 'backend'), _IMAGE_SECTION_SITES)
def test_every_backend_passes_a_level_to_its_image_section(relative: str, backend: str) -> None:
    """No backend opens its per-image section without a level.

    Read from the source rather than exercised, because the omission is only
    observable in a configuration the backend's own tests need not use, and
    because a new backend should be caught by being added to the list rather
    than by someone remembering this rule.
    """
    unfloored = [
        node.lineno
        for node in _image_section_opens(relative)
        if not any(keyword.arg == 'level' for keyword in node.keywords)
    ]
    assert unfloored == []


def _image_section_opens(relative: str) -> list[ast.Call]:
    """Return every ``open(handler=...)`` call in a module.

    Parameters:
        relative: Path of the module under the package.

    Returns:
        The calls that open a section with handlers attached.
    """
    tree = ast.parse((_SRC / relative).read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'open'
        and any(keyword.arg == 'handler' for keyword in node.keywords)
    ]


@pytest.mark.parametrize(('relative', 'backend'), _IMAGE_SECTION_SITES)
def test_every_backend_floors_at_the_image_level(relative: str, backend: str) -> None:
    """And the level it passes is the image level, not something of its own.

    Read from the AST rather than matched in the call's text, so that
    reformatting the argument list cannot change what this checks.
    """
    wrong = [
        node.lineno
        for node in _image_section_opens(relative)
        for keyword in node.keywords
        if keyword.arg == 'level' and not _is_image_section_level(keyword.value)
    ]
    assert wrong == []


def _is_image_section_level(value: ast.expr) -> bool:
    """Whether an argument expression is a call to ``image_section_level()``.

    Parameters:
        value: The expression passed as ``level``.

    Returns:
        True when it is that call.
    """
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == 'image_section_level'
    )
