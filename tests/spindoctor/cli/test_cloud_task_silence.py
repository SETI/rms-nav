"""Tests that a cloud task emits nothing on a real terminal.

The rest of the logging tests capture with ``capsys``, which replaces
``sys.stdout`` and ``sys.stderr`` inside the test process.  That is enough for
records written through Python, but the guarantee here is about the file
descriptors a worker's terminal actually is, and the handler that
``cloud_tasks`` installs on the root logger binds a stream at construction.
So this one runs a real subprocess, with ``logging.basicConfig`` installed
exactly as a worker installs it, and measures the bytes on each descriptor.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Every level, so nothing is passing merely for being below a threshold, and
# an exception, whose traceback is the bulkiest thing a per-image log carries.
_CHILD = """
import logging
import sys

logging.basicConfig(level=logging.DEBUG)

from filecache import FCPath
from spindoctor.config import IMAGE_LOGGER, MAIN_LOGGER
from spindoctor.config.config import Config
from spindoctor.config.logging_config import (
    build_cloud_task_logging,
    build_image_log_handlers,
)
import argparse

config = Config()
config.read_config()
log_root = FCPath(sys.argv[1])

run_logging = build_cloud_task_logging(
    'sd_offset',
    argparse.Namespace(log_root=log_root.as_posix(), log_level=['DEBUG']),
    config,
)
handlers, path = build_image_log_handlers(
    'nav', 'vol/N1', run_logging.sinks, run_logging.levels, timestamp='STAMP'
)
try:
    with IMAGE_LOGGER.open('IMAGE', handler=handlers):
        IMAGE_LOGGER.debug('CANARY-DEBUG')
        IMAGE_LOGGER.info('CANARY-INFO')
        IMAGE_LOGGER.warning('CANARY-WARNING')
        IMAGE_LOGGER.error('CANARY-ERROR')
        IMAGE_LOGGER.critical('CANARY-CRITICAL')
        try:
            raise RuntimeError('CANARY-EXCEPTION')
        except RuntimeError:
            IMAGE_LOGGER.exception('CANARY-EXCEPTION')
    MAIN_LOGGER.info('CANARY-MAIN')
    IMAGE_LOGGER.warning('CANARY-OUT-OF-SCOPE')
finally:
    for handler in handlers:
        handler.close()

with open(sys.argv[2], 'w') as stream:
    stream.write(path.as_posix())
"""


@pytest.fixture(scope='module')
def task_output(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str, str]:
    """Run one cloud task in a worker-like subprocess.

    Module-scoped: the subprocess costs a full interpreter start and every
    assertion below reads the same run.

    Parameters:
        tmp_path_factory: Fixture used to make the run's directory.

    Yields:
        Tuple of the child's stdout, its stderr, and its image log text.
    """
    directory = tmp_path_factory.mktemp('cloud_task_silence')
    script = directory / 'child.py'
    script.write_text(_CHILD)
    path_file = directory / 'log_path.txt'

    completed = subprocess.run(
        [sys.executable, str(script), str(directory / 'logs'), str(path_file)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    log_text = Path(path_file.read_text()).read_text()
    return completed.stdout, completed.stderr, log_text


def test_a_cloud_task_writes_nothing_to_stdout(task_output: tuple[str, str, str]) -> None:
    """Not one byte reaches the descriptor the worker reports progress on."""
    assert task_output[0] == ''


def test_a_cloud_task_writes_nothing_to_stderr(task_output: tuple[str, str, str]) -> None:
    """Nor to stderr, where the root handler would otherwise re-emit it all."""
    assert task_output[1] == ''


@pytest.mark.parametrize(
    'canary',
    [
        'CANARY-DEBUG',
        'CANARY-INFO',
        'CANARY-WARNING',
        'CANARY-ERROR',
        'CANARY-CRITICAL',
        'CANARY-EXCEPTION',
    ],
)
def test_the_image_log_is_complete(task_output: tuple[str, str, str], canary: str) -> None:
    """Silence on the terminal is not silence everywhere.

    Every level reaches the per-image file, so what the terminal is spared is
    duplication rather than the record itself.
    """
    assert canary in task_output[2]


def test_the_exception_keeps_its_traceback(task_output: tuple[str, str, str]) -> None:
    """The traceback survives too, since the image log is where it now lives."""
    assert 'RuntimeError' in task_output[2]
