"""What a consuming program's run log says about the index it was given.

A program that reads navigation records through an index writes the URL down
twice: once inside the command line the run-environment banner records, and
once on a line of its own naming the index the run resolved.  Both reach the
run's log file and its terminal, and a connection URL can carry a database
password.  The second line is what is pinned here, on the bytes each program
that writes one really persists.

A ``sqlite:`` URL is returned by the masking rule exactly as it came -- it names
a filesystem path, which has no credentials -- so only a server URL can hold
these programs to masking anything at all.  The runs below therefore name a
PostgreSQL index, and they name it in a mode that reads no navigation record,
so the line is written and nothing tries to open a database that does not exist.
"""

from pathlib import Path

import pytest
from tests.spindoctor.cli.conftest import backplane_argv, mosaic_argv, run_program

from spindoctor.cli import sd_backplanes, sd_mosaic

LEFT = 'sup3r'
"""First half of the password, distinctive enough that finding it is a leak."""

RIGHT = 's3cr3t'
"""Second half, so a rule that hides only part of a password is still caught."""

_PASSWORD = f'{LEFT}%40%3A%2F%3F%23{RIGHT}'
"""The password as a URL carries it, around every character that delimits one.

An ``@`` ends the credentials, a ``:`` starts a port, a ``/`` starts a path, a
``?`` starts a query and a ``#`` starts a fragment, so a rule that stops at the
first of them leaves the rest of the password in the log.
"""

_HOST = 'db.example:5432/spindoctor'
"""Everything after the credentials, which the line is read to learn."""

INDEX_URL = f'postgresql+psycopg://us%40er:{_PASSWORD}@{_HOST}'
"""A server index URL under a user name that itself carries an at-sign."""

# Each program, with the command line that reaches its banner without opening
# an index: the backplane stage under --dry-run, and the mosaic pass under a
# dry run that also skips the mosaic combination.
_RUNS = {
    'sd_backplanes': (sd_backplanes, backplane_argv, ('--dry-run',)),
    'sd_mosaic': (sd_mosaic, mosaic_argv, ('--dry-run', '--skip-mosaic')),
}


def _run_log(program: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run one program over no images and return the log file it persisted.

    The file rather than the terminal: an unattended run is where a password
    would sit longest, and it is the copy handed to whoever is asked to look at
    a failed run.

    Parameters:
        program: Which of the two consuming programs to run.
        tmp_path: Directory the run's roots are placed under.
        monkeypatch: Patcher, used for ``sys.argv``.

    Returns:
        The text of the run's main log file.
    """
    module, argv_for, flags = _RUNS[program]
    log_root = tmp_path / 'logs'
    run_program(
        module,
        argv_for(tmp_path, INDEX_URL, '--log-root', log_root.as_posix(), *flags),
        monkeypatch,
    )
    written = sorted(log_root.rglob('main_*.log'))
    assert len(written) == 1
    return written[0].read_text(encoding='utf-8')


@pytest.mark.parametrize('program', sorted(_RUNS))
def test_the_index_line_a_run_logs_carries_no_password(
    program: str,
    tmp_path: Path,
    datasetless: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The line naming the resolved index is masked like the command line is.

    Parameters:
        program: Which consuming program is under test.
        tmp_path: Directory the run's roots are placed under.
        datasetless: Fixture emptying the enumeration.
        monkeypatch: Patcher, used for ``sys.argv``.
    """
    assert LEFT not in _run_log(program, tmp_path, monkeypatch)


@pytest.mark.parametrize('program', sorted(_RUNS))
def test_no_tail_of_that_password_reaches_the_run_log_either(
    program: str,
    tmp_path: Path,
    datasetless: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rule stopping at the first URL delimiter would leave a working password.

    Parameters:
        program: Which consuming program is under test.
        tmp_path: Directory the run's roots are placed under.
        datasetless: Fixture emptying the enumeration.
        monkeypatch: Patcher, used for ``sys.argv``.
    """
    assert RIGHT not in _run_log(program, tmp_path, monkeypatch)


@pytest.mark.parametrize('program', sorted(_RUNS))
def test_the_line_still_names_the_index_it_resolved(
    program: str,
    tmp_path: Path,
    datasetless: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control: a run that logged no URL at all would pass both above.

    Which index a run resolved, and therefore which of the three resolution
    levels supplied it, is the whole reason the line is written.

    Parameters:
        program: Which consuming program is under test.
        tmp_path: Directory the run's roots are placed under.
        datasetless: Fixture emptying the enumeration.
        monkeypatch: Patcher, used for ``sys.argv``.
    """
    line = f'Results index: postgresql+psycopg://us%40er:***@{_HOST}'
    assert line in _run_log(program, tmp_path, monkeypatch)
