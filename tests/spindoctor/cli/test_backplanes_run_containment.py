"""What one unusable image costs a backplane run.

Backplane generation is per-image work with no cross-image state, so one
image's failure is that image's failure.  A driver that let it out ends the
enumeration, discards every image after it with no record of how many there
were, and makes the rerun start from the beginning -- which is the difference
between a batch over a volume finishing with a count of what went wrong and a
batch stopping at the first hand-edited document in it.

The program is driven through its own ``main``, with only the dataset
enumeration and the per-image stage replaced, so what is exercised is the loop
the program really runs.
"""

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from spindoctor.cli import sd_backplanes
from spindoctor.dataset.dataset import ImageFile, ImageFiles

_STUBS = ('COISS_2001/N1000000000_1_CALIB', 'COISS_2001/N1000000001_1_CALIB')


class _TwoImages:
    """A dataset enumerating two images and taking no selection arguments."""

    def add_selection_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add nothing: this run selects both of its images unconditionally.

        Parameters:
            parser: The program's parser.
        """

    def yield_image_files_from_arguments(self, arguments: argparse.Namespace) -> Iterator[Any]:
        """Yield one batch per image, in a fixed order.

        Parameters:
            arguments: Ignored.

        Yields:
            One single-image batch per stub.
        """
        for stub in _STUBS:
            name = stub.rsplit('/', 1)[-1]
            yield ImageFiles(
                image_files=[
                    ImageFile(
                        image_file_url=FCPath(f'/holdings/{name}.IMG'),
                        label_file_url=FCPath(f'/holdings/{name}.LBL'),
                        results_path_stub=stub,
                        index_file_row={},
                    )
                ]
            )


def _argv(tmp_path: Path) -> list[str]:
    """Return a backplane command line naming both roots and no index.

    Parameters:
        tmp_path: Directory the roots are placed under.

    Returns:
        The arguments, without the program name.
    """
    return [
        'coiss_saturn',
        '--nav-results-root',
        (tmp_path / 'nav').as_posix(),
        '--backplane-results-root',
        (tmp_path / 'backplanes').as_posix(),
        '--log-root',
        (tmp_path / 'logs').as_posix(),
        '--no-log-main-to-console',
    ]


def _run_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: Any) -> str:
    """Run the driver over two images with the per-image stage replaced.

    Parameters:
        tmp_path: Directory the roots are placed under.
        monkeypatch: Patcher, which reverts after the test.
        stage: Stands in for ``generate_backplanes_image_files``.

    Returns:
        The text of the run's log, which is where a batch reports what became
        of each of its images.
    """
    monkeypatch.setattr(sd_backplanes, 'dataset_name_to_class', lambda _name: _TwoImages)
    monkeypatch.setattr(sd_backplanes, 'generate_backplanes_image_files', stage)
    monkeypatch.setattr('sys.argv', ['sd_backplanes', *_argv(tmp_path)])
    sd_backplanes.main()
    return '\n'.join(path.read_text() for path in sorted((tmp_path / 'logs').rglob('*.log')))


def _failing_first(attempted: list[str]) -> Any:
    """Build a stage that raises for the first image it is handed.

    Parameters:
        attempted: List each attempt's stub is appended to.

    Returns:
        The stage.
    """

    def stage(_obs_class: Any, image_files: ImageFiles, **_kwargs: Any) -> dict[str, Any]:
        stub = image_files.image_files[0].results_path_stub
        attempted.append(stub)
        if stub == _STUBS[0]:
            raise ValueError('the record is shaped like a defect')
        return {'status': 'success'}

    return stage


def test_one_unreadable_image_does_not_end_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The image after the failure is still attempted.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    attempted: list[str] = []
    _run_with(tmp_path, monkeypatch, _failing_first(attempted))
    assert attempted == list(_STUBS)


def test_the_run_says_which_image_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure counted and not reported is a total that does not add up.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    log_text = _run_with(tmp_path, monkeypatch, _failing_first([]))
    assert 'N1000000000_1_CALIB.LBL' in log_text


def test_the_run_says_what_went_wrong_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming the image without naming the cause sends the reader to a log per image.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    log_text = _run_with(tmp_path, monkeypatch, _failing_first([]))
    assert 'the record is shaped like a defect' in log_text


def test_the_run_records_the_traceback_of_an_unexpected_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An image can fail before it has a log of its own, leaving this the only account.

    The pointing lookup and the observation setup both run before the stage
    opens the image's log, so a defect in either is diagnosable only from what
    the run's log kept.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    log_text = _run_with(tmp_path, monkeypatch, _failing_first([]))
    assert 'Traceback (most recent call last)' in log_text


def test_that_traceback_names_the_frame_that_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The message alone does not say where it came from, which is what a traceback is for.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    log_text = _run_with(tmp_path, monkeypatch, _failing_first([]))
    assert 'in stage' in log_text


def test_that_traceback_names_the_exception_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two failures can carry the same message and differ only in their type.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    log_text = _run_with(tmp_path, monkeypatch, _failing_first([]))
    assert 'ValueError: the record is shaped like a defect' in log_text


def test_the_closing_summary_counts_the_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that failed everything must not read like one that processed everything.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    log_text = _run_with(tmp_path, monkeypatch, _failing_first([]))
    assert '1 done, 0 skipped, 1 failed' in log_text


def test_an_image_nothing_navigated_is_skipped_rather_than_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing navigated it, which is an expected outcome and not a defect.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """

    def stage(_obs_class: Any, image_files: ImageFiles, **_kwargs: Any) -> dict[str, Any]:
        if image_files.image_files[0].results_path_stub == _STUBS[0]:
            raise FileNotFoundError('no navigation record for this image')
        return {'status': 'success'}

    log_text = _run_with(tmp_path, monkeypatch, stage)
    assert '1 done, 1 skipped, 0 failed' in log_text


def test_a_skipped_navigation_is_counted_as_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And so is an image whose navigation did not succeed, which writes no product.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """

    def stage(_obs_class: Any, image_files: ImageFiles, **_kwargs: Any) -> dict[str, Any]:
        if image_files.image_files[0].results_path_stub == _STUBS[0]:
            return {'status': 'skipped', 'nav_status': 'error'}
        return {'status': 'success'}

    log_text = _run_with(tmp_path, monkeypatch, stage)
    assert '1 done, 1 skipped, 0 failed' in log_text
