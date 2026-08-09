"""Fixtures and helpers shared by the driver test modules at this level.

Most of what is here belongs to the ``sd_create_ck`` end-to-end tests.  Those
live in two modules -- the products of a clean run and its refusals -- and both
run the real program over prepared trees.  The trees, and the guard that undoes
what a run furnished into the process-global SPICE pool, are built here; the
plain builders they use live in ``sd_create_ck_helpers``.

:func:`help_text` and :func:`cloud_task_parser` are shared more widely: a
program's command-line surface is asserted against the parser the program
builds for itself rather than against a reconstruction of it, and one spelling
of that serves every module that asks.  An interactive program is run with
``--help``, which is the surface a user meets; a cloud-task driver builds its
parser inside ``async_main`` and hands it straight to the worker, so the worker
is intercepted and the parser taken from it.

Nothing here is autouse, so the other test packages under this directory are
untouched.
"""

import argparse
import asyncio
import contextlib
import importlib
import io
import sys
from collections.abc import Iterator
from pathlib import Path

import cspyce
import pytest
from tests.spindoctor.cli.ck.ck_helpers import (
    CASSINI_CAMERA_FRAME,
    CASSINI_CK_FRAME_ID,
    baseline_angular_velocity,
    baseline_attitude,
    image_metadata,
    write_baseline_ck,
    write_support_kernels,
)
from tests.spindoctor.cli.sd_create_ck_helpers import (
    BASELINE_A,
    CASSINI_SCLK_ID,
    IMAGE_A_ET,
    IMAGE_B_ET,
    KERNEL_NAMES,
    camera_attitude,
    corrected,
    image_document,
    write_kernels,
    write_metadata,
)


def help_text(program: str, argv: list[str]) -> str:
    """Return what ``program --help`` prints.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.
        argv: Arguments preceding ``--help``, for a program that reads its
            dataset or mode from argv before parsing.

    Returns:
        The help text.
    """
    module = importlib.import_module(f'spindoctor.cli.{program}')
    buffer = io.StringIO()
    saved = sys.argv
    sys.argv = [program, *argv, '--help']
    try:
        with contextlib.redirect_stdout(buffer), contextlib.suppress(SystemExit):
            module.main()
    finally:
        sys.argv = saved
    return buffer.getvalue()


def cloud_task_parser(program: str) -> argparse.ArgumentParser:
    """Return the parser a cloud-task driver builds for itself.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.

    Returns:
        The parser the driver would have run with.
    """
    module = importlib.import_module(f'spindoctor.cli.{program}')
    captured: dict[str, argparse.ArgumentParser] = {}

    class _CapturedError(Exception):
        """Raised to stop the driver once its parser has been seen."""

    def _intercept(*args: object, **kwargs: object) -> None:
        parser = kwargs.get('argparser')
        assert isinstance(parser, argparse.ArgumentParser)
        captured['parser'] = parser
        raise _CapturedError

    real_worker = module.Worker
    module.Worker = _intercept  # type: ignore[attr-defined]
    try:
        with contextlib.suppress(_CapturedError):
            asyncio.run(module.async_main())
    finally:
        module.Worker = real_worker  # type: ignore[attr-defined]
    return captured['parser']


class _Furnished:
    """The pool as it was before a run, so a run's additions can be undone."""

    def __init__(self) -> None:
        """Record every kernel currently furnished."""
        self.before = self._loaded()

    @staticmethod
    def _loaded() -> list[str]:
        """Return the paths of every furnished kernel, in load order."""
        return [str(cspyce.kdata(at, 'ALL')[0]) for at in range(int(cspyce.ktotal('ALL')))]

    def restore(self) -> None:
        """Unload every kernel furnished since this object was built."""
        for path in reversed(self._loaded()):
            if path not in self.before:
                cspyce.unload(path)


@pytest.fixture
def pool_restored() -> Iterator[None]:
    """Undo whatever the driver furnished, leaving the process pool as found."""
    guard = _Furnished()
    try:
        yield
    finally:
        guard.restore()


@pytest.fixture
def run_tree(tmp_path: Path) -> dict[str, Path]:
    """Build a kernel directory, a results root and four images for one run.

    Returns:
        The run's directories, under the keys ``kernels``, ``results`` and
        ``output``.
    """
    kernels = tmp_path / 'kernels'
    results = tmp_path / 'results'
    output = tmp_path / 'output'
    original_a, original_b = write_kernels(kernels)
    drifted = corrected(original_a)
    write_metadata(
        results,
        'vol/A_CALIB',
        image_document(
            image_name='A_CALIB',
            midtime=IMAGE_A_ET,
            cmatrix_original=original_a,
            cmatrix=corrected(original_a),
        ),
    )
    write_metadata(
        results,
        'vol/B_CALIB',
        image_document(
            image_name='B_CALIB',
            midtime=IMAGE_B_ET,
            cmatrix_original=original_b,
            cmatrix=corrected(original_b),
        ),
    )
    write_metadata(
        results,
        'vol/C_CALIB',
        image_document(
            image_name='C_CALIB',
            midtime=IMAGE_A_ET,
            cmatrix_original=original_a,
            cmatrix=None,
            status='failed',
        ),
    )
    write_metadata(
        results,
        'vol/D_CALIB',
        image_document(
            image_name='D_CALIB',
            midtime=IMAGE_A_ET,
            cmatrix_original=drifted,
            cmatrix=corrected(drifted),
        ),
    )
    return {'kernels': kernels, 'results': results, 'output': output}


@pytest.fixture
def refused_second_file_tree(tmp_path: Path) -> dict[str, Path]:
    """Build a run whose second output file cannot be built at all.

    Two images navigate against two baselines, and the second baseline carries
    no angular velocity, which no segment can express.  The corrected files are
    written in name order, so the first one is buildable and the second is the
    refusal: a run that wrote as it went would leave the first file behind.

    Returns:
        The run's directories, under the keys ``kernels``, ``results`` and
        ``output``.
    """
    kernels = tmp_path / 'kernels'
    results = tmp_path / 'results'
    output = tmp_path / 'output'
    original_a, original_b = write_kernels(kernels, angular_velocity_in_b=False)
    for stub, name, midtime, original in (
        ('vol/A_CALIB', 'A_CALIB', IMAGE_A_ET, original_a),
        ('vol/B_CALIB', 'B_CALIB', IMAGE_B_ET, original_b),
    ):
        write_metadata(
            results,
            stub,
            image_document(
                image_name=name,
                midtime=midtime,
                cmatrix_original=original,
                cmatrix=corrected(original),
            ),
        )
    return {'kernels': kernels, 'results': results, 'output': output}


@pytest.fixture
def straddling_tree(tmp_path: Path) -> dict[str, Path]:
    """Build a run whose one image outlasts the baseline that reproduces it.

    The baseline covers a second either side of the exposure midtime and the
    exposure runs for four, so the midtime reproduces -- which is what pairs
    the image with this baseline -- and the segment's start and stop records
    then have no pointing to read.

    Returns:
        The run's directories, under the keys ``kernels``, ``results`` and
        ``output``.
    """
    kernels = tmp_path / 'kernels'
    results = tmp_path / 'results'
    output = tmp_path / 'output'
    support = write_support_kernels(kernels)
    for path in support:
        cspyce.furnsh(str(path))
    baseline = kernels / BASELINE_A
    try:
        write_baseline_ck(
            baseline,
            ck_frame_id=CASSINI_CK_FRAME_ID,
            sclk_id=CASSINI_SCLK_ID,
            epochs=[IMAGE_A_ET - 1.0, IMAGE_A_ET, IMAGE_A_ET + 1.0],
            attitude=baseline_attitude,
            angular_velocity=baseline_angular_velocity,
        )
        cspyce.furnsh(str(baseline))
        original = camera_attitude(IMAGE_A_ET)
    finally:
        for path in reversed([*support, baseline]):
            cspyce.unload(str(path))
    metadata = image_metadata(
        image_name='G_CALIB',
        cmatrix=corrected(original),
        cmatrix_original=original,
        camera_frame=CASSINI_CAMERA_FRAME,
        ck_frame_id=CASSINI_CK_FRAME_ID,
        start_et=IMAGE_A_ET - 2.0,
        stop_et=IMAGE_A_ET + 2.0,
        status='success',
        instrument='coiss',
        camera='NAC',
        shutter_mode='NACONLY',
        kernels=KERNEL_NAMES,
        sclk_midtime='1/1484573295.118',
        offset=(-3.25, 1.125),
        sigma_px=(0.0625, 0.0313),
        confidence=0.8125,
        confidence_rank='high',
        status_reason='ensemble_agreement',
    )
    write_metadata(results, 'vol/G_CALIB', metadata)
    return {'kernels': kernels, 'results': results, 'output': output}
