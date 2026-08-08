"""Real-holdings integration tests for ``spindoctor.cli.ck.assignment``.

Assignment is the phase's whole point and the part real kernels can falsify:
whether the file an image navigated against is the one picked out of everything
the holdings offer.  These tests build the index over the real Cassini and
Voyager C-kernel directories and run the assignment against real epochs, real
basenames and real frame kernels.

An image's recorded uncorrected attitude is reconstructed here the way the
pipeline records it -- ``pxform`` at the midtime for Cassini, the
tolerance-snapped platform lookup composed with the fixed platform-to-camera
rotation for Voyager -- rather than by navigating an image, because loading one
through oops furnishes C-kernels into the process, and assignment refuses to
run with any furnished.  What is real here is the kernels, the epochs and the
names; what is computed is the one number a navigation run would have recorded.

Not navigating an image here is not enough on its own, because the kernel pool
is process-global and nothing unloads what an earlier test furnished.  Every
other test in the integration tier navigates a real image, so in a full run
these would find dozens of C-kernels already in the pool and refuse to run at
all.  The pool is therefore emptied for this module and furnished again
afterwards, so what these tests measure is what they furnished.
"""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.integration

_RESOURCES = os.environ.get('OOPS_RESOURCES', '')
_SPICE_ROOT = Path(_RESOURCES) / 'SPICE'
_CASSINI_ROOT = _SPICE_ROOT / 'Cassini'
_VOYAGER_ROOT = _SPICE_ROOT / 'Voyager'

if (
    len(_RESOURCES) == 0
    or not (_CASSINI_ROOT / 'CK-reconstructed').is_dir()
    or not (_VOYAGER_ROOT / 'CK').is_dir()
):
    pytest.skip(
        'OOPS_RESOURCES does not name a local SPICE tree holding the Cassini and Voyager '
        'C-kernels; skipping C-kernel assignment holdings tests',
        allow_module_level=True,
    )

import cspyce  # noqa: E402  (guarded import)

from spindoctor.cli.ck.assignment import assign_images  # noqa: E402  (guarded import)
from spindoctor.cli.ck.images import (  # noqa: E402  (guarded import)
    ImageEntry,
    OmissionReason,
)
from spindoctor.cli.ck.index import (  # noqa: E402  (guarded import)
    CkIndex,
    KernelClass,
    build_ck_index,
)
from spindoctor.cli.ck.pointing import NDArrayFloatType  # noqa: E402  (guarded import)
from tests.kernel_pool import isolated_kernel_pool  # noqa: E402  (guarded import)

_CASSINI_CK_DIRS = (
    'CK-reconstructed',
    'CK-gapfill',
    'CK-predicted',
    'CK-predicted-v02',
    'CK-cruise',
    'CK-jup',
)
_CASSINI_CK_FRAME_ID = -82000
_CASSINI_SCLK_ID = -82
_NAC_FRAME = 'CASSINI_ISS_NAC'
_WAC_FRAME = 'CASSINI_ISS_WAC'

_VOYAGER_CK_FRAME_ID = -32100
_VOYAGER_SCLK_ID = -32
_VOYAGER_CAMERA_FRAME = 'VG2_ISSNA'
_VOYAGER_KERNEL = 'vg2_sat_version1_type1_iss_sedr.bc'

# A real simultaneous Cassini exposure from COISS_2038: both cameras, one
# spacecraft clock reading, both frames in the volume's index with
# SHUTTER_MODE_ID = BOTSIM.
_BOTSIM_CLOCK = '1569962762.000'
_BOTSIM_NAC = 'N1569962762_1.IMG'
_BOTSIM_WAC = 'W1569962762_1.IMG'

# The kernel that actually supplies that epoch, and the three others whose
# coverage windows also contain it.
_RECONSTRUCTED_AT_BOTSIM = '07272_07277ra.bc'
_GAPFILL_AT_BOTSIM = '07001_08001pa_gapfill_v14.bc'
_PREDICTED_AT_BOTSIM = '07265_07304pe_live.bc'
_AS_FLOWN_AT_BOTSIM = '07265_07304py_as_flown.bc'

# An epoch a day and a half earlier at which the two predicted kernels agree to
# well inside the reproduction bound while the reconstructed one does not, so
# the tie-break between two files of one class decides which carries the
# segment.
_TWO_PREDICTED_AGREE_ET = 244411914.0

# Every exposure below is short; the value only widens a Voyager lookup
# tolerance, and these tests do not depend on that.
_EXPOSURE_S = 2.0

# How far past a real Voyager pointing record to place an exposure: inside the
# base tolerance of 800 ticks, and outside it.
_INSIDE_BASE_TOLERANCE_TICKS = 100.0
_OUTSIDE_BASE_TOLERANCE_TICKS = 900.0

# Locating a record in a discrete kernel means asking at the widest tolerance
# the navigated lookup could have used, across the segment's window.
_FALLBACK_TOL_TICKS = 80000.0
_RECORD_SEARCH_STEPS = 200

# The platform-to-camera rotation is a constant the frame kernel defines, read
# at the epoch oops reads it at.
_FIXED_ROTATION_ET = 0.0


def _furnish_all(paths: list[Path]) -> Iterator[None]:
    """Furnish kernels for one test and unload exactly those afterwards.

    Parameters:
        paths: The kernels to furnish, in order.

    Yields:
        Nothing; the kernels are furnished for the body of the test.
    """
    for path in paths:
        cspyce.furnsh(str(path))
    try:
        yield
    finally:
        for path in reversed(paths):
            cspyce.unload(str(path))


@pytest.fixture(scope='module', autouse=True)
def empty_kernel_pool() -> Iterator[None]:
    """Run this module's tests against a pool holding nothing they did not furnish.

    Assignment refuses to run with any C-kernel furnished, which is the point:
    any other C-kernel answers the same lookups as the candidate under test.
    That refusal is what a full integration run would hit here, since every
    other test in the tier navigates a real image and leaves its kernels
    behind.

    Yields:
        Nothing; the module's tests run against an empty pool.
    """
    with isolated_kernel_pool():
        yield


@pytest.fixture
def cassini_pool() -> Iterator[None]:
    """Furnish the real Cassini leapseconds, clock and frame kernels."""
    kernels = [_SPICE_ROOT / 'leapseconds.ker']
    kernels.extend(sorted((_CASSINI_ROOT / 'SCLK').glob('*.tsc')))
    kernels.append(_CASSINI_ROOT / 'FK' / 'cas_v43.tf')
    yield from _furnish_all(kernels)


@pytest.fixture
def voyager_pool() -> Iterator[None]:
    """Furnish the real Voyager leapseconds, clock and frame kernels."""
    kernels = [_SPICE_ROOT / 'leapseconds.ker']
    kernels.extend(sorted((_VOYAGER_ROOT / 'SCLK').glob('*.tsc')))
    kernels.extend(sorted((_VOYAGER_ROOT / 'FK').glob('*.tf')))
    yield from _furnish_all(kernels)


@pytest.fixture
def cassini_index(cassini_pool: None) -> CkIndex:
    """Index every real Cassini C-kernel directory.

    Parameters:
        cassini_pool: The furnished kernels the coverage read needs.

    Returns:
        The index.
    """
    return build_ck_index([_CASSINI_ROOT / name for name in _CASSINI_CK_DIRS])


@pytest.fixture
def voyager_index(voyager_pool: None) -> CkIndex:
    """Index the real Voyager C-kernel directory.

    Parameters:
        voyager_pool: The furnished kernels the coverage read needs.

    Returns:
        The index.
    """
    return build_ck_index([_VOYAGER_ROOT / 'CK'])


def _attitude_from(path: Path, camera_frame: str, et: float) -> NDArrayFloatType:
    """Read the J2000-to-camera attitude one kernel gives at one epoch.

    The kernel is furnished for the read and unloaded again, so the pool is
    left as assignment requires it: supporting kernels only.

    Parameters:
        path: The C-kernel to read.
        camera_frame: SPICE name of the camera frame.
        et: TDB seconds past J2000.

    Returns:
        The 3x3 rotation.
    """
    cspyce.furnsh(str(path))
    try:
        return np.asarray(cspyce.pxform('J2000', camera_frame, et), dtype=np.float64)
    finally:
        cspyce.unload(str(path))


def _cassini_entry(
    image_name: str,
    *,
    cmatrix_original: NDArrayFloatType,
    kernels: tuple[str, ...],
    start_et: float,
    camera: str,
    camera_frame: str = _NAC_FRAME,
    shutter_mode: str | None = None,
) -> ImageEntry:
    """Build one Cassini image's entry from a recorded uncorrected attitude.

    Parameters:
        image_name: Basename of the image.
        cmatrix_original: The uncorrected attitude the image recorded.
        kernels: The kernel basenames its provenance recorded.
        start_et: Exposure start, TDB seconds past J2000.
        camera: The camera that took it.
        camera_frame: SPICE name of that camera's frame.
        shutter_mode: The recorded shutter mode, or None to omit it.

    Returns:
        The entry.
    """
    flat = [float(value) for value in np.asarray(cmatrix_original).reshape(9)]
    observation: dict[str, Any] = {'image_name': image_name, 'camera': camera}
    if shutter_mode is not None:
        observation['shutter_mode'] = shutter_mode
    metadata = {
        'status': 'success',
        'observation': observation,
        'navigation_result': {
            'pointing': {
                # The corrected attitude is irrelevant to assignment, which
                # tests candidates against the uncorrected one; a real
                # correction is a few microradians, so the recorded value here
                # is the uncorrected attitude itself.
                'cmatrix': flat,
                'cmatrix_original': flat,
                'camera_frame': camera_frame,
                'ck_frame_id': _CASSINI_CK_FRAME_ID,
            },
            'times': {
                'start_et': start_et,
                'stop_et': start_et + _EXPOSURE_S,
                'midtime_et': start_et + _EXPOSURE_S / 2.0,
                'exposure_s': _EXPOSURE_S,
            },
            'provenance': {'spice_kernels': sorted(kernels)},
        },
    }
    return ImageEntry.from_metadata(metadata)


def _botsim_et() -> float:
    """Return the epoch of the real simultaneous exposure under test.

    Returns:
        TDB seconds past J2000 of the pair's spacecraft clock reading.
    """
    return float(cspyce.scs2e(_CASSINI_SCLK_ID, _BOTSIM_CLOCK))


def _all_cassini_basenames(index: CkIndex) -> tuple[str, ...]:
    """Return every basename the Cassini index holds.

    Recording all of them is what a batch run's provenance looks like: a
    superset accumulated across images, with no load order preserved.

    Parameters:
        index: The Cassini index.

    Returns:
        The basenames.
    """
    return tuple(sorted({ck_file.basename for ck_file in index.files}))


def test_the_real_epoch_offers_four_candidates(cassini_index: CkIndex) -> None:
    """Four real kernels cover the exposure, and only one supplied its pointing.

    This is the premise of the test below: the coverage filter cannot decide
    between them, which is why reproduction has to.
    """
    candidates = cassini_index.candidates(
        basenames=_all_cassini_basenames(cassini_index),
        ck_frame_id=_CASSINI_CK_FRAME_ID,
        et=_botsim_et(),
    )
    assert {ck_file.basename for ck_file in candidates} == {
        _RECONSTRUCTED_AT_BOTSIM,
        _GAPFILL_AT_BOTSIM,
        _PREDICTED_AT_BOTSIM,
        _AS_FLOWN_AT_BOTSIM,
    }


def test_a_real_image_is_paired_with_the_kernel_that_supplied_it(
    cassini_index: CkIndex,
) -> None:
    """The reconstructed kernel is picked out of the four that cover the epoch.

    The gapfill kernel advertises the epoch and has no pointing there at all,
    and the two predicted kernels answer with an attitude tens of microradians
    away -- four orders outside the reproduction bound.
    """
    et = _botsim_et()
    recorded = _attitude_from(
        _CASSINI_ROOT / 'CK-reconstructed' / _RECONSTRUCTED_AT_BOTSIM, _NAC_FRAME, et
    )
    entry = _cassini_entry(
        _BOTSIM_NAC,
        cmatrix_original=recorded,
        kernels=_all_cassini_basenames(cassini_index),
        start_et=et - _EXPOSURE_S / 2.0,
        camera='NAC',
    )
    assignments = assign_images([entry], cassini_index)
    assert assignments[0].baseline is not None
    assert assignments[0].baseline.basename == _RECONSTRUCTED_AT_BOTSIM


def test_a_real_image_the_holdings_no_longer_supply_is_refused(
    cassini_index: CkIndex,
) -> None:
    """An attitude no real candidate answers is refused rather than corrected.

    The recorded attitude is the one the predicted kernels give a day and a
    half from this epoch, so every candidate covering this one answers
    something else.  That is the baseline-drift detector working.
    """
    et = _botsim_et()
    elsewhere = _attitude_from(
        _CASSINI_ROOT / 'CK-predicted' / _PREDICTED_AT_BOTSIM, _NAC_FRAME, _TWO_PREDICTED_AGREE_ET
    )
    entry = _cassini_entry(
        _BOTSIM_NAC,
        cmatrix_original=elsewhere,
        kernels=_all_cassini_basenames(cassini_index),
        start_et=et - _EXPOSURE_S / 2.0,
        camera='NAC',
    )
    assignments = assign_images([entry], cassini_index)
    assert assignments[0].omission_reason is OmissionReason.NO_REPRODUCING_BASELINE


def test_two_real_kernels_of_one_class_are_separated_by_name(
    cassini_index: CkIndex,
) -> None:
    """Where two real kernels of a class agree, the greatest basename carries the segment.

    The two predicted sets agree at this epoch to about 5e-11 radians, inside
    the reproduction bound, so both reproduce and the tie-break is what
    decides.
    """
    recorded = _attitude_from(
        _CASSINI_ROOT / 'CK-predicted' / _PREDICTED_AT_BOTSIM, _NAC_FRAME, _TWO_PREDICTED_AGREE_ET
    )
    entry = _cassini_entry(
        _BOTSIM_NAC,
        cmatrix_original=recorded,
        kernels=_all_cassini_basenames(cassini_index),
        start_et=_TWO_PREDICTED_AGREE_ET - _EXPOSURE_S / 2.0,
        camera='NAC',
    )
    assignments = assign_images([entry], cassini_index)
    assert assignments[0].baseline is not None
    assert assignments[0].baseline.basename == _AS_FLOWN_AT_BOTSIM


def test_the_winning_class_outranks_the_other_real_classes(cassini_index: CkIndex) -> None:
    """The kernel picked for the real exposure is the reconstructed one by class."""
    et = _botsim_et()
    recorded = _attitude_from(
        _CASSINI_ROOT / 'CK-reconstructed' / _RECONSTRUCTED_AT_BOTSIM, _NAC_FRAME, et
    )
    entry = _cassini_entry(
        _BOTSIM_NAC,
        cmatrix_original=recorded,
        kernels=_all_cassini_basenames(cassini_index),
        start_et=et - _EXPOSURE_S / 2.0,
        camera='NAC',
    )
    assignments = assign_images([entry], cassini_index)
    assert assignments[0].baseline is not None
    assert assignments[0].baseline.kernel_class is KernelClass.RECONSTRUCTED


def test_a_real_simultaneous_pair_yields_one_segment_and_one_loser(
    cassini_index: CkIndex,
) -> None:
    """Both frames of a real BOTSIM exposure reproduce; only the narrow angle one is written."""
    et = _botsim_et()
    basenames = _all_cassini_basenames(cassini_index)
    baseline = _CASSINI_ROOT / 'CK-reconstructed' / _RECONSTRUCTED_AT_BOTSIM
    entries = [
        _cassini_entry(
            _BOTSIM_NAC,
            cmatrix_original=_attitude_from(baseline, _NAC_FRAME, et),
            kernels=basenames,
            start_et=et - _EXPOSURE_S / 2.0,
            camera='NAC',
            camera_frame=_NAC_FRAME,
            shutter_mode='BOTSIM',
        ),
        _cassini_entry(
            _BOTSIM_WAC,
            cmatrix_original=_attitude_from(baseline, _WAC_FRAME, et),
            kernels=basenames,
            start_et=et - _EXPOSURE_S / 2.0,
            camera='WAC',
            camera_frame=_WAC_FRAME,
            shutter_mode='BOTSIM',
        ),
    ]
    assignments = assign_images(entries, cassini_index)
    assert assignments[0].baseline is not None
    assert assignments[1].baseline is None
    assert assignments[1].omission_reason is OmissionReason.BOTSIM_LOSER


def _voyager_record(voyager_pool: None) -> tuple[float, NDArrayFloatType]:
    """Find one real Voyager pointing record and read the attitude it holds.

    The record is located with the widest tolerance the navigated lookup could
    have used, which is the only way to find one: a discrete kernel answers
    nothing between its records, and SPICE offers no way to enumerate them.
    The attitude is composed the way the pipeline composes a frozen Voyager
    attitude, from the platform lookup and the fixed platform-to-camera
    rotation, because the camera frame chains on a discrete platform frame that
    a zero-tolerance frame evaluation cannot resolve.

    Parameters:
        voyager_pool: The furnished supporting kernels.

    Returns:
        The record's encoded clock tick and the J2000-to-camera attitude there.
    """
    path = _VOYAGER_ROOT / 'CK' / _VOYAGER_KERNEL
    window = list(cspyce.ckcov(str(path), _VOYAGER_CK_FRAME_ID, False, 'SEGMENT', 0.0, 'TDB'))
    platform_to_camera = np.asarray(
        cspyce.pxform(
            str(cspyce.frmnam(_VOYAGER_CK_FRAME_ID)), _VOYAGER_CAMERA_FRAME, _FIXED_ROTATION_ET
        ),
        dtype=np.float64,
    )
    cspyce.furnsh(str(path))
    try:
        for et in np.linspace(window[0], window[1], _RECORD_SEARCH_STEPS):
            try:
                cmat, clkout = cspyce.ckgp(
                    _VOYAGER_CK_FRAME_ID,
                    float(cspyce.sce2c(_VOYAGER_SCLK_ID, float(et))),
                    _FALLBACK_TOL_TICKS,
                    'J2000',
                )
            except OSError:
                continue
            attitude: NDArrayFloatType = platform_to_camera @ np.asarray(cmat, dtype=np.float64)
            return float(clkout), attitude
        raise AssertionError('no pointing record found in the Voyager baseline')
    finally:
        cspyce.unload(str(path))


def _voyager_entry(midtime_et: float, attitude: NDArrayFloatType) -> ImageEntry:
    """Build a Voyager image's entry recording a frozen attitude.

    Parameters:
        midtime_et: The exposure midtime, TDB seconds past J2000.
        attitude: The J2000-to-camera attitude the frozen lookup gave.

    Returns:
        The entry.
    """
    flat = [float(value) for value in np.asarray(attitude).reshape(9)]
    metadata = {
        'status': 'success',
        'observation': {'image_name': 'C1205021_CALIB.IMG', 'camera': 'NAC'},
        'navigation_result': {
            'pointing': {
                'cmatrix': flat,
                'cmatrix_original': flat,
                'camera_frame': _VOYAGER_CAMERA_FRAME,
                'ck_frame_id': _VOYAGER_CK_FRAME_ID,
            },
            'times': {
                'start_et': midtime_et - _EXPOSURE_S / 2.0,
                'stop_et': midtime_et + _EXPOSURE_S / 2.0,
                'midtime_et': midtime_et,
                'exposure_s': _EXPOSURE_S,
            },
            'provenance': {'spice_kernels': [_VOYAGER_KERNEL]},
        },
    }
    return ImageEntry.from_metadata(metadata)


def test_a_real_discrete_baseline_answers_nothing_without_a_tolerance(
    voyager_pool: None,
) -> None:
    """A real Voyager baseline holds pointing only at its records.

    This is the premise of the two tests below and the reason the snapped
    lookup carries a tolerance at all: the records do not even sit on whole
    clock ticks, so a lookup with no tolerance misses one that is right there.
    """
    tick, _attitude = _voyager_record(voyager_pool)
    path = _VOYAGER_ROOT / 'CK' / _VOYAGER_KERNEL
    cspyce.furnsh(str(path))
    try:
        et = float(cspyce.sct2e(_VOYAGER_SCLK_ID, tick))
        whole_tick = float(cspyce.sce2t(_VOYAGER_SCLK_ID, et))
        with pytest.raises(OSError, match='CKINSUFFDATA'):
            cspyce.ckgp(_VOYAGER_CK_FRAME_ID, whole_tick, 0.0, 'J2000')
    finally:
        cspyce.unload(str(path))


@pytest.mark.parametrize(
    'offset_ticks',
    [_INSIDE_BASE_TOLERANCE_TICKS, _OUTSIDE_BASE_TOLERANCE_TICKS],
    ids=['inside-base-tolerance', 'inside-fallback-tolerance'],
)
def test_a_real_voyager_image_is_paired_through_a_snapped_lookup(
    voyager_index: CkIndex, voyager_pool: None, offset_ticks: float
) -> None:
    """A Voyager exposure near a real record is paired with the kernel holding it.

    One exposure sits inside the tolerance of the lookup that froze the
    observation frame; the other sits outside it and inside the tolerance of
    the frame oops falls back to.  Both are reproduced by the same real
    kernel.

    Parameters:
        offset_ticks: How far past the record the exposure midtime sits.
    """
    tick, attitude = _voyager_record(voyager_pool)
    midtime = float(cspyce.sct2e(_VOYAGER_SCLK_ID, tick + offset_ticks))
    assignments = assign_images([_voyager_entry(midtime, attitude)], voyager_index)
    assert assignments[0].baseline is not None
    assert assignments[0].baseline.basename == _VOYAGER_KERNEL
