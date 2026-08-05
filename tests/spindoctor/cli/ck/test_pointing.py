"""Hermetic tests for ``spindoctor.cli.ck.pointing``.

The writer's input type is the boundary between the navigation metadata and
SPICE, so these tests pin what it accepts out of a metadata dict and what it
refuses.  Nothing here furnishes a kernel.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.cli.ck.pointing import ImagePointing

# A recognizable, deliberately non-symmetric rotation: a quarter turn about Z.
_QUARTER_TURN = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

_START_ET = 5.0e8
_EXPOSURE_S = 2.0


def _metadata(**overrides: Any) -> dict[str, Any]:
    """Build a navigation metadata dict shaped like the pipeline's own.

    Parameters:
        overrides: Values replacing entries of the ``pointing`` or ``times``
            blocks, or of the top-level ``observation`` block.

    Returns:
        The metadata dict.
    """
    pointing: dict[str, Any] = {
        'cmatrix': list(_QUARTER_TURN),
        'cmatrix_original': list(_QUARTER_TURN),
        'camera_frame': 'CASSINI_ISS_NAC',
        'camera_frame_id': -82360,
        'ck_frame_id': -82000,
    }
    times: dict[str, Any] = {
        'start_et': _START_ET,
        'stop_et': _START_ET + _EXPOSURE_S,
        'midtime_et': _START_ET + _EXPOSURE_S / 2.0,
        'exposure_s': _EXPOSURE_S,
        'sclk_start': '1/1484573293.055',
        'sclk_midtime': '1/1484573295.118',
        'sclk_stop': '1/1484573297.181',
    }
    observation: dict[str, Any] = {
        'image_name': 'N1484573295_1.IMG',
        'instrument': 'COISS',
        'camera': 'NAC',
        'shutter_mode': 'NACONLY',
    }
    for key, value in overrides.items():
        if key in pointing:
            pointing[key] = value
        elif key in times:
            times[key] = value
        else:
            observation[key] = value
    return {
        'status': 'success',
        'observation': observation,
        'navigation_result': {'pointing': pointing, 'times': times},
    }


def test_from_metadata_reads_the_recorded_block() -> None:
    """Every field the writer needs comes out of the metadata unchanged."""
    pointing = ImagePointing.from_metadata(_metadata())
    assert pointing.image_name == 'N1484573295_1.IMG'
    assert pointing.camera_frame == 'CASSINI_ISS_NAC'
    assert pointing.ck_frame_id == -82000
    assert pointing.start_et == _START_ET
    assert pointing.stop_et == _START_ET + _EXPOSURE_S
    assert pointing.midtime_et == _START_ET + _EXPOSURE_S / 2.0
    assert pointing.exposure_s == _EXPOSURE_S
    assert np.array_equal(pointing.cmatrix, np.asarray(_QUARTER_TURN).reshape(3, 3))


def test_from_metadata_stores_the_matrix_read_only() -> None:
    """The stored C-matrix cannot be mutated behind the writer's back."""
    pointing = ImagePointing.from_metadata(_metadata())
    assert pointing.cmatrix.flags.writeable is False


def test_from_metadata_refuses_a_result_with_no_corrected_matrix() -> None:
    """An image that navigated without a corrected attitude has no segment to write."""
    metadata = _metadata()
    del metadata['navigation_result']['pointing']['cmatrix']
    with pytest.raises(ValueError, match="pointing has no 'cmatrix' field"):
        ImagePointing.from_metadata(metadata)


def test_from_metadata_refuses_a_result_with_no_pointing_block() -> None:
    """Metadata predating the recorded pointing is refused, not guessed at."""
    metadata = _metadata()
    del metadata['navigation_result']['pointing']
    with pytest.raises(ValueError, match="navigation_result has no 'pointing' field"):
        ImagePointing.from_metadata(metadata)


def test_from_metadata_refuses_a_times_block_of_the_wrong_shape() -> None:
    """A section that is not a section is named as such."""
    metadata = _metadata()
    metadata['navigation_result']['times'] = [1.0, 2.0]
    with pytest.raises(ValueError, match=r'navigation_result\.times is list, not a section'):
        ImagePointing.from_metadata(metadata)


def test_from_metadata_refuses_a_missing_epoch() -> None:
    """An absent exposure epoch is refused rather than defaulted."""
    metadata = _metadata()
    del metadata['navigation_result']['times']['midtime_et']
    with pytest.raises(ValueError, match="times has no 'midtime_et' field"):
        ImagePointing.from_metadata(metadata)


def test_from_metadata_refuses_a_matrix_that_is_not_a_rotation() -> None:
    """A C-matrix that lost orthonormality is refused, not orthonormalized."""
    scaled = [value * 1.5 for value in _QUARTER_TURN]
    with pytest.raises(ValueError, match='cmatrix is not a proper rotation'):
        ImagePointing.from_metadata(_metadata(cmatrix=scaled))


def test_from_metadata_refuses_a_non_finite_matrix() -> None:
    """A non-finite C-matrix is refused before any tolerance test can pass it."""
    broken = list(_QUARTER_TURN)
    broken[0] = float('nan')
    with pytest.raises(ValueError, match='cmatrix holds a non-finite value'):
        ImagePointing.from_metadata(_metadata(cmatrix=broken))


def test_from_metadata_refuses_epochs_out_of_order() -> None:
    """A midtime outside the exposure is refused."""
    with pytest.raises(ValueError, match='exposure epochs are out of order'):
        ImagePointing.from_metadata(_metadata(midtime_et=_START_ET - 1.0))


def test_from_metadata_refuses_a_negative_exposure() -> None:
    """A negative exposure duration is refused."""
    with pytest.raises(ValueError, match='exposure_s is negative'):
        ImagePointing.from_metadata(_metadata(exposure_s=-1.0))


@pytest.mark.parametrize(
    'exposure_s',
    [float('nan'), float('inf'), float('-inf')],
    ids=['nan', 'inf', 'negative-inf'],
)
def test_from_metadata_refuses_a_non_finite_exposure(exposure_s: float) -> None:
    """A non-finite duration is refused rather than carried into the cadence.

    A NaN in particular passes a negativity test, since every comparison
    against it is False, and would reach the record-cadence arithmetic
    unnoticed.

    Parameters:
        exposure_s: Non-finite duration recorded in the metadata.
    """
    with pytest.raises(ValueError, match='exposure_s is not finite'):
        ImagePointing.from_metadata(_metadata(exposure_s=exposure_s))


@pytest.mark.parametrize(
    'field', ['start_et', 'stop_et', 'midtime_et'], ids=['start', 'stop', 'midtime']
)
def test_from_metadata_refuses_a_non_finite_epoch(field: str) -> None:
    """A non-finite epoch is refused before it reaches a clock encoding.

    The ordering check catches a NaN epoch only as a side effect of every
    comparison against NaN being False; an infinite one satisfies the ordering
    outright.

    Parameters:
        field: Name of the exposure epoch set to infinity.
    """
    with pytest.raises(ValueError, match=f'{field} is not finite'):
        ImagePointing.from_metadata(_metadata(**{field: float('inf')}))


def test_from_metadata_refuses_an_empty_image_name() -> None:
    """A segment must name the image it corrects."""
    with pytest.raises(ValueError, match='image_name is empty'):
        ImagePointing.from_metadata(_metadata(image_name=''))


def test_refuses_a_matrix_that_is_not_orthonormal() -> None:
    """A shear has determinant 1 and is still not an attitude."""
    shear = [1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match='cmatrix is not orthonormal'):
        ImagePointing.from_metadata(_metadata(cmatrix=shear))


def test_refuses_a_matrix_of_the_wrong_shape() -> None:
    """Anything but a 3x3 is refused by name rather than by a numpy message."""
    with pytest.raises(ValueError, match=r'cmatrix is not a 3x3 matrix; got shape \(2, 2\)'):
        ImagePointing(
            image_name='N1484573295_1.IMG',
            cmatrix=np.eye(2),
            camera_frame='CASSINI_ISS_NAC',
            ck_frame_id=-82000,
            start_et=_START_ET,
            stop_et=_START_ET + _EXPOSURE_S,
            midtime_et=_START_ET + _EXPOSURE_S / 2.0,
            exposure_s=_EXPOSURE_S,
        )
