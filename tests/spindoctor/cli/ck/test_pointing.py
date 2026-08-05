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


def test_from_metadata_reads_the_uncorrected_matrix() -> None:
    """The uncorrected attitude is carried too; it identifies the baseline kernel."""
    pointing = ImagePointing.from_metadata(_metadata())
    assert np.array_equal(pointing.cmatrix_original, np.asarray(_QUARTER_TURN).reshape(3, 3))


def test_from_metadata_refuses_a_missing_uncorrected_matrix() -> None:
    """Without it no candidate kernel can be tested against the image."""
    metadata = _metadata()
    del metadata['navigation_result']['pointing']['cmatrix_original']
    with pytest.raises(ValueError, match="pointing has no 'cmatrix_original' field"):
        ImagePointing.from_metadata(metadata)


def test_from_metadata_refuses_a_misshapen_uncorrected_matrix() -> None:
    """The refusal names the field it read, not the other one."""
    with pytest.raises(ValueError, match='cmatrix_original must be nine row-major floats'):
        ImagePointing.from_metadata(_metadata(cmatrix_original=[[float(v)] for v in _QUARTER_TURN]))


def test_from_metadata_refuses_an_uncorrected_matrix_that_is_not_a_rotation() -> None:
    """A baseline that is not an attitude cannot have been one."""
    scaled = [value * 1.5 for value in _QUARTER_TURN]
    with pytest.raises(ValueError, match='cmatrix_original is not a proper rotation'):
        ImagePointing.from_metadata(_metadata(cmatrix_original=scaled))


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
    'cmatrix',
    [
        [[float(v) for v in _QUARTER_TURN]],
        [[[float(v)] for v in _QUARTER_TURN[0:3]]] * 3,
        [float(v) for v in _QUARTER_TURN][:8],
    ],
    ids=['nested-1x9', 'nested-3x3x1', 'eight-values'],
)
def test_from_metadata_refuses_a_misshapen_cmatrix(cmatrix: list[Any]) -> None:
    """A nine-value array of the wrong shape is refused, not reshaped.

    ``reshape(3, 3)`` accepts every nine-element array, so a document nested
    one level too deep would otherwise be read as if it were well formed.

    Parameters:
        cmatrix: A recorded C-matrix whose shape the schema does not write.
    """
    with pytest.raises(ValueError, match='cmatrix must be nine row-major floats'):
        ImagePointing.from_metadata(_metadata(cmatrix=cmatrix))


def test_from_metadata_reads_a_flat_nine_value_cmatrix() -> None:
    """The canonical flat form the curator writes is read as a 3x3."""
    pointing = ImagePointing.from_metadata(_metadata())
    assert pointing.cmatrix.shape == (3, 3)


@pytest.mark.parametrize(
    'field', ['image_name', 'camera_frame'], ids=['image-name', 'camera-frame']
)
def test_from_metadata_refuses_a_null_text_field(field: str) -> None:
    """A null where text belongs is refused rather than coerced.

    ``str(None)`` is the text ``'None'``, which is neither empty nor
    obviously wrong, so a null image name would otherwise identify a written
    segment and pass every downstream check.

    Parameters:
        field: Name of the text field set to a JSON null.
    """
    with pytest.raises(TypeError, match=f'{field!r} is NoneType, not a string'):
        ImagePointing.from_metadata(_metadata(**{field: None}))


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
            cmatrix_original=np.eye(3),
            camera_frame='CASSINI_ISS_NAC',
            ck_frame_id=-82000,
            start_et=_START_ET,
            stop_et=_START_ET + _EXPOSURE_S,
            midtime_et=_START_ET + _EXPOSURE_S / 2.0,
            exposure_s=_EXPOSURE_S,
        )


@pytest.mark.parametrize(
    'ck_frame_id',
    ['-82000', -82000.0, -82000.9, True, None],
    ids=['text', 'whole-float', 'truncating-float', 'boolean', 'null'],
)
def test_from_metadata_refuses_a_ck_object_that_is_not_an_integer(ck_frame_id: Any) -> None:
    """An object id is never coerced into being one.

    ``int('-82000')`` and ``int(-82000.9)`` both produce a valid Cassini bus
    id, the second by truncating a value that was never that id, and
    ``int(True)`` produces 1.  Each would resolve a clock, encode time tags,
    and write a segment against an object the metadata never recorded.

    Parameters:
        ck_frame_id: A recorded object id of the wrong kind.
    """
    with pytest.raises(TypeError, match=r"'ck_frame_id' is \w+, not an integer"):
        ImagePointing.from_metadata(_metadata(ck_frame_id=ck_frame_id))


@pytest.mark.parametrize(
    'field',
    ['start_et', 'stop_et', 'midtime_et', 'exposure_s'],
    ids=['start', 'stop', 'mid', 'exp'],
)
@pytest.mark.parametrize('value', ['0.0', True, None], ids=['text', 'boolean', 'null'])
def test_from_metadata_refuses_an_epoch_that_is_not_a_number(field: str, value: Any) -> None:
    """An epoch is never coerced into being one.

    ``float('0.0')`` and ``float(True)`` both succeed, so an epoch recorded as
    text or as a JSON ``true`` would reach a clock encoding as a plausible
    number.

    Parameters:
        field: Name of the time field set to the wrong kind of value.
        value: A recorded value that is not a number.
    """
    with pytest.raises(TypeError, match=f'{field!r} is .*, not a number'):
        ImagePointing.from_metadata(_metadata(**{field: value}))


def test_from_metadata_accepts_a_whole_number_epoch() -> None:
    """JSON writes an exact epoch without a decimal point, and it is widened."""
    pointing = ImagePointing.from_metadata(_metadata(exposure_s=2))
    assert pointing.exposure_s == 2.0


def test_from_metadata_refuses_a_matrix_off_orthonormality_by_a_microradian() -> None:
    """The rotation bound is a nanoradian, not something a defect can hide under.

    A shear a thousand times smaller than a pixel of any camera here is still
    not an attitude, and it would be written into a kernel other tools trust.
    """
    nearly = [1.0, 1e-6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match='cmatrix is not orthonormal'):
        ImagePointing.from_metadata(_metadata(cmatrix=nearly))


@pytest.mark.parametrize(
    'cmatrix',
    [
        [True, False, False, False, True, False, False, False, True],
        [[True, False, False], [False, True, False], [False, False, True]],
        ['1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '1.0'],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, None],
    ],
    ids=['flat-booleans', 'nested-booleans', 'numeric-strings', 'null-element'],
)
def test_from_metadata_refuses_a_matrix_of_the_wrong_kind(cmatrix: list[Any]) -> None:
    """A matrix element is never coerced into being a number.

    Nine booleans convert to a flawless identity, and nine numeric strings to
    whatever they spell, so both would satisfy the determinant, orthonormality
    and finiteness guards and be written into a kernel.

    Parameters:
        cmatrix: A recorded matrix whose elements are not numbers.
    """
    with pytest.raises(TypeError, match='cmatrix holds a'):
        ImagePointing.from_metadata(_metadata(cmatrix=cmatrix))


def test_from_metadata_refuses_an_empty_camera_frame() -> None:
    """A frame with no name cannot be looked up, and says so here rather than in SPICE."""
    with pytest.raises(ValueError, match='camera_frame is empty'):
        ImagePointing.from_metadata(_metadata(camera_frame=''))
