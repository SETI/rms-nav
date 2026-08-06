"""Hermetic tests for ``spindoctor.cli.ck.images``.

These pin what the generator reads out of one image's metadata, which images it
judges eligible, and which of two simultaneous exposures yields.  Nothing here
furnishes a kernel: eligibility is a property of the record, not of SPICE.
"""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.cli.ck.conftest import (
    CASSINI_CAMERA_FRAME,
    CASSINI_CK_FRAME_ID,
    ET0,
    VOYAGER_CAMERA_FRAME,
    VOYAGER_CK_FRAME_ID,
    axis_rotation,
    image_metadata,
)

from spindoctor.cli.ck.images import ImageEntry, OmissionReason, botsim_losers
from spindoctor.cli.ck.pointing import ImagePointing

# Two recognizably different attitudes, so a test that confused the corrected
# matrix with the uncorrected one would not pass by symmetry.
_CORRECTED = axis_rotation(np.array([0.1, -0.7, 0.3]), 1.1)
_UNCORRECTED = axis_rotation(np.array([0.1, -0.7, 0.3]), 1.2)

_EXPOSURE_S = 2.0
_KERNELS = ('03236_04002ra.bc', 'naif0012.tls')


def _metadata(**overrides: Any) -> dict[str, Any]:
    """Build an eligible Cassini image's metadata, with fields replaced.

    Parameters:
        overrides: Keyword arguments passed through to the metadata builder,
            replacing its defaults.

    Returns:
        The metadata dict.
    """
    defaults: dict[str, Any] = {
        'image_name': 'N1484573295_1.IMG',
        'cmatrix': _CORRECTED,
        'cmatrix_original': _UNCORRECTED,
        'camera_frame': CASSINI_CAMERA_FRAME,
        'ck_frame_id': CASSINI_CK_FRAME_ID,
        'start_et': ET0,
        'stop_et': ET0 + _EXPOSURE_S,
        'camera': 'NAC',
        'shutter_mode': 'NACONLY',
        'kernels': _KERNELS,
    }
    defaults.update(overrides)
    return image_metadata(**defaults)


def _entry(**overrides: Any) -> ImageEntry:
    """Read an entry from an eligible image's metadata, with fields replaced.

    Parameters:
        overrides: Keyword arguments passed through to the metadata builder.

    Returns:
        The entry.
    """
    return ImageEntry.from_metadata(_metadata(**overrides))


def test_from_metadata_reads_an_eligible_image() -> None:
    """Every field the generator needs comes out of the metadata unchanged."""
    entry = _entry()
    assert entry.image_name == 'N1484573295_1.IMG'
    assert entry.status == 'success'
    assert entry.camera == 'NAC'
    assert entry.shutter_mode == 'NACONLY'
    assert entry.rotation_fitted is False
    assert entry.kernel_basenames == _KERNELS
    assert entry.ineligibility_reason is None
    assert entry.is_eligible is True


def test_from_metadata_reads_both_recorded_matrices() -> None:
    """The uncorrected matrix is carried too: it identifies the baseline."""
    entry = _entry()
    assert entry.pointing is not None
    assert np.allclose(entry.pointing.cmatrix_original, _UNCORRECTED, rtol=0.0, atol=0.0)


def test_a_conflicted_result_is_eligible() -> None:
    """A conflicted result is written; the report carries the conflict."""
    assert _entry(status='conflicted').is_eligible is True


def test_a_failed_result_is_not_eligible() -> None:
    """An image that did not navigate has no correction to write."""
    entry = _entry(status='failed')
    assert entry.ineligibility_reason is OmissionReason.NOT_ELIGIBLE


def test_a_fitted_rotation_is_reported_as_unsupported() -> None:
    """A fitted rotation turns about a pivot no result records."""
    entry = _entry(cmatrix=None, rotation_deg=0.25)
    assert entry.ineligibility_reason is OmissionReason.ROTATION_UNSUPPORTED


def test_a_fitted_rotation_of_zero_is_still_a_fitted_rotation() -> None:
    """The field's presence is the fact, not its value."""
    entry = _entry(cmatrix=None, rotation_deg=0.0)
    assert entry.ineligibility_reason is OmissionReason.ROTATION_UNSUPPORTED


def test_a_failed_result_with_a_fitted_rotation_is_reported_by_its_status() -> None:
    """The status is judged first: an image that did not navigate is not eligible."""
    entry = _entry(status='failed', cmatrix=None, rotation_deg=0.25)
    assert entry.ineligibility_reason is OmissionReason.NOT_ELIGIBLE


def test_a_result_without_a_corrected_matrix_is_not_eligible() -> None:
    """An image that navigated without an offset carries no corrected attitude."""
    entry = _entry(cmatrix=None)
    assert entry.ineligibility_reason is OmissionReason.NOT_ELIGIBLE


def test_a_navigated_image_with_no_pointing_block_is_not_eligible() -> None:
    """A run whose attitude computation failed records a result and no pointing.

    The pipeline reports that failure and leaves the block out rather than
    recording a wrong attitude, so the image navigated but cannot be written.
    """
    metadata = _metadata()
    del metadata['navigation_result']['pointing']
    entry = ImageEntry.from_metadata(metadata)
    assert entry.ineligibility_reason is OmissionReason.NOT_ELIGIBLE


def test_a_load_error_document_is_not_eligible() -> None:
    """An image that never loaded has no navigation result at all."""
    metadata = _metadata()
    del metadata['navigation_result']
    metadata['status'] = 'error'
    entry = ImageEntry.from_metadata(metadata)
    assert entry.ineligibility_reason is OmissionReason.NOT_ELIGIBLE


def test_an_ineligible_image_carries_no_kernel_names() -> None:
    """Nothing is placed for an image that gets no segment."""
    assert _entry(status='failed').kernel_basenames == ()


def test_an_absent_camera_is_read_as_none() -> None:
    """A host that reports no camera leaves the field out."""
    assert _entry(camera=None).camera is None


def test_an_absent_shutter_mode_is_read_as_none() -> None:
    """A host with no shutter mode leaves the field out."""
    assert _entry(shutter_mode=None).shutter_mode is None


@pytest.mark.parametrize('field', ['camera', 'shutter_mode'], ids=['camera', 'shutter-mode'])
def test_a_null_optional_text_field_is_refused(field: str) -> None:
    """A null where text belongs is refused rather than coerced.

    ``str(None)`` is the text ``'None'``, which would pair as a camera name
    and decide which of two simultaneous exposures keeps its correction.

    Parameters:
        field: Name of the observation field set to a JSON null.
    """
    metadata = _metadata()
    metadata['observation'][field] = None
    with pytest.raises(TypeError, match=f'{field!r} is NoneType, not a string'):
        ImageEntry.from_metadata(metadata)


def test_a_missing_status_is_refused() -> None:
    """Eligibility turns on the status, so it cannot be assumed."""
    metadata = _metadata()
    del metadata['status']
    with pytest.raises(ValueError, match="metadata has no 'status' field"):
        ImageEntry.from_metadata(metadata)


def test_a_pointing_block_that_is_not_a_block_is_refused() -> None:
    """A malformed document fails loudly rather than reading as ineligible."""
    metadata = _metadata()
    metadata['navigation_result']['pointing'] = ['not', 'a', 'block']
    with pytest.raises(ValueError, match=r'navigation_result\.pointing is list, not a section'):
        ImageEntry.from_metadata(metadata)


def test_an_eligible_image_with_no_provenance_is_refused() -> None:
    """An eligible image records the kernels it navigated with."""
    metadata = _metadata(kernels=None)
    with pytest.raises(ValueError, match="navigation_result has no 'provenance' field"):
        ImageEntry.from_metadata(metadata)


def test_a_kernel_list_that_is_not_a_list_is_refused() -> None:
    """The recorded kernels are a list of names, not a name."""
    metadata = _metadata()
    metadata['navigation_result']['provenance']['spice_kernels'] = '03236_04002ra.bc'
    with pytest.raises(TypeError, match="'spice_kernels' is str, not a list"):
        ImageEntry.from_metadata(metadata)


def test_a_null_in_the_kernel_list_is_refused() -> None:
    """A null kernel name would resolve against nothing and be read as absent."""
    metadata = _metadata()
    metadata['navigation_result']['provenance']['spice_kernels'] = ['a.bc', None]
    with pytest.raises(TypeError, match="'spice_kernels' holds a NoneType, not a string"):
        ImageEntry.from_metadata(metadata)


def test_an_empty_kernel_list_is_refused() -> None:
    """An image with a corrected attitude was navigated against kernels.

    Recording none of them is a defect in the record, and it is refused as a
    missing provenance block is: reporting it instead would say the image's
    baseline had drifted, which is what that report is for.
    """
    with pytest.raises(ValueError, match="'spice_kernels' is empty"):
        _entry(kernels=())


def test_an_entry_cannot_be_both_eligible_and_not() -> None:
    """A pointing solution and a reason it has none contradict each other."""
    pointing = ImagePointing(
        image_name='N1484573295_1.IMG',
        cmatrix=_CORRECTED,
        cmatrix_original=_UNCORRECTED,
        camera_frame=CASSINI_CAMERA_FRAME,
        ck_frame_id=CASSINI_CK_FRAME_ID,
        start_et=ET0,
        stop_et=ET0 + _EXPOSURE_S,
        midtime_et=ET0 + _EXPOSURE_S / 2.0,
        exposure_s=_EXPOSURE_S,
    )
    with pytest.raises(ValueError, match='not both and not neither'):
        ImageEntry(
            image_name='N1484573295_1.IMG',
            status='success',
            camera='NAC',
            shutter_mode='NACONLY',
            rotation_fitted=False,
            kernel_basenames=(),
            pointing=pointing,
            ineligibility_reason=OmissionReason.NOT_ELIGIBLE,
        )


def test_an_entry_needs_a_pointing_or_a_reason() -> None:
    """An image with neither would appear in the report with no disposition."""
    with pytest.raises(ValueError, match='not both and not neither'):
        ImageEntry(
            image_name='N1484573295_1.IMG',
            status='success',
            camera='NAC',
            shutter_mode='NACONLY',
            rotation_fitted=False,
            kernel_basenames=(),
            pointing=None,
            ineligibility_reason=None,
        )


def test_the_omission_reason_set_is_closed() -> None:
    """The reasons a consumer must handle are exactly these five."""
    assert {reason.value for reason in OmissionReason} == {
        'not_eligible',
        'botsim_loser',
        'rotation_unsupported',
        'no_reproducing_baseline',
        'degenerate_exposure',
    }


def _botsim_entry(image_name: str, camera: str, start_et: float, **overrides: Any) -> ImageEntry:
    """Read an entry for one member of a simultaneous exposure.

    Parameters:
        image_name: Basename recorded for the image.
        camera: The camera that took it.
        start_et: Exposure start, TDB seconds past J2000.
        overrides: Further keyword arguments for the metadata builder.

    Returns:
        The entry.
    """
    return _entry(
        image_name=image_name,
        camera=camera,
        shutter_mode='BOTSIM',
        start_et=start_et,
        stop_et=start_et + _EXPOSURE_S,
        **overrides,
    )


def test_a_simultaneous_pair_makes_the_wide_angle_frame_yield() -> None:
    """One bus attitude cannot honor two corrections; the narrow angle wins."""
    entries = [
        _botsim_entry('N1484573295_1.IMG', 'NAC', ET0),
        _botsim_entry('W1484573295_1.IMG', 'WAC', ET0 + 0.1),
    ]
    assert botsim_losers(entries) == frozenset({'W1484573295_1.IMG'})


def test_a_pair_exactly_at_the_window_edge_still_pairs() -> None:
    """The window is inclusive of its bound."""
    entries = [
        _botsim_entry('N1484573295_1.IMG', 'NAC', ET0),
        _botsim_entry('W1484573295_1.IMG', 'WAC', ET0 + 1.0),
    ]
    assert botsim_losers(entries) == frozenset({'W1484573295_1.IMG'})


def test_exposures_further_apart_than_the_window_do_not_pair() -> None:
    """Two frames a second and a half apart are two events."""
    entries = [
        _botsim_entry('N1484573295_1.IMG', 'NAC', ET0),
        _botsim_entry('W1484573295_1.IMG', 'WAC', ET0 + 1.5),
    ]
    assert botsim_losers(entries) == frozenset()


def test_two_frames_from_the_same_camera_do_not_pair() -> None:
    """The rule is about one exposure seen by both cameras."""
    entries = [
        _botsim_entry('W1484573295_1.IMG', 'WAC', ET0),
        _botsim_entry('W1484573296_1.IMG', 'WAC', ET0 + 0.1),
    ]
    assert botsim_losers(entries) == frozenset()


def test_a_wide_angle_frame_whose_partner_did_not_navigate_keeps_its_correction() -> None:
    """The narrow angle frame wins only when it is eligible itself."""
    entries = [
        _botsim_entry('N1484573295_1.IMG', 'NAC', ET0, status='failed'),
        _botsim_entry('W1484573295_1.IMG', 'WAC', ET0 + 0.1),
    ]
    assert botsim_losers(entries) == frozenset()


def test_frames_not_taken_in_the_simultaneous_mode_do_not_pair() -> None:
    """The shutter mode is what says the two exposures are one event."""
    entries = [
        _entry(image_name='N1484573295_1.IMG', camera='NAC', shutter_mode='NACONLY'),
        _entry(image_name='W1484573295_1.IMG', camera='WAC', shutter_mode='WACONLY'),
    ]
    assert botsim_losers(entries) == frozenset()


def test_frames_correcting_different_objects_do_not_pair() -> None:
    """Two spacecraft do not share an attitude however close their exposures are."""
    entries = [
        _botsim_entry('N1484573295_1.IMG', 'NAC', ET0),
        _botsim_entry(
            'C1205021_CALIB.IMG',
            'WAC',
            ET0 + 0.1,
            camera_frame=VOYAGER_CAMERA_FRAME,
            ck_frame_id=VOYAGER_CK_FRAME_ID,
        ),
    ]
    assert botsim_losers(entries) == frozenset()


def test_a_frame_with_no_camera_recorded_does_not_pair() -> None:
    """The pairing turns on which camera took the frame."""
    entries = [
        _botsim_entry('N1484573295_1.IMG', 'NAC', ET0),
        _entry(image_name='W1484573295_1.IMG', camera=None, shutter_mode='BOTSIM'),
    ]
    assert botsim_losers(entries) == frozenset()


def test_two_wide_angle_frames_around_one_narrow_angle_frame_both_yield() -> None:
    """Every frame pairing with an eligible narrow angle frame yields to it."""
    entries = [
        _botsim_entry('N1484573295_1.IMG', 'NAC', ET0),
        _botsim_entry('W1484573295_1.IMG', 'WAC', ET0 - 0.5),
        _botsim_entry('W1484573296_1.IMG', 'WAC', ET0 + 0.5),
    ]
    assert botsim_losers(entries) == frozenset({'W1484573295_1.IMG', 'W1484573296_1.IMG'})


def test_a_pair_whose_winner_starts_last_still_pairs() -> None:
    """The window is inclusive on both sides of the frame that yields."""
    entries = [
        _botsim_entry('N1484573295_1.IMG', 'NAC', ET0 + 1.0),
        _botsim_entry('W1484573295_1.IMG', 'WAC', ET0),
    ]
    assert botsim_losers(entries) == frozenset({'W1484573295_1.IMG'})


def test_a_frame_with_an_empty_camera_does_not_pair() -> None:
    """A camera that names nothing cannot be the one that yields."""
    entries = [
        _botsim_entry('N1484573295_1.IMG', 'NAC', ET0),
        _botsim_entry('W1484573295_1.IMG', '', ET0 + 0.1),
    ]
    assert botsim_losers(entries) == frozenset()


def test_a_frame_from_another_instrument_does_not_yield() -> None:
    """The rule is about the two cameras that share a bus attitude, not everything else.

    Reported as a loser, a frame from any other instrument would be omitted
    for yielding to a pair it was never part of.
    """
    entries = [
        _botsim_entry('N1484573295_1.IMG', 'NAC', ET0),
        _botsim_entry('U1484573295_1.IMG', 'UVIS', ET0 + 0.1),
    ]
    assert botsim_losers(entries) == frozenset()
