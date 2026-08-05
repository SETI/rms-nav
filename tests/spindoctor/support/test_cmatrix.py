"""Hermetic tests for ``spindoctor.support.cmatrix``.

Every test here builds its own FOV and its own attitudes, so nothing depends
on SPICE kernels or on a real observation.  The conventions under test are
the ones a hermetic test can silently get wrong in both directions at once:
the sign of the tangent-plane offset, the direction of the rotation
``cspyce.axisar`` builds, and the conjugation of the oops-frame correction
into the SPICE frame.  Each is pinned by recovering the planted offset back
out of the recorded C-matrix with an independent inverse.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import cspyce
import numpy as np
import oops
import pytest
from tests.cmatrix_helpers import offset_from_correction

from spindoctor.obs import ObsSnapshotInst
from spindoctor.support.cmatrix import (
    _CASSINI_CK_FRAME_ID,
    _CASSINI_SCLK_ID,
    _GALILEO_CK_FRAME_ID,
    _GALILEO_SCLK_ID,
    _LORRI_CK_FRAME_ID,
    _LORRI_SCLK_ID,
    AttitudeBaseline,
    _build_pointing_solution,
    _camera_frame_id,
    _check_flip,
    _FrameIdentity,
    _oops_correction_matrix,
    _pxform,
    _sclk_id,
    _sclk_string,
    _spice_cmatrix,
    _validate_rotation,
    compute_pointing,
)
from spindoctor.support.exceptions import NavPointingError

# A Cassini-NAC-like square camera: 1024 pixels at 6 microradians each.  A
# FlatFOV maps its boresight pixel to xy exactly (0, 0), which is what makes
# the zero-offset identity guard exercisable.
_PIXEL_RAD = 6.0e-6
_SHAPE = (1024, 1024)

# A planted offset with two distinct, non-zero, opposite-signed components,
# so a sign flip or an axis swap cannot go unnoticed.
_PLANTED_OFFSET = (8.68, -17.37)

# The 180-degree flip oops applies on top of the SPICE Cassini ISS frames.
_CASSINI_FLIP = np.diag([-1.0, -1.0, 1.0])

# Voyager 1's scan-platform CK object, which is the wrong one for a Voyager 2
# image: ``ckmeta`` answers -31 for it, the other spacecraft's clock.
_VOYAGER1_CK_FRAME_ID = -31100
_VOYAGER2_SCLK_ID = -32

# A CK object no mission owns, used to pin what ``ckmeta`` does with one.
_NONEXISTENT_CK_FRAME_ID = -999999

# A spacecraft clock no furnished kernel can ever describe.
_NONEXISTENT_SCLK_ID = -999

_IDENTITY = np.eye(3)


def _fov() -> oops.fov.FOV:
    """Build the synthetic camera every test in this module points."""
    return oops.fov.FlatFOV((_PIXEL_RAD, _PIXEL_RAD), _SHAPE)


def _some_attitude() -> np.ndarray:
    """Return an arbitrary, deliberately non-axis-aligned J2000-to-camera rotation."""
    ra, dec, twist = 0.7, -0.4, 1.9
    rot_z = np.array(
        [[np.cos(ra), np.sin(ra), 0.0], [-np.sin(ra), np.cos(ra), 0.0], [0.0, 0.0, 1.0]]
    )
    rot_y = np.array(
        [[np.cos(dec), 0.0, -np.sin(dec)], [0.0, 1.0, 0.0], [np.sin(dec), 0.0, np.cos(dec)]]
    )
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(twist), np.sin(twist)], [0.0, -np.sin(twist), np.cos(twist)]]
    )
    attitude: np.ndarray = rot_x @ rot_y @ rot_z
    return attitude


def _outcome_of(check: Callable[[], None]) -> str:
    """Report whether a validator accepted or refused its input.

    The validators under test signal acceptance by returning and refusal by
    raising, so a test that only calls one asserts nothing.  This turns the
    two outcomes into a value a test can assert on.
    """
    try:
        check()
    except NavPointingError:
        return 'refused'
    return 'accepted'


def _baseline(
    cmatrix_original: np.ndarray, oops_from_spice: np.ndarray = _IDENTITY
) -> AttitudeBaseline:
    """Build a synthetic AttitudeBaseline around a given attitude and flip."""
    return AttitudeBaseline(
        cmatrix_original=cmatrix_original,
        oops_from_spice=oops_from_spice,
        camera_frame='TEST_CAMERA',
        camera_frame_id=-999999,
        ck_frame_id=-999000,
        start_et=100.0,
        stop_et=100.5,
        midtime_et=100.25,
        exposure_s=0.5,
        sclk_start='1/100.000',
        sclk_midtime='1/100.250',
        sclk_stop='1/100.500',
    )


def test_correction_maps_the_uncorrected_boresight_onto_the_corrected_one() -> None:
    """``M . d`` is the direction the unmodified FOV assigns to the boresight."""
    fov = _fov()
    correction = _oops_correction_matrix(fov, _PLANTED_OFFSET)
    xy_los = fov.xy_from_uv(fov.uv_los)
    uv_los = fov.uv_los
    xy_offset = fov.xy_from_uv(
        oops.Pair((uv_los.vals[0] + _PLANTED_OFFSET[1], uv_los.vals[1] + _PLANTED_OFFSET[0]))
    )
    uncorrected = np.asarray(fov.los_from_xy(xy_los - xy_offset).unit().vals, np.float64)
    corrected = np.asarray(fov.los_from_xy(xy_los).unit().vals, np.float64)
    assert correction @ uncorrected == pytest.approx(corrected, abs=1e-15)


def test_correction_is_a_proper_rotation() -> None:
    """The correction matrix has unit determinant."""
    correction = _oops_correction_matrix(_fov(), _PLANTED_OFFSET)
    assert float(np.linalg.det(correction)) == pytest.approx(1.0, abs=1e-14)


def test_planted_offset_is_recovered_from_the_recorded_cmatrix() -> None:
    """A planted offset survives the round trip through the recorded C-matrix.

    With no flip between the oops and SPICE frames, inverting the recorded
    correction must return the planted ``(dv, du)``.  A flipped ``xy_offset``
    sign, or a transposed correction, returns the negated offset instead.
    """
    fov = _fov()
    original = _some_attitude()
    solution = _build_pointing_solution(
        _baseline(original), fov, offset_px=_PLANTED_OFFSET, rotation_fitted=False
    )
    assert solution.cmatrix is not None
    correction = np.asarray(solution.cmatrix, np.float64) @ original.T
    recovered = offset_from_correction(fov, correction)
    assert recovered[0] == pytest.approx(_PLANTED_OFFSET[0], abs=1e-9)
    assert recovered[1] == pytest.approx(_PLANTED_OFFSET[1], abs=1e-9)


def test_zero_offset_leaves_the_cmatrix_identical_to_the_original() -> None:
    """A zero offset takes the identity guard and changes nothing at all."""
    original = _some_attitude()
    solution = _build_pointing_solution(
        _baseline(original), _fov(), offset_px=(0.0, 0.0), rotation_fitted=False
    )
    assert solution.cmatrix is not None
    assert np.array_equal(solution.cmatrix, solution.baseline.cmatrix_original)


def test_zero_offset_correction_is_exactly_the_identity() -> None:
    """The correction for a zero offset is the identity matrix, bit for bit."""
    assert np.array_equal(_oops_correction_matrix(_fov(), (0.0, 0.0)), _IDENTITY)


def test_cassini_style_flip_reproduces_the_offset_in_the_spice_convention() -> None:
    """The recorded C-matrix carries the planted offset through a 180-degree flip.

    With ``R = diag(-1, -1, 1)`` between the oops observation frame and the
    SPICE camera frame, converting both recorded attitudes back into the oops
    convention must yield exactly the correction the offset implies.  Dropping
    the ``R`` conjugation leaves a correction of the right magnitude whose
    tangent-plane components are both negated.
    """
    fov = _fov()
    original = _some_attitude()
    solution = _build_pointing_solution(
        _baseline(original, _CASSINI_FLIP), fov, offset_px=_PLANTED_OFFSET, rotation_fitted=False
    )
    assert solution.cmatrix is not None
    corrected_oops = _CASSINI_FLIP @ np.asarray(solution.cmatrix, np.float64)
    original_oops = _CASSINI_FLIP @ original
    recovered = offset_from_correction(fov, corrected_oops @ original_oops.T)
    assert recovered[0] == pytest.approx(_PLANTED_OFFSET[0], abs=1e-9)
    assert recovered[1] == pytest.approx(_PLANTED_OFFSET[1], abs=1e-9)


def test_flip_conjugation_changes_the_recorded_cmatrix() -> None:
    """Conjugating through the flip is not a no-op for a Cassini-style flip.

    Guards the test above: if ``R^T M R`` happened to equal ``M`` here, that
    test would pass with the conjugation dropped.
    """
    fov = _fov()
    original = _some_attitude()
    correction = _oops_correction_matrix(fov, _PLANTED_OFFSET)
    conjugated = _spice_cmatrix(original, correction, _CASSINI_FLIP)
    unconjugated = correction @ original
    assert float(np.max(np.abs(conjugated - unconjugated))) > 1e-6


def test_fitted_rotation_records_the_original_but_no_corrected_cmatrix() -> None:
    """A result carrying a fitted camera rotation gets no corrected attitude."""
    original = _some_attitude()
    solution = _build_pointing_solution(
        _baseline(original), _fov(), offset_px=_PLANTED_OFFSET, rotation_fitted=True
    )
    assert solution.cmatrix is None
    assert np.array_equal(solution.baseline.cmatrix_original, original)


def test_missing_offset_records_the_original_but_no_corrected_cmatrix() -> None:
    """A result with no offset gets no corrected attitude."""
    original = _some_attitude()
    solution = _build_pointing_solution(
        _baseline(original), _fov(), offset_px=None, rotation_fitted=False
    )
    assert solution.cmatrix is None
    assert np.array_equal(solution.baseline.cmatrix_original, original)


def test_baseline_with_a_bad_determinant_raises() -> None:
    """A baseline attitude that is not a proper rotation is refused."""
    reflected = _some_attitude()
    reflected[0] = -reflected[0]
    with pytest.raises(NavPointingError, match='cmatrix_original is not a proper rotation'):
        _build_pointing_solution(
            _baseline(reflected), _fov(), offset_px=_PLANTED_OFFSET, rotation_fitted=False
        )


def test_baseline_that_is_not_orthonormal_raises() -> None:
    """A unit-determinant baseline that is not orthonormal is refused."""
    sheared = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(NavPointingError, match='cmatrix_original is not orthonormal'):
        _build_pointing_solution(
            _baseline(sheared), _fov(), offset_px=_PLANTED_OFFSET, rotation_fitted=False
        )


def test_baseline_with_a_bad_determinant_raises_without_an_offset() -> None:
    """The baseline is checked even when no corrected attitude is produced."""
    reflected = _some_attitude()
    reflected[0] = -reflected[0]
    with pytest.raises(NavPointingError, match='cmatrix_original is not a proper rotation'):
        _build_pointing_solution(
            _baseline(reflected), _fov(), offset_px=None, rotation_fitted=False
        )


def test_corrected_cmatrix_that_is_not_a_rotation_raises() -> None:
    """A correction that is not a rotation is refused by the result check."""
    with pytest.raises(NavPointingError, match='cmatrix is not a proper rotation'):
        _spice_cmatrix(_some_attitude(), np.diag([1.0, 1.0, 2.0]), _IDENTITY)


def test_attitude_baseline_rejects_a_matrix_of_the_wrong_shape() -> None:
    """A baseline built from a non-3x3 matrix is refused at construction."""
    with pytest.raises(NavPointingError, match='expected a 3x3 matrix'):
        _baseline(np.eye(4))


def test_recorded_matrices_are_read_only() -> None:
    """Recorded C-matrices cannot be mutated through the returned solution."""
    solution = _build_pointing_solution(
        _baseline(_some_attitude()), _fov(), offset_px=_PLANTED_OFFSET, rotation_fitted=False
    )
    assert solution.cmatrix is not None
    with pytest.raises(ValueError, match='read-only'):
        solution.cmatrix[0, 0] = 0.0


def test_a_sub_pixel_offset_still_produces_a_real_correction() -> None:
    """A small but genuine offset is corrected, not swallowed by the guard.

    Pins the degenerate-axis threshold from below: raising it far enough to
    absorb a real sub-pixel offset would silently record every such image as
    needing no correction at all.
    """
    fov = _fov()
    small = (0.05, -0.02)
    correction = _oops_correction_matrix(fov, small)
    assert not np.array_equal(correction, _IDENTITY)
    recovered = offset_from_correction(fov, correction)
    assert recovered[0] == pytest.approx(small[0], abs=1e-9)
    assert recovered[1] == pytest.approx(small[1], abs=1e-9)


def _quarter_turn_about_z() -> np.ndarray:
    """A 90-degree rotation about Z: a proper rotation that is not involutory."""
    return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_conjugation_direction_is_pinned_by_a_non_involutory_flip() -> None:
    """The conjugation is ``R^T M R``, not ``R M R^T``.

    Every flip in the instrument table is diagonal and therefore its own
    inverse, so the two orders agree on real data and no real-frame test can
    tell them apart.  A quarter turn about Z can.
    """
    fov = _fov()
    flip = _quarter_turn_about_z()
    original = _some_attitude()
    solution = _build_pointing_solution(
        _baseline(original, flip), fov, offset_px=_PLANTED_OFFSET, rotation_fitted=False
    )
    assert solution.cmatrix is not None
    recovered = offset_from_correction(
        fov, (flip @ np.asarray(solution.cmatrix, np.float64)) @ (flip @ original).T
    )
    assert recovered[0] == pytest.approx(_PLANTED_OFFSET[0], abs=1e-9)
    assert recovered[1] == pytest.approx(_PLANTED_OFFSET[1], abs=1e-9)


def test_the_two_conjugation_orders_differ_under_a_non_involutory_flip() -> None:
    """Guards the test above: the quarter turn really does tell them apart."""
    flip = _quarter_turn_about_z()
    correction = _oops_correction_matrix(_fov(), _PLANTED_OFFSET)
    forward = flip.T @ correction @ flip
    reversed_order = flip @ correction @ flip.T
    assert float(np.max(np.abs(forward - reversed_order))) > 1e-6


def test_a_nan_baseline_is_rejected() -> None:
    """A NaN attitude is refused rather than serialized as an invalid JSON token."""
    corrupted = _some_attitude()
    corrupted[1, 1] = np.nan
    with pytest.raises(NavPointingError, match='cmatrix_original holds a non-finite value'):
        _build_pointing_solution(
            _baseline(corrupted), _fov(), offset_px=_PLANTED_OFFSET, rotation_fitted=False
        )


def test_an_infinite_baseline_is_rejected() -> None:
    """An infinite attitude entry is refused for the same reason."""
    corrupted = _some_attitude()
    corrupted[0, 2] = np.inf
    with pytest.raises(NavPointingError, match='cmatrix_original holds a non-finite value'):
        _build_pointing_solution(
            _baseline(corrupted), _fov(), offset_px=None, rotation_fitted=False
        )


def test_a_nan_correction_is_rejected_by_the_result_check() -> None:
    """A NaN reaching the corrected matrix is refused too."""
    corrupted = np.eye(3) * np.nan
    with pytest.raises(NavPointingError, match='cmatrix holds a non-finite value'):
        _spice_cmatrix(_some_attitude(), corrupted, _IDENTITY)


def _cassini_identity() -> _FrameIdentity:
    """The Cassini NAC frame identity, for the flip-check and clock tests."""
    return _FrameIdentity(
        camera_frame='CASSINI_ISS_NAC',
        ck_frame_id=_CASSINI_CK_FRAME_ID,
        sclk_id=_CASSINI_SCLK_ID,
        oops_from_spice=_CASSINI_FLIP,
        frozen_oops_attitude=False,
    )


def test_a_mismatched_flip_is_refused() -> None:
    """A measured flip that is not the instrument's constant raises."""
    with pytest.raises(NavPointingError, match='differs from the expected'):
        _check_flip(_IDENTITY, _cassini_identity())


def test_the_mismatched_flip_message_names_the_camera_frame() -> None:
    """The refusal says which frame disagreed, so it is attributable."""
    with pytest.raises(NavPointingError, match='CASSINI_ISS_NAC'):
        _check_flip(_IDENTITY, _cassini_identity())


def test_a_flip_within_tolerance_is_accepted() -> None:
    """A flip differing by less than the tolerance is accepted silently."""
    perturbed = _CASSINI_FLIP + np.full((3, 3), 1e-12)
    assert _outcome_of(lambda: _check_flip(perturbed, _cassini_identity())) == 'accepted'


def test_a_flip_just_outside_tolerance_is_refused() -> None:
    """A flip differing by more than the tolerance raises."""
    perturbed = _CASSINI_FLIP.copy()
    perturbed[0, 1] = 1e-8
    with pytest.raises(NavPointingError, match='differs from the expected'):
        _check_flip(perturbed, _cassini_identity())


def test_the_missions_own_clock_resolves_and_is_accepted() -> None:
    """The clock SPICE resolves for a mission's CK object is the expected one.

    Pins the expected constant against what ``ckmeta`` actually computes: a
    typo in either the CK object or the clock id fails here rather than
    producing time strings from a plausible-looking wrong clock.
    """
    assert _sclk_id(_cassini_identity()) == _CASSINI_SCLK_ID


def _voyager2_identity_with_the_voyager1_ck_object() -> _FrameIdentity:
    """A Voyager 2 identity carrying Voyager 1's CK object.

    One Voyager instrument key serves two spacecraft, so naming the wrong
    spacecraft's CK object is the realistic way this mismatch arises.
    """
    return _FrameIdentity(
        camera_frame='VG2_ISSNA',
        ck_frame_id=_VOYAGER1_CK_FRAME_ID,
        sclk_id=_VOYAGER2_SCLK_ID,
        oops_from_spice=_IDENTITY,
        frozen_oops_attitude=True,
    )


def test_a_clock_that_is_not_the_missions_is_refused() -> None:
    """A CK object resolving to another spacecraft's clock raises."""
    with pytest.raises(NavPointingError, match='resolves to spacecraft clock -31'):
        _sclk_id(_voyager2_identity_with_the_voyager1_ck_object())


def test_the_refused_clock_message_names_the_expected_clock() -> None:
    """The refusal says which clock was expected, so it is attributable."""
    with pytest.raises(NavPointingError, match='not the -32'):
        _sclk_id(_voyager2_identity_with_the_voyager1_ck_object())


def test_ckmeta_computes_a_clock_for_a_ck_object_that_does_not_exist() -> None:
    """``ckmeta`` computes rather than validates, which is why the check exists.

    A nonexistent CK object still yields a plausible clock id and no error,
    so trusting the round trip would write time strings from the wrong clock
    while every call reported success.
    """
    assert int(cspyce.ckmeta(_NONEXISTENT_CK_FRAME_ID, 'SCLK')) == -999


@pytest.mark.parametrize(
    ('ck_frame_id', 'sclk_id'),
    [
        (_CASSINI_CK_FRAME_ID, _CASSINI_SCLK_ID),
        (_GALILEO_CK_FRAME_ID, _GALILEO_SCLK_ID),
        (_LORRI_CK_FRAME_ID, _LORRI_SCLK_ID),
    ],
    ids=['cassini', 'galileo', 'lorri'],
)
def test_each_recorded_ck_object_and_clock_pair_agrees_with_spice(
    ck_frame_id: int, sclk_id: int
) -> None:
    """Every mission's CK object resolves to the clock recorded beside it.

    ``ckmeta`` needs no kernels, so the pairing is checkable here as well as
    on real frames.  A typo in either half of a pair would otherwise only
    surface as time strings encoding another spacecraft's clock.
    """
    assert int(cspyce.ckmeta(ck_frame_id, 'SCLK')) == sclk_id


def test_a_frame_the_kernel_pool_does_not_know_is_a_pointing_failure() -> None:
    """A rotation SPICE cannot supply is reported as the computation's own failure."""
    with pytest.raises(NavPointingError, match='cannot supply the J2000 to NO_SUCH_FRAME'):
        _pxform('NO_SUCH_FRAME', 0.0)


def test_the_spice_error_behind_a_pointing_failure_is_kept_as_the_cause() -> None:
    """The original SPICE exception survives the conversion, so it stays debuggable."""
    with pytest.raises(NavPointingError) as exc_info:
        _pxform('NO_SUCH_FRAME', 0.0)
    assert isinstance(exc_info.value.__cause__, LookupError)


def test_a_frame_name_with_no_spice_id_is_a_pointing_failure() -> None:
    """A frame name that resolves to no id is reported, not left to surface later."""
    with pytest.raises(NavPointingError, match='NO_SUCH_FRAME has no SPICE id'):
        _camera_frame_id('NO_SUCH_FRAME')


def test_a_clock_that_cannot_encode_an_epoch_is_a_pointing_failure() -> None:
    """A clock with no furnished SCLK kernel is reported rather than raising raw."""
    with pytest.raises(NavPointingError, match='cannot encode et'):
        _sclk_string(_NONEXISTENT_SCLK_ID, 0.0)


def test_a_host_with_no_spice_camera_frame_records_nothing() -> None:
    """An observation whose instrument has no frame mapping yields no solution.

    A simulated image has no spacecraft and no furnished camera frame; it is
    returned as "nothing to record" rather than raising, so a simulated
    navigation runs unchanged.
    """
    unmapped = cast(ObsSnapshotInst, object())
    assert compute_pointing(unmapped, offset_px=_PLANTED_OFFSET, rotation_fitted=False) is None


def test_a_determinant_just_inside_tolerance_is_accepted() -> None:
    """A matrix whose determinant is within the tolerance of 1 is accepted.

    Pins the rotation tolerance from below.  Scaling a rotation by
    ``1 + 1e-11`` moves its determinant by about ``3e-11``, comfortably
    inside the documented ``1e-9``.
    """
    nearly = _some_attitude() * (1.0 + 1e-11)
    assert _outcome_of(lambda: _validate_rotation(nearly, 'test matrix')) == 'accepted'


def test_a_determinant_just_outside_tolerance_is_refused() -> None:
    """A determinant further than the tolerance from 1 raises.

    Pins the rotation tolerance from above.  Scaling by ``1 + 1e-8`` moves
    the determinant by about ``3e-8``, an order of magnitude outside the
    documented ``1e-9``, so widening the tolerance fails this test.
    """
    with pytest.raises(NavPointingError, match='is not a proper rotation'):
        _validate_rotation(_some_attitude() * (1.0 + 1e-8), 'test matrix')


def test_a_shear_just_inside_tolerance_is_accepted() -> None:
    """A unit-determinant shear smaller than the tolerance is accepted.

    A shear of ``s`` leaves the determinant exactly 1 and moves
    ``max|C C^T - I|`` by about ``s``, so it exercises the orthonormality
    bound alone.
    """
    sheared = np.array([[1.0, 1e-11, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert _outcome_of(lambda: _validate_rotation(sheared, 'test matrix')) == 'accepted'


def test_a_shear_just_outside_tolerance_is_refused() -> None:
    """A unit-determinant shear larger than the tolerance raises."""
    sheared = np.array([[1.0, 1e-8, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(NavPointingError, match='is not orthonormal'):
        _validate_rotation(sheared, 'test matrix')
