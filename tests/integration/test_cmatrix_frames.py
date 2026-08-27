"""Real-frame integration tests for ``spindoctor.support.cmatrix``.

The relation between an oops observation frame and the SPICE camera frame it
is built on cannot be checked hermetically: it only exists once the mission's
kernels are furnished and the host has built its frame.  These tests measure
that relation on one frame per instrument and pin it to the constant the
implementation asserts, at the exposure start, midtime and stop.

Voyager gets its own test because oops does not build its frame from a frame
chain at all: it freezes a tolerance-snapped pointing lookup, so ``pxform``
at the midtime does not reproduce it and the recorded uncorrected attitude
must be the frozen one.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import cast

import numpy as np
import oops
import pytest
from filecache import FCPath

pytestmark = pytest.mark.integration

if 'PDS3_HOLDINGS_DIR' not in os.environ:
    pytest.skip(
        'PDS3_HOLDINGS_DIR is not set; skipping C-matrix frame integration tests',
        allow_module_level=True,
    )

import cspyce  # noqa: E402  (guarded import)

from spindoctor.obs import (  # noqa: E402
    ObsCassiniISS,
    ObsGalileoSSI,
    ObsNewHorizonsLORRI,
    ObsSnapshotInst,
    ObsVoyagerISS,
)
from spindoctor.support.cmatrix import (  # noqa: E402
    _attitude_baseline,
    _frame_identity,
    _FrameIdentity,
    _sclk_id,
    compute_pointing,
)
from spindoctor.support.exceptions import NavPointingError  # noqa: E402  (guarded import)
from tests.cmatrix_helpers import (  # noqa: E402  (guarded import)
    observation_attitude,
    offset_from_correction,
)

_CASSINI_NAC = 'calibrated/COISS_2xxx/COISS_2038/data/1572094226_1572114418/N1572105349_1_CALIB.IMG'
_CASSINI_WAC = 'volumes/COISS_2xxx/COISS_2099/data/1822057149_1822284412/W1822132529_1.IMG'
_GALILEO_SSI = 'volumes/GO_0xxx/GO_0003/RAW_CAL/C0059897400R.IMG'
_LORRI = 'volumes/NHxxLO_xxxx/NHJULO_2001/data/20070110_003071/lor_0030713597_0x633_sci.fit'
_VOYAGER_NAC = 'volumes/VGISS_8xxx/VGISS_8210/DATA/C12050XX/C1205021_GEOMED.IMG'

_FLIP_TOL = 1e-9

# A deliberately asymmetric offset, so a sign or axis error cannot cancel.
_OFFSET = (8.68, -17.37)

# A sub-pixel offset, well above the degenerate-axis guard but far below one
# pixel, so the guard cannot be widened without this failing.
_SMALL_OFFSET = (0.05, -0.02)

# Tolerance on recovering a planted offset back out of a recorded C-matrix.
# The exact rigid rotation is not exactly a uniform tangent-plane shift, so the
# inverse carries the second-order difference.  Measured across these five
# frames at both offsets below, the largest residual is 9.1e-4 px (Cassini
# WAC); every directional error these tests guard against is off by twice the
# offset, so a 1e-2 px bound separates the two by four orders of magnitude.
_RECOVERY_TOL_PX = 1.0e-2

_CASSINI_FLIP = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
_LORRI_FLIP = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
_NO_FLIP = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

_CASSINI_FLIP_ARRAY = np.asarray(_CASSINI_FLIP)


def _url(relative: str) -> FCPath:
    """Resolve a holdings-relative path against ``PDS3_HOLDINGS_DIR``."""
    return FCPath(f'{os.environ["PDS3_HOLDINGS_DIR"].rstrip("/")}/{relative}')


def _load(obs_class: type[ObsSnapshotInst], relative: str) -> ObsSnapshotInst:
    """Load one holdings image through its instrument class."""
    return cast(ObsSnapshotInst, obs_class.from_file(_url(relative)))


def _measured_flip(obs: ObsSnapshotInst, camera_frame: str, et: float) -> np.ndarray:
    """Measure ``R = C_oops . C_spice^T`` directly from SPICE at one epoch."""
    spice = np.asarray(cspyce.pxform('J2000', camera_frame, et), np.float64)
    flip: np.ndarray = observation_attitude(obs, et) @ spice.T
    return flip


def _rotation_about_z(angle_rad: float) -> np.ndarray:
    """Build a small proper rotation about the Z axis."""
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _cassini_nac_identity() -> _FrameIdentity:
    """The Cassini NAC frame facts, for driving the baseline guards directly."""
    return _FrameIdentity(
        ck_frame_id=-82000,
        sclk_id=-82,
        frozen_oops_attitude=False,
    )


class _StubTransform:
    """Stands in for an oops ``Transform``, carrying only its matrix.

    The matrix is a real ``Matrix3`` rather than a stand-in, because the
    accessor this feeds composes with it.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = oops.Matrix3(matrix)


class _StubFrame:
    """Stands in for an oops frame whose attitude a test dictates per epoch."""

    def __init__(self, attitude: Callable[[float], np.ndarray]) -> None:
        self._attitude = attitude

    def wrt(self, reference: object) -> _StubFrame:
        """Return self; the stub is always expressed relative to J2000."""
        return self

    def transform_at_time(self, et: float) -> _StubTransform:
        """Return the attitude the test dictated for this epoch."""
        return _StubTransform(self._attitude(float(et)))


class _StubObs:
    """A real observation's times wrapped around a test-dictated frame.

    Only the attributes ``_attitude_baseline`` reads are provided.  The times
    come from a real frame so the SPICE clock and frame lookups it also makes
    resolve against the furnished kernels.

    The accessor is borrowed from oops rather than reimplemented, so what the
    test exercises is what a real observation would do with the dictated frame.
    """

    get_spice_cmatrix = oops.obs.Observation.get_spice_cmatrix

    def __init__(self, attitude: Callable[[float], np.ndarray], obs: ObsSnapshotInst) -> None:
        self.frame = _StubFrame(attitude)
        self.time = (float(obs.time[0]), float(obs.time[1]))
        self.midtime = float(obs.midtime)
        self.texp = float(obs.texp)
        # The subfields the real host declares, carried through unchanged so
        # that only the frame is dictated by the test.
        self.spice_to_frame = obs.spice_to_frame
        self.spice_frame_name = obs.spice_frame_name
        self.spice_frame_id = obs.spice_frame_id


@pytest.mark.parametrize(
    ('obs_class', 'relative', 'camera_frame', 'expected_flip', 'ck_frame_id'),
    [
        (ObsCassiniISS, _CASSINI_NAC, 'CASSINI_ISS_NAC', _CASSINI_FLIP, -82000),
        (ObsCassiniISS, _CASSINI_WAC, 'CASSINI_ISS_WAC', _CASSINI_FLIP, -82000),
        (ObsNewHorizonsLORRI, _LORRI, 'NH_LORRI', _LORRI_FLIP, -98000),
        (ObsGalileoSSI, _GALILEO_SSI, 'GLL_SCAN_PLATFORM', _NO_FLIP, -77001),
    ],
    ids=['cassini_nac', 'cassini_wac', 'lorri', 'galileo_ssi'],
)
def test_measured_flip_matches_the_expected_constant(
    obs_class: type[ObsSnapshotInst],
    relative: str,
    camera_frame: str,
    expected_flip: list[list[float]],
    ck_frame_id: int,
) -> None:
    """The recorded flip is the instrument's documented constant.

    Also pins the frame name and CK object the solution records, since a
    correct flip against the wrong frame would be a silent mis-attribution.

    Parameters:
        obs_class: The instrument class the image is loaded through.
        relative: Holdings-relative path of the image.
        camera_frame: SPICE name of the frame the solution must record.
        expected_flip: The instrument's documented constant flip.
        ck_frame_id: SPICE id of the object the solution must record.
    """
    obs = _load(obs_class, relative)
    solution = compute_pointing(obs, offset_px=_OFFSET, rotation_fitted=False)
    assert solution is not None
    assert solution.baseline.camera_frame == camera_frame
    assert solution.baseline.ck_frame_id == ck_frame_id
    measured = np.asarray(solution.baseline.oops_from_spice, np.float64)
    assert float(np.max(np.abs(measured - np.asarray(expected_flip)))) < _FLIP_TOL


@pytest.mark.parametrize(
    ('obs_class', 'relative', 'camera_frame'),
    [
        (ObsCassiniISS, _CASSINI_NAC, 'CASSINI_ISS_NAC'),
        (ObsCassiniISS, _CASSINI_WAC, 'CASSINI_ISS_WAC'),
        (ObsNewHorizonsLORRI, _LORRI, 'NH_LORRI'),
        (ObsGalileoSSI, _GALILEO_SSI, 'GLL_SCAN_PLATFORM'),
    ],
    ids=['cassini_nac', 'cassini_wac', 'lorri', 'galileo_ssi'],
)
def test_measured_flip_is_identical_at_start_mid_and_stop(
    obs_class: type[ObsSnapshotInst], relative: str, camera_frame: str
) -> None:
    """The flip does not vary across the exposure.

    The correction is applied as a constant conjugation, which is only valid
    if the two frames really are rigidly attached to each other.

    Parameters:
        obs_class: The instrument class the image is loaded through.
        relative: Holdings-relative path of the image.
        camera_frame: SPICE name of the frame the flip is measured against.
    """
    obs = _load(obs_class, relative)
    at_mid = _measured_flip(obs, camera_frame, float(obs.midtime))
    at_start = _measured_flip(obs, camera_frame, float(obs.time[0]))
    at_stop = _measured_flip(obs, camera_frame, float(obs.time[1]))
    assert float(np.max(np.abs(at_start - at_mid))) < _FLIP_TOL
    assert float(np.max(np.abs(at_stop - at_mid))) < _FLIP_TOL


def test_voyager_records_the_frozen_oops_attitude() -> None:
    """Voyager's uncorrected attitude is the frozen observation frame itself."""
    obs = _load(ObsVoyagerISS, _VOYAGER_NAC)
    solution = compute_pointing(obs, offset_px=_OFFSET, rotation_fitted=False)
    assert solution is not None
    frozen = observation_attitude(obs, float(obs.midtime))
    recorded = np.asarray(solution.baseline.cmatrix_original, np.float64)
    assert float(np.max(np.abs(recorded - frozen))) == 0.0


def test_voyager_pxform_cannot_reproduce_the_frozen_attitude() -> None:
    """A plain ``pxform`` at the Voyager midtime does not resolve at all.

    Pins the reason Voyager takes its own path: the scan-platform CK is
    sparse, so the frame chain has no data at the exact midtime and only a
    tolerance-snapped lookup succeeds.
    """
    obs = _load(ObsVoyagerISS, _VOYAGER_NAC)
    with pytest.raises(RuntimeError, match='insufficient information'):
        cspyce.pxform('J2000', 'VG2_ISSNA', float(obs.midtime))


def test_voyager_records_its_per_spacecraft_frames() -> None:
    """Voyager frame identities are derived per spacecraft, not per instrument."""
    obs = _load(ObsVoyagerISS, _VOYAGER_NAC)
    solution = compute_pointing(obs, offset_px=_OFFSET, rotation_fitted=False)
    assert solution is not None
    assert solution.baseline.camera_frame == 'VG2_ISSNA'
    assert solution.baseline.camera_frame_id == -32101
    assert solution.baseline.ck_frame_id == -32100


def test_voyager_flip_is_the_identity() -> None:
    """No flip is applied on the Voyager path, by construction."""
    obs = _load(ObsVoyagerISS, _VOYAGER_NAC)
    solution = compute_pointing(obs, offset_px=_OFFSET, rotation_fitted=False)
    assert solution is not None
    measured = np.asarray(solution.baseline.oops_from_spice, np.float64)
    assert float(np.max(np.abs(measured - np.eye(3)))) == 0.0


@pytest.mark.parametrize(
    ('obs_class', 'relative'),
    [
        (ObsCassiniISS, _CASSINI_NAC),
        (ObsNewHorizonsLORRI, _LORRI),
        (ObsGalileoSSI, _GALILEO_SSI),
        (ObsVoyagerISS, _VOYAGER_NAC),
    ],
    ids=['cassini_nac', 'lorri', 'galileo_ssi', 'voyager_nac'],
)
def test_recorded_times_bracket_the_midtime(
    obs_class: type[ObsSnapshotInst], relative: str
) -> None:
    """The recorded epochs are the observation's own exposure window.

    Parameters:
        obs_class: The instrument class the image is loaded through.
        relative: Holdings-relative path of the image.
    """
    obs = _load(obs_class, relative)
    solution = compute_pointing(obs, offset_px=_OFFSET, rotation_fitted=False)
    assert solution is not None
    baseline = solution.baseline
    assert baseline.start_et == float(obs.time[0])
    assert baseline.stop_et == float(obs.time[1])
    assert baseline.midtime_et == float(obs.midtime)
    assert baseline.exposure_s == pytest.approx(float(obs.texp))


@pytest.mark.parametrize(
    ('obs_class', 'relative'),
    [
        (ObsCassiniISS, _CASSINI_NAC),
        (ObsNewHorizonsLORRI, _LORRI),
        (ObsGalileoSSI, _GALILEO_SSI),
        (ObsVoyagerISS, _VOYAGER_NAC),
    ],
    ids=['cassini_nac', 'lorri', 'galileo_ssi', 'voyager_nac'],
)
def test_recorded_clock_strings_are_distinct_and_ordered(
    obs_class: type[ObsSnapshotInst], relative: str
) -> None:
    """Start, midtime and stop encode to three different clock readings.

    Parameters:
        obs_class: The instrument class the image is loaded through.
        relative: Holdings-relative path of the image.
    """
    obs = _load(obs_class, relative)
    solution = compute_pointing(obs, offset_px=_OFFSET, rotation_fitted=False)
    assert solution is not None
    baseline = solution.baseline
    assert baseline.sclk_start < baseline.sclk_midtime
    assert baseline.sclk_midtime < baseline.sclk_stop


@pytest.mark.parametrize(
    ('obs_class', 'relative', 'sclk_id'),
    [
        (ObsCassiniISS, _CASSINI_NAC, -82),
        (ObsNewHorizonsLORRI, _LORRI, -98),
        (ObsGalileoSSI, _GALILEO_SSI, -77),
        (ObsVoyagerISS, _VOYAGER_NAC, -32),
    ],
    ids=['cassini_nac', 'lorri', 'galileo_ssi', 'voyager_nac'],
)
def test_the_recorded_ck_object_resolves_to_the_missions_clock(
    obs_class: type[ObsSnapshotInst], relative: str, sclk_id: int
) -> None:
    """Each mission's CK object resolves to the spacecraft clock it expects.

    ``ckmeta`` computes a clock id from a CK object id instead of validating
    it, so a wrong CK object would yield a plausible clock and clock strings
    encoding another spacecraft's time.  This pins the pair per mission
    against what SPICE actually resolves.

    Parameters:
        obs_class: The instrument class the image is loaded through.
        relative: Holdings-relative path of the image.
        sclk_id: The spacecraft clock the mission's CK object must resolve to.
    """
    obs = _load(obs_class, relative)
    identity = _frame_identity(obs)
    assert identity is not None
    assert identity.sclk_id == sclk_id
    assert _sclk_id(identity, str(obs.spice_frame_name)) == sclk_id


@pytest.mark.parametrize(
    ('obs_class', 'relative'),
    [
        (ObsCassiniISS, _CASSINI_NAC),
        (ObsCassiniISS, _CASSINI_WAC),
        (ObsNewHorizonsLORRI, _LORRI),
        (ObsGalileoSSI, _GALILEO_SSI),
        (ObsVoyagerISS, _VOYAGER_NAC),
    ],
    ids=['cassini_nac', 'cassini_wac', 'lorri', 'galileo_ssi', 'voyager_nac'],
)
def test_recorded_cmatrix_recovers_the_planted_offset(
    obs_class: type[ObsSnapshotInst], relative: str
) -> None:
    """The recorded C-matrix returns the exact ``(dv, du)`` that produced it.

    The inverse runs on the real distorted FOV of each instrument and against
    the real measured flip, so every directional convention in the chain is
    pinned at once: a flipped ``xy_offset``, a transposed correction, a
    dropped or reversed ``R`` conjugation, and a reversed composition all
    return a different offset.

    Parameters:
        obs_class: The instrument class the image is loaded through.
        relative: Holdings-relative path of the image.
    """
    obs = _load(obs_class, relative)
    solution = compute_pointing(obs, offset_px=_OFFSET, rotation_fitted=False)
    assert solution is not None
    assert solution.cmatrix is not None
    flip = np.asarray(solution.baseline.oops_from_spice, np.float64)
    corrected_oops = flip @ np.asarray(solution.cmatrix, np.float64)
    original_oops = flip @ np.asarray(solution.baseline.cmatrix_original, np.float64)
    recovered = offset_from_correction(obs.fov, corrected_oops @ original_oops.T)
    assert recovered[0] == pytest.approx(_OFFSET[0], abs=_RECOVERY_TOL_PX)
    assert recovered[1] == pytest.approx(_OFFSET[1], abs=_RECOVERY_TOL_PX)


@pytest.mark.parametrize(
    ('obs_class', 'relative'),
    [
        (ObsCassiniISS, _CASSINI_NAC),
        (ObsCassiniISS, _CASSINI_WAC),
        (ObsNewHorizonsLORRI, _LORRI),
        (ObsGalileoSSI, _GALILEO_SSI),
        (ObsVoyagerISS, _VOYAGER_NAC),
    ],
    ids=['cassini_nac', 'cassini_wac', 'lorri', 'galileo_ssi', 'voyager_nac'],
)
def test_recorded_cmatrix_recovers_a_sub_pixel_offset(
    obs_class: type[ObsSnapshotInst], relative: str
) -> None:
    """A sub-pixel offset survives the round trip on a real FOV too.

    The degenerate-axis guard must not absorb a real measurement, and the
    second-order rotation-versus-shift difference is proportional to the
    offset, so a small offset recovers tighter than a large one.

    Parameters:
        obs_class: The instrument class the image is loaded through.
        relative: Holdings-relative path of the image.
    """
    obs = _load(obs_class, relative)
    solution = compute_pointing(obs, offset_px=_SMALL_OFFSET, rotation_fitted=False)
    assert solution is not None
    assert solution.cmatrix is not None
    flip = np.asarray(solution.baseline.oops_from_spice, np.float64)
    corrected_oops = flip @ np.asarray(solution.cmatrix, np.float64)
    original_oops = flip @ np.asarray(solution.baseline.cmatrix_original, np.float64)
    recovered = offset_from_correction(obs.fov, corrected_oops @ original_oops.T)
    assert recovered[0] == pytest.approx(_SMALL_OFFSET[0], abs=_RECOVERY_TOL_PX)
    assert recovered[1] == pytest.approx(_SMALL_OFFSET[1], abs=_RECOVERY_TOL_PX)


def test_epoch_varying_flip_is_refused() -> None:
    """A flip that changes across the exposure is refused, not averaged away.

    The correction is applied to a kernel as one constant body-fixed
    rotation, which is only valid while the two frames are rigidly attached.
    A stub observation whose frame drifts against the real SPICE camera frame
    exercises the guard that would otherwise let a drifting frame through.

    The guard measures every epoch against the constant the host declares, so
    a frame that is right at the midtime and wrong at the edges is refused at
    the edge rather than by a separate constancy comparison.
    """
    obs = _load(ObsCassiniISS, _CASSINI_NAC)
    midtime = float(obs.midtime)
    drift = _rotation_about_z(1.0e-6)

    def attitude(et: float) -> np.ndarray:
        spice = np.asarray(cspyce.pxform('J2000', 'CASSINI_ISS_NAC', et), np.float64)
        flip = _CASSINI_FLIP_ARRAY if et == midtime else drift @ _CASSINI_FLIP_ARRAY
        product: np.ndarray = flip @ spice
        return product

    stub = cast(ObsSnapshotInst, _StubObs(attitude, obs))
    with pytest.raises(NavPointingError, match='the host declares'):
        _attitude_baseline(stub, _cassini_nac_identity())


def test_a_wrong_flip_measured_from_the_frame_is_refused() -> None:
    """An observation frame that is not the expected flip of the SPICE frame raises.

    Pins the direction the flip is measured in as well as its value: measuring
    ``C_spice^T . C_oops`` instead of ``C_oops . C_spice^T`` produces a matrix
    that is not the instrument's constant and lands here.
    """
    obs = _load(ObsCassiniISS, _CASSINI_NAC)

    def attitude(et: float) -> np.ndarray:
        spice: np.ndarray = np.asarray(cspyce.pxform('J2000', 'CASSINI_ISS_NAC', et), np.float64)
        return spice

    stub = cast(ObsSnapshotInst, _StubObs(attitude, obs))
    with pytest.raises(NavPointingError, match='the host declares'):
        _attitude_baseline(stub, _cassini_nac_identity())
