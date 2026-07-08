from typing import Any

import numpy as np
import oops
import pytest
from tests.config import (
    REQUIRES_EXTERNAL_DATA,
    URL_CASSINI_ISS_STARS_01,
    URL_CASSINI_ISS_TITAN_01,
    URL_VOYAGER_ISS_IO_01,
    URL_VOYAGER_ISS_URANUS_01,
)

import spindoctor.obs.obs_inst_cassini_iss as obstcoiss
import spindoctor.obs.obs_inst_voyager_iss as obstvgiss
import spindoctor.obs.obs_snapshot as obs_snapshot

pytestmark = REQUIRES_EXTERNAL_DATA


@pytest.fixture
def obs_coiss_stars_01(config_fixture: Any) -> obstcoiss.ObsCassiniISS:
    """Cassini ISS observation instance for testing."""
    return obstcoiss.ObsCassiniISS.from_file(URL_CASSINI_ISS_STARS_01)


@pytest.fixture
def obs_coiss_titan_01(config_fixture: Any) -> obstcoiss.ObsCassiniISS:
    """Cassini ISS observation instance for testing."""
    return obstcoiss.ObsCassiniISS.from_file(URL_CASSINI_ISS_TITAN_01)


@pytest.fixture
def obs_vgiss_io_01(config_fixture: Any) -> obstvgiss.ObsVoyagerISS:
    """Voyager ISS Io observation instance for testing."""
    return obstvgiss.ObsVoyagerISS.from_file(URL_VOYAGER_ISS_IO_01)


@pytest.fixture
def obs_vgiss_uranus_01(config_fixture: Any) -> obstvgiss.ObsVoyagerISS:
    """Voyager ISS Uranus observation instance for testing."""
    return obstvgiss.ObsVoyagerISS.from_file(URL_VOYAGER_ISS_URANUS_01)


def test_obs_snapshot_init(obs_coiss_stars_01: obstcoiss.ObsCassiniISS) -> None:
    s = obs_snapshot.ObsSnapshot(obs_coiss_stars_01)
    assert s.data_shape_uv == (1024, 1024)
    assert s.fov_vu_min == (0, 0)
    assert s.fov_vu_max == (1023, 1023)
    assert s.extfov_margin_vu == (0, 0)
    assert s.extdata.shape == (1024, 1024)
    assert s.extdata_shape_uv == (1024, 1024)
    assert s.extfov_vu_min == (0, 0)
    assert s.extfov_vu_max == (1023, 1023)

    s = obs_snapshot.ObsSnapshot(obs_coiss_stars_01, extfov_margin_vu=10)
    assert s.data_shape_uv == (1024, 1024)
    assert s.fov_vu_min == (0, 0)
    assert s.fov_vu_max == (1023, 1023)
    assert s.extfov_margin_vu == (10, 10)
    assert s.extdata.shape == (1044, 1044)
    assert s.extdata_shape_uv == (1044, 1044)
    assert s.extfov_vu_min == (-10, -10)
    assert s.extfov_vu_max == (1033, 1033)

    s = obs_snapshot.ObsSnapshot(obs_coiss_stars_01, extfov_margin_vu=(10, 20))
    assert s.data_shape_uv == (1024, 1024)
    assert s.fov_vu_min == (0, 0)
    assert s.fov_vu_max == (1023, 1023)
    assert s.extfov_margin_vu == (10, 20)
    assert s.extdata.shape == (1044, 1064)
    assert s.extdata_shape_uv == (1064, 1044)
    assert s.extfov_vu_min == (-10, -20)
    assert s.extfov_vu_max == (1033, 1043)


def test_obs_snapshot_bp_mg(obs_coiss_stars_01: obstcoiss.ObsCassiniISS) -> None:
    s = obs_snapshot.ObsSnapshot(obs_coiss_stars_01)
    assert s.bp is s.bp
    assert s.bp.meshgrid.shape == (1024, 1024)
    assert s.bp.meshgrid.uv.vals[0, 0, 0] == 0.5
    assert s.bp.meshgrid.uv.vals[0, 0, 1] == 0.5
    assert s.bp.meshgrid.uv.vals[1023, 1023, 0] == 1023.5
    assert s.bp.meshgrid.uv.vals[1023, 1023, 1] == 1023.5
    assert s.ext_bp is s.bp
    assert s.corner_bp is s.corner_bp
    assert s.corner_bp.meshgrid.shape == (2, 2)
    assert s.corner_bp.meshgrid.uv.vals[0, 0, 0] == 0
    assert s.corner_bp.meshgrid.uv.vals[0, 0, 1] == 0
    assert s.corner_bp.meshgrid.uv.vals[1, 1, 0] == 1024
    assert s.corner_bp.meshgrid.uv.vals[1, 1, 1] == 1024
    assert s.ext_corner_bp is s.corner_bp
    assert s.center_bp is s.center_bp
    assert s.center_bp.meshgrid.shape == (1, 1)
    assert s.center_bp.meshgrid.uv.vals[0, 0, 0] == 512
    assert s.center_bp.meshgrid.uv.vals[0, 0, 1] == 512

    s = obs_snapshot.ObsSnapshot(obs_coiss_stars_01, extfov_margin_vu=(10, 20))
    assert s.bp is s.bp
    assert s.bp.meshgrid.shape == (1024, 1024)
    assert s.bp.meshgrid.uv.vals[0, 0, 0] == 0.5
    assert s.bp.meshgrid.uv.vals[0, 0, 1] == 0.5
    assert s.bp.meshgrid.uv.vals[1023, 1023, 0] == 1023.5
    assert s.bp.meshgrid.uv.vals[1023, 1023, 1] == 1023.5
    assert s.ext_bp is not s.bp
    assert s.ext_bp.meshgrid.shape == (1044, 1064)
    assert s.ext_bp.meshgrid.uv.vals[0, 0, 0] == -19.5
    assert s.ext_bp.meshgrid.uv.vals[0, 0, 1] == -9.5
    assert s.ext_bp.meshgrid.uv.vals[1043, 1063, 0] == 1043.5
    assert s.ext_bp.meshgrid.uv.vals[1043, 1063, 1] == 1033.5
    assert s.corner_bp is s.corner_bp
    assert s.corner_bp.meshgrid.shape == (2, 2)
    assert s.corner_bp.meshgrid.uv.vals[0, 0, 0] == 0
    assert s.corner_bp.meshgrid.uv.vals[0, 0, 1] == 0
    assert s.corner_bp.meshgrid.uv.vals[1, 1, 0] == 1024
    assert s.corner_bp.meshgrid.uv.vals[1, 1, 1] == 1024
    assert s.ext_corner_bp is not s.corner_bp
    assert s.ext_corner_bp.meshgrid.shape == (2, 2)
    assert s.ext_corner_bp.meshgrid.uv.vals[0, 0, 0] == -20
    assert s.ext_corner_bp.meshgrid.uv.vals[0, 0, 1] == -10
    assert s.ext_corner_bp.meshgrid.uv.vals[1, 1, 0] == 1044
    assert s.ext_corner_bp.meshgrid.uv.vals[1, 1, 1] == 1034
    assert s.center_bp is s.center_bp
    assert s.center_bp.meshgrid.shape == (1, 1)
    assert s.center_bp.meshgrid.uv.vals[0, 0, 0] == 512
    assert s.center_bp.meshgrid.uv.vals[0, 0, 1] == 512


def test_obs_snapshot_ra_dec_limits_wrap(
    obs_coiss_titan_01: obstcoiss.ObsCassiniISS,
) -> None:
    s = obs_snapshot.ObsSnapshot(obs_coiss_titan_01)
    ra_min, ra_max, dec_min, dec_max = s.ra_dec_limits()
    assert ra_min * oops.DPR == pytest.approx(358.2124125206211)
    assert ra_max * oops.DPR == pytest.approx(4.265581158319197)
    assert dec_min * oops.DPR == pytest.approx(-49.15956373477803)
    assert dec_max * oops.DPR == pytest.approx(-45.044159571378906)


def test_obs_snapshot_ra_dec_limits_ext_wrap(
    obs_coiss_titan_01: obstcoiss.ObsCassiniISS,
) -> None:
    s = obs_snapshot.ObsSnapshot(obs_coiss_titan_01, extfov_margin_vu=10)
    ra_min, ra_max, dec_min, dec_max = s.ra_dec_limits_ext()
    assert ra_min * oops.DPR == pytest.approx(358.1569510257557)
    assert ra_max * oops.DPR == pytest.approx(4.327186052757717)
    assert dec_min * oops.DPR == pytest.approx(-49.19857831505729)
    assert dec_max * oops.DPR == pytest.approx(-45.0037137426367)


def test_obs_snapshot_ra_dec_limits(
    obs_vgiss_uranus_01: obstvgiss.ObsVoyagerISS,
) -> None:
    s = obs_snapshot.ObsSnapshot(obs_vgiss_uranus_01)
    ra_min, ra_max, dec_min, dec_max = s.ra_dec_limits()
    assert ra_min * oops.DPR == pytest.approx(115.13781882464525)
    assert ra_max * oops.DPR == pytest.approx(115.64874257283917)
    assert dec_min * oops.DPR == pytest.approx(18.330307457361243)
    assert dec_max * oops.DPR == pytest.approx(18.814618908448328)


def test_obs_snapshot_ra_dec_limits_ext(
    obs_vgiss_uranus_01: obstvgiss.ObsVoyagerISS,
) -> None:
    s = obs_snapshot.ObsSnapshot(obs_vgiss_uranus_01, extfov_margin_vu=10)
    ra_min, ra_max, dec_min, dec_max = s.ra_dec_limits_ext()
    assert ra_min * oops.DPR == pytest.approx(115.13272208854077)
    assert ra_max * oops.DPR == pytest.approx(115.65386420280674)
    assert dec_min * oops.DPR == pytest.approx(18.325459463553006)
    assert dec_max * oops.DPR == pytest.approx(18.819456881754636)


def test_obs_snapshot_distance(obs_vgiss_io_01: obstvgiss.ObsVoyagerISS) -> None:
    s = obs_snapshot.ObsSnapshot(obs_vgiss_io_01)
    assert s.sun_body_distance('JUPITER') == pytest.approx(797366152.953817)
    assert s.body_distance('IO') == pytest.approx(1338438.5304067382)


def test_extract_offset_array_within_margin_returns_translated_slice(
    obs_coiss_stars_01: obstcoiss.ObsCassiniISS,
) -> None:
    """In-bounds offset returns the appropriately translated slice."""
    s = obs_snapshot.ObsSnapshot(obs_coiss_stars_01, extfov_margin_vu=(5, 10))
    array = np.arange(s.extdata_shape_vu[0] * s.extdata_shape_vu[1], dtype=np.float64).reshape(
        s.extdata_shape_vu
    )
    out = s.extract_offset_array(array, (2.0, -3.0))
    expected = array[5 - 2 : 5 - 2 + 1024, 10 + 3 : 10 + 3 + 1024]
    assert out.shape == (1024, 1024)
    assert np.array_equal(out, expected)


def test_extract_offset_array_offset_beyond_margin_zero_fills_clipped_region(
    obs_coiss_stars_01: obstcoiss.ObsCassiniISS,
) -> None:
    """Offset > margin zero-fills the out-of-extfov rows instead of raising.

    With ``offset = (10.72, -15.0)`` and ``extfov_margin_vu = (5, 10)`` the
    requested slice runs from ``v0 = -6`` to ``v1 = 1018`` (six rows above
    the extfov in V) and from ``u0 = 25`` to ``u1 = 1049`` (five columns
    past the right edge of the extfov in U).  The in-bounds region must
    match ``array``, the out-of-bounds rows / cols must be zero, and the
    call must not raise.
    """
    s = obs_snapshot.ObsSnapshot(obs_coiss_stars_01, extfov_margin_vu=(5, 10))
    array = np.ones(s.extdata_shape_vu, dtype=np.float64)
    out = s.extract_offset_array(array, (10.72, -15.0))
    assert out.shape == (1024, 1024)
    # First 6 rows of the output are zero-filled (slice begins at v0=-6).
    assert np.all(out[:6, :] == 0.0)
    # Last 5 columns of the output are zero-filled (slice ends past u1=1044).
    assert np.all(out[:, -5:] == 0.0)
    # The interior is populated from ``array``.
    assert np.all(out[6:, :-5] == 1.0)


def test_extract_offset_array_offset_far_outside_extfov_returns_all_zeros(
    obs_coiss_stars_01: obstcoiss.ObsCassiniISS,
) -> None:
    """A pathological offset entirely outside the extfov yields a zero array."""
    s = obs_snapshot.ObsSnapshot(obs_coiss_stars_01, extfov_margin_vu=(5, 10))
    array = np.ones(s.extdata_shape_vu, dtype=np.float64)
    out = s.extract_offset_array(array, (5000.0, 0.0))
    assert out.shape == (1024, 1024)
    assert np.all(out == 0.0)


def test_extract_offset_array_preserves_dtype(
    obs_coiss_stars_01: obstcoiss.ObsCassiniISS,
) -> None:
    """Boolean overlays stay boolean; integer overlays stay integer."""
    s = obs_snapshot.ObsSnapshot(obs_coiss_stars_01, extfov_margin_vu=(2, 2))
    bool_array = np.ones(s.extdata_shape_vu, dtype=bool)
    out_bool = s.extract_offset_array(bool_array, (10.0, 0.0))
    assert out_bool.dtype == bool
    assert not out_bool[0, 0]  # zero-filled (False) past the margin
    int_array = np.full(s.extdata_shape_vu, 7, dtype=np.int32)
    out_int = s.extract_offset_array(int_array, (0.0, 0.0))
    assert out_int.dtype == np.int32
