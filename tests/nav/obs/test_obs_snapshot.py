import oops
import pytest

import nav.inst.inst_cassini_iss as instcoiss
import nav.inst.inst_voyager_iss as instvgiss
import nav.obs.obs_snapshot as obs_snapshot
from tests.config import (URL_CASSINI_ISS_STARS_01,
                          URL_VOYAGER_ISS_IO_01,
                          URL_VOYAGER_ISS_URANUS_01)


@pytest.fixture
def obs_coiss1(config_fixture):
    """Cassini ISS observation instance for testing."""
    return instcoiss.InstCassiniISS.from_file(URL_CASSINI_ISS_STARS_01)


@pytest.fixture
def obs_vgiss1(config_fixture):
    """Voyager ISS Io observation instance for testing."""
    return instvgiss.InstVoyagerISS.from_file(URL_VOYAGER_ISS_IO_01)


@pytest.fixture
def obs_vgiss2(config_fixture):
    """Voyager ISS Uranus observation instance for testing."""
    return instvgiss.InstVoyagerISS.from_file(URL_VOYAGER_ISS_URANUS_01)


def test_obs_snapshot_init(obs_coiss1) -> None:
    s = obs_snapshot.ObsSnapshot(obs_coiss1)
    assert s.data_shape_uv == (256, 256)
    assert s.fov_vu_min == (0, 0)
    assert s.fov_vu_max == (255, 255)
    assert s.extfov_margin_vu == (0, 0)
    assert s.extdata.shape == (256, 256)
    assert s.extdata_shape_uv == (256, 256)
    assert s.extfov_vu_min == (0, 0)
    assert s.extfov_vu_max == (255, 255)

    s = obs_snapshot.ObsSnapshot(obs_coiss1, extfov_margin_vu=10)
    assert s.data_shape_uv == (256, 256)
    assert s.fov_vu_min == (0, 0)
    assert s.fov_vu_max == (255, 255)
    assert s.extfov_margin_vu == (10, 10)
    assert s.extdata.shape == (276, 276)
    assert s.extdata_shape_uv == (276, 276)
    assert s.extfov_vu_min == (-10, -10)
    assert s.extfov_vu_max == (265, 265)

    s = obs_snapshot.ObsSnapshot(obs_coiss1, extfov_margin_vu=(10, 20))
    assert s.data_shape_uv == (256, 256)
    assert s.fov_vu_min == (0, 0)
    assert s.fov_vu_max == (255, 255)
    assert s.extfov_margin_vu == (10, 20)
    assert s.extdata.shape == (276, 296)
    assert s.extdata_shape_uv == (296, 276)
    assert s.extfov_vu_min == (-10, -20)
    assert s.extfov_vu_max == (265, 275)


def test_obs_snapshot_bp_mg(obs_coiss1) -> None:
    s = obs_snapshot.ObsSnapshot(obs_coiss1)
    assert s.bp is s.bp
    assert s.bp.meshgrid.shape == (256, 256)
    assert s.bp.meshgrid.uv.vals[0, 0, 0] == 0.5
    assert s.bp.meshgrid.uv.vals[0, 0, 1] == 0.5
    assert s.bp.meshgrid.uv.vals[255, 255, 0] == 255.5
    assert s.bp.meshgrid.uv.vals[255, 255, 1] == 255.5
    assert s.ext_bp is s.bp
    assert s.corner_bp is s.corner_bp
    assert s.corner_bp.meshgrid.shape == (2, 2)
    assert s.corner_bp.meshgrid.uv.vals[0, 0, 0] == 0
    assert s.corner_bp.meshgrid.uv.vals[0, 0, 1] == 0
    assert s.corner_bp.meshgrid.uv.vals[1, 1, 0] == 256
    assert s.corner_bp.meshgrid.uv.vals[1, 1, 1] == 256
    assert s.ext_corner_bp is s.corner_bp
    assert s.center_bp is s.center_bp
    assert s.center_bp.meshgrid.shape == (1, 1)
    assert s.center_bp.meshgrid.uv.vals[0, 0, 0] == 128
    assert s.center_bp.meshgrid.uv.vals[0, 0, 1] == 128

    s = obs_snapshot.ObsSnapshot(obs_coiss1, extfov_margin_vu=(10, 20))
    assert s.bp is s.bp
    assert s.bp.meshgrid.shape == (256, 256)
    assert s.bp.meshgrid.uv.vals[0, 0, 0] == 0.5
    assert s.bp.meshgrid.uv.vals[0, 0, 1] == 0.5
    assert s.bp.meshgrid.uv.vals[255, 255, 0] == 255.5
    assert s.bp.meshgrid.uv.vals[255, 255, 1] == 255.5
    assert s.ext_bp is not s.bp
    assert s.ext_bp.meshgrid.shape == (276, 296)
    assert s.ext_bp.meshgrid.uv.vals[0, 0, 0] == -19.5
    assert s.ext_bp.meshgrid.uv.vals[0, 0, 1] == -9.5
    assert s.ext_bp.meshgrid.uv.vals[275, 295, 0] == 275.5
    assert s.ext_bp.meshgrid.uv.vals[275, 295, 1] == 265.5
    assert s.corner_bp is s.corner_bp
    assert s.corner_bp.meshgrid.shape == (2, 2)
    assert s.corner_bp.meshgrid.uv.vals[0, 0, 0] == 0
    assert s.corner_bp.meshgrid.uv.vals[0, 0, 1] == 0
    assert s.corner_bp.meshgrid.uv.vals[1, 1, 0] == 256
    assert s.corner_bp.meshgrid.uv.vals[1, 1, 1] == 256
    assert s.ext_corner_bp is not s.corner_bp
    assert s.ext_corner_bp.meshgrid.shape == (2, 2)
    assert s.ext_corner_bp.meshgrid.uv.vals[0, 0, 0] == -20
    assert s.ext_corner_bp.meshgrid.uv.vals[0, 0, 1] == -10
    assert s.ext_corner_bp.meshgrid.uv.vals[1, 1, 0] == 276
    assert s.ext_corner_bp.meshgrid.uv.vals[1, 1, 1] == 266
    assert s.center_bp is s.center_bp
    assert s.center_bp.meshgrid.shape == (1, 1)
    assert s.center_bp.meshgrid.uv.vals[0, 0, 0] == 128
    assert s.center_bp.meshgrid.uv.vals[0, 0, 1] == 128


def test_obs_snapshot_ra_dec_limits_wrap(obs_coiss1) -> None:
    s = obs_snapshot.ObsSnapshot(obs_coiss1)
    ra_min, ra_max, dec_min, dec_max = s.ra_dec_limits()
    assert ra_min * oops.DPR == pytest.approx(359.96968466526647)
    assert ra_max * oops.DPR == pytest.approx(0.4436640310675228)
    assert dec_min * oops.DPR == pytest.approx(33.4586221077806)
    assert dec_max * oops.DPR == pytest.approx(33.85318503454683)


def test_obs_snapshot_ra_dec_limits_ext_wrap(obs_coiss1) -> None:
    s = obs_snapshot.ObsSnapshot(obs_coiss1, extfov_margin_vu=10)
    ra_min, ra_max, dec_min, dec_max = s.ra_dec_limits_ext()
    assert ra_min * oops.DPR == pytest.approx(359.95125729565814)
    assert ra_max * oops.DPR == pytest.approx(0.46221430761720933)
    assert dec_min * oops.DPR == pytest.approx(33.44320702355352)
    assert dec_max * oops.DPR == pytest.approx(33.86855152602122)


def test_obs_snapshot_ra_dec_limits(obs_vgiss2) -> None:
    s = obs_snapshot.ObsSnapshot(obs_vgiss2)
    ra_min, ra_max, dec_min, dec_max = s.ra_dec_limits()
    assert ra_min * oops.DPR == pytest.approx(115.13781882464525)
    assert ra_max * oops.DPR == pytest.approx(115.64874257283917)
    assert dec_min * oops.DPR == pytest.approx(18.330307457361243)
    assert dec_max * oops.DPR == pytest.approx(18.814618908448328)


def test_obs_snapshot_ra_dec_limits_ext(obs_vgiss2) -> None:
    s = obs_snapshot.ObsSnapshot(obs_vgiss2, extfov_margin_vu=10)
    ra_min, ra_max, dec_min, dec_max = s.ra_dec_limits_ext()
    assert ra_min * oops.DPR == pytest.approx(115.13272208854077)
    assert ra_max * oops.DPR == pytest.approx(115.65386420280674)
    assert dec_min * oops.DPR == pytest.approx(18.325459463553006)
    assert dec_max * oops.DPR == pytest.approx(18.819456881754636)


def test_obs_snapshot_distance(obs_vgiss1) -> None:
    s = obs_snapshot.ObsSnapshot(obs_vgiss1)
    assert s.sun_body_distance('JUPITER') == pytest.approx(797366152.953817)
    assert s.body_distance('IO') == pytest.approx(1338438.5304067382)
