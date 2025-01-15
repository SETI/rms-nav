import oops
import pytest

import nav.inst.inst_cassini_iss as instcoiss
import nav.inst.inst_voyager_iss as instvgiss
import nav.obs.obs_snapshot as obs_snapshot
from tests.config import URL_CASSINI_ISS_02, URL_VOYAGER_ISS_01, URL_VOYAGER_ISS_02

# This image was chosen because it is 256x256 (and thus faster to create a Backplane)
# and the RA wraps around.
OBS_COISS2 = instcoiss.InstCassiniISS.from_file(URL_CASSINI_ISS_02)

# This image was chosen because it is of Io
OBS_VGISS1 = instvgiss.InstVoyagerISS.from_file(URL_VOYAGER_ISS_01)

# This image was chosen because it's a different instrument and the RA does not wrap
# around and also it is of Uranus
OBS_VGISS2 = instvgiss.InstVoyagerISS.from_file(URL_VOYAGER_ISS_02)


def test_obs_snapshot_init():
    s = obs_snapshot.ObsSnapshot(OBS_COISS2)
    assert s._data_shape_uv == (256, 256)
    assert s._fov_uv_min == (0, 0)
    assert s._fov_uv_max == (255, 255)
    assert s._extfov_margin_uv == (0, 0)
    assert s._extdata.shape == (256, 256)
    assert s._extdata_shape_uv == (256, 256)
    assert s._extfov_uv_min == (0, 0)
    assert s._extfov_uv_max == (255, 255)

    s = obs_snapshot.ObsSnapshot(OBS_COISS2, extfov_margin=10)
    assert s._data_shape_uv == (256, 256)
    assert s._fov_uv_min == (0, 0)
    assert s._fov_uv_max == (255, 255)
    assert s._extfov_margin_uv == (10, 10)
    assert s._extdata.shape == (276, 276)
    assert s._extdata_shape_uv == (276, 276)
    assert s._extfov_uv_min == (-10, -10)
    assert s._extfov_uv_max == (265, 265)

    s = obs_snapshot.ObsSnapshot(OBS_COISS2, extfov_margin=(10, 20))
    assert s._data_shape_uv == (256, 256)
    assert s._fov_uv_min == (0, 0)
    assert s._fov_uv_max == (255, 255)
    assert s._extfov_margin_uv == (10, 20)
    assert s._extdata.shape == (296, 276)
    assert s._extdata_shape_uv == (276, 296)
    assert s._extfov_uv_min == (-10, -20)
    assert s._extfov_uv_max == (265, 275)


def test_obs_snapshot_bp_mg():
    s = obs_snapshot.ObsSnapshot(OBS_COISS2)
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

    s = obs_snapshot.ObsSnapshot(OBS_COISS2, extfov_margin=(10, 20))
    assert s.bp is s.bp
    assert s.bp.meshgrid.shape == (256, 256)
    assert s.bp.meshgrid.uv.vals[0, 0, 0] == 0.5
    assert s.bp.meshgrid.uv.vals[0, 0, 1] == 0.5
    assert s.bp.meshgrid.uv.vals[255, 255, 0] == 255.5
    assert s.bp.meshgrid.uv.vals[255, 255, 1] == 255.5
    assert s.ext_bp is not s.bp
    assert s.ext_bp.meshgrid.shape == (296, 276)
    assert s.ext_bp.meshgrid.uv.vals[0, 0, 0] == -9.5
    assert s.ext_bp.meshgrid.uv.vals[0, 0, 1] == -19.5
    assert s.ext_bp.meshgrid.uv.vals[295, 275, 0] == 265.5
    assert s.ext_bp.meshgrid.uv.vals[295, 275, 1] == 275.5
    assert s.corner_bp is s.corner_bp
    assert s.corner_bp.meshgrid.shape == (2, 2)
    assert s.corner_bp.meshgrid.uv.vals[0, 0, 0] == 0
    assert s.corner_bp.meshgrid.uv.vals[0, 0, 1] == 0
    assert s.corner_bp.meshgrid.uv.vals[1, 1, 0] == 256
    assert s.corner_bp.meshgrid.uv.vals[1, 1, 1] == 256
    assert s.ext_corner_bp is not s.corner_bp
    assert s.ext_corner_bp.meshgrid.shape == (2, 2)
    assert s.ext_corner_bp.meshgrid.uv.vals[0, 0, 0] == -10
    assert s.ext_corner_bp.meshgrid.uv.vals[0, 0, 1] == -20
    assert s.ext_corner_bp.meshgrid.uv.vals[1, 1, 0] == 266
    assert s.ext_corner_bp.meshgrid.uv.vals[1, 1, 1] == 276
    assert s.center_bp is s.center_bp
    assert s.center_bp.meshgrid.shape == (1, 1)
    assert s.center_bp.meshgrid.uv.vals[0, 0, 0] == 128
    assert s.center_bp.meshgrid.uv.vals[0, 0, 1] == 128


def test_obs_snapshot_ra_dec_limits_wrap():
    s = obs_snapshot.ObsSnapshot(OBS_COISS2)
    ra_min, ra_max, dec_min, dec_max = s.ra_dec_limits()
    assert ra_min * oops.DPR == pytest.approx(359.96968466526647)
    assert ra_max * oops.DPR == pytest.approx(0.4436640310675228)
    assert dec_min * oops.DPR == pytest.approx(33.4586221077806)
    assert dec_max * oops.DPR == pytest.approx(33.85318503454683)


def test_obs_snapshot_ra_dec_limits_ext_wrap():
    s = obs_snapshot.ObsSnapshot(OBS_COISS2, extfov_margin=10)
    ra_min, ra_max, dec_min, dec_max = s.ra_dec_limits_ext()
    assert ra_min * oops.DPR == pytest.approx(359.95125729565814)
    assert ra_max * oops.DPR == pytest.approx(0.46221430761720933)
    assert dec_min * oops.DPR == pytest.approx(33.44320702355352)
    assert dec_max * oops.DPR == pytest.approx(33.86855152602122)


def test_obs_snapshot_ra_dec_limits():
    s = obs_snapshot.ObsSnapshot(OBS_VGISS2)
    ra_min, ra_max, dec_min, dec_max = s.ra_dec_limits()
    assert ra_min * oops.DPR == pytest.approx(115.15217011450416)
    assert ra_max * oops.DPR == pytest.approx(115.75520315049783)
    assert dec_min * oops.DPR == pytest.approx(18.33519459450846)
    assert dec_max * oops.DPR == pytest.approx(18.906653573574005)


def test_obs_snapshot_ra_dec_limits_ext():
    s = obs_snapshot.ObsSnapshot(OBS_VGISS2, extfov_margin=10)
    ra_min, ra_max, dec_min, dec_max = s.ra_dec_limits_ext()
    assert ra_min * oops.DPR == pytest.approx(115.14615714065772)
    assert ra_max * oops.DPR == pytest.approx(115.76125065791138)
    assert dec_min * oops.DPR == pytest.approx(18.329473234091115)
    assert dec_max * oops.DPR == pytest.approx(18.912360962666266)


def test_obs_snapshot_distance():
    s = obs_snapshot.ObsSnapshot(OBS_VGISS1)
    assert s.sun_body_distance('JUPITER') == pytest.approx(797366152.953817)
    assert s.body_distance('IO') == pytest.approx(1338438.5304067382)
