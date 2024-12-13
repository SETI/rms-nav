import oops

import nav.inst.inst_cassini_iss as instcoiss
import nav.obs.obs_snapshot as obs_snapshot

# This image was chosen because it is 256x256 (and thus faster to create a Backplane)
# and the RA wraps around.
URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2091/data/1787135320_1787144464/N1787139424_1_CALIB.IMG'
INST = instcoiss.InstCassiniISS.from_file(URL)
OBS = INST.obs


def test_obs_snapshot_init():
    s = obs_snapshot.ObsSnapshot(OBS)
    assert s._data_shape_uv == (256, 256)
    assert s._fov_uv_min == (0, 0)
    assert s._fov_uv_max == (255, 255)
    assert s._extfov_margin == (0, 0)
    assert s._extdata.shape == (256, 256)
    assert s._extdata_shape_uv == (256, 256)
    assert s._extfov_uv_min == (0, 0)
    assert s._extfov_uv_max == (255, 255)

    s = obs_snapshot.ObsSnapshot(OBS, extfov_margin=10)
    assert s._data_shape_uv == (256, 256)
    assert s._fov_uv_min == (0, 0)
    assert s._fov_uv_max == (255, 255)
    assert s._extfov_margin == (10, 10)
    assert s._extdata.shape == (276, 276)
    assert s._extdata_shape_uv == (276, 276)
    assert s._extfov_uv_min == (-10, -10)
    assert s._extfov_uv_max == (265, 265)

    s = obs_snapshot.ObsSnapshot(OBS, extfov_margin=(10, 20))
    assert s._data_shape_uv == (256, 256)
    assert s._fov_uv_min == (0, 0)
    assert s._fov_uv_max == (255, 255)
    assert s._extfov_margin == (10, 20)
    assert s._extdata.shape == (296, 276)
    assert s._extdata_shape_uv == (276, 296)
    assert s._extfov_uv_min == (-10, -20)
    assert s._extfov_uv_max == (265, 275)


def test_obs_snapshot_bp_mg():
    s = obs_snapshot.ObsSnapshot(OBS)
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

    s = obs_snapshot.ObsSnapshot(OBS, extfov_margin=(10, 20))
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


def test_obs_snapshot_ra_dec_limits():
    s = obs_snapshot.ObsSnapshot(OBS)
    ra_min, ra_max, dec_min, dec_max = s.ra_dec_limits()
    assert ra_min * oops.DPR == 359.96968466526647
    assert ra_max * oops.DPR == 0.4436640310675228
    assert dec_min * oops.DPR == 33.4586221077806
    assert dec_max * oops.DPR == 33.85318503454683


def test_obs_snapshot_ra_dec_limits_ext():
    s = obs_snapshot.ObsSnapshot(OBS, extfov_margin=10)
    ra_min, ra_max, dec_min, dec_max = s.ra_dec_limits_ext()
    assert ra_min * oops.DPR == 359.95125729565814
    assert ra_max * oops.DPR == 0.46221430761720933
    assert dec_min * oops.DPR == 33.44320702355352
    assert dec_max * oops.DPR == 33.86855152602122


def test_obs_snapshot_sun_distance():
    s = obs_snapshot.ObsSnapshot(OBS)
    assert s.sun_distance('SATURN') == 9.929003393012831
