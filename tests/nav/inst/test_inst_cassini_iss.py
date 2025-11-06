import nav.obs.obs_inst_cassini_iss as obstcoiss
from tests.config import URL_CASSINI_ISS_RHEA_01


def test_cassini_iss_basic() -> None:
    obs = obstcoiss.ObsCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01)
    assert obs.midtime == 196177280.54761
