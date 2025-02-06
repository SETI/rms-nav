import nav.inst.inst_cassini_iss as instcoiss
from tests.config import URL_CASSINI_ISS_RHEA_01


def test_cassini_iss_basic():
    obs = instcoiss.InstCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01)
    assert obs.midtime == 196177280.54761
