import nav.inst.inst_cassini_iss as instcoiss
from tests.config import URL_CASSINI_ISS_01


def test_cassini_iss_basic():
    inst = instcoiss.InstCassiniISS.from_file(URL_CASSINI_ISS_01)
    assert inst.obs.midtime == 196177280.54761
