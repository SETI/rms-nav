import nav.inst.inst_cassini_iss as instcoiss


def test_cassini_iss_basic():
    url = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx_v2/COISS_2021/data/1521584844_1521609901/W1521598221_1_CALIB.IMG'
    inst = instcoiss.InstCassiniISS.from_file(url)
    assert inst.obs.midtime == 196177280.54761
