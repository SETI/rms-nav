import nav.inst.inst_voyager_iss as instvgiss
from tests.config import URL_VOYAGER_ISS_01


def test_voyager_iss_basic():
    inst = instvgiss.InstVoyagerISS.from_file(URL_VOYAGER_ISS_01)
    assert inst.obs.midtime == -646429822.8760977
