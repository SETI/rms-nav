import nav.inst.inst_voyager_iss as instvgiss
from tests.config import URL_VOYAGER_ISS_IO_01


def test_voyager_iss_basic():
    obs = instvgiss.InstVoyagerISS.from_file(URL_VOYAGER_ISS_IO_01)
    assert obs.midtime == -646429822.8760977
