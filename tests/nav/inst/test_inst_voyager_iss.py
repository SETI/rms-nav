from tests.config import URL_VOYAGER_ISS_IO_01

import nav.obs.obs_inst_voyager_iss as obstvgiss


def test_voyager_iss_basic() -> None:
    obs = obstvgiss.ObsVoyagerISS.from_file(URL_VOYAGER_ISS_IO_01)
    assert obs.midtime == -646429822.8760977
