from tests.config import URL_GALILEO_SSI_IO_01

import nav.obs.obs_inst_galileo_ssi as obstgossi


def test_galileo_ssi_basic() -> None:
    obs = obstgossi.ObsGalileoSSI.from_file(URL_GALILEO_SSI_IO_01)
    assert obs.midtime == -110923771.01052806
