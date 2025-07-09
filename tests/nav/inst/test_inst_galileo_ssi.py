import nav.inst.inst_galileo_ssi as instgossi
from tests.config import URL_GALILEO_SSI_IO_01


def test_galileo_ssi_basic() -> None:
    obs = instgossi.InstGalileoSSI.from_file(URL_GALILEO_SSI_IO_01)
    assert obs.midtime == -110923771.01052806
