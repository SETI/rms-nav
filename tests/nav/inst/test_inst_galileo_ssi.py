import nav.inst.inst_galileo_ssi as instgossi
from tests.config import URL_GALILEO_SSI_01


def test_galileo_ssi_basic():
    inst = instgossi.InstGalileoSSI.from_file(URL_GALILEO_SSI_01)
    assert inst.obs.midtime == -110923771.01052806
