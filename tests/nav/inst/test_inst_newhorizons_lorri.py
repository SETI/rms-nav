import nav.inst.inst_newhorizons_lorri as instnhlorri
from tests.config import URL_NEWHORIZONS_LORRI_01


def test_newhorizons_lorri_basic():
    obs = instnhlorri.InstNewHorizonsLORRI.from_file(URL_NEWHORIZONS_LORRI_01)
    assert obs.midtime == 490113790.9641424
