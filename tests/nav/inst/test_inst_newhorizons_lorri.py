import nav.obs.obs_inst_newhorizons_lorri as obstnhlorri
from tests.config import URL_NEWHORIZONS_LORRI_CHARON_01


def test_newhorizons_lorri_basic() -> None:
    obs = obstnhlorri.ObsNewHorizonsLORRI.from_file(URL_NEWHORIZONS_LORRI_CHARON_01)
    assert obs.midtime == 490113790.9641424
