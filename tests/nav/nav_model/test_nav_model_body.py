import pytest

import nav.obs.obs_inst_cassini_iss as obstcoiss
from nav.nav_model.nav_model_body import NavModelBody
from tests.config import URL_CASSINI_ISS_RHEA_01


@pytest.fixture
def obs_rhea(config_fixture):
    """Cassini ISS Rhea observation instance for testing."""
    return obstcoiss.ObsCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01,
                                             extfov_margin_vu=(200, 100))


def test_nav_model_body(obs_rhea) -> None:
    body = NavModelBody(obs_rhea, 'RHEA')
    body.create_model()
    # TODO Add tests here
