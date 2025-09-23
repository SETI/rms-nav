import pytest

import nav.inst.inst_cassini_iss as instcoiss
from nav.nav_model.nav_model_body import NavModelBody
from tests.config import URL_CASSINI_ISS_RHEA_01


@pytest.fixture
def obs_rhea(config_fixture):
    """Cassini ISS Rhea observation instance for testing."""
    return instcoiss.InstCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01,
                                              extfov_margin_vu=(200, 100))


def test_nav_model_body(obs_rhea) -> None:
    body = NavModelBody(obs_rhea, 'RHEA')
    body.create_model()
    # TODO Add tests here
