import pytest

import nav.obs.obs_inst_cassini_iss as obstcoiss
from nav.nav_model.nav_model_body import NavModelBody
from tests.config import URL_CASSINI_ISS_RHEA_01


@pytest.fixture
def obs_rhea(config_fixture: None) -> obstcoiss.ObsCassiniISS:
    """Cassini ISS Rhea observation instance for testing."""
    return obstcoiss.ObsCassiniISS.from_file(
        URL_CASSINI_ISS_RHEA_01, extfov_margin_vu=(200, 100)
    )


def test_nav_model_body(obs_rhea: obstcoiss.ObsCassiniISS) -> None:
    """Body model creates one NavModelResult in models after create_model."""
    body = NavModelBody('body:rhea', obs_rhea, 'RHEA')
    body.create_model()
    assert len(body.models) == 1
    result = body.models[0]
    assert result.model_img is not None
    assert result.model_mask is not None
    assert result.range is not None
    assert result.confidence == 1.0
