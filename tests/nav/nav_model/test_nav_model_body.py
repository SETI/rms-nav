import nav.inst.inst_cassini_iss as instcoiss
from nav.nav_model.nav_model_body import NavModelBody
import nav.obs.obs_snapshot as obs_snapshot
from tests.config import URL_CASSINI_ISS_RHEA_01

OBS = instcoiss.InstCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01)


def test_nav_model_body():
    s = obs_snapshot.ObsSnapshot(OBS, extfov_margin_vu=(200, 100))
    body = NavModelBody(s, 'RHEA')
    body.create_model()
    print(body.metadata)
