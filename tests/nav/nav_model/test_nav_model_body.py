import nav.inst.inst_cassini_iss as instcoiss
from nav.nav_model.nav_model_body import NavModelBody
import nav.obs.obs_snapshot as obs_snapshot
from tests.config import URL_CASSINI_ISS_01

OBS = instcoiss.InstCassiniISS.from_file(URL_CASSINI_ISS_01)


def test_nav_model_body():
    s = obs_snapshot.ObsSnapshot(OBS, extfov_margin=(200,100))
    body = NavModelBody(s, 'RHEA')
    model, metadata, annotation = body.create_model()
    print(metadata)
