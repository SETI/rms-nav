from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import oops

# Add the repository root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nav.annotation import Annotations
import nav.inst.inst_cassini_iss as instcoiss
from nav.nav_model import NavModelBody
import nav.obs.obs_snapshot as obs_snapshot
from nav.util.file import dump_yaml
from tests.config import URL_CASSINI_ISS_RHEA_01

oops.config.PATH_PHOTONS.dlt_precision = 1
oops.config.PATH_PHOTONS.max_iterations = 1
oops.config.SURFACE_PHOTONS.dlt_precision = 1
oops.config.SURFACE_PHOTONS.max_iterations = 1
oops.config.LOGGING.fov_iterations = True          # Log iterations of FOV solvers
oops.config.LOGGING.path_iterations = True         # Log iterations of Path photon solvers
oops.config.LOGGING.surface_iterations = True      # Log iterations of Surface photon solvers
oops.config.LOGGING.observation_iterations = True  # Log iterations of Observation solvers
oops.config.LOGGING.event_time_collapse = True     # Report event time collapse
oops.config.LOGGING.surface_time_collapse = True   # Report surface time collapse

# OBS = instcoiss.InstCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01); bodies = ['RHEA']
# OBS = instcoiss.InstCassiniISS.from_file('https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2042/data/1584035653_1584189857/N1584039961_2_CALIB.IMG'); bodies = ['ENCELADUS']
# OBS = instcoiss.InstCassiniISS.from_file('https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2067/data/1673418976_1673425869/W1673423216_1_CALIB.IMG'); bodies = ['DIONE', 'EPIMETHEUS', 'PROMETHEUS' ,'RHEA', 'TETHYS']
# OBS = instcoiss.InstCassiniISS.from_file('https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2066/data/1669905491_1670314125/N1670313079_1_CALIB.IMG'); bodies = ['DIONE', 'TETHYS']
OBS = instcoiss.InstCassiniISS.from_file('https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2067/data/1678241258_1678386243/N1678279645_1_CALIB.IMG'); bodies = ['SATURN', 'EPIMETHEUS', 'TETHYS']
annotations = Annotations()

extfov_margin = (100, 200)
s = obs_snapshot.ObsSnapshot(OBS, extfov_margin_vu=extfov_margin)

for body in bodies:
    body_model = NavModelBody(s, body)
    body_model.create_model()
    annotations.add_annotation(body_model.annotation)

    dump_yaml(body_model.metadata)

offset = (0, 0)

overlay = annotations.combine(extfov_margin, offset=offset)
img = OBS.data

res = np.zeros(img.shape + (3,), dtype=np.uint8)

black_point = np.min(img)
white_point = np.max(img)

img_stretched = np.clip((img - black_point) / (white_point - black_point) * 255,
                        0, 255).astype(np.uint8)

res[:, :, 1] = img_stretched

res[overlay != 0] = overlay[overlay != 0]

model_mask = body_model.model_mask

plt.imshow(res)
# plt.figure()
# plt.imshow(model_mask)
plt.show()
