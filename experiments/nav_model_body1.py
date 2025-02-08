from pathlib import Path
import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add the repository root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nav.annotation import Annotations
import nav.inst.inst_cassini_iss as instcoiss
from nav.nav_model import NavModelBody
import nav.obs.obs_snapshot as obs_snapshot
from nav.util.file import dump_yaml
from tests.config import URL_CASSINI_ISS_RHEA_01

# OBS = instcoiss.InstCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01); body = 'RHEA'
OBS = instcoiss.InstCassiniISS.from_file('https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2042/data/1584035653_1584189857/N1584039961_2_CALIB.IMG'); body = 'ENCELADUS'
annotations = Annotations()

extfov_margin = (100, 200)
s = obs_snapshot.ObsSnapshot(OBS, extfov_margin_vu=extfov_margin)
body_model = NavModelBody(s, body)
body_model.create_model()
annotations.add_annotation(body_model.annotation)

dump_yaml(body_model.metadata)
# pprint.pprint(metadata)
# plt.imshow(model)
# plt.show()

offset = (0, 20)

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
plt.figure()
plt.imshow(model_mask)
plt.show()
