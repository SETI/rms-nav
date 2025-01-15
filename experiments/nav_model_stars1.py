from pathlib import Path
import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add the repository root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nav.annotation import Annotations
import nav.inst.inst_cassini_iss as instcoiss
from nav.nav_model import NavModelStars
import nav.obs.obs_snapshot as obs_snapshot
from tests.config import URL_CASSINI_ISS_02

INST = instcoiss.InstCassiniISS.from_file(URL_CASSINI_ISS_02)
OBS = INST.obs

annotations = Annotations()

extfov_margin = (200, 100)
s = obs_snapshot.ObsSnapshot(OBS, extfov_margin=extfov_margin)
stars = NavModelStars(s)
model, metadata, annotation = stars.create_model()
annotations.add_annotation(annotation)

pprint.pprint(metadata)
# plt.imshow(model)
# plt.show()

overlay = annotations.combine()

offset = (-1,-1)

img = OBS.data

res = np.zeros(img.shape + (3,), dtype=np.uint8)

offset_overlay = overlay[extfov_margin[1]+offset[1] : overlay.shape[0]-extfov_margin[1]+offset[1],
                         extfov_margin[0]+offset[0] : overlay.shape[1]-extfov_margin[0]+offset[0]]

black_point = np.min(img)
white_point = np.max(img)

img_stretched = np.clip((img - black_point) / (white_point - black_point) * 255,
                        0, 255).astype(np.uint8)

res[:, :, 1] = img_stretched

res[offset_overlay != 0] = offset_overlay[offset_overlay != 0]

plt.imshow(res)
plt.show()
