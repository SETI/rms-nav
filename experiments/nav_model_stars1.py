from pathlib import Path
import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add the repository root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nav.annotation import Annotations
import nav.inst.inst_cassini_iss as instcoiss
from nav.nav_model import NavModelStars
import nav.obs.obs_snapshot as obs_snapshot
from tests.config import URL_CASSINI_ISS_STARS_02

URL = URL_CASSINI_ISS_STARS_02
OBS = instcoiss.InstCassiniISS.from_file(URL)

annotations = Annotations()

offset = (-9, 28)
extfov_margin = (200, 100)
s = obs_snapshot.ObsSnapshot(OBS, extfov_margin_vu=extfov_margin)
stars = NavModelStars(s)
stars.create_model()
annotations = stars.annotations

overlay = annotations.combine(extfov_margin, offset=offset
                            #   text_use_avoid_mask=False,
                            #   text_show_all_positions=True,
                            #   text_avoid_other_text=False
                              )
img = OBS.data.astype(np.float64)

res = np.zeros(img.shape + (3,), dtype=np.uint8)

img_sorted = sorted(list(img.flatten()))
blackpoint = img_sorted[np.clip(int(len(img_sorted)*0.005),
                                0, len(img_sorted)-1)]
whitepoint = img_sorted[np.clip(int(len(img_sorted)*0.995),
                                0, len(img_sorted)-1)]
print(blackpoint, whitepoint)
gamma = 0.5

img_stretched = np.floor((np.maximum(img-blackpoint, 0) /
                          (whitepoint-blackpoint))**gamma * 256)
img_stretched = np.clip(img_stretched, 0, 255) # Clip black and white

img_stretched = img_stretched.astype(np.uint8)

res[:, :, 0] = img_stretched
res[:, :, 1] = img_stretched
res[:, :, 2] = img_stretched

if overlay is not None:
    overlay[overlay < 128] = 0
    mask = np.any(overlay, axis=2)
    res[mask, :] = overlay[mask, :]

im = Image.fromarray(res)
fn = URL.split('/')[-1].split('.')[0]
im.save(f'/home/rfrench/{fn}.png')

plt.imshow(res)
# plt.figure()
# model_mask = body_model.model_mask
# plt.imshow(model_mask)
plt.show()
