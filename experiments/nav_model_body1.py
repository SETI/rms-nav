from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import oops
from PIL import Image

# Add the repository root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nav.annotation import Annotations
from nav.inst import inst_id_to_class
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

offset = (0, 0)

# URL = URL_CASSINI_ISS_RHEA_01; bodies = ['RHEA']
# inst_id = 'coiss'; URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2042/data/1584035653_1584189857/N1584039961_2_CALIB.IMG'; bodies = ['ENCELADUS']
inst_id = 'coiss'; URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2067/data/1673418976_1673425869/W1673423216_1_CALIB.IMG'; bodies = ['DIONE', 'EPIMETHEUS', 'PROMETHEUS' ,'RHEA', 'TETHYS']
# inst_id = 'coiss'; URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2066/data/1669905491_1670314125/N1670313079_1_CALIB.IMG'; bodies = ['DIONE', 'TETHYS']
# inst_id = 'coiss'; URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2067/data/1678241258_1678386243/N1678279645_1_CALIB.IMG'; bodies = ['SATURN', 'EPIMETHEUS', 'TETHYS']

# inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0022/I24/IO/C0520821352R.IMG'; bodies = ['IO', 'JUPITER']
# inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0023/I31/IO/C0615816324R.IMG'; bodies = ['IO']
# inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0003/MOON/C0061059500R.IMG'; bodies = ['MOON']
# inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0005/EARTH/C0061498700R.IMG'; bodies = ['EARTH']
# inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0019/C10/EUROPA/C0416073113R.IMG'; bodies = ['EUROPA']

# inst_id = 'vgiss'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/VGISS_5xxx/VGISS_5112/DATA/C16101XX/C1610143_GEOMED.IMG'; bodies = ['EUROPA', 'IO', 'JUPITER']; offset = (-46, 131)
# inst_id = 'vgiss'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/VGISS_5xxx/VGISS_5112/DATA/C16101XX/C1610143_CALIB.IMG'; bodies = ['EUROPA', 'IO', 'JUPITER']

# inst_id = 'nhlorri'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/NHxxLO_xxxx/NHPELO_2001/data/20150625_029751/lor_0297516223_0x633_sci.fit'; bodies = ['CHARON', 'PLUTO']
# inst_id = 'nhlorri'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/NHxxLO_xxxx/NHPELO_1001/data/20150625_029751/lor_0297516223_0x633_eng.fit'; bodies = ['CHARON', 'PLUTO', 'STYX']

inst_class = inst_id_to_class(inst_id)
OBS = inst_class.from_file(URL)
annotations = Annotations()

extfov_margin = (200, 200)
# extfov_margin = (0, 0)
s = obs_snapshot.ObsSnapshot(OBS, extfov_margin_vu=extfov_margin)

if True:
    for body in bodies:
        body_model = NavModelBody(s, body)
        body_model.create_model()
        annotations.add_annotations(body_model.annotations)

        dump_yaml(body_model.metadata)

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
                          (whitepoint-blackpoint))**gamma*256)
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
