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
from nav.nav_master import NavMaster
from nav.nav_model import NavModelBody
import nav.obs.obs_snapshot as obs_snapshot
from nav.support.file import dump_yaml
from tests.config import URL_CASSINI_ISS_RHEA_01

# URL = URL_CASSINI_ISS_RHEA_01; bodies = ['RHEA']
# inst_id = 'coiss'; URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2042/data/1584035653_1584189857/N1584039961_2_CALIB.IMG'; bodies = ['ENCELADUS']
# inst_id = 'coiss'; URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2067/data/1673418976_1673425869/W1673423216_1_CALIB.IMG'; bodies = ['DIONE', 'EPIMETHEUS', 'PROMETHEUS' ,'RHEA', 'TETHYS']
# inst_id = 'coiss'; URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2066/data/1669905491_1670314125/N1670313079_1_CALIB.IMG'; bodies = ['DIONE', 'TETHYS']
# inst_id = 'coiss'; URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2067/data/1678241258_1678386243/N1678279645_1_CALIB.IMG'; bodies = ['SATURN', 'EPIMETHEUS', 'TETHYS']

# inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0022/I24/IO/C0520821352R.IMG'; bodies = ['IO', 'JUPITER']
# inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0023/I31/IO/C0615816324R.IMG'; bodies = ['IO']
# inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0003/MOON/C0061059500R.IMG'; bodies = ['MOON']
# inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0005/EARTH/C0061498700R.IMG'; bodies = ['EARTH']
inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0019/C10/EUROPA/C0416073113R.IMG'; bodies = ['EUROPA']

# inst_id = 'vgiss'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/VGISS_5xxx/VGISS_5112/DATA/C16101XX/C1610143_GEOMED.IMG'; bodies = ['EUROPA', 'IO', 'JUPITER']
# inst_id = 'vgiss'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/VGISS_5xxx/VGISS_5112/DATA/C16101XX/C1610143_CALIB.IMG'; bodies = ['EUROPA', 'IO', 'JUPITER']

# inst_id = 'nhlorri'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/NHxxLO_xxxx/NHPELO_2001/data/20150625_029751/lor_0297516223_0x633_sci.fit'; bodies = ['CHARON', 'PLUTO']
# inst_id = 'nhlorri'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/NHxxLO_xxxx/NHPELO_1001/data/20150625_029751/lor_0297516223_0x633_eng.fit'; bodies = ['CHARON', 'PLUTO', 'STYX']

inst_class = inst_id_to_class(inst_id)
OBS = inst_class.from_file(URL)

extfov_margin = (200, 200)
# extfov_margin = (0, 0)
s = obs_snapshot.ObsSnapshot(OBS, extfov_margin_vu=extfov_margin)

nm = NavMaster(s)
nm.compute_all_models()

nm.navigate()

nm.create_overlay()
