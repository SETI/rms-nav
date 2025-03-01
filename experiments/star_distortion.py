# N1521879918_2 CLR star field - used for distortion analysis
# W1521879918_2 CLR star field - used for distortion analysis
# N1521880278_2 CLR star field - used for distortion analysis
# W1521880278_2 CLR star field - used for distortion analysis
# N1521880638_2 CLR star field - used for distortion analysis
# W1521880638_2 CLR star field - used for distortion analysis
# N1521880998_2 CLR star field - used for distortion analysis
# W1521880998_2 CLR star field - used for distortion analysis
# N1521881358_2 CLR star field - used for distortion analysis
# W1521881358_2 CLR star field - used for distortion analysis


from pathlib import Path
import sys

# Add the repository root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nav.inst import inst_id_to_class
from nav.nav_master import NavMaster

extfov_margin_vu = (300, 300)

# inst_id = 'coiss'
# URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2021/data/1521798868_1521893025/N1521879918_2_CALIB.IMG'
# URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2021/data/1521798868_1521893025/W1521879918_2_CALIB.IMG'

# inst_id = 'gossi'
# URL = 'https://pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C0059869345R.IMG'
# URL = 'https://pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C0059874200R.IMG'

inst_id = 'vgiss'
URL = 'https://pds-rings.seti.org/holdings/volumes/VGISS_8xxx/VGISS_8210/DATA/C12051XX/C1205111_GEOMED.IMG'

inst_class = inst_id_to_class(inst_id)
OBS = inst_class.from_file(URL, extfov_margin_vu=extfov_margin_vu)

nm = NavMaster(OBS)
nm.compute_all_models()

nm.navigate()

nm.create_overlay()
