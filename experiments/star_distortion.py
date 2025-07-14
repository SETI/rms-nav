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

import matplotlib.pyplot as plt
import numpy as np

from psfmodel import GaussianPSF

# Add the repository root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nav.config import DEFAULT_CONFIG
from nav.inst import INST_NAME_TO_CLASS_MAPPING
from nav.nav_master import NavMaster
from nav.process import process_one_image

DEFAULT_CONFIG.read_config()

inst_id = 'coiss'
# URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2021/data/1521798868_1521893025/N1521879918_2_CALIB.IMG'
# nstars = 400 # 500
# URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2021/data/1521798868_1521893025/W1521879918_2_CALIB.IMG'

# inst_id = 'gossi'
# nstars = 30
# URL = 'https://pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C0059869345R.IMG'
# URL = 'https://pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C0059874200R.IMG'
# URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0004/RAW_CAL/C0061337645R.IMG'
# URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C0059867900R.IMG'
# URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C0059730600R.IMG'  # Alp CMa

# inst_id = 'vgiss'
# nstars = 30
#URL = 'https://pds-rings.seti.org/holdings/volumes/VGISS_8xxx/VGISS_8210/DATA/C12051XX/C1205111_GEOMED.IMG'
# URL = 'https://pds-rings.seti.org/holdings/volumes/VGISS_5xxx/VGISS_5209/DATA/C20326XX/C2032609_GEOMED.IMG'
# URL = 'https://opus.pds-rings.seti.org/holdings/volumes/VGISS_8xxx/VGISS_8210/DATA/C12050XX/C1205037_GEOMED.IMG'  # Pleides


DEFAULT_CONFIG.stars.max_stars = nstars

inst_class = INST_NAME_TO_CLASS_MAPPING[inst_id.upper()]
OBS = inst_class.from_file(URL, fast_distortion=None)

nm = NavMaster(OBS)
nm.compute_all_models()

nm.navigate()

offset = nm.final_offset
if offset is None:
    print('No offset found')
    sys.exit(1)

img = nm.obs.data.astype(np.float64)

star_model = nm.star_models[0]
star_list = star_model.metadata['star_list']

psf = GaussianPSF(sigma=1.0)
# psf._debug_opt = 10

plt.figure()

u_list = []
v_list = []

for star in star_list:
    u = star.u - offset[1]
    v = star.v - offset[0]
    u_int = int(u)
    v_int = int(v)
    u_frac = u - u_int
    v_frac = v - v_int

    # sub_img = img[v_int-4:v_int+5, u_int-4:u_int+5]
    # plt.imshow(sub_img)
    # plt.show()

    ret = psf.find_position(img, (13, 13), (v, u))
    if ret is None:
        print(f'Star {star.unique_number} VMAG {star.vmag} failed')
        continue
    opt_y, opt_x, metadata = ret
    diff_x = float(opt_x-u)
    diff_y = float(opt_y-v)
    u_list.append(diff_x)
    v_list.append(diff_y)
    print(f'Star {star.unique_number} VMAG {star.vmag} offset {diff_x}, {diff_y}')

print()
mean_u = np.mean(u_list)
mean_v = np.mean(v_list)
print(f'Mean delta offset {mean_u}, {mean_v}')

for star in star_list:
    u = star.u - offset[1] + mean_u
    v = star.v - offset[0] + mean_v
    u_int = int(u)
    v_int = int(v)
    u_frac = u - u_int
    v_frac = v - v_int

    # sub_img = img[v_int-4:v_int+5, u_int-4:u_int+5]
    # plt.imshow(sub_img)
    # plt.show()

    ret = psf.find_position(img, (9, 9), (v, u))
    if ret is None:
        print(f'Star {star.unique_number} VMAG {star.vmag} failed')
        continue
    opt_y, opt_x, metadata = ret
    u_list.append(opt_x)
    v_list.append(opt_y)
    diff_x = float(opt_x-u)
    diff_y = float(opt_y-v)
    print(f'Star {star.unique_number} VMAG {star.vmag} offset {diff_x}, {diff_y}')
    plt.quiver(u, v, diff_x*1000, diff_y*1000, angles='xy', scale_units='xy', color='#80ff80')


# plt.show()

nm.create_overlay()
