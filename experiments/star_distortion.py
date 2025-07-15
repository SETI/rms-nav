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

from pdslogger import PdsLogger

from nav.config import DEFAULT_CONFIG, DEFAULT_LOGGER
from nav.inst import INST_NAME_TO_CLASS_MAPPING
from nav.nav_master import NavMaster
from nav.process import process_one_image

DEFAULT_CONFIG.read_config()

# Set logger to debug
DEFAULT_LOGGER.set_level('DEBUG')

inst_id = 'coiss'

# nstars = 400
# psf_size = (9, 9)
# perc = 75
# URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2021/data/1521798868_1521893025/N1521879918_2_CALIB.IMG'

# nstars = 400
# psf_size = (13, 13)
# perc = 90
# URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2021/data/1521798868_1521893025/W1521879918_2_CALIB.IMG'

# inst_id = 'gossi'
# nstars = 30
# psf_size = (11, 11)
# perc = 90
# URL = 'https://pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C0059869345R.IMG'
# URL = 'https://pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C0059874200R.IMG'
# URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0004/RAW_CAL/C0061337645R.IMG'
# URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C0059867900R.IMG'
# URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C0059730600R.IMG'  # Alp CMa

# inst_id = 'vgiss'
# nstars = 15
# psf_size = (15, 15)
# perc = 100
# URL = 'https://pds-rings.seti.org/holdings/volumes/VGISS_8xxx/VGISS_8210/DATA/C12051XX/C1205111_GEOMED.IMG'
# URL = 'https://pds-rings.seti.org/holdings/volumes/VGISS_5xxx/VGISS_5209/DATA/C20326XX/C2032609_GEOMED.IMG'
# URL = 'https://opus.pds-rings.seti.org/holdings/volumes/VGISS_8xxx/VGISS_8210/DATA/C12050XX/C1205037_GEOMED.IMG'  # Pleides

inst_id = 'nhlorri'
nstars = 100
psf_size = (15, 15)
perc = 100
# URL = 'https://pds-rings.seti.org/holdings/volumes/NHxxLO_xxxx/NHPCLO_2001/data/20081015_008636/lor_0086360869_0x630_sci.fit'
# URL = 'https://opus.pds-rings.seti.org/holdings/volumes/NHxxLO_xxxx/NHPCLO_2001/data/20081015_008636/lor_0086365265_0x630_sci.fit'
# URL = 'https://opus.pds-rings.seti.org/holdings/volumes/NHxxLO_xxxx/NHPCLO_2001/data/20130702_023509/lor_0235099619_0x630_sci.fit'
# URL = 'https://opus.pds-rings.seti.org/holdings/volumes/NHxxLO_xxxx/NHJULO_2001/data/20070227_003485/lor_0034858514_0x630_sci.fit'  # Callisto
# URL = 'https://opus.pds-rings.seti.org/holdings/volumes/NHxxLO_xxxx/NHJULO_2001/data/20070122_003173/lor_0031739639_0x630_sci.fit'  # Jupiter
URL = 'https://opus.pds-rings.seti.org/holdings/volumes/NHxxLO_xxxx/NHPELO_2001/data/20150701_029803/lor_0298030149_0x630_sci.fit'

DEFAULT_CONFIG.stars.max_stars = nstars

inst_class = INST_NAME_TO_CLASS_MAPPING[inst_id.upper()]
OBS = inst_class.from_file(URL, fast_distortion=None)

# plt.imshow(OBS.data)
# plt.show()

nm = NavMaster(OBS)
nm.compute_all_models()

nm.navigate()

# body_model = nm.body_models[0]
# plt.imshow(body_model.model_img)
# plt.show()

# nm._final_offset = (-12, 5); print('** OVERRIDE FINAL OFFSET **')
offset = nm.final_offset
if offset is None:
    print('No offset found')
    sys.exit(1)
print(f'Offset: {offset}')

img = nm.obs.data.astype(np.float64)

star_model = nm.star_models[0]
star_list = star_model.metadata['star_list']

psf = GaussianPSF(sigma=2.0)
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

    ret = psf.find_position(img, psf_size, (v, u))
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

u_diff_list = []
v_diff_list = []
u_list = []
v_list = []
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

    ret = psf.find_position(img, (11, 11), (v, u))
    if ret is None:
        print(f'Star {star.unique_number} VMAG {star.vmag} failed')
        continue
    opt_y, opt_x, metadata = ret
    diff_x = float(opt_x-u)
    diff_y = float(opt_y-v)
    u_list.append(u)
    v_list.append(v)
    u_diff_list.append(diff_x)
    v_diff_list.append(diff_y)
    print(f'Star {star.unique_number} VMAG {star.vmag} offset {diff_x}, {diff_y}')

u_arr = np.array(u_list)
v_arr = np.array(v_list)
u_diff_arr = np.array(u_diff_list)
v_diff_arr = np.array(v_diff_list)
mag_arr = np.sqrt(u_diff_arr**2 + v_diff_arr**2)
mag_perc = np.percentile(mag_arr, perc)

for u, v, u_diff, v_diff in zip(u_arr, v_arr, u_diff_arr, v_diff_arr):
    mag = np.sqrt(u_diff**2 + v_diff**2)
    if mag <= mag_perc:
        plt.quiver(u, v, u_diff*100, v_diff*100, angles='xy', scale_units='xy', scale=1,
                   color='#80ff80', alpha=0.6)


# plt.show()

overlay = nm.create_overlay()
plt.imshow(overlay)
plt.show()
