import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# Add the repository root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nav.inst import inst_name_to_class

from nav.config import DEFAULT_CONFIG
from nav.nav_master import NavMaster


def main():
    DEFAULT_CONFIG.read_config()

    # URL = URL_CASSINI_ISS_RHEA_01; bodies = ['RHEA']
    # inst_id = 'coiss'; URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2042/data/1584035653_1584189857/N1584039961_2_CALIB.IMG'; bodies = ['ENCELADUS']
    # inst_id = 'coiss'; URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2067/data/1673418976_1673425869/W1673423216_1_CALIB.IMG'; bodies = ['DIONE', 'EPIMETHEUS', 'PROMETHEUS' ,'RHEA', 'TETHYS']
    # inst_id = 'coiss'; URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2066/data/1669905491_1670314125/N1670313079_1_CALIB.IMG'; bodies = ['DIONE', 'TETHYS']
    # inst_id = 'coiss'; URL = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2067/data/1678241258_1678386243/N1678279645_1_CALIB.IMG'; bodies = ['SATURN', 'EPIMETHEUS', 'TETHYS']
    # inst_id = 'coiss'; URL = 'https://opus.pds-rings.seti.org/holdings/calibrated/COISS_1xxx/COISS_1001/data/1313633670_1327290527/N1313633773_1_CALIB.IMG'  # Moon, fails reading in oops
    # inst_id = 'coiss'; URL = 'https://opus.pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2030/data/1553898219_1554036373/N1554025173_2_CALIB.IMG'  # Pluto taken from Saturn
    # inst_id = 'coiss'; URL = 'https://opus.pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2010/data/1489087056_1489133407/N1489100503_1_CALIB.IMG'

    # inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0022/I24/IO/C0520821352R.IMG'; bodies = ['IO', 'JUPITER']
    # inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0023/I31/IO/C0615816324R.IMG'; bodies = ['IO']
    # inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0003/MOON/C0061059500R.IMG'; bodies = ['MOON']
    # inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0005/EARTH/C0061498700R.IMG'; bodies = ['EARTH']
    # inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0019/C10/EUROPA/C0416073113R.IMG'; bodies = ['EUROPA']
    # inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0017/C3/IO/C0368641300R.IMG'  # Io
    # inst_id = 'gossi'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0021/C21/IO/C0506465900R.IMG'  # Io

    # inst_id = 'vgiss'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/VGISS_5xxx/VGISS_5112/DATA/C16101XX/C1610143_GEOMED.IMG'; bodies = ['EUROPA', 'IO', 'JUPITER']
    # inst_id = 'vgiss'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/VGISS_8xxx/VGISS_8209/DATA/C11661XX/C1166148_GEOMED.IMG'  # Neptune crescent with Triton hidden on top
    # inst_id = 'vgiss'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/VGISS_8xxx/VGISS_8208/DATA/C11488XX/C1148852_GEOMED.IMG'  # Triton next to Neptune

    # inst_id = 'nhlorri'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/NHxxLO_xxxx/NHPELO_2001/data/20150625_029751/lor_0297516223_0x633_sci.fit'; bodies = ['CHARON', 'PLUTO']
    # inst_id = 'nhlorri'; URL = 'https://opus.pds-rings.seti.org/holdings/volumes/NHxxLO_xxxx/NHPELO_1001/data/20150625_029751/lor_0297516223_0x633_eng.fit'; bodies = ['CHARON', 'PLUTO', 'STYX']
    inst_id = 'nhlorri'
    URL = 'https://opus.pds-rings.seti.org/holdings/volumes/NHxxLO_xxxx/NHPELO_2001/data/20150707_029861/lor_0298615084_0x630_sci.fit'

    inst_class = inst_name_to_class(inst_id)
    s = inst_class.from_file(URL)

    nm = NavMaster(s)

    nm.navigate()

    res = nm.create_overlay()

    filename = URL.split('/')[-1].split('.')[0].replace('_CALIB', '')
    im = Image.fromarray(res)
    im.save(f'{filename}.png')

    plt.imshow(res)
    plt.show()


if __name__ == '__main__':
    main()
