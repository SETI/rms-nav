from pathlib import Path
import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add the repository root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nav.annotation import Annotations
from nav.inst import inst_name_to_class
from nav.nav_master import NavMaster
from tests.config import (URL_CASSINI_ISS_STARS_01,
                          URL_CASSINI_ISS_STARS_02,
                          URL_GALILEO_SSI_STARS_01,
                          URL_GALILEO_SSI_STARS_02,
                          URL_VOYAGER_ISS_STARS_01,
                          URL_VOYAGER_ISS_STARS_02)

def main():
    offset = (0, 0)
    extfov_margin_vu = (300, 300)

    # inst_id = 'coiss'; URL = URL_CASSINI_ISS_STARS_01
    inst_id = 'coiss'; URL = URL_CASSINI_ISS_STARS_02

    # inst_id = 'gossi'; URL = URL_GALILEO_SSI_STARS_01; offset = (85, 391); extfov_margin = (200, 500)
    # inst_id = 'gossi'; URL = URL_GALILEO_SSI_STARS_02; offset = (0, 0) #; extfov_margin = (200, 500)

    # inst_id = 'vgiss'; URL = URL_VOYAGER_ISS_STARS_01; offset = (0, 0) ; extfov_margin = (1000, 1000)
    # inst_id = 'vgiss'; URL = URL_VOYAGER_ISS_STARS_02; offset = (1, 12) #; extfov_margin = (1000, 1000)

    inst_class = inst_name_to_class(inst_id)
    OBS = inst_class.from_file(URL, extfov_margin_vu=extfov_margin_vu)

    nm = NavMaster(OBS)
    nm.compute_all_models()

    nm.navigate()

    nm.create_overlay()


if __name__ == '__main__':
    main()
