from pathlib import Path
import sys


# Add the repository root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nav.obs import inst_name_to_obs_class
from nav.nav_master import NavMaster
from tests.config import (
    URL_CASSINI_ISS_STARS_01,
)


def main():
    inst_id = 'coiss'
    URL = URL_CASSINI_ISS_STARS_01
    # inst_id = 'coiss'; URL = URL_CASSINI_ISS_STARS_02

    # inst_id = 'gossi'; URL = URL_GALILEO_SSI_STARS_01
    # inst_id = 'gossi'; URL = URL_GALILEO_SSI_STARS_02

    # inst_id = 'vgiss'; URL = URL_VOYAGER_ISS_STARS_01
    # inst_id = 'vgiss'; URL = URL_VOYAGER_ISS_STARS_02

    inst_class = inst_name_to_obs_class(inst_id)
    OBS = inst_class.from_file(URL)

    nm = NavMaster(OBS)
    nm.compute_all_models()

    nm.navigate()

    nm.create_overlay()


if __name__ == '__main__':
    main()
