from .obs import Obs  # noqa: F401
from .obs_snapshot import ObsSnapshot  # noqa: F401

from .obs_cassini_iss import ObsCassiniISS  # noqa: F401
from .obs_galileo_ssi import ObsGalileoSSI  # noqa: F401
from .obs_newhorizons_lorri import ObsNewHorizonsLORRI  # noqa: F401
from .obs_voyager_iss import ObsVoyagerISS  # noqa: F401

INST_NAME_TO_OBS_CLASS_MAPPING = {
    'coiss': ObsCassiniISS,
    'gossi': ObsGalileoSSI,
    'nhlorri': ObsNewHorizonsLORRI,
    'vgiss': ObsVoyagerISS,
}


def inst_name_to_obs_class(name: str) -> type[Obs]:
    """Convert an instrument name to the corresponding class.

    Parameters:
        name: The name of the instrument.

    Returns:
        The class corresponding to the instrument name.
    """

    return INST_NAME_TO_OBS_CLASS_MAPPING[name.lower()]


__all__ = ['Obs',
           'ObsSnapshot',
           'ObsCassiniISS',
           'ObsGalileoSSI',
           'ObsNewHorizonsLORRI',
           'ObsVoyagerISS',
           'inst_name_to_obs_class']
