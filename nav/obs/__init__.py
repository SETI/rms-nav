from .obs import Obs  # noqa: F401
from .obs_snapshot import ObsSnapshot  # noqa: F401
from .obs_snapshot_inst import ObsSnapshotInst  # noqa: F401

from .obs_inst_cassini_iss import ObsCassiniISS  # noqa: F401
from .obs_inst_galileo_ssi import ObsGalileoSSI  # noqa: F401
from .obs_inst_newhorizons_lorri import ObsNewHorizonsLORRI  # noqa: F401
from .obs_inst_voyager_iss import ObsVoyagerISS  # noqa: F401


_INST_NAME_TO_OBS_CLASS_MAPPING = {
    'coiss': ObsCassiniISS,
    'gossi': ObsGalileoSSI,
    'nhlorri': ObsNewHorizonsLORRI,
    'vgiss': ObsVoyagerISS,
}


def inst_names() -> list[str]:
    """Return a list of all instrument names."""
    return sorted(_INST_NAME_TO_OBS_CLASS_MAPPING.keys())


def inst_name_to_obs_class(name: str) -> type[ObsSnapshotInst]:
    """Convert an instrument name to the corresponding class.

    Parameters:
        name: The name of the instrument.

    Returns:
        The class corresponding to the instrument name.
    """
    return _INST_NAME_TO_OBS_CLASS_MAPPING[name.lower()]


__all__ = ['Obs',
           'ObsSnapshot',
           'ObsSnapshotInst',
           'ObsCassiniISS',
           'ObsGalileoSSI',
           'ObsNewHorizonsLORRI',
           'ObsVoyagerISS',
           'inst_names',
           'inst_name_to_obs_class']
