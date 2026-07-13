from .obs import Obs
from .obs_inst_cassini_iss import ObsCassiniISS
from .obs_inst_galileo_ssi import ObsGalileoSSI
from .obs_inst_newhorizons_lorri import ObsNewHorizonsLORRI
from .obs_inst_sim import ObsSim
from .obs_inst_voyager_iss import ObsVoyagerISS
from .obs_snapshot import ObsSnapshot
from .obs_snapshot_inst import ObsSnapshotInst

_INST_NAME_TO_OBS_CLASS_MAPPING: dict[str, type[ObsSnapshotInst]] = {
    'coiss': ObsCassiniISS,
    'gossi': ObsGalileoSSI,
    'nhlorri': ObsNewHorizonsLORRI,
    'sim': ObsSim,
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

    Raises:
        KeyError: If ``name`` does not match a registered instrument; the
            message lists the valid names.
    """
    try:
        return _INST_NAME_TO_OBS_CLASS_MAPPING[name.lower()]
    except KeyError:
        valid = ', '.join(sorted(_INST_NAME_TO_OBS_CLASS_MAPPING))
        raise KeyError(f'unknown instrument name {name!r}; valid names: {valid}') from None


def obs_class_to_inst_name(obs_class: type) -> str:
    """Convert an observation class to its registered instrument name.

    The reverse of :func:`inst_name_to_obs_class`.  Metadata writers use
    this to record which instrument produced a result, so an unregistered
    class maps to ``'unknown'`` rather than raising -- instrument identity
    is bookkeeping and must never abort a navigation run.

    Parameters:
        obs_class: An ``ObsSnapshotInst`` subclass (or any class; only
            exact matches against the registry count).

    Returns:
        The registered instrument name (``'coiss'``, ``'gossi'``,
        ``'nhlorri'``, ``'sim'``, or ``'vgiss'``), or ``'unknown'`` when
        the class is not registered.
    """
    for name, cls in _INST_NAME_TO_OBS_CLASS_MAPPING.items():
        if cls is obs_class:
            return name
    return 'unknown'


__all__ = [
    'Obs',
    'ObsCassiniISS',
    'ObsGalileoSSI',
    'ObsNewHorizonsLORRI',
    'ObsSim',
    'ObsSnapshot',
    'ObsSnapshotInst',
    'ObsVoyagerISS',
    'inst_name_to_obs_class',
    'inst_names',
    'obs_class_to_inst_name',
]
