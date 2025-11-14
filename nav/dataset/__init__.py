from .dataset import DataSet  # noqa: F401
from .dataset_pds3_cassini_iss import DataSetPDS3CassiniISS  # noqa: F401
from .dataset_pds3_galileo_ssi import DataSetPDS3GalileoSSI  # noqa: F401
from .dataset_pds3_newhorizons_lorri import DataSetPDS3NewHorizonsLORRI  # noqa: F401
from .dataset_pds3_voyager_iss import DataSetPDS3VoyagerISS  # noqa: F401
from .dataset_sim import DataSetSim  # noqa: F401


_DATASET_NAME_TO_CLASS_MAPPING: dict[str, type[DataSet]] = {
    'coiss': DataSetPDS3CassiniISS,
    'coiss_pds3': DataSetPDS3CassiniISS,
    'gossi': DataSetPDS3GalileoSSI,
    'gossi_pds3': DataSetPDS3GalileoSSI,
    'nhlorri': DataSetPDS3NewHorizonsLORRI,
    'nhlorri_pds3': DataSetPDS3NewHorizonsLORRI,
    'vgiss': DataSetPDS3VoyagerISS,
    'vgiss_pds3': DataSetPDS3VoyagerISS,
    'sim': DataSetSim,
}


_DATASET_NAME_TO_INST_NAME_MAPPING = {
    'coiss': 'coiss',
    'coiss_pds3': 'coiss',
    'gossi': 'gossi',
    'gossi_pds3': 'gossi',
    'nhlorri': 'nhlorri',
    'nhlorri_pds3': 'nhlorri',
    'vgiss': 'vgiss',
    'vgiss_pds3': 'vgiss',
    'sim': 'sim',
}


assert (sorted(_DATASET_NAME_TO_CLASS_MAPPING.keys()) ==
        sorted(_DATASET_NAME_TO_INST_NAME_MAPPING.keys())), \
            'Dataset names are inconsistent'


def dataset_names() -> list[str]:
    """Return a list of all dataset names."""
    return sorted(_DATASET_NAME_TO_CLASS_MAPPING.keys())


def dataset_name_to_class(name: str) -> type[DataSet]:
    """Convert a dataset name to the corresponding class.

    Parameters:
        name: The name of the dataset.

    Returns:
        The class corresponding to the dataset name.

    Raises:
        KeyError: If the dataset name is not found.
    """
    return _DATASET_NAME_TO_CLASS_MAPPING[name.lower()]


def dataset_name_to_inst_name(name: str) -> str:
    """Convert a dataset name to the corresponding instrument name.

    Parameters:
        name: The name of the dataset.

    Returns:
        The instrument name corresponding to the dataset name.

    Raises:
        KeyError: If the dataset name is not found.
    """
    return _DATASET_NAME_TO_INST_NAME_MAPPING[name.lower()]


__all__ = ['DataSet',
           'DataSetPDS3CassiniISS',
           'DataSetPDS3GalileoSSI',
           'DataSetPDS3NewHorizonsLORRI',
           'DataSetPDS3VoyagerISS',
           'DataSetSim',
           'dataset_names',
           'dataset_name_to_class']
