from .dataset import DataSet  # noqa: F401
from .dataset_cassini_iss import DataSetCassiniISS  # noqa: F401
from .dataset_galileo_ssi import DataSetGalileoSSI  # noqa: F401
from .dataset_newhorizons_lorri import DataSetNewHorizonsLORRI  # noqa: F401
from .dataset_voyager_iss import DataSetVoyagerISS  # noqa: F401


def dataset_id_to_class(dataset_id: str) -> DataSet:
    match dataset_id.upper():
        case 'COISS', 'COISS_PDS3':
            return DataSetCassiniISS
        case 'GOSSI', 'GOSSI_PDS3':
            return DataSetGalileoSSI
        case 'NHLORRI', 'NHLORRI_PDS3':
            return DataSetNewHorizonsLORRI
        case 'VGISS', 'VGISS_PDS3':
            return DataSetVoyagerISS
    raise ValueError(f'Unknown dataset id: {dataset_id}')


__all__ = ['DataSet',
           'DataSetCassiniISS',
           'DataSetGalileoSSI',
           'DataSetNewHorizonsLORRI',
           'DataSetVoyagerISS']
