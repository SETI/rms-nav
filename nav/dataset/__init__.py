from .dataset import DataSet  # noqa: F401
from .dataset_cassini_iss import DataSetCassiniISS  # noqa: F401
from .dataset_galileo_ssi import DataSetGalileoSSI  # noqa: F401
from .dataset_newhorizons_lorri import DataSetNewHorizonsLORRI  # noqa: F401
from .dataset_voyager_iss import DataSetVoyagerISS  # noqa: F401


DATASET_NAME_TO_CLASS_MAPPING = {
    'COISS': DataSetCassiniISS,
    'COISS_PDS3': DataSetCassiniISS,
    'GOSSI': DataSetGalileoSSI,
    'GOSSI_PDS3': DataSetGalileoSSI,
    'NHLORRI': DataSetNewHorizonsLORRI,
    'NHLORRI_PDS3': DataSetNewHorizonsLORRI,
    'VGISS': DataSetVoyagerISS,
    'VGISS_PDS3': DataSetVoyagerISS,
}


__all__ = ['DataSet',
           'DataSetCassiniISS',
           'DataSetGalileoSSI',
           'DataSetNewHorizonsLORRI',
           'DataSetVoyagerISS']
