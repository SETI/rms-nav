from .inst import Inst  # noqa: F401
from .inst_cassini_iss import InstCassiniISS  # noqa: F401
from .inst_galileo_ssi import InstGalileoSSI  # noqa: F401
from .inst_newhorizons_lorri import InstNewHorizonsLORRI  # noqa: F401
from .inst_voyager_iss import InstVoyagerISS  # noqa: F401

def inst_id_to_class(inst_id: str) -> Inst:
    match inst_id.upper():
        case 'COISS' | 'COISS_PDS3':
            return InstCassiniISS
        case 'GOSSI' | 'GOSSI_PDS3':
            return InstGalileoSSI
        case 'NHLORRI' | 'NHLORRI_PDS3':
            return InstNewHorizonsLORRI
        case 'VGISS' | 'VGISS_PDS3':
            return InstVoyagerISS
    raise ValueError(f'Unknown instrument id: {inst_id}')

__all__ = ['Inst',
           'InstCassiniISS',
           'InstGalileoSSI',
           'InstNewHorizonsLORRI',
           'InstVoyagerISS']
