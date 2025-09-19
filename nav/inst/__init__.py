from .inst import Inst  # noqa: F401
from .inst_cassini_iss import InstCassiniISS  # noqa: F401
from .inst_galileo_ssi import InstGalileoSSI  # noqa: F401
from .inst_newhorizons_lorri import InstNewHorizonsLORRI  # noqa: F401
from .inst_voyager_iss import InstVoyagerISS  # noqa: F401


INST_NAME_TO_CLASS_MAPPING = {
    'coiss': InstCassiniISS,
    'gossi': InstGalileoSSI,
    'nhlorri': InstNewHorizonsLORRI,
    'vgiss': InstVoyagerISS,
}


def inst_name_to_class(name: str) -> type[Inst]:
    return INST_NAME_TO_CLASS_MAPPING[name]


__all__ = ['Inst',
           'InstCassiniISS',
           'InstGalileoSSI',
           'InstNewHorizonsLORRI',
           'InstVoyagerISS']
