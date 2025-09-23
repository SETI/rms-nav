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
    """Convert an instrument name to the corresponding class.

    Parameters:
        name: The name of the instrument.

    Returns:
        The class corresponding to the instrument name.

    Raises:
        KeyError: If the instrument name is not found.
    """

    return INST_NAME_TO_CLASS_MAPPING[name.lower()]


__all__ = ['Inst',
           'InstCassiniISS',
           'InstGalileoSSI',
           'InstNewHorizonsLORRI',
           'InstVoyagerISS',
           'inst_name_to_class']
