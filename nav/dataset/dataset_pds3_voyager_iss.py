from typing import Any

from .dataset_pds3 import DataSetPDS3
from nav.support.misc import safe_lstrip_zero


class DataSetPDS3VoyagerISS(DataSetPDS3):

    @staticmethod
    def _get_img_name_from_filespec(filespec: str) -> str | None:
        """Extract the image name (with no extension) from a file specification.

        Parameters:
            filespec: The file specification string to parse.

        Raises:
            ValueError: If the file specification format is invalid.
        """

        parts = filespec.split('/')
        if len(parts) != 3:
            raise ValueError(f'Bad Primary File Spec "{filespec}" - expected 3 '
                             'directory levels')
        if parts[0].upper() != 'DATA':
            raise ValueError(f'Bad Primary File Spec "{filespec}" - expected "DATA"')
        range_dir = parts[1]
        img_name = parts[2]
        if len(range_dir) != 8 or range_dir[0] != 'C':
            raise ValueError(f'Bad Primary File Spec "{filespec}" - expected "Cddddddd"')
        if not img_name.endswith('_GEOMED.LBL'):
            return None
        return img_name.rsplit('_GEOMED')[0]

    @staticmethod
    def _img_name_valid(img_name: str) -> bool:
        """True if an image name is valid for this instrument.

        Parameters:
            img_name: The name of the image.

        Returns:
            True if the image name is valid for this instrument, False otherwise.
        """

        img_name = img_name.upper()

        # Cddddddd
        if len(img_name) != 8 or img_name[0] != 'C':
            return False
        try:
            _ = int(safe_lstrip_zero(img_name[1:]))
        except ValueError:
            return False

        return True

    @staticmethod
    def _extract_img_number(img_name: str) -> int:
        """Extract the image number from an image name.

        Parameters:
            img_name: The name of the image.

        Returns:
            The image number.

        Raises:
            ValueError: If the image name format is invalid.
        """

        if not DataSetPDS3VoyagerISS._img_name_valid(img_name):
            raise ValueError(f'Invalid image name "{img_name}"')

        return int(safe_lstrip_zero(img_name[1:]))

    _MIN_5xxx_VOL1 = 5101
    _MAX_5xxx_VOL1 = 5120
    _MIN_5xxx_VOL2 = 5201
    _MAX_5xxx_VOL2 = 5214
    _MIN_6xxx_VOL1 = 6101
    _MAX_6xxx_VOL1 = 6121
    _MIN_6xxx_VOL2 = 6201
    _MAX_6xxx_VOL2 = 6215
    _MIN_7xxx_VOL2 = 7201
    _MAX_7xxx_VOL2 = 7207
    _MIN_8xxx_VOL2 = 8201
    _MAX_8xxx_VOL2 = 8210

    _DATASET_LAYOUT = {
        'all_volume_names': [f'VGISS_{x:04d}' for x in
                             list(range(_MIN_5xxx_VOL1, _MAX_5xxx_VOL1+1)) +
                             list(range(_MIN_5xxx_VOL2, _MAX_5xxx_VOL2+1)) +
                             list(range(_MIN_6xxx_VOL1, _MAX_6xxx_VOL1+1)) +
                             list(range(_MIN_6xxx_VOL2, _MAX_6xxx_VOL2+1)) +
                             list(range(_MIN_7xxx_VOL2, _MAX_7xxx_VOL2+1)) +
                             list(range(_MIN_8xxx_VOL2, _MAX_8xxx_VOL2+1))],
        'volset_and_volume': lambda v: f'VGISS_{v[6]}xxx/{v}',
        'volume_to_index': lambda v: f'VGISS_{v[6]}xxx/{v}/{v}_index.lbl',
        'index_columns': ('FILE_SPECIFICATION_NAME',),
        'volumes_dir_name': 'volumes',
    }

    def __init__(self,
                 *args: Any,
                 **kwargs: Any) -> None:
        """Initializes a Voyager ISS dataset handler.

        Parameters:
            *args: Positional arguments to pass to the parent class.
            **kwargs: Keyword arguments to pass to the parent class.
        """
        super().__init__(*args, logger_name='DataSetVoyagerISS', **kwargs)
