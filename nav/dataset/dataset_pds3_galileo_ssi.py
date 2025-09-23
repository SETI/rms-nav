from typing import Any

from .dataset_pds3 import DataSetPDS3
from nav.support.misc import safe_lstrip_zero


class DataSetPDS3GalileoSSI(DataSetPDS3):

    # Methods inherited from DataSetPDS3

    @staticmethod
    def _get_img_name_from_filespec(filespec: str) -> str | None:
        """Extract the image name (with no extension) from a file specification.

        Parameters:
            filespec: The file specification string to parse.

        Raises:
            ValueError: If the file specification format is invalid.
        """

        parts = filespec.split('/')
        if not (2 <= len(parts) <= 3):
            raise ValueError(f'Bad Primary File Spec "{filespec}" - expected 2 or 3 '
                             'directory levels')
        if parts[0].upper() in ('REDO', 'RAW_CAL', 'VENUS', 'EARTH', 'MOON',
                                'GASPRA', 'IDA'):
            img_name = parts[1]
        elif parts[0].upper() in ('C9', 'C10', 'C20', 'C21', 'C22', 'C30',
                                  'E4', 'E6', 'E11', 'E12', 'E14', 'E15',
                                  'E17', 'E18', 'E19', 'E26',
                                  'G1', 'G2', 'G7', 'G8', 'G28', 'G29',
                                  'I24', 'I25', 'I27', 'I31', 'I32', 'I33',
                                  'J0'):
            img_name = parts[2]
        else:
            raise ValueError(f'Bad Primary File Spec "{filespec}" - bad target directory')
        if not img_name.endswith('.IMG'):
            return None
        img_name = img_name.rsplit('.', maxsplit=1)[0]
        return img_name

    @staticmethod
    def _img_name_valid(img_name: str) -> bool:
        """True if an image name is valid for this instrument.

        Parameters:
            img_name: The name of the image.

        Returns:
            True if the image name is valid for this instrument, False otherwise.
        """

        img_name = img_name.upper()

        # Cdddddddddd[RS]
        if len(img_name) != 12 or img_name[0] != 'C' or img_name[11] not in 'RS':
            return False
        try:
            _ = int(safe_lstrip_zero(img_name[1:11]))
        except ValueError:
            return False

        return True

    @staticmethod
    def _extract_img_number(img_name: str) -> int:
        """Extract the image number from an image name.

        Parameters:
            img_name: The name of the image. Can be just the image name, the image
                filename, or the full file spec.

        Returns:
            The image number.

        Raises:
            ValueError: If the image name format is invalid.
        """

        if not DataSetPDS3GalileoSSI._img_name_valid(img_name):
            raise ValueError(f'Invalid image name "{img_name}"')

        return int(img_name[1:11].lstrip('0'))

    _MIN_0xxx_VOL = 2
    _MAX_0xxx_VOL = 23

    _DATASET_LAYOUT = {
        'all_volume_names': [f'GO_{x:04d}' for x in
                             list(range(_MIN_0xxx_VOL, _MAX_0xxx_VOL+1))],
        'volset_and_volume': lambda v: f'GO_0xxx/{v}',
        'volume_to_index': lambda v: f'GO_0xxx/{v}/{v}_index.lbl',
        'index_columns': ('FILE_SPECIFICATION_NAME',),
        'volumes_dir_name': 'volumes',
    }

    def __init__(self,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, logger_name='DataSetGalileoSSI', **kwargs)
