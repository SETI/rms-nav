from typing import Any

from .dataset_pds3 import DataSetPDS3


class DataSetPDS3NewHorizonsLORRI(DataSetPDS3):

    @staticmethod
    def _get_img_name_from_filespec(filespec: str) -> str | None:
        """Extract the image name (with no extension) from a file specification.

        Parameters:
            filespec: The file specification string to parse.

        Raises:
            ValueError: If the file specification format is invalid.
        """

        parts = filespec.split('/')
        if parts[0].upper() != 'DATA':
            raise ValueError(f'Bad Primary File Spec "{filespec}"')
        range_dir = parts[1]
        img_name = parts[2]
        if len(range_dir) != 15 or range_dir[8] != '_':
            raise ValueError(f'Bad Primary File Spec "{filespec}"')
        if not img_name.endswith('_sci.lbl'):
            return None
        return img_name.rsplit('_sci')[0]

    @staticmethod
    def _img_name_valid(img_name: str) -> bool:
        """True if an image name is valid for this instrument.

        Parameters:
            img_name: The name of the image. Can be just the image name, the image filename,
                or the full file spec.

        Returns:
            True if the image name is valid for this instrument, False otherwise.
        """

        img_name = img_name.upper()

        # lor_0297859350_0x633
        if len(img_name) != 20 or not img_name.startswith('LOR_') or img_name[14:17] != '_0X':
            return False
        try:
            _ = int(img_name[4:14].lstrip('0'))
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

        if not DataSetPDS3NewHorizonsLORRI._img_name_valid(img_name):
            raise ValueError(f'Invalid image name "{img_name}"')

        return int(img_name[4:14].lstrip('0'))

    _DATASET_LAYOUT = {
        'all_volume_names': ['NHLALO_2001', 'NHJULO_2001',
                             'NHPCLO_2001', 'NHPELO_2001',
                             'NHKCLO_2001', 'NHKELO_2001',
                             'NHK2LO_2001'],
        'volset_and_volume': lambda v: f'NHxxLO_xxxx/{v}',
        'volume_to_index': lambda v: f'NHxxLO_xxxx/{v}/{v}_index.lbl',
        'index_columns': ('FILE_SPECIFICATION_NAME',),
        'volumes_dir_name': 'volumes',
    }

    def __init__(self,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, logger_name='DataSetNewHorizonsLORRI', **kwargs)
