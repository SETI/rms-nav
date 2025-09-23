import argparse
from typing import Any, Optional

from .dataset_pds3 import DataSetPDS3
from nav.support.misc import safe_lstrip_zero


class DataSetPDS3CassiniISS(DataSetPDS3):
    """Implements dataset access for Cassini ISS (Imaging Science Subsystem) data.

    This class provides specialized functionality for accessing and parsing Cassini
    ISS image data stored in PDS3 format.
    """

    # Methods inherited from DataSetPDS3

    @staticmethod
    def _get_img_name_from_filespec(filespec: str) -> str | None:
        """Extract the image name (with no extension) from a file specification.

        Parameters:
            filespec: The file specification string to parse.

        Returns:
            The image name if valid. None if the name is valid but should not be
            processed.

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
        if len(range_dir) != 21 or range_dir[10] != '_':
            raise ValueError(f'Bad Primary File Spec "{filespec}" - '
                             'expected "DATA/dddddddddd_dddddddddd"')
        img_name = img_name.rsplit('.')[0]
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

        # [NW]dddddddddd[_d[d]]
        if img_name[0] not in 'NW':
            return False
        if len(img_name) == 11:
            try:
                _ = int(img_name[1:])
                return True
            except ValueError:
                return False
        if 13 <= len(img_name) <= 14:
            if img_name[11] != '_':
                return False
            try:
                _ = int(img_name[1:11])
                _ = int(img_name[12:])
                return True
            except ValueError:
                return False
        return True

    @staticmethod
    def _extract_img_number(img_name: str) -> int:
        """Extract the image number from an image name.

        Parameters:
            img_name: The name of the image. Can be just the image name, the image filename,
                or the full file spec.

        Returns:
            The image number.

        Raises:
            ValueError: If the image name format is invalid.
        """

        if not DataSetPDS3CassiniISS._img_name_valid(img_name):
            raise ValueError(f'Invalid image name "{img_name}"')

        return int(safe_lstrip_zero(img_name[1:11]))

    @staticmethod
    def _map_filename(img_filename: str) -> str:
        """Map an img filename as found in the primary filespec to the filename to use.

        Parameters:
            img_filename: The filename to map.

        Returns:
            The mapped filename.

        Raises:
            ValueError: If the image filename format is invalid.
        """
        if not img_filename.upper().endswith(('.LBL', '.IMG')):
            raise ValueError(f'Invalid image filename "{img_filename}"')
        return img_filename[:-4] + '_CALIB' + img_filename[-4:]

    _MIN_1xxx_VOL = 1001
    _MAX_1xxx_VOL = 1009
    _MIN_2xxx_VOL = 2001
    _MAX_2xxx_VOL = 2116

    _DATASET_LAYOUT = {
        'all_volume_names': [f'COISS_{x:04d}' for x in
                             list(range(_MIN_1xxx_VOL, _MAX_1xxx_VOL+1)) +
                             list(range(_MIN_2xxx_VOL, _MAX_2xxx_VOL+1))],
        'volset_and_volume': lambda v: f'COISS_{v[6]}xxx/{v}',
        'volume_to_index': lambda v: f'COISS_{v[6]}xxx/{v}/{v}_index.lbl',
        'index_columns': ('FILE_SPECIFICATION_NAME',),
        'volumes_dir_name': 'calibrated',
        'map_filename_func': _map_filename,
    }

    def __init__(self,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, logger_name='DataSetCassiniISS', **kwargs)

    @staticmethod
    def add_selection_arguments(cmdparser: argparse.ArgumentParser,
                                group: Optional[argparse._ArgumentGroup] = None) -> None:
        """Adds Cassini ISS-specific command-line arguments for image selection.

        Parameters:
            cmdparser: The argument parser to add arguments to.
            group: Optional argument group to add arguments to. If None, creates a new group.
        """

        # First add the PDS3 arguments
        super(DataSetPDS3CassiniISS, DataSetPDS3CassiniISS).add_selection_arguments(cmdparser,
                                                                                    group)

        if group is None:
            group = cmdparser.add_argument_group('Image selection (Cassini ISS-specific)')
        group.add_argument(
            '--camera', type=str, default=None, choices=['nac', 'NAC', 'wac', 'WAC'],
            help='Only process images with the given camera type')

    def _check_additional_image_selection_criteria(self,
                                                   img_path: str,
                                                   img_name: str,
                                                   img_num: int,
                                                   arguments: Optional[argparse.Namespace] = None
                                                   ) -> bool:
        """Check additional image selection criteria.

        Parameters:
            img_path: The path to the image.
            img_name: The name of the image.
            img_num: The number of the image.
            arguments: The parsed arguments.

        Returns:
            True if the image should be processed, False otherwise.
        """

        if arguments is None or arguments.camera is None:
            return True
        if arguments.camera.lower() == 'nac':
            return img_name[0] == 'N'
        if arguments.camera.lower() == 'wac':
            return img_name[0] == 'W'
        raise ValueError(f'Invalid camera type "{arguments.camera}"')
