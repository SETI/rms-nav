import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional, cast

from .dataset import ImageFile, ImageFiles
from .dataset_pds3 import DataSetPDS3
from nav.config import Config
from nav.support.misc import safe_lstrip_zero


class DataSetPDS3CassiniISS(DataSetPDS3):
    """Implements dataset access for PDS3 Cassini ISS (Imaging Science Subsystem) data.

    This class provides specialized functionality for accessing and parsing Cassini
    ISS image data stored in PDS3 format.
    """

    _MIN_1xxx_VOL = 1001
    _MAX_1xxx_VOL = 1009
    _MIN_2xxx_VOL = 2001
    _MAX_2xxx_VOL = 2116
    _ALL_VOLUME_NAMES = tuple(f'COISS_{x:04d}' for x in
                              list(range(_MIN_1xxx_VOL, _MAX_1xxx_VOL+1)) +
                              list(range(_MIN_2xxx_VOL, _MAX_2xxx_VOL+1)))
    _INDEX_COLUMNS = ('FILE_SPECIFICATION_NAME',)
    _VOLUMES_DIR_NAME = 'calibrated'

    # Methods inherited from DataSetPDS3

    @staticmethod
    def _get_label_filespec_from_index(row: dict[str, Any]) -> str:
        """Extracts the label file specification from a row from an index table.

        Parameters:
            row: Dictionary containing PDS3 index table row data.

        Returns:
            The file specification string from the row.
        """

        filespec = cast(str, row['FILE_SPECIFICATION_NAME'])
        # Intentionally uppercase only
        if not filespec.endswith('.IMG'):
            raise ValueError(f'Bad Primary File Spec "{filespec}" - '
                             'expected ".IMG"')
        # Intentionally uppercase only
        return filespec.replace('.IMG', '_CALIB.LBL')

    @staticmethod
    def _get_image_filespec_from_label_filespec(label_filespec: str) -> str:
        """Extracts the image file specification from a label file specification.

        Parameters:
            label_filespec: The label file specification string to parse.

        Returns:
            The image file specification string.
        """
        # Intentionally uppercase only
        return label_filespec.replace('.LBL', '.IMG')

    @staticmethod
    def _get_img_name_from_label_filespec(filespec: str) -> str | None:
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
        img_name = img_name.replace('_CALIB', '')

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
        return False

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
    def _volset_and_volume(volume: str) -> str:
        """Get the volset and volume name.

        Parameters:
            volume: The volume name.
        """
        return f'COISS_{volume[6]}xxx/{volume}'

    @staticmethod
    def _volume_to_index(volume: str) -> str:
        """Get the index file name for a volume.

        Parameters:
            volume: The volume name.
        """
        # Intentionally lowercase only
        return f'COISS_{volume[6]}xxx/{volume}/{volume}_index.lbl'

    @staticmethod
    def _results_path_stub(volume: str, filespec: str) -> str:
        """Get the results path stub for an image filespec.

        Parameters:
            volume: The volume name.
            filespec: The filespec of the image.

        Returns:
            The results path stub. This is {volume}/{filespec} without the .IMG extension.
        """

        return str(Path(f'{volume}/{filespec}').with_suffix(''))

    def _check_additional_image_selection_criteria(self,
                                                   _img_path: str,
                                                   img_name: str,
                                                   _img_num: int,
                                                   arguments: Optional[argparse.Namespace] = None
                                                   ) -> bool:
        """Check additional image selection criteria.

        Parameters:
            _img_path: The path to the image.
            img_name: The name of the image.
            _img_num: The number of the image.
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

    # Public methods

    def __init__(self,
                 *,
                 config: Optional[Config] = None) -> None:
        """Initializes a Cassini ISS dataset handler.

        Parameters:
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """
        super().__init__(config=config)

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

    def yield_image_files_index(self, **kwargs: Any) -> Iterator[ImageFiles]:
        """Yield filenames given search criteria using index files.

        Parameters:
            group: How to group the results. Only 'botsim' is supported.
            **kwargs: Arbitrary keyword arguments, usually used to restrict the search.

        Yields:
            ImageFiles objects containing information about groups of selected
            image files.
        """

        group = kwargs.pop('group', None)

        if group is None:
            for imagefile in self._yield_image_files_index(**kwargs):
                yield ImageFiles(image_files=[imagefile])
            return

        if group != 'botsim':
            raise ValueError(f'Invalid group type "{group}"')

        # Combine BOTSIM files
        last_imagefile: ImageFile | None = None
        for imagefile in self._yield_image_files_index(additional_index_columns=('SHUTTER_MODE_ID',
                                                                                 'IMAGE_NUMBER'),
                                                       **kwargs):
            if last_imagefile is not None:
                if (last_imagefile.index_file_row['SHUTTER_MODE_ID'] != 'BOTSIM' or
                    imagefile.index_file_row['SHUTTER_MODE_ID'] != 'BOTSIM'):
                    yield ImageFiles(image_files=[last_imagefile])
                    last_imagefile = imagefile
                else:
                    last_img_num = int(last_imagefile.index_file_row['IMAGE_NUMBER'])
                    img_num = int(imagefile.index_file_row['IMAGE_NUMBER'])
                    if abs(img_num - last_img_num) <= 3:  # Allow 3 seconds slop
                        if last_imagefile.image_file_name[0] == 'N':
                            nac_imagefile = last_imagefile
                            wac_imagefile = imagefile
                        else:
                            wac_imagefile = last_imagefile
                            nac_imagefile = imagefile
                        yield ImageFiles(image_files=[nac_imagefile, wac_imagefile])
                        last_imagefile = None
                    else:
                        last_imagefile = imagefile
            else:
                last_imagefile = imagefile

        if last_imagefile is not None:
            yield ImageFiles(image_files=[last_imagefile])

    @staticmethod
    def supported_grouping() -> list[str]:
        """Returns the list of supported grouping types.

        Returns:
            The list of supported grouping types. For Cassini ISS, only 'botsim'
            is supported.
        """
        return ['botsim']
