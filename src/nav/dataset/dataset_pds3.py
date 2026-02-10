from abc import abstractmethod
import argparse
import csv
from functools import lru_cache
import os
from pathlib import Path
import random
from typing import Any, Iterator, Optional, cast

from filecache import FCPath, FileCache
from pdstable import PdsTable

from .dataset import DataSet, ImageFile, ImageFiles
from nav.config import Config
from nav.support.misc import flatten_list


class DataSetPDS3(DataSet):
    """Parent class for PDS3 datasets.

    This class provides functionality common to all PDS3 datasets.
    """

    # Data definitions overriden by subclasses
    _ALL_VOLUME_NAMES: tuple[str, ...] = ()
    _INDEX_COLUMNS: tuple[str, ...] = ()
    _VOLUMES_DIR_NAME: str = ''

    def __init__(
        self,
        pds3_holdings_root: Optional[str | Path | FCPath] = None,
        *,
        index_filecache: Optional[FileCache] = None,
        pds3_holdings_filecache: Optional[FileCache] = None,
        config: Optional[Config] = None,
    ) -> None:
        """Initializes a PDS3 dataset with directory and cache settings.

        Parameters:
            pds3_holdings_root: Path to PDS3 holdings directory. If None, uses PDS3_HOLDINGS_DIR
                environment variable. May be a URL accepted by FCPath.
            index_filecache: FileCache object to use for index files. If None, creates a new one.
            pds3_holdings_filecache: FileCache object to use for PDS3 holdings files. If None,
                creates a new one.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.

        Raises:
            ValueError: If pds3_holdings_root is None and PDS3_HOLDINGS_DIR environment variable
                is not set.
        """

        super().__init__(config=config)

        if index_filecache is None:
            self._index_filecache = FileCache('nav_pds3_index')  # Index shared; MP safe
        else:
            self._index_filecache = index_filecache

        if pds3_holdings_filecache is None:
            self._pds3_holdings_filecache = FileCache(None)  # Data not shared
        else:
            self._pds3_holdings_filecache = pds3_holdings_filecache

        if pds3_holdings_root is not None:
            self._pds3_holdings_root: FCPath | None = (
                self._pds3_holdings_filecache.new_path(pds3_holdings_root)
            )
        else:
            self._pds3_holdings_root = None

    @property
    def pds3_holdings_root(self) -> FCPath:
        """The PDS3 holdings directory; may be a URL."""
        if self._pds3_holdings_root is not None:
            return self._pds3_holdings_root

        pds3_holdings_root = None
        try:
            pds3_holdings_root = self.config.environment.pds3_holdings_root
        except AttributeError:
            pass
        if pds3_holdings_root is None:
            pds3_holdings_root = os.getenv('PDS3_HOLDINGS_DIR')
        if pds3_holdings_root is None:
            raise ValueError(
                'One of configuration variable "pds3_holdings_root" or '
                'PDS3_HOLDINGS_DIR environment variable must be set'
            )
        self._pds3_holdings_root = self._pds3_holdings_filecache.new_path(
            pds3_holdings_root
        )

        return self._pds3_holdings_root

    def __str__(self) -> str:
        return f'DataSetPDS3(pds3_holdings_root={self._pds3_holdings_root})'

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    @abstractmethod
    def _get_label_filespec_from_index(row: dict[str, Any]) -> str:
        """Extracts the label file specification from a row from an index table.

        Parameters:
            row: Dictionary containing PDS3 index table row data.

        Returns:
            The file specification string from the row.
        """
        ...

    @staticmethod
    @abstractmethod
    def _get_image_filespec_from_label_filespec(label_filespec: str) -> str:
        """Extracts the image file specification from a label file specification.

        Parameters:
            label_filespec: The label file specification string to parse.

        Returns:
            The image file specification string.
        """
        ...

    @staticmethod
    @abstractmethod
    def _get_img_name_from_label_filespec(filespec: str) -> str | None:
        """Extracts the image name (with no extension) from a file specification.

        Parameters:
            filespec: The file specification string to parse.

        Returns:
            The image name if valid. None if the name is valid but should not be
            processed.

        Raises:
            ValueError: If the file specification format is invalid.
        """
        ...

    @staticmethod
    @abstractmethod
    def _img_name_valid(img_name: str) -> bool:
        """True if an image name is valid for this instrument.

        Parameters:
            img_name: The name of the image.

        Returns:
            True if the image name is valid for this instrument, False otherwise.
        """
        ...

    @staticmethod
    @abstractmethod
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
        ...

    @staticmethod
    @abstractmethod
    def _volset_and_volume(volume: str) -> str:
        """Get the volset and volume name.

        Parameters:
            volume: The volume name.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _volume_to_index(volume: str) -> str:
        """Get the index file name for a volume.

        Parameters:
            volume: The volume name.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _results_path_stub(volume: str, filespec: str) -> str:
        """Get the results path stub for an image filespec."""
        raise NotImplementedError

    @staticmethod
    def add_selection_arguments(
        cmdparser: argparse.ArgumentParser,
        group: Optional[argparse._ArgumentGroup] = None,
    ) -> None:
        """Adds PDS3-specific command-line arguments for image selection.

        Parameters:
            cmdparser: The argument parser to add arguments to.
            group: Optional argument group to add arguments to. If None, creates a new group.
        """

        if group is None:
            group = cmdparser.add_argument_group('Image selection (PDS3-specific)')
        group.add_argument(
            'img_name',
            action='append',
            nargs='*',
            type=str,
            help='Specific image name(s) to process',
        )
        # group.add_argument(
        #     '--planet', default='saturn',
        #     type=_validate_planet,
        #     help=f"""Which planet to process: jupiter, saturn, uranus, neptune
        #              (saturn is the default)""")
        group.add_argument(
            '--first-image-num',
            type=int,
            default=None,
            metavar='IMAGE_NUM',
            help="""The starting image number; only images with this number or greater will be
            processed""",
        )
        group.add_argument(
            '--last-image-num',
            type=int,
            default=None,
            metavar='IMAGE_NUM',
            help="""The ending image number; only images with this number or less will be
            processed""",
        )
        group.add_argument(
            '--volumes',
            action='append',
            help="""One or more entire PDS3 volume names; only images in these
            volumes or volume subdirectories will be processed. Can accept multiple values
            separated by commas or multiple arguments.""",
        )
        group.add_argument(
            '--first-volume',
            type=str,
            default=None,
            metavar='VOL_NAME',
            help="""The starting PDS3 volume name; only images in this volume or chronologically
            later will be processed""",
        )
        group.add_argument(
            '--last-volume',
            type=str,
            default=None,
            metavar='VOL_NAME',
            help="""The ending PDS3 volume name; only images in this volume or chronologically
            earlier will be processed""",
        )
        group.add_argument(
            '--image-filespec-csv',
            action='append',
            help="""A CSV file that contains filespecs of images to process; a header row
            is required and must contain a column named 'Primary File Spec' or 'primaryfilespec'.
            The list is still subject to other selection criteria.""",
        )
        group.add_argument(
            '--image-file-list',
            action='append',
            help="""A file that contains filespecs or names of images to process;
            the list is still subject to other selection criteria.""",
        )
        # group.add_argument(
        #     '--has-offset-file', action='store_true', default=False,
        #     help='Only process images that already have an offset file')
        # group.add_argument(
        #     '--has-no-offset-file', action='store_true', default=False,
        #     help='Only process images that don\'t already have an offset file')
        # group.add_argument(
        #     '--has-png-file', action='store_true', default=False,
        #     help='Only process images that already have a PNG file')
        # group.add_argument(
        #     '--has-no-png-file', action='store_true', default=False,
        #     help='Only process images that don\'t already have a PNGfile')
        # group.add_argument(
        #     '--has-offset-error', action='store_true', default=False,
        #     help="""Only process images if the offset file exists and
        #             indicates a fatal error""")
        # group.add_argument(
        #     '--has-offset-nonspice-error', action='store_true', default=False,
        #     help="""Only process images if the offset file exists and
        #             indicates a fatal error other than missing SPICE data""")
        # group.add_argument(
        #     '--has-offset-spice-error', action='store_true', default=False,
        #     help="""Only process images if the offset file exists and
        #             indicates a fatal error from missing SPICE data""")
        # group.add_argument(
        #     '--selection-expr', type=str, metavar='EXPR',
        #     help='Expression to evaluate to decide whether to reprocess an offset')
        group.add_argument(
            '--choose-random-images',
            type=int,
            default=None,
            metavar='N',
            help='Choose N random images to process within other constraints',
        )
        # group.add_argument(
        #     '--show-image-list-only', action='store_true', default=False,
        #     help="""Just show a list of files that would be processed without doing
        #             any actual processing"""
        # )

    def _validate_selection_arguments(self, arguments: argparse.Namespace) -> None:
        """Validates user arguments that can't be checked during initial parsing.

        Parameters:
            arguments: The parsed arguments to validate.
        """

        # TODO This method is currently unused and should be used
        # For some reason mypy can't see the img_name field
        for img_name in flatten_list(arguments.img_name):
            if not self._img_name_valid(img_name):
                raise argparse.ArgumentTypeError(f'Invalid image name {img_name}')

    def yield_image_files_from_arguments(
        self, arguments: argparse.Namespace
    ) -> Iterator[ImageFiles]:
        """Given parsed arguments, yield all selected filenames.

        Parameters:
            arguments: The parsed arguments structure.

        Yields:
            ImageFiles objects containing information about groups of selected
            image files.
        """

        # Start with wanting all images
        img_name_list: list[str] = []
        img_filespec_list: list[str] = []
        # TODO It's a problem that these are image filespecs but elsewhere we are expecting label

        # Limit to the user-specific list of images, if any
        if arguments.img_name is not None and flatten_list(arguments.img_name):
            img_name_list = [x.upper() for x in flatten_list(arguments.img_name)]

        # Also limit to the list of images in the FileSpec CSV file, if any
        if arguments.image_filespec_csv:
            for filename in arguments.image_filespec_csv:
                with open(filename, 'r') as csvfile:
                    csvreader = csv.reader(csvfile)
                    header = next(csvreader)
                    for colnum in range(len(header)):
                        if (
                            header[colnum] == 'Primary File Spec'
                            or header[colnum] == 'primaryfilespec'
                        ):
                            break
                    else:
                        raise ValueError(
                            f'Badly formatted CSV file "{filename}" '
                            '- no Primary File Spec header'
                        )
                    for row in csvreader:
                        filespec = row[colnum]
                        img_filespec_list.append(filespec)

        # Also limit to the list of images in the filelist file, if any
        if arguments.image_file_list:
            for filename in arguments.image_file_list:
                with open(filename, 'r') as fp:
                    for line in fp:
                        line = line.strip()
                        if len(line) == 0 or line[0] == '#':
                            continue
                        # Ignore anything after the filename
                        filename = line.split(' ')[0]
                        if not self._img_name_valid(filename):
                            raise ValueError(
                                f'Bad filename in filelist file "{filename}": {line}'
                            )
                        img_name_list.append(filename)

        first_image_number = arguments.first_image_num
        last_image_number = arguments.last_image_num
        first_volume_number = arguments.first_volume
        last_volume_number = arguments.last_volume
        volumes = None
        if arguments.volumes:
            volumes = [x for y in arguments.volumes for x in y.split(',')]
            volumes.sort()

        # if use_index_files:
        #     yield_function = yield_image_filenames_index
        # else:
        #     yield_function = yield_image_filenames

        # last_image_name = None
        # last_image_path = None

        for ret in self.yield_image_files_index(
            img_start_num=first_image_number,
            img_end_num=last_image_number,
            vol_start=first_volume_number,
            vol_end=last_volume_number,
            volumes=volumes,
            img_name_list=img_name_list,
            img_filespec_list=img_filespec_list,
            # TODO
            # force_has_offset_file=arguments.has_offset_file,
            # force_has_no_offset_file=arguments.has_no_offset_file,
            # force_has_png_file=arguments.has_png_file,
            # force_has_no_png_file=arguments.has_no_png_file,
            # force_has_offset_error=arguments.has_offset_error,
            # force_has_offset_spice_error=arguments.has_offset_spice_error,
            # force_has_offset_nonspice_error=arguments.has_offset_nonspice_error,
            # selection_expr=arguments.selection_expr,
            choose_random_images=arguments.choose_random_images,
            arguments=arguments,
        ):
            yield ret
        #     # Before returning a matching image, see if we need to combine BOTSIM
        #     # images. We do this by looking at adjacent pairs of returned images to
        #     # see if they match.
        #     _, image_name = os.path.split(image_path)
        #     image_name = image_name[img_lim_start:img_lim_end]
        #     if (combine_botsim and
        #         last_image_name is not None and
        #         last_image_name[0] == 'N' and
        #         image_name[0] == 'W' and
        #         image_name[1:] == last_image_name[1:]):
        #         yield (last_image_path, image_path)
        #         last_image_path = None
        #         last_image_name = None
        #     else:
        #         if last_image_path is not None:
        #             if combine_botsim:
        #                 yield (last_image_path, None)
        #             else:
        #                 yield last_image_path
        #         last_image_path = image_path
        #         last_image_name = image_name

        # if last_image_path is not None:
        #     if combine_botsim:
        #         yield (last_image_path, None)
        #     else:
        #         yield last_image_path

    @lru_cache(maxsize=3)
    def _read_pds_table(
        self, fn: str, columns: Optional[tuple[str, ...]] = None
    ) -> PdsTable:
        """Reads a PDS table file with caching.

        Parameters:
            fn: Path to the PDS table file.
            columns: Optional tuple of column names to read. If None, all columns
                     are read. This is useful for improving performance.

        Returns:
            The parsed PdsTable object.
        """

        return PdsTable(fn, columns=columns, label_method='fast')

    def _yield_image_files_index(self, **kwargs: Any) -> Iterator[ImageFile]:
        """Yield filenames given search criteria using index files.

        This function assumes that the dataset is in a set of PDS3 volumes laid out like
        the PDS Ring-Moon Systems Node archive:

            $(PDS3_HOLDINGS_DIR)/volumes/{volume_set}/{volume}/
            {sub_dirs}/{image}.[IMG,LBL]

        and that the index files are laid out as:

            $(PDS3_HOLDINGS_DIR)/metadata/(volume set)/(volume)/{volume}_index.[lbl,tab]

        Parameters:
            img_start_num: Optional[int] = None,
            img_end_num: Optional[int] = None,
            vol_start: Optional[str] = None,
            vol_end: Optional[str] = None,
            volumes: Optional[list[str]] = None,
            camera: Optional[str] = None,
            img_name_list: Optional[list[str]] = None,
            image_filespec_list: Optional[list[str]] = None,
            force_has_offset_file: bool = False,
            force_has_no_offset_file: bool = False,
            force_has_png_file: bool = False,
            force_has_no_png_file: bool = False,
            force_has_offset_error: bool = False,
            force_has_offset_spice_error: bool = False,
            force_has_offset_nonspice_error: bool = False,
            selection_expr: Optional[str] = None,
            choose_random_images: int | None = False,
            max_filenames: Optional[int] = None,
            suffix: Optional[str] = None,
            planets: Optional[str] = None

        Yields:
            ImageFile objects containing information about selected image files.
        """

        kwargs = kwargs.copy()
        img_start_num: Optional[int] = kwargs.pop('img_start_num', None)
        img_end_num: Optional[int] = kwargs.pop('img_end_num', None)
        vol_start: Optional[str] = kwargs.pop('vol_start', None)
        vol_end: Optional[str] = kwargs.pop('vol_end', None)
        volumes: Optional[list[str]] = kwargs.pop('volumes', None)
        camera: Optional[str] = kwargs.pop('camera', None)
        img_name_list: Optional[list[str]] = kwargs.pop('img_name_list', None)
        img_filespec_list: Optional[list[str]] = kwargs.pop('img_filespec_list', None)
        choose_random_images: Optional[int] = kwargs.pop('choose_random_images', None)
        max_filenames: Optional[int] = kwargs.pop('max_filenames', None)
        arguments: Optional[argparse.Namespace] = kwargs.pop('arguments', None)
        additional_index_columns: tuple[str, ...] = kwargs.pop(
            'additional_index_columns', ()
        )

        if len(kwargs) > 0:
            raise ValueError(f'Unexpected keyword arguments: {kwargs}')

        logger = self._logger

        logger.info(f'*** Image number range: {img_start_num} - {img_end_num}')
        logger.info(f'*** Volume range:       {vol_start} - {vol_end}')
        logger.info(f'*** Camera:             {camera}')
        # logger.info('*** Results root directory:  %s', CB_RESULTS_ROOT)
        # logger.info('*** Instrument host:         %s', arguments.instrument_host)
        # if arguments.image_full_path:
        #     log('*** Images explicitly from full paths:')
        #     for image_path in arguments.image_full_path:
        #         log('        %s', image_path)
        # log('*** Already has offset file: %s', arguments.has_offset_file)
        # log('*** Has no offset file:      %s', arguments.has_no_offset_file)
        # log('*** Already has PNG file:    %s', arguments.has_png_file)
        # log('*** Has no PNG file:         %s', arguments.has_no_png_file)
        if img_name_list:
            logger.info('*** Explicit image names:')
            for explicit_img_name in img_name_list:
                logger.info(f'        {explicit_img_name}')
        if img_filespec_list:
            logger.info('*** Explicit image filespecs:')
            for explicit_img_filespec in img_filespec_list:
                logger.info(f'        {explicit_img_filespec}')
        if volumes is not None and volumes != []:
            logger.info('*** Images restricted to volumes:')
            for volume in volumes:
                for vol in volume.split(','):
                    logger.info(f'        {vol}')

        all_volume_names = self._ALL_VOLUME_NAMES
        index_columns = self._INDEX_COLUMNS + additional_index_columns
        volumes_dir_name = self._VOLUMES_DIR_NAME

        # Restrict volumes to given "volumes" argument
        if volumes is not None:
            for vol in volumes:
                if vol not in all_volume_names:
                    raise ValueError(f'Illegal volume name: {vol}')
            # This keeps the order of the provided volumes
            valid_volumes = [v for v in volumes if v in all_volume_names]
        else:
            valid_volumes = list(all_volume_names)

        # Restrict volumes to given "vol_start" and "vol_end" arguments
        # keeping original order
        if vol_start is not None:
            vol_start_idx = all_volume_names.index(vol_start)
        if vol_end is not None:
            vol_end_idx = all_volume_names.index(vol_end)
        valid_volumes = [
            v
            for v in valid_volumes
            if (
                (vol_start is None or vol_start_idx <= all_volume_names.index(v))
                and (vol_end is None or all_volume_names.index(v) <= vol_end_idx)
            )
        ]

        # URLs to the volume raw directory and index directory
        volume_raw_dir_url = self.pds3_holdings_root / volumes_dir_name
        index_dir_url = self.pds3_holdings_root / 'metadata'

        # Validate the image_name_list and image_filespec_list
        if img_name_list:
            for explicit_img_name in img_name_list:
                if not self._img_name_valid(explicit_img_name):
                    raise ValueError(f'Invalid image name "{explicit_img_name}"')
        if img_filespec_list:
            for explicit_img_filespec in img_filespec_list:
                if not self._get_img_name_from_label_filespec(explicit_img_filespec):
                    raise ValueError(
                        f'Invalid image filespec "{explicit_img_filespec}"'
                    )
            img_filespec_list = [
                x.rsplit('.', maxsplit=1)[0] for x in img_filespec_list
            ]

        # Optimize the first and last image number based on image_name_list and image_filespec_list
        # This is just to improve performance
        if img_name_list:
            img_start_num = max(
                0 if img_start_num is None else img_start_num,
                min([self._extract_img_number(x) for x in img_name_list]),
            )
            img_end_num = min(
                999999999999 if img_end_num is None else img_end_num,
                max([self._extract_img_number(x) for x in img_name_list]),
            )

        if img_filespec_list:
            img_start_num = max(
                0 if img_start_num is None else img_start_num,
                min([self._extract_img_number(x) for x in img_filespec_list]),
            )
            img_end_num = min(
                999999999999 if img_end_num is None else img_end_num,
                max([self._extract_img_number(x) for x in img_filespec_list]),
            )

        # TODO When yielding via an index, we don't get to optimize searching for
        # offset/png files. We just always look through the index files, and then check
        # out the offset/png files later. This could definitely be improved.

        # search_volume_path = None
        # search_suffix = None

        # if (force_has_offset_error or force_has_offset_nonspice_error or
        #     force_has_offset_spice_error):
        #     force_has_offset_file = True

        # assert not (force_has_offset_file and force_has_png_file)
        # if force_has_offset_file:
        #     # If we need an offset file, then we're actually looking for files in
        #     # the offsets directories, not the image directories. This is much faster
        #     # than going through the image directories and checking each one when there
        #     # aren't a lot of offset files.
        #     search_volume_path = clean_join(CB_RESULTS_ROOT, 'offsets')
        #     search_suffix = '-OFFSET.dat'
        # if force_has_png_file:
        #     # If we need an offset file, then we're actually looking for files in
        #     # the png directories, not the image directories. This is much faster
        #     # than going through the image directories and checking each one when there
        #     # aren't a lot of png files.
        #     search_volume_path = clean_join(CB_RESULTS_ROOT, 'png')
        #     search_suffix = '.png'
        # # The directory format if searching_offset or searching_png is:
        # #       <search_volume_path>/<VOLUME>/<RANGE>/<IMGNAME><search_suffix>
        # # The directory format if NOT searching_offset or searching_png is:
        # #       <search_volume_path>/<VOLUME>/[data]/<RANGE>/IMGNAME_<SUFFIX>

        # What part of the filename do we look at to get the image number?

        # index_volume_path = clean_join(CB_HOLDINGS_ROOT,
        #                             instrument_host_config['index_volume_path'][planet])
        # index_path = instrument_host_config['index_path']
        # volume_prefix = instrument_host_config['volume_prefix']

        # logger.debug('Index files exist in: %s', index_volume_path)
        # logger.debug('Data exists in: %s', data_volume_path)

        # Limit the number of returned yields from this method if necessary
        limit_yields = choose_random_images if choose_random_images else None
        if max_filenames is not None:
            if limit_yields is None:
                limit_yields = max_filenames
            else:
                limit_yields = min(limit_yields, max_filenames)
        num_yields = 0

        while True:
            done = False

            # If choosing random images, replace the volumes with a single random volume
            # but keep the original volume list for the next iteration
            valid_volumes_to_use = valid_volumes
            if choose_random_images:
                valid_volumes_to_use = [
                    valid_volumes_to_use[
                        random.randint(0, len(valid_volumes_to_use) - 1)
                    ]
                ]

            for search_vol in valid_volumes_to_use:
                # Find and retrieve the volume index label/table
                index_label_url = index_dir_url / self._volume_to_index(search_vol)
                index_tab_url = index_label_url.with_suffix('.tab')
                # This will raise a FileNotFoundError if the index file label or table
                # can't be found
                # TODO Implement actual error handling
                # We have to convert the FCPaths to Posix strings here so that FileCache.retrieve()
                # can use them. Note that if for some reason there was a specific FileCache given
                # for pds3_holdings_root, it will be overriden by self._index_filecache.
                # TODO Needs to return exceptions instead of a single FileNotFoundError
                # so we can tell the user what's actually going on.
                ret = self._index_filecache.retrieve(
                    [index_label_url.as_posix(), index_tab_url.as_posix()]
                )
                index_label_localpath, _ = cast(list[Path], ret)

                # if search_volume_path is not None:
                #     search_vol_fulldir = clean_join(search_volume_path, search_vol)
                #     # If we require an offset/png file but the directory is missing,
                #     # there's no point in looking further
                #     if not os.path.isdir(search_vol_fulldir):
                #         continue

                # Read the index table
                index_tab = self._read_pds_table(
                    index_label_localpath, columns=index_columns
                )
                rows = index_tab.dicts_by_row()

                # If choosing random images, replace the rows with a single random row
                if choose_random_images:
                    rows = [rows[random.randint(0, len(rows) - 1)]]

                for row in rows:
                    label_filespec = self._get_label_filespec_from_index(row)
                    img_filespec = self._get_image_filespec_from_label_filespec(
                        label_filespec
                    )

                    # Check that the image filespec is in the requested list
                    if img_filespec_list:
                        if (
                            label_filespec.rsplit('.', maxsplit=1)[0]
                            not in img_filespec_list
                        ):
                            continue

                    # Get the image name
                    try:
                        img_name = self._get_img_name_from_label_filespec(
                            label_filespec
                        )
                    except ValueError:
                        logger.error(
                            'IMGNAME: Index file "%s" contains bad Primary File Spec "%s"',
                            index_tab_url,
                            label_filespec,
                        )
                        continue
                    if img_name is None:
                        continue  # Not a name we should process

                    # Check that the image name is in the requested list
                    if img_name_list:
                        for restrict_name in img_name_list:
                            if img_name.lower().startswith(restrict_name.lower()):
                                break
                        else:
                            continue

                    # if raw_suffix is not None:
                    #     image_name = image_name.replace(raw_suffix, suffix)

                    # Get the image number and test that it's in range
                    # There's no point in checking the range dir for image number
                    # since we have to go through the entire index either way
                    try:
                        img_num = self._extract_img_number(img_name)
                    except ValueError as err:
                        raise ValueError(
                            f'IMGNUM: Index file "{index_tab_url}" contains bad '
                            f'path "{label_filespec}"'
                        ) from err
                    if img_end_num is not None and img_num > img_end_num:
                        if choose_random_images:
                            continue
                        # Images are in monotonically increasing order, so we can just
                        # quit now for efficiency
                        done = True
                        break
                    if img_start_num is not None and img_num < img_start_num:
                        continue

                    # Check that the image meets any additional selection criteria specific
                    # to this dataset
                    label_url = (
                        volume_raw_dir_url
                        / self._volset_and_volume(search_vol)
                        / label_filespec
                    )
                    img_url = (
                        volume_raw_dir_url
                        / self._volset_and_volume(search_vol)
                        / img_filespec
                    )
                    if not self._check_additional_image_selection_criteria(
                        img_url.as_posix(), img_name, img_num, arguments
                    ):
                        continue

                    # if force_has_offset_file:
                    #     offset_path = img_to_offset_path(img_path, instrument_host)
                    #     if not os.path.isfile(offset_path):
                    #         continue
                    # if force_has_no_offset_file:
                    #     offset_path = img_to_offset_path(img_path, instrument_host)
                    #     if os.path.isfile(offset_path):
                    #         continue
                    # if force_has_png_file:
                    #     png_path = img_to_png_path(img_path, instrument_host)
                    #     if not os.path.isfile(png_path):
                    #         continue
                    # if force_has_no_png_file:
                    #     png_path = img_to_png_path(img_path, instrument_host)
                    #     if os.path.isfile(png_path):
                    #         continue
                    # if (force_has_offset_error or force_has_offset_spice_error or
                    #     force_has_offset_nonspice_error):
                    #     if not _check_for_offset_errors(img_path, instrument_host,
                    #                                     planet,
                    #                                     force_has_offset_error,
                    #                                     force_has_offset_nonspice_error,
                    #                                     force_has_offset_spice_error):
                    #         continue
                    # if selection_expr is not None:
                    #     # User-provided Python code to check the metadata
                    #     metadata = read_offset_metadata(
                    #                 img_path, instrument_host, planet,
                    #                 type_pref='force_plain',
                    #                 overlay=False)
                    #     bootstrap_metadata = read_offset_metadata(
                    #                 img_path, instrument_host, planet,
                    #                 type_pref='force_bootstrap',
                    #                 overlay=False)
                    #     botsim_metadata = read_offset_metadata(
                    #                 img_path, instrument_host, planet,
                    #                 type_pref='force_botsim',
                    #                 overlay=False)
                    #     if metadata is None or not eval(selection_expr):
                    #         continue
                    imagefile = ImageFile(
                        image_file_url=img_url,
                        label_file_url=label_url,
                        index_file_row=row,
                        results_path_stub=self._results_path_stub(
                            search_vol, label_filespec
                        ),
                    )
                    yield imagefile

                    num_yields += 1
                    if limit_yields is not None and num_yields >= limit_yields:
                        return
                    if choose_random_images:
                        # Only return one image before cycling back for a new volume
                        break
                if choose_random_images or done:
                    break
            if not choose_random_images:
                break

    def _check_additional_image_selection_criteria(
        self,
        img_path: str,
        img_name: str,
        img_num: int,
        arguments: Optional[argparse.Namespace] = None,
    ) -> bool:
        """Check additional image selection criteria. Overridden by subclasses.

        Parameters:
            img_path: The path to the image.
            img_name: The name of the image.
            img_num: The number of the image.
            arguments: The parsed arguments.

        Returns:
            True if the image should be processed, False otherwise.
        """

        return True

    def yield_image_files_index(self, **kwargs: Any) -> Iterator[ImageFiles]:
        """Yield filenames given search criteria using index files. Overridden by subclasses.

        Parameters:
            **kwargs: Arbitrary keyword arguments, usually used to restrict the search.

        Yields:
            ImageFiles objects containing information about groups of selected
            image files.
        """

        for imagefile in self._yield_image_files_index(**kwargs):
            yield ImageFiles(image_files=[imagefile])

    @staticmethod
    def supported_grouping() -> list[str]:
        """Returns the list of supported grouping types.

        Returns:
            The list of supported grouping types.
        """
        return []
