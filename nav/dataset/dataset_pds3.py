import argparse
import csv
import os
from pathlib import Path
import random
import re

from typing import Optional

from filecache import FCPath, FileCache
from pdstable import PdsTable

from .dataset import DataSet


_INDEX_FILECACHE = FileCache('nav_pds3_index')


class DataSetPDS3(DataSet):

    def __init__(self):
        ...

    @staticmethod
    def image_name_valid(name: str) -> bool:
        """True if an image name is valid for this instrument."""

        assert NotImplementedError

    def add_selection_arguments(cmdparser: argparse.ArgumentParser,
                                group: Optional[str] = None) -> None:

        if group is None:
            group = cmdparser.add_argument_group('Image selection')
        group.add_argument(
            'image_name', action='append', nargs='*', type=str,
            help='Specific image name(s) to process')
        # group.add_argument(
        #     '--planet', default='saturn',
        #     type=_validate_planet,
        #     help=f"""Which planet to process: jupiter, saturn, uranus, neptune
        #              (saturn is the default)""")
        group.add_argument(
            '--first-image-num', type=int, default=None, metavar='IMAGE_NUM',
            help='The starting image number')
        group.add_argument(
            '--last-image-num', type=int, default=None, metavar='IMAGE_NUM',
            help='The ending image number')
        group.add_argument(
            '--camera-type', type=str, default=None,
            help='Only process images with the given camera type')
        group.add_argument(
            '--volume', action='append',
            help='An entire PDS3 volume or volume/range_subdir')
        group.add_argument(
            '--first-volume-num', type=int, default=None, metavar='VOL_NUM',
            help='The starting PDS3 volume number')
        group.add_argument(
            '--last-volume-num', type=int, default=None, metavar='VOL_NUM',
            help='The ending PDS3 volume number')
        group.add_argument(
            '--image-full-path', action='append', nargs='*',
            help='The full path for an image')
        group.add_argument(
            '--image-pds-csv', action='append',
            help="""A CSV file downloaded from PDS that contains filespecs of images
                    to process""")
        group.add_argument(
            '--image-filelist', action='append',
            help="""A file that contains image names of images to process""")
        group.add_argument(
            '--strict-file-order', action='store_true', default=False,
            help="""With --image-filelist or --image-pds-csv, return filename in
                    the order in the file, not numerical order""")
        group.add_argument(
            '--has-offset-file', action='store_true', default=False,
            help='Only process images that already have an offset file')
        group.add_argument(
            '--has-no-offset-file', action='store_true', default=False,
            help='Only process images that don\'t already have an offset file')
        group.add_argument(
            '--has-png-file', action='store_true', default=False,
            help='Only process images that already have a PNG file')
        group.add_argument(
            '--has-no-png-file', action='store_true', default=False,
            help='Only process images that don\'t already have a PNGfile')
        group.add_argument(
            '--has-offset-error', action='store_true', default=False,
            help="""Only process images if the offset file exists and
                    indicates a fatal error""")
        group.add_argument(
            '--has-offset-nonspice-error', action='store_true', default=False,
            help="""Only process images if the offset file exists and
                    indicates a fatal error other than missing SPICE data""")
        group.add_argument(
            '--has-offset-spice-error', action='store_true', default=False,
            help="""Only process images if the offset file exists and
                    indicates a fatal error from missing SPICE data""")
        group.add_argument(
            '--selection-expr', type=str, metavar='EXPR',
            help='Expression to evaluate to decide whether to reprocess an offset')
        group.add_argument(
            '--choose-random-images', type=int, default=None, metavar='N',
            help='Choose random images to process within other constraints')
        group.add_argument(
            '--show-image-list-only', action='store_true', default=False,
            help="""Just show a list of files that would be processed without doing
                    any actual processing"""
        )

    def _validate_selection_arguments(self, arguments):
        """Validate the user arguments that can't be checked during initial parsing."""
        if arguments.camera_type is not None:
            if arguments.camera_type.upper() not in ('NAC', 'WAC'):
                raise argparse.ArgumentTypeError('--camera-type must be NAC or WAC')
        # arguments.image_name is a list of lists
        for image_name in _flatten(arguments.image_name):
            if not self.valid_image_name(image_name, arguments.instrument_host):
                raise argparse.ArgumentTypeError(
                    f'Invalid image name {image_name} with instrument host '
                    f'{arguments.instrument_host}')

# def log_arguments(arguments, log):
#     """Log the details of the command line arguments."""
#     log('*** Results root directory:  %s', CB_RESULTS_ROOT)
#     log('*** Instrument host:         %s', arguments.instrument_host)
#     if arguments.image_full_path:
#         log('*** Images explicitly from full paths:')
#         for image_path in arguments.image_full_path:
#             log('        %s', image_path)
#     log('*** Image #s:                %010d - %010d',
#         arguments.first_image_num,
#         arguments.last_image_num)
#     try:
#         v1 = '%04d' % arguments.first_volume_num
#     except TypeError:
#         v1 = 'None'
#     try:
#         v2 = '%04d' % arguments.last_volume_num
#     except TypeError:
#         v2 = 'None'
#     log('*** Volume #s:               %s - %s', v1, v2)
#     log('*** NAC only:                %s', arguments.nac_only)
#     log('*** WAC only:                %s', arguments.wac_only)
#     log('*** Already has offset file: %s', arguments.has_offset_file)
#     log('*** Has no offset file:      %s', arguments.has_no_offset_file)
#     log('*** Already has PNG file:    %s', arguments.has_png_file)
#     log('*** Has no PNG file:         %s', arguments.has_no_png_file)
#     if (arguments.image_name is not None and arguments.image_name != [] and
#         arguments.image_name[0] != []):
#         log('*** Images restricted to list:')
#         for filename in arguments.image_name[0]:
#             log('        %s', filename)
#     if arguments.volume is not None and arguments.volume != []:
#         log('*** Images restricted to volumes:')
#         for volume in arguments.volume:
#             for vol in volume.split(','):
#                 log('        %s', vol)
#     if arguments.image_pds_csv:
#         log('*** Images restricted to those from PDS CSV:')
#         for filename in arguments.image_pds_csv:
#             log('        %s', filename)
#     if arguments.image_filelist:
#         log('*** Images restricted to those from file:')
#         for filename in arguments.image_filelist:
#             log('        %s', filename)

# def yield_image_filenames_from_arguments(arguments, use_index_files=False,
#                                          combine_botsim=False):
#     """Given parsed arguments, yield all selected filenames.

#     arguments               The parsed arguments structure.
#     use_index_files         If True, look up images in the index files instead
#                             of directly on the disk.
#     combine_botsim          If True, look for adjacent NAC/WAC images with the
#                             same image numbers and return them together as a
#                             tuple.
#     """
#     instrument_host_config = INSTRUMENT_HOST_CONFIG[arguments.instrument_host]

#     if arguments.image_full_path:
#         # An explicit list of image paths overrides all other selectors.
#         assert not combine_botsim
#         if len(arguments.image_full_path) > 0:
#             for image_path in arguments.image_full_path[0]:
#                 yield image_path
#             return

#     # Start with wanting all images
#     restrict_image_list = []

#     # Limit to the user-specific list of images, if any
#     if arguments.image_name is not None and _flatten(arguments.image_name):
#         restrict_image_list = [x.upper() for x in _flatten(arguments.image_name)]

#     # Also limit to the list of images in the PDS CSV file, if any
#     if arguments.image_pds_csv:
#         for filename in arguments.image_pds_csv:
#             with open(filename, 'r') as csvfile:
#                 csvreader = csv.reader(csvfile)
#                 header = next(csvreader)
#                 for colnum in range(len(header)):
#                     if (header[colnum] == 'Primary File Spec' or
#                         header[colnum] == 'primaryfilespec'):
#                         break
#                 else:
#                     print('Badly formatted CSV file - no Primary File Spec header',
#                           filename)
#                     sys.exit(-1)
#                 for row in csvreader:
#                     filespec = row[colnum]
#                     # XXX Should come from INSTRUMENT_HOST_CONFIG
#                     filespec = (filespec
#                                 .replace('.IMG', '')
#                                 .replace('_CALIB', '')
#                                 .replace('_RAW', '')
#                                 .replace('_GEOMED', '')
#                                 .replace('.LBL', '')
#                                )
#                     _, filespec = os.path.split(filespec)
#                     restrict_image_list.append(filespec)

#     # Also limit to the list of images in the filelist file, if any
#     if arguments.image_filelist:
#         for filename in arguments.image_filelist:
#             with open(filename, 'r') as fp:
#                 for line in fp:
#                     line = line.strip()
#                     if len(line) == 0 or line[0] == '#':
#                         continue
#                     # Ignore anything after the filename
#                     line = line.split(' ')[0]
#                     if not valid_image_name(line, arguments.instrument_host):
#                         print('Bad filename for instrument host '
#                               f'{arguments.instrument_host} - {line}')
#                         sys.exit(-1)
#                     restrict_image_list.append(line)

#     # Normally we walk through the directory structure or the index file to
#     # find images, then see if they're in the restrict_image_list. But this
#     # yields images in filesystem order, not the order specified by the user.
#     # If the user cares, we have to yield these images directly, so create new
#     # arguments that look like the user just provided the list of filenames
#     # directly.
#     if arguments.strict_file_order:
#         new_arguments = copy.deepcopy(arguments)
#         new_arguments.strict_order = False
#         new_arguments.image_filelist = None
#         new_arguments.image_pds_csv = None
#         for filename in restrict_image_list:
#             new_arguments.image_name = [[filename]]
#             for ret in yield_image_filenames_from_arguments(
#                                           new_arguments,
#                                           use_index_files=use_index_files,
#                                           combine_botsim=combine_botsim):
#                 yield ret
#         return

#     # Works for both Cassini and Voyager
#     restrict_camera = None
#     if arguments.instrument_host in ('cassini', 'voyager'):
#         restrict_camera = 'NW'
#         if arguments.nac_only:
#             restrict_camera = 'N'
#         if arguments.wac_only:
#             restrict_camera = 'W'

#     first_image_number = arguments.first_image_num
#     last_image_number = arguments.last_image_num
#     first_volume_number = arguments.first_volume_num
#     last_volume_number = arguments.last_volume_num
#     volumes = None
#     if arguments.volume:
#         volumes = [x for y in arguments.volume for x in y.split(',')]

#     # What part of the filename do we look at to get the image number?
#     img_lim_start = instrument_host_config['filename_image_number_start']
#     img_lim_end = instrument_host_config['filename_image_number_end']

#     if restrict_image_list:
#         first_image_number = max(first_image_number,
#                                  min([int(x[img_lim_start:img_lim_end])
#                                       for x in restrict_image_list]))
#         last_image_number = min(last_image_number,
#                                 max([int(x[img_lim_start:img_lim_end])
#                                      for x in restrict_image_list]))

#     if use_index_files:
#         yield_function = yield_image_filenames_index
#     else:
#         yield_function = yield_image_filenames

#     last_image_name = None
#     last_image_path = None

#     for image_path in yield_function(
#                 first_image_number,
#                 last_image_number,
#                 first_volume_number,
#                 last_volume_number,
#                 volumes,
#                 restrict_camera,
#                 restrict_image_list,
#                 force_has_offset_file=arguments.has_offset_file,
#                 force_has_no_offset_file=arguments.has_no_offset_file,
#                 force_has_png_file=arguments.has_png_file,
#                 force_has_no_png_file=arguments.has_no_png_file,
#                 force_has_offset_error=arguments.has_offset_error,
#                 force_has_offset_spice_error=arguments.has_offset_spice_error,
#                 force_has_offset_nonspice_error=arguments.has_offset_nonspice_error,
#                 selection_expr=arguments.selection_expr,
#                 choose_random_images=arguments.choose_random_images,
#                 instrument_host=arguments.instrument_host,
#                 planet=arguments.planet):
#         # Before returning a matching image, see if we need to combine BOTSIM
#         # images. We do this by looking at adjacent pairs of returned images to
#         # see if they match.
#         _, image_name = os.path.split(image_path)
#         image_name = image_name[img_lim_start:img_lim_end]
#         if (combine_botsim and
#             last_image_name is not None and
#             last_image_name[0] == 'N' and
#             image_name[0] == 'W' and
#             image_name[1:] == last_image_name[1:]):
#             yield (last_image_path, image_path)
#             last_image_path = None
#             last_image_name = None
#         else:
#             if last_image_path is not None:
#                 if combine_botsim:
#                     yield (last_image_path, None)
#                 else:
#                     yield last_image_path
#             last_image_path = image_path
#             last_image_name = image_name

#     if last_image_path is not None:
#         if combine_botsim:
#             yield (last_image_path, None)
#         else:
#             yield last_image_path


    def yield_image_filenames_index(self,
                                    img_start_num: Optional[int] = None,
                                    img_end_num: Optional[int] = None,
                                    vol_start: Optional[str] = None,
                                    vol_end: Optional[str] = None,
                                    volumes: Optional[list[int]] = None,
                                    camera: Optional[str] = None,
                                    restrict_list: Optional[list[str]] = None,
                                    force_has_offset_file: bool = False,
                                    force_has_no_offset_file: bool = False,
                                    force_has_png_file: bool = False,
                                    force_has_no_png_file: bool = False,
                                    force_has_offset_error: bool = False,
                                    force_has_offset_spice_error: bool = False,
                                    force_has_offset_nonspice_error: bool = False,
                                    selection_expr: Optional[str] = None,
                                    choose_random_images: bool | int = False,
                                    volume_raw_dir: Optional[str | Path | FCPath] = None,
                                    index_dir: Optional[str | Path | FCPath] = None,
                                    suffix: Optional[str] = None,
                                    planets: Optional[str] = None):
        """Yield filenames given search criteria using index files.

        This function assumes that the dataset is in a set of PDS3 volumes laid out like
        the PDS Ring-Moon Systems Node archive:

            $(PDS3_HOLDINGS_DIR)/volumes/{volume_set}/{volume}/
            {sub_dirs}/{image}.[IMG,LBL]

        and that the index files are laid out as:

            $(PDS3_HOLDINGS_DIR)/metadata/(volume set)/(volume)/{volume}_index.[lbl,tab]

        """

        # logger = logging.getLogger(main_module_name() +
        #                         '.yield_image_filenames_index')

        # instrument_host_config = INSTRUMENT_HOST_CONFIG[instrument_host]

        dataset_layout = self._DATASET_LAYOUT
        all_volume_nums = dataset_layout['all_volume_nums']
        is_valid_volume_name = dataset_layout['is_valid_volume_name']
        extract_image_number = dataset_layout['extract_image_number']
        parse_filespec = dataset_layout['parse_filespec']
        volset_and_volume = dataset_layout['volset_and_volume']
        volume_to_index = dataset_layout['volume_to_index']

        if volumes is not None:
            for vol in volumes:
                if not is_valid_volume_name(vol):
                    raise ValueError(f'Illegal volume name: {vol}')
            valid_volume_nums = [v for v in volumes if v in all_volume_nums]
        else:
            valid_volume_nums = all_volume_nums

        valid_volume_nums = [v for v in valid_volume_nums
                             if ((vol_start is not None and vol_start <= v) and
                                 (vol_end is not None and v <= vol_end))]
        valid_volumes = [v for v in valid_volume_nums]

        if volume_raw_dir is None and index_dir is None:
            pds3_holdings_dir = os.getenv('PDS3_HOLDINGS_DIR')
            if pds3_holdings_dir is None:
                raise ValueError('PDS3_HOLDINGS_DIR environment variable not set')
            pds3_holdings_dir = _INDEX_FILECACHE.new_path(pds3_holdings_dir)

        if volume_raw_dir is None:
            volume_raw_dir = pds3_holdings_dir / 'volumes'
        else:
            volume_raw_dir = FCPath(volume_raw_dir)

        if index_dir is None:
            index_dir = pds3_holdings_dir / 'metadata'
        else:
            index_dir = FCPath(index_dir)

        # When yielding via an index, we don't get to optimize searching for
        # offset/png files. We just always look through the index files, and then
        # check out the offset/png files later.
        # This could definitely be improved. TODO

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

        limit_yields = choose_random_images if choose_random_images else None
        cur_yields = 0
        while True:
            done = False
            if choose_random_images:
                # Random -> shuffle valid_volumes since we only take the first one
                random.shuffle(valid_volumes)
            for search_vol in valid_volumes:
                index_label_abspath = index_dir / volume_to_index(search_vol)
                index_tab_abspath = index_label_abspath.with_suffix('.tab')
                # This will raise a FileNotFoundError if the index file label or table
                # can't be found
                _INDEX_FILECACHE.retrieve([index_label_abspath, index_tab_abspath])

                # if search_volume_path is not None:
                #     search_vol_fulldir = clean_join(search_volume_path, search_vol)
                #     # If we require an offset/png file but the directory is missing,
                #     # there's no point in looking further
                #     if not os.path.isdir(search_vol_fulldir):
                #         continue

                index_tab = PdsTable(index_label_abspath.get_local_path())
                filespecs = index_tab.get_column('FILE_SPECIFICATION_NAME')
                if choose_random_images:
                    random.shuffle(filespecs)
                for filespec in filespecs:
                    img_name = parse_filespec(filespec, volumes, search_vol,
                                              index_tab_abspath)
                    if img_name is None:
                        continue

                    # if raw_suffix is not None:
                    #     img_name = img_name.replace(raw_suffix, suffix)
                    # There's no point in checking the range dir for image number
                    # since we have to go through the entire index either way
                    img_num = extract_image_number(img_name)
                    if img_end_num is not None and img_num > img_end_num:
                        # Images are in monotonically increasing order, so we can just
                        # quit now
                        done = True
                        break
                    if img_start_num is not None and img_num < img_start_num:
                        continue
                    if camera is not None and img_name[0] not in camera:
                        continue
                    # if (restrict_list and
                    #     img_name[:img_name.find(suffix)] not in restrict_list and
                    #     img_name[img_lim_start:img_lim_end] not in restrict_list):
                    #     continue
                    img_path = volume_raw_dir / search_vol / filespec
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
                    yield img_path
                    cur_yields += 1
                    if limit_yields is not None and cur_yields >= limit_yields:
                        return
                    if choose_random_images:
                        # Only return one image before cycling back for a new volume
                        break
                if choose_random_images or done:
                    break
            if not choose_random_images:
                break
