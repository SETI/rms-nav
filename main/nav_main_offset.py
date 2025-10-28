################################################################################
# nav_main_offset.py
#
# This is the main top-level driver for offset finding. It enumerates one or
# more images either from scanning an index file, a holdings directory,
# or from an AWS SQS queue, and for each computes the offset and saves the
# resulting offset and preview image files. TODO
################################################################################

import argparse
import cProfile
# import datetime
import io
# import logging
import os
import pstats
# import subprocess
from pathlib import Path
import sys
import time

from filecache import FileCache

# Add the repository root to the path
# XXX This should eventually go away
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nav.dataset import (DATASET_NAME_TO_CLASS_MAPPING,
                         dataset_name_to_class,
                         dataset_name_to_inst_name)
from nav.config import DEFAULT_CONFIG
from nav.config.logger import DEFAULT_LOGGER
from nav.obs import inst_name_to_obs_class
from nav.navigate_image_files import navigate_image_files

import tkinter  # TODO Change to only install if needed
# import traceback

import spicedb

# import nav.aws
# import nav.config
# import nav.file
# import nav.file_oops
# import nav.gui_offset_data
# import nav.logging_setup
# import nav.misc
# from   nav.nav import Navigation
# import nav.offset
# import nav.web


MAIN_LOG_NAME = 'nav_main_offset'

DATASET = None
DATASET_NAME = None


################################################################################
#
# ARGUMENT PARSING
#
################################################################################

def parse_args(command_list: list[str]) -> argparse.Namespace:
    global DATASET
    global DATASET_NAME

    if len(command_list) < 1:
        print('Usage: python nav_main_offset.py <dataset_name> [args]')
        sys.exit(1)

    DATASET_NAME = command_list[0].lower()

    if not DATASET_NAME in DATASET_NAME_TO_CLASS_MAPPING:
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(DATASET_NAME_TO_CLASS_MAPPING.keys())}')
        print('Usage: python nav_main_offset.py <dataset_name> [args]')
        sys.exit(1)

    DATASET = dataset_name_to_class(DATASET_NAME)()
    try:
        DATASET = dataset_name_to_class(DATASET_NAME)()
    except KeyError:
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(DATASET_NAME_TO_CLASS_MAPPING.keys())}')
        print('Usage: python nav_main_offset.py <dataset_name> [args]')
        sys.exit(1)

    cmdparser = argparse.ArgumentParser(
        description='Navigation & Backplane Main Interface for Offsets',
        epilog="""Default behavior is to perform an offset pass on all Cassini images
                that don't have associated offset files""")

    # Arguments about the environment
    environment_group = cmdparser.add_argument_group('Environment')
    environment_group.add_argument(
        '--config-file', action='append', default=None,
        help="""The configuration file(s) to use to override settings in the
        the default configuration file; may be specified multiple times""")
    environment_group.add_argument(
        '--pds3-holdings-root', type=str, default=None,
        help="""The root directory of the PDS3 holdings; overrides the PDS3_HOLDINGS_DIR
        environment variable and the pds3_holdings_root configuration variable""")
    environment_group.add_argument(
        '--results-root', type=str, default=None,
        help="""The root directory of the results; overrides the NAV_RESULTS_ROOT
        environment variable and the results_root configuration variable""")

    # Arguments about the general navigation process
    nav_group = cmdparser.add_argument_group('Navigation')
    nav_group.add_argument(
        '--force-offset', action='store_true', default=False,
        help='Force offset computation even if the offset file exists')
    nav_group.add_argument(
        '--display-offset-results', action='store_true', default=False,
        help='Graphically display the results of the offset process')
    nav_group.add_argument(
        '--force-offset-amount', type=str, metavar='U,V',
        help='Force the offset to be u,v')
    nav_group.add_argument(
        '--stars-only', action='store_true', default=False,
        help="""Navigate only using stars;
                implies --no-allow-rings --no-allow-moons --no-allow-central-body""")
    nav_group.add_argument(
        '--allow-stars', action=argparse.BooleanOptionalAction, default=True,
        help='Include stars in navigation')
    nav_group.add_argument(
        '--rings-only', action='store_true', default=False,
        help="""Navigate only using rings;
                implies --no-allow-stars --no-allow-moons --no-allow-central-body""")
    nav_group.add_argument(
        '--allow-rings', action=argparse.BooleanOptionalAction, default=True,
        help='Include rings in navigation')
    nav_group.add_argument(
        '--moons-only', action='store_true', default=False,
        help="""Navigate only using moons;
                implies --no-allow-stars --no-allow-rings --no-allow-central-body""")
    nav_group.add_argument(
        '--allow-moons', action=argparse.BooleanOptionalAction, default=True,
        help='Include moons in navigation')
    nav_group.add_argument(
        '--central-planet-only', action='store_true', default=False,
        help="""Navigate only using the central planet (e.g. Saturn, Jupiter);
                implies --no-allow-stars --no-allow-rings --no-allow-moons""")
    nav_group.add_argument(
        '--allow-central-planet', action=argparse.BooleanOptionalAction, default=True,
        help='Include the central planet in navigation')
    nav_group.add_argument(
        '--body-cartographic-data', dest='body_cartographic_data', action='append',
        metavar='BODY=MOSAIC',
        help="""The mosaic providing cartographic data for the given BODY
                (not currently supported)""")
    nav_group.add_argument(
        '--use-predicted-kernels', action=argparse.BooleanOptionalAction, default=False,
        help='Use predicted CK kernels')
    nav_group.add_argument(
        '--use-gapfill-kernels', action=argparse.BooleanOptionalAction, default=False,
        help='Use gapfill kernels')
    nav_group.add_argument(
        '--use-kernel', action='append',
        help='Use specified CK kernel(s)')
    nav_group.add_argument(
        '--use-cassini-nac-wac-offset', action=argparse.BooleanOptionalAction,
        default=True,
        help='Use the computed offset between NAC and WAC frames')

    # Arguments about offset, overlay, and PNG file generation
    output_group = cmdparser.add_argument_group('Output')
    output_group.add_argument(
        '--write-offset-file', action=argparse.BooleanOptionalAction, default=True,
        help='Generate an offset file; no implies --no-overlay-file')
    output_group.add_argument(
        '--write-overlay-file', action=argparse.BooleanOptionalAction, default=True,
        help='Generate an overlay file')
    output_group.add_argument(
        '--write-png-file', action=argparse.BooleanOptionalAction, default=True,
        help='Generate a PNG file')
    output_group.add_argument(
        '--no-write-results', action='store_true', default=False,
        help="""Don't write offset, overlay, or PNG files;
                implies --no-write-offset-file --no-write-overlay-file
                --no-write-png-file""")
    output_group.add_argument(
        '--png-also-bw', action=argparse.BooleanOptionalAction, default=False,
        help='Produce a black and white PNG file along with the color one')
    output_group.add_argument(
        '--png-blackpoint', type=float, default=None,
        help='Set the blackpoint for the PNG file')
    output_group.add_argument(
        '--png-whitepoint', type=float, default=None,
        help='Set the whitepoint for the PNG file')
    output_group.add_argument(
        '--png-gamma', type=float, default=None,
        help='Set the gamma for the PNG file')
    output_group.add_argument(
        '--metadata-label-font', type=str, default=None, metavar='FONTFILE,SIZE',
        help='Set the font for the PNG metadata info')
    output_group.add_argument(
        '--stars-label-font', type=str, default=None, metavar='FONTFILE,SIZE',
        help='Set the font for star labels')
    output_group.add_argument(
        '--rings-label-font', type=str, default=None, metavar='FONTFILE,SIZE',
        help='Set the font for ring labels')
    output_group.add_argument(
        '--bodies-label-font', type=str, default=None, metavar='FONTFILE,SIZE',
        help='Set the font for body labels (moons and central planet)')
    output_group.add_argument(
        '--label-rings-backplane', action=argparse.BooleanOptionalAction, default=False,
        help='Label backplane longitude and radius on ring images')
    output_group.add_argument(
        '--show-star-streaks', action=argparse.BooleanOptionalAction, default=False,
        help='Show star streaks in the overlay and PNG files')
    output_group.add_argument(
        '--dry-run', action='store_true', default=False,
        help="Don't actually process any images or write any output files")

    # Add all the arguments related to selecting files
    DATASET.add_selection_arguments(cmdparser)

    # Misc arguments
    misc_group = cmdparser.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        '--profile', action=argparse.BooleanOptionalAction, default=False,
        help='Do performance profiling')

    arguments = cmdparser.parse_args(command_list[1:])

    return arguments


def exit_processing():
    end_time = time.time()

    # main_logger.info('Total files processed %d', NUM_FILES_PROCESSED)
    # main_logger.info('Total files skipped %d', NUM_FILES_SKIPPED)
    # main_logger.info('Total elapsed time %.2f sec', end_time-START_TIME)

    # if arguments.profile and arguments.max_subprocesses == 0:
    #     pr.disable()
    #     s = io.StringIO()
    #     sortby = 'cumulative'
    #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #     ps.print_stats()
    #     ps.print_callers()
    #     main_logger.info('Profile results:\n%s', s.getvalue())

    # nav.logging_setup.close_main_logging()

    # if (arguments.results_in_s3 and
    #     arguments.main_loglevel.upper() != 'NONE'):
    #     nav.aws.aws_copy_to_s3(main_log_path_local,
    #                            arguments.aws_results_bucket, main_log_path,
    #                            main_logger)
    #     nav.file.safe_remove(main_log_path_local)

    sys.exit(0)


###############################################################################
#
# MAIN
#
###############################################################################

def main():

    command_list = sys.argv[1:]
    arguments = parse_args(command_list)

    if arguments.stars_only:
        arguments.allow_rings = False
        arguments.allow_moons = False
        arguments.allow_central_planet = False
    if arguments.rings_only:
        arguments.allow_stars = False
        arguments.allow_moons = False
        arguments.allow_central_planet = False
    if arguments.moons_only:
        arguments.allow_stars = False
        arguments.allow_rings = False
        arguments.allow_central_planet = False
    if arguments.central_planet_only:
        arguments.allow_stars = False
        arguments.allow_rings = False
        arguments.allow_moons = False
    if arguments.no_write_results:
        arguments.write_offset_file = False
        arguments.write_overlay_file = False
        arguments.write_png_file = False

    if arguments.profile:
        pr = cProfile.Profile()
        pr.enable()

    if arguments.display_offset_results:
        root = tkinter.Tk()
        root.withdraw()

    # Read the default configuration file and then any override files provided
    # on the command line
    DEFAULT_CONFIG.read_config()
    if arguments.config_file:
        for config_file in arguments.config_file:
            DEFAULT_CONFIG.update_config(config_file)

    # Derive the results root
    results_root_str = arguments.results_root
    if results_root_str is None:
        try:
            results_root_str = DEFAULT_CONFIG.environment.results_root
        except AttributeError:
            pass
    if results_root_str is None:
        results_root_str = os.getenv('NAV_RESULTS_ROOT')
    if results_root_str is None:
        raise ValueError('One of configuration variable "results_root" or '
                         'NAV_RESULTS_ROOT environment variable must be set')
    results_root = FileCache('nav_results').new_path(results_root_str)

    # main_log_path = arguments.main_logfile
    # main_log_path_local = main_log_path
    # if (arguments.results_in_s3 and
    #     arguments.main_loglevel.upper() != 'NONE' and
    #     main_log_path is None):
    #     main_log_path_local = '/tmp/mainlog.txt' # For CloudWatch logs
    #     main_log_datetime = datetime.datetime.now().isoformat()[:-7]
    #     main_log_datetime = main_log_datetime.replace(':','-')
    #     main_log_path = 'logs/'+MAIN_LOG_NAME+'/'+main_log_datetime+'-'
    #     if nav.aws.AWS_HOST_INSTANCE_ID is not None:
    #         main_log_path += nav.aws.AWS_HOST_INSTANCE_ID
    #     else:
    #         main_log_path += '-'+str(os.getpid())
    #     main_log_path += '.log'

    main_logger = DEFAULT_LOGGER
    # main_logger, image_logger = nav.logging_setup.setup_main_logging(
    #             MAIN_LOG_NAME, arguments.main_logfile_level,
    #             arguments.main_console_level, main_log_path_local,
    #             arguments.image_logfile_level, arguments.image_console_level)

    # image_loglevel = nav.logging_setup.decode_level(arguments.image_logfile_level)


    global START_TIME, NUM_FILES_PROCESSED, NUM_FILES_SKIPPED, NUM_FILES_COMPLETED
    START_TIME = time.time()
    NUM_FILES_PROCESSED = 0
    NUM_FILES_SKIPPED = 0
    NUM_FILES_COMPLETED = 0

    # main_logger.info('**********************************')
    # main_logger.info('*** BEGINNING MAIN OFFSET PASS ***')
    # main_logger.info('**********************************')
    # main_logger.info('')
    # main_logger.info('Host Local Name: %s', nav.aws.AWS_HOST_FQDN)
    # if nav.aws.AWS_ON_EC2_INSTANCE:
    #     main_logger.info('Host Public Name: %s (%s) in %s', nav.aws.AWS_HOST_PUBLIC_NAME,
    #                      nav.aws.AWS_HOST_PUBLIC_IPV4, nav.aws.AWS_HOST_ZONE)
    #     main_logger.info('Host AMI ID: %s', nav.aws.AWS_HOST_AMI_ID)
    #     main_logger.info('Host Instance Type: %s', nav.aws.AWS_HOST_INSTANCE_TYPE)
    #     main_logger.info('Host Instance ID: %s', nav.aws.AWS_HOST_INSTANCE_ID)
    # main_logger.info('GIT Status:   %s', nav.misc.current_git_version())
    # main_logger.info('')
    # main_logger.info('Command line: %s', ' '.join(command_list))
    # main_logger.info('')
    # main_logger.info('Subprocesses:  %d', arguments.max_subprocesses)
    # main_logger.info('')
    # main_logger.info('Allow stars:   %s', str(arguments.allow_stars))
    # main_logger.info('Allow rings:   %s', str(arguments.allow_rings))
    # main_logger.info('Allow moons:   %s', str(arguments.allow_moons))
    # main_logger.info('Allow planet:  %s', str(arguments.allow_central_planet))
    # main_logger.info('BOTSIM offset: %s', str(force_offset_amount))
    # main_logger.info('Pred kernels:  %s', str(arguments.use_predicted_kernels))
    # if kernel_type == 'none':
    #     main_logger.info('Specific kernels:')
    #     for kernel in arguments.use_kernel:
    #         main_logger.info('  %s', kernel)

    try:
        INST_NAME = dataset_name_to_inst_name(DATASET_NAME)
    except KeyError:
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(DATASET_NAME_TO_CLASS_MAPPING.keys())}')
        sys.exit(1)

    obs_class = inst_name_to_obs_class(INST_NAME)

    for imagefiles in DATASET.yield_image_files_from_arguments(arguments):
        assert len(imagefiles.image_files) == 1
        image_path = imagefiles.image_files[0].image_file_path
        if arguments.dry_run:
            main_logger.info('Would process: %s', image_path)
            continue

        if navigate_image_files(
                obs_class,
                imagefiles,
                results_root=results_root,
                # allow_stars=arguments.allow_stars,
                # allow_rings=arguments.allow_rings,
                # allow_moons=arguments.allow_moons,
                # allow_central_planet=arguments.allow_central_planet,
                # force_offset_amount=force_offset_amount,
                # cartographic_data=cartographic_data,
                # bootstrapped=bootstrapped,
                # loaded_kernel_type=loaded_kernel_type
                ):
            NUM_FILES_PROCESSED += 1
        else:
            NUM_FILES_SKIPPED += 1

    exit_processing()

if __name__ == '__main__':
    main()
