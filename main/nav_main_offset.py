#!/usr/bin/env python3
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
import os
import sys
import time

from filecache import FileCache, FCPath

from nav.dataset.dataset import DataSet
from nav.dataset import (dataset_names,
                         dataset_name_to_class,
                         dataset_name_to_inst_name)
from nav.config import DEFAULT_CONFIG
from nav.config.logger import DEFAULT_LOGGER
from nav.support.file import json_as_string
from nav.obs import inst_name_to_obs_class
from nav.navigate_image_files import navigate_image_files


DATASET: DataSet | None = None
DATASET_NAME: str | None = None


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

    if not DATASET_NAME in dataset_names():
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: python nav_main_offset.py <dataset_name> [args]')
        sys.exit(1)

    try:
        DATASET = dataset_name_to_class(DATASET_NAME)()
    except KeyError:
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
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
        '--nav-models', type=str, default=None,
        help='Comma-separated list of model names to use')
    nav_group.add_argument(
        '--nav-techniques', type=str, default=None,
        help='Comma-separated list of navigation technique names to use')

    # Arguments about output file generation
    output_group = cmdparser.add_argument_group('Output')
    # output_group.add_argument(
    #     '--write-offset-file', action=argparse.BooleanOptionalAction, default=True,
    #     help='Generate an offset file; no implies --no-overlay-file')
    # output_group.add_argument(
    #     '--write-overlay-file', action=argparse.BooleanOptionalAction, default=True,
    #     help='Generate an overlay file')
    # output_group.add_argument(
    #     '--write-png-file', action=argparse.BooleanOptionalAction, default=True,
    #     help='Generate a PNG file')
    output_group.add_argument(
        '--output-cloud-tasks-file', type=str, default=None,
        help="""Write a JSON file containing task descriptions for all selected images that
        is suitable for loading into a cloud_tasks queue; do not perform any other processing.""")
    output_group.add_argument(
        '--dry-run', action='store_true', default=False,
        help="Don't actually process the images, just print what would be done")
    output_group.add_argument(
        '--no-write-output-files', action='store_true', default=False,
        help="Don't write any output files")

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
    # main_logger.info('Total files processed %d', NUM_FILES_PROCESSED)
    # main_logger.info('Total files skipped %d', NUM_FILES_SKIPPED)

    sys.exit(0)


###############################################################################
#
# MAIN
#
###############################################################################

def main():

    command_list = sys.argv[1:]
    arguments = parse_args(command_list)

    if arguments.profile:
        pr = cProfile.Profile()
        pr.enable()

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

    try:
        INST_NAME = dataset_name_to_inst_name(DATASET_NAME)
    except KeyError:
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        sys.exit(1)

    obs_class = inst_name_to_obs_class(INST_NAME)

    if arguments.output_cloud_tasks_file:
        task_arguments = {
            'nav_models': arguments.nav_models.split(',')
                if arguments.nav_models is not None else None,
            'nav_techniques': arguments.nav_techniques.split(',')
                if arguments.nav_techniques is not None else None,
        }
        tasks_json = []
        for imagefile_idx, imagefiles in enumerate(
                DATASET.yield_image_files_from_arguments(arguments)):
            task_id = f'{DATASET_NAME}-{imagefiles.image_files[0].label_file_name}-{imagefile_idx}'
            task_files = []
            for image_file in imagefiles.image_files:
                task_files.append({
                    'image_file_url': image_file.image_file_url.as_posix(),
                    'label_file_url': image_file.label_file_url.as_posix(),
                    'results_path_stub': image_file.results_path_stub,
                    'index_file_row': image_file.index_file_row,
                })
            task_info = {
                'task_id': task_id,
                'data': {
                    'arguments': task_arguments,
                    'dataset_name': DATASET_NAME,
                    'files': task_files,
                }
            }
            tasks_json.append(task_info)

        cloud_tasks_path = FCPath(arguments.output_cloud_tasks_file)
        with cloud_tasks_path.open('w') as f:
            json_string = json_as_string(tasks_json)
            f.write(json_string)
        sys.exit(0)

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
                nav_models=arguments.nav_models.split(',')
                    if arguments.nav_models is not None else None,
                nav_techniques=arguments.nav_techniques.split(',')
                    if arguments.nav_techniques is not None else None,
                write_output_files=not arguments.no_write_output_files
                ):
            NUM_FILES_PROCESSED += 1
        else:
            NUM_FILES_SKIPPED += 1

    exit_processing()

if __name__ == '__main__':
    main()
