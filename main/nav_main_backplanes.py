#!/usr/bin/env python3
################################################################################
# nav_main_backplanes.py
#
# Top-level driver for backplane generation. Enumerates images via datasets and
# for each, generates body and ring backplanes based on prior offset metadata.
################################################################################

import argparse
import os
import sys
from typing import cast

from filecache import FileCache
import pdslogger

from nav.dataset.dataset import DataSet
from nav.dataset import (dataset_names,
                         dataset_name_to_class,
                         dataset_name_to_inst_name)
from nav.config import DEFAULT_CONFIG
from nav.config.logger import DEFAULT_LOGGER
from nav.obs import inst_name_to_obs_class

from nav.backplanes.backplanes import generate_backplanes_image_files


DATASET: DataSet | None = None
DATASET_NAME: str | None = None
MAIN_LOGGER: pdslogger.PdsLogger | None = None


def parse_args(command_list: list[str]) -> argparse.Namespace:
    global DATASET
    global DATASET_NAME

    if len(command_list) < 1:
        print('Usage: python nav_main_backplanes.py <dataset_name> [args]')
        sys.exit(1)

    DATASET_NAME = command_list[0].lower()

    if DATASET_NAME not in dataset_names():
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: python nav_main_backplanes.py <dataset_name> [args]')
        sys.exit(1)

    try:
        DATASET = dataset_name_to_class(DATASET_NAME)()
    except KeyError:
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: python nav_main_backplanes.py <dataset_name> [args]')
        sys.exit(1)

    cmdparser = argparse.ArgumentParser(
        description='Backplanes Main Interface',
        epilog="""Default behavior is to generate body/ring backplanes for selected images
                  using prior navigation offsets stored in metadata_root.""")

    # Environment
    environment_group = cmdparser.add_argument_group('Environment')
    environment_group.add_argument(
        '--config-file', action='append', default=['nav_default_config.yaml'],
        help="""The configuration file(s) to use to override default settings;
        may be specified multiple times (default: ./nav_default_config.yaml)""")
    environment_group.add_argument(
        '--pds3-holdings-root', type=str, default=None,
        help='Root directory of PDS3 holdings; overrides PDS3_HOLDINGS_DIR or config')
    environment_group.add_argument(
        '--nav-results-root', type=str, default=None,
        help="""Root directory for prior navigation metadata files (_metadata.json);
        overrides NAV_RESULTS_ROOT and the nav_results_root configuration variable""")
    environment_group.add_argument(
        '--backplane-results-root', type=str, default=None,
        help="""Root directory for backplane results; overrides the NAV_BACKPLANES_ROOT
        environment variable and the backplane_results_root configuration variable""")

    # Output
    output_group = cmdparser.add_argument_group('Output')
    output_group.add_argument(
        '--dry-run', action='store_true', default=False,
        help="Don't process images, just print what would be done")
    output_group.add_argument(
        '--no-write-output-files', action='store_true', default=False,
        help="Don't write any output files")

    # Dataset selection
    DATASET.add_selection_arguments(cmdparser)

    # Misc
    misc_group = cmdparser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--profile', action='store_true', default=False,
                            help='Enable profiling')

    arguments = cmdparser.parse_args(command_list[1:])
    return arguments


def main() -> None:
    command_list = sys.argv[1:]
    arguments = parse_args(command_list)

    # Read configuration files
    DEFAULT_CONFIG.read_config()
    if arguments.config_file:
        for config_file in arguments.config_file:
            DEFAULT_CONFIG.update_config(config_file)

    # Derive roots
    nav_results_root_str = arguments.nav_results_root
    if nav_results_root_str is None:
        # Allow config override if present
        try:
            nav_results_root_str = DEFAULT_CONFIG.environment.nav_results_root
        except AttributeError:
            pass
    if nav_results_root_str is None:
        raise ValueError('One of --nav-results-root, the configuration variable '
                         '"nav_results_root" or the NAV_RESULTS_ROOT environment variable must be '
                         'set')
    nav_results_root = FileCache('nav_results').new_path(nav_results_root_str)

    backplane_results_root_str = arguments.backplane_results_root
    if backplane_results_root_str is None:
        try:
            backplane_results_root_str = DEFAULT_CONFIG.environment.backplane_results_root
        except AttributeError:
            pass
    if backplane_results_root_str is None:
        backplane_results_root_str = os.getenv('BACKPLANE_RESULTS_ROOT')
    if backplane_results_root_str is None:
        raise ValueError('One of --backplane-results-root, the configuration variable '
                         '"backplane_results_root" or the BACKPLANE_RESULTS_ROOT environment '
                         'variable must be set')
    backplane_results_root = FileCache('nav_results').new_path(backplane_results_root_str)

    global MAIN_LOGGER
    MAIN_LOGGER = DEFAULT_LOGGER

    assert DATASET is not None
    inst_name = dataset_name_to_inst_name(cast(str, DATASET_NAME))
    obs_class = inst_name_to_obs_class(inst_name)

    for imagefiles in DATASET.yield_image_files_from_arguments(arguments):
        assert len(imagefiles.image_files) == 1
        image_path = imagefiles.image_files[0].image_file_path
        if arguments.dry_run:
            MAIN_LOGGER.info('Would process: %s', image_path)
            continue

        ok, metadata = generate_backplanes_image_files(
            obs_class,
            imagefiles,
            nav_results_root=nav_results_root,
            backplane_results_root=backplane_results_root,
            write_output_files=not arguments.no_write_output_files
        )
        if not ok:
            MAIN_LOGGER.info('Skipped: %s (%s)', image_path, metadata.get('status_error', ''))


if __name__ == '__main__':
    main()
