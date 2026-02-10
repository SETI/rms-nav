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

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, package_source_path)

from nav.dataset.dataset import DataSet
from nav.dataset import dataset_names, dataset_name_to_class, dataset_name_to_inst_name
from nav.config import (
    DEFAULT_CONFIG,
    DEFAULT_LOGGER,
    get_backplane_results_root,
    get_nav_results_root,
    load_default_and_user_config,
)
from nav.obs import inst_name_to_obs_class

from backplanes.backplanes import generate_backplanes_image_files


DATASET: DataSet | None = None
DATASET_NAME: str | None = None
MAIN_LOGGER: pdslogger.PdsLogger | None = None


def parse_args(command_list: list[str]) -> argparse.Namespace:
    global DATASET
    global DATASET_NAME

    if len(command_list) < 1:
        print('Usage: nav_main_backplanes <dataset_name> [args]')
        sys.exit(1)

    DATASET_NAME = command_list[0].lower()

    if DATASET_NAME not in dataset_names():
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: nav_main_backplanes <dataset_name> [args]')
        sys.exit(1)

    try:
        DATASET = dataset_name_to_class(DATASET_NAME)()
    except KeyError:
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: nav_main_backplanes <dataset_name> [args]')
        sys.exit(1)

    cmdparser = argparse.ArgumentParser(
        description='Backplanes Main Interface',
        epilog="""Default behavior is to generate body/ring backplanes for selected images
                  using prior navigation offsets stored in metadata_root.""",
    )

    # Environment
    environment_group = cmdparser.add_argument_group('Environment')
    environment_group.add_argument(
        '--config-file',
        action='append',
        default=None,
        help="""The configuration file(s) to use to override default settings;
        may be specified multiple times. If not provided, attempts to load
        ./nav_default_config.yaml if present.""",
    )
    environment_group.add_argument(
        '--pds3-holdings-root',
        type=str,
        default=None,
        help='Root directory of PDS3 holdings; overrides PDS3_HOLDINGS_DIR or config',
    )
    environment_group.add_argument(
        '--nav-results-root',
        type=str,
        default=None,
        help="""Root directory for prior navigation metadata files (_metadata.json);
        overrides NAV_RESULTS_ROOT and the nav_results_root configuration variable""",
    )
    environment_group.add_argument(
        '--backplane-results-root',
        type=str,
        default=None,
        help="""Root directory for backplane results; overrides the BACKPLANE_RESULTS_ROOT
        environment variable and the backplane_results_root configuration variable""",
    )

    # Output
    output_group = cmdparser.add_argument_group('Output')
    output_group.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help="Don't process images, just print what would be done",
    )
    output_group.add_argument(
        '--no-write-output-files',
        action='store_true',
        default=False,
        help="Don't write any output files",
    )

    # Dataset selection
    DATASET.add_selection_arguments(cmdparser)

    # Misc
    misc_group = cmdparser.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        '--profile', action='store_true', default=False, help='Enable profiling'
    )

    arguments = cmdparser.parse_args(command_list[1:])
    return arguments


def main() -> None:
    command_list = sys.argv[1:]
    arguments = parse_args(command_list)

    # Read configuration files
    load_default_and_user_config(arguments, DEFAULT_CONFIG)

    # Derive roots
    nav_results_root_str = get_nav_results_root(arguments, DEFAULT_CONFIG)
    nav_results_root = FileCache(None).new_path(nav_results_root_str)

    backplane_results_root_str = get_backplane_results_root(arguments, DEFAULT_CONFIG)
    backplane_results_root = FileCache(None).new_path(backplane_results_root_str)

    global MAIN_LOGGER
    MAIN_LOGGER = DEFAULT_LOGGER

    MAIN_LOGGER.info('Starting backplanes generation')
    MAIN_LOGGER.info('Dataset: %s', DATASET_NAME)
    MAIN_LOGGER.info('Nav results root: %s', nav_results_root.as_posix())
    MAIN_LOGGER.info('Backplane results root: %s', backplane_results_root.as_posix())
    MAIN_LOGGER.info('Dry run: %s', arguments.dry_run)
    MAIN_LOGGER.info('No write output files: %s', arguments.no_write_output_files)
    MAIN_LOGGER.info('Arguments: %s', command_list)

    assert DATASET is not None
    inst_name = dataset_name_to_inst_name(cast(str, DATASET_NAME))
    obs_class = inst_name_to_obs_class(inst_name)

    for imagefiles in DATASET.yield_image_files_from_arguments(arguments):
        assert len(imagefiles.image_files) == 1
        if arguments.dry_run:
            MAIN_LOGGER.info(
                'Would process: %s', imagefiles.image_files[0].label_file_url.as_posix()
            )
            continue

        try:
            generate_backplanes_image_files(
                obs_class,
                imagefiles,
                nav_results_root=nav_results_root,
                backplane_results_root=backplane_results_root,
                write_output_files=not arguments.no_write_output_files,
            )
        except FileNotFoundError as e:
            MAIN_LOGGER.error(
                'Skipped due to missing metadata: %s (%s)',
                imagefiles.image_files[0].label_file_url.as_posix(),
                str(e),
            )
            continue


if __name__ == '__main__':
    main()
