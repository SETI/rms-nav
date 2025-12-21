#!/usr/bin/env python3
################################################################################
# nav_create_bundle.py
#
# Top-level driver for PDS4 bundle generation. Enumerates images via datasets
# and for each, generates PDS4 labels and metadata files. Also supports
# generating collection files and global index files.
################################################################################

import argparse
import os
import sys

from filecache import FileCache, FCPath
import pdslogger

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, package_source_path)

from nav.dataset.dataset import DataSet
from nav.dataset import dataset_names, dataset_name_to_class
from nav.config import DEFAULT_CONFIG
from nav.config.logger import DEFAULT_LOGGER

from pds4.data import generate_bundle_data_files
from pds4.collections import generate_collection_files, generate_global_index_files


DATASET: DataSet | None = None
DATASET_NAME: str | None = None
MAIN_LOGGER: pdslogger.PdsLogger | None = None


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to an argument parser.

    Parameters:
        parser: The argument parser to add arguments to.
    """
    environment_group = parser.add_argument_group('Environment')
    environment_group.add_argument(
        '--config-file', action='append', default=None,
        help="""The configuration file(s) to use to override default settings;
        may be specified multiple times. If not provided, attempts to load
        ./nav_default_config.yaml if present.""")
    environment_group.add_argument(
        '--bundle-results-root', type=str, default=None,
        required=False,
        help="""Root directory for bundle results; overrides the BUNDLE_RESULTS_ROOT
        environment variable and the bundle_results_root configuration variable""")


def parse_args_labels(command_list: list[str]) -> argparse.Namespace:
    """Parse arguments for the labels subcommand."""
    global DATASET
    global DATASET_NAME

    if len(command_list) < 1:
        print('Usage: python nav_create_bundle.py labels <dataset_name> [args]')
        sys.exit(1)

    DATASET_NAME = command_list[0].lower()

    if DATASET_NAME not in dataset_names():
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: python nav_create_bundle.py labels <dataset_name> [args]')
        sys.exit(1)

    try:
        DATASET = dataset_name_to_class(DATASET_NAME)()
    except KeyError:
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: python nav_create_bundle.py labels <dataset_name> [args]')
        sys.exit(1)

    cmdparser = argparse.ArgumentParser(
        description='PDS4 Bundle Generation - Labels',
        epilog="""Generate PDS4 labels and metadata files for selected images.""")

    # Common arguments
    add_common_arguments(cmdparser)

    # Environment (additional arguments specific to labels)
    environment_group = cmdparser.add_argument_group('Environment')
    environment_group.add_argument(
        '--pds3-holdings-root', type=str, default=None,
        help='Root directory of PDS3 holdings; overrides PDS3_HOLDINGS_DIR or config')
    environment_group.add_argument(
        '--nav-results-root', type=str, default=None,
        help="""Root directory for prior navigation metadata files (_metadata.json);
        overrides NAV_RESULTS_ROOT and the nav_results_root configuration variable""")
    environment_group.add_argument(
        '--backplane-results-root', type=str, default=None,
        help="""Root directory for backplane results; overrides the BACKPLANE_RESULTS_ROOT
        environment variable and the backplane_results_root configuration variable""")

    # Output
    output_group = cmdparser.add_argument_group('Output')
    output_group.add_argument(
        '--dry-run', action='store_true', default=False,
        help="Don't process images, just print what would be done")

    # Dataset selection
    DATASET.add_selection_arguments(cmdparser)

    arguments = cmdparser.parse_args(command_list[1:])
    return arguments


def parse_args_summary(command_list: list[str]) -> argparse.Namespace:
    """Parse arguments for the summary subcommand."""
    cmdparser = argparse.ArgumentParser(
        description='PDS4 Bundle Generation - Summary',
        epilog="""Generate collection files and global index files for completed bundle.""")

    add_common_arguments(cmdparser)

    # Dataset selection (needed to get dataset instance)
    environment_group = cmdparser.add_argument_group('Environment')
    environment_group.add_argument(
        '--dataset-name', type=str, default=None,
        required=True,
        help='Dataset name (e.g., coiss)')

    arguments = cmdparser.parse_args(command_list)
    return arguments


def derive_bundle_results_root(arguments: argparse.Namespace) -> FCPath:
    """Derive bundle results root with priority: command line > env > config.

    Parameters:
        arguments: Parsed arguments.

    Returns:
        FCPath for bundle results root.

    Raises:
        ValueError: If bundle_results_root cannot be determined.
    """
    bundle_results_root_str = arguments.bundle_results_root
    if bundle_results_root_str is None:
        try:
            bundle_results_root_str = DEFAULT_CONFIG.environment.bundle_results_root
        except AttributeError:
            pass
    if bundle_results_root_str is None:
        bundle_results_root_str = os.getenv('BUNDLE_RESULTS_ROOT')
    if bundle_results_root_str is None:
        raise ValueError('One of --bundle-results-root, the configuration variable '
                         '"bundle_results_root" or the BUNDLE_RESULTS_ROOT environment '
                         'variable must be set')
    return FileCache('bundle_results').new_path(bundle_results_root_str)


def main_labels() -> None:
    """Main function for labels subcommand."""
    command_list = sys.argv[2:]  # Skip 'labels'
    arguments = parse_args_labels(command_list)

    # Read configuration files
    DEFAULT_CONFIG.read_config()
    if arguments.config_file:
        for config_file in arguments.config_file:
            DEFAULT_CONFIG.update_config(config_file)
    else:
        try:
            DEFAULT_CONFIG.update_config('nav_default_config.yaml')
        except FileNotFoundError:
            pass

    # Derive roots
    nav_results_root_str = arguments.nav_results_root
    if nav_results_root_str is None:
        try:
            nav_results_root_str = DEFAULT_CONFIG.environment.nav_results_root
        except AttributeError:
            pass
    if nav_results_root_str is None:
        nav_results_root_str = os.getenv('NAV_RESULTS_ROOT')
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
    backplane_results_root = FileCache('backplane_results').new_path(backplane_results_root_str)

    bundle_results_root = derive_bundle_results_root(arguments)

    global MAIN_LOGGER
    MAIN_LOGGER = DEFAULT_LOGGER

    assert DATASET is not None

    for imagefiles in DATASET.yield_image_files_from_arguments(arguments):
        assert len(imagefiles.image_files) == 1
        if arguments.dry_run:
            MAIN_LOGGER.info('Would process: %s',
                             imagefiles.image_files[0].label_file_url.as_posix())
            continue

        ok, metadata = generate_bundle_data_files(
            dataset=DATASET,
            image_files=imagefiles,
            nav_results_root=nav_results_root,
            backplane_results_root=backplane_results_root,
            bundle_results_root=bundle_results_root,
            logger=MAIN_LOGGER,
            write_output_files=not arguments.dry_run,
        )
        if not ok:
            MAIN_LOGGER.info('Skipped: %s (%s)',
                             imagefiles.image_files[0].label_file_url.as_posix(),
                             metadata.get('status_error', ''))


def main_summary() -> None:
    """Main function for summary subcommand."""
    command_list = sys.argv[2:]  # Skip 'summary'
    arguments = parse_args_summary(command_list)

    # Read configuration files
    DEFAULT_CONFIG.read_config()
    if arguments.config_file:
        for config_file in arguments.config_file:
            DEFAULT_CONFIG.update_config(config_file)
    else:
        try:
            DEFAULT_CONFIG.update_config('nav_default_config.yaml')
        except FileNotFoundError:
            pass

    bundle_results_root = derive_bundle_results_root(arguments)

    global MAIN_LOGGER
    MAIN_LOGGER = DEFAULT_LOGGER

    # Get dataset instance
    dataset_name = arguments.dataset_name.lower()
    if dataset_name not in dataset_names():
        print(f'Unknown dataset "{dataset_name}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        sys.exit(1)

    try:
        dataset = dataset_name_to_class(dataset_name)()
    except KeyError:
        print(f'Unknown dataset "{dataset_name}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        sys.exit(1)

    # Generate collection files
    ok, metadata = generate_collection_files(
        bundle_results_root=bundle_results_root,
        dataset=dataset,
        logger=MAIN_LOGGER,
    )
    if not ok:
        MAIN_LOGGER.error('Failed to generate collection files: %s',
                          metadata.get('status_exception', 'unknown error'))
        sys.exit(1)

    # Generate global index files
    ok, metadata = generate_global_index_files(
        bundle_results_root=bundle_results_root,
        dataset=dataset,
        logger=MAIN_LOGGER,
    )
    if not ok:
        MAIN_LOGGER.error('Failed to generate global index files: %s',
                          metadata.get('status_exception', 'unknown error'))
        sys.exit(1)

    MAIN_LOGGER.info('Summary generation complete')


def main() -> None:
    """Main entry point with subparsers."""
    if len(sys.argv) < 2:
        print('Usage: python nav_create_bundle.py <labels|summary> [args]')
        sys.exit(1)

    subcommand = sys.argv[1].lower()

    if subcommand == 'labels':
        main_labels()
    elif subcommand == 'summary':
        main_summary()
    else:
        print(f'Unknown subcommand "{subcommand}"')
        print('Usage: python nav_create_bundle.py <labels|summary> [args]')
        sys.exit(1)


if __name__ == '__main__':
    main()
