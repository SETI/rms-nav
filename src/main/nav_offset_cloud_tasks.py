#!/usr/bin/env python3
################################################################################
# nav_main_offset_cloud_tasks.py
#
# This is the main top-level driver for offset finding when the list of images
# to process is provided by the cloud_tasks module.
################################################################################

import argparse
import asyncio
import os
import sys
from typing import Any, cast

from cloud_tasks.worker import Worker, WorkerData
from filecache import FileCache, FCPath

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, package_source_path)

from nav.dataset.dataset import ImageFile, ImageFiles
from nav.dataset import dataset_name_to_inst_name
from nav.config import (DEFAULT_CONFIG,
                        get_nav_results_root,
                        load_default_and_user_config)
from nav.obs import inst_name_to_obs_class
from nav.navigate_image_files import navigate_image_files


def process_task(
    task_id: str, task_data: dict[str, Any], worker_data: WorkerData
) -> tuple[bool, Any]:
    """Navigate a single batch of image files.

    Parameters:
        task_id: The ID of the task.
        task_data: The data for the task. It is expected to contain the following keys:
            - "arguments": The arguments for the task. It is expected to contain the following keys:
                - "nav_models": The models to use for navigation, or None if all models are to be
                  used.
                - "nav_techniques": The techniques to use for navigation, or None if all techniques
                  are to be used.
            - "dataset_name": The name of the dataset.
            - "files": The files to process. It is expected to contain the following keys:
                - "image_file_url": The URL of the image file.
                - "label_file_url": The URL of the label file.
                - "results_path_stub": The path stub for the results.
                - "index_file_row": The row from the index file for the image file.
        worker_data: The data for the worker.

    Returns:
        Tuple of (retry, result)
    """

    arguments = cast(argparse.Namespace, worker_data.args)
    load_default_and_user_config(arguments, DEFAULT_CONFIG)

    try:
        nav_results_root_str = get_nav_results_root(arguments, DEFAULT_CONFIG)
    except ValueError:
        return False, {
            'status': 'error',
            'status_error': 'no_nav_root'
        }
    nav_results_root = FileCache(None).new_path(nav_results_root_str)

    nav_models = task_data.get('arguments', {}).get('nav_models', None)
    nav_techniques = task_data.get('arguments', {}).get('nav_techniques', None)
    dataset_name = task_data.get('dataset_name', None)
    if dataset_name is None:
        return False, {
            'status': 'error',
            'status_error': 'no_dataset_name'
        }
    try:
        inst_name = dataset_name_to_inst_name(dataset_name)
    except KeyError:
        return False, {
            'status': 'error',
            'status_error': 'unknown_dataset',
            'status_exception': f'Unknown dataset "{dataset_name}"'
        }

    obs_class = inst_name_to_obs_class(inst_name)
    files = task_data.get('files', None)
    if files is None:
        return False, {
            'status': 'error',
            'status_error': 'no_files'
        }

    image_files = []
    for file in files:
        image_file_url = file.get('image_file_url', None)
        label_file_url = file.get('label_file_url', None)
        results_path_stub = file.get('results_path_stub', None)
        index_file_row = file.get('index_file_row', None)
        extra_params = file.get('extra_params', {})
        if image_file_url is None:
            return False, {
                'status': 'error',
                'status_error': 'no_image_file_url'
            }
        if label_file_url is None:
            return False, {
                'status': 'error',
                'status_error': 'no_label_file_url'
            }
        if results_path_stub is None:
            return False, {
                'status': 'error',
                'status_error': 'no_results_path_stub'
            }
        image_file = ImageFile(
            image_file_url=FCPath(image_file_url),
            label_file_url=FCPath(label_file_url),
            results_path_stub=results_path_stub,
            index_file_row=index_file_row,
            extra_params=extra_params,
        )
        image_files.append(image_file)

    _, metadata = navigate_image_files(obs_class,
                                       ImageFiles(image_files=image_files),
                                       nav_results_root=nav_results_root,
                                       nav_models=nav_models,
                                       nav_techniques=nav_techniques)

    return False, metadata  # No retry under any circumstances


async def async_main() -> None:
    argparser = argparse.ArgumentParser(
        description='Navigation & Backplane Main Interface for Offsets '
                    'Cloud Tasks version)')

    # Arguments about the environment
    environment_group = argparser.add_argument_group('Environment')
    environment_group.add_argument(
        '--config-file', action='append', default=None,
        help="""The configuration file(s) to use to override default settings;
        may be specified multiple times. If not provided, attempts to load
        ./nav_default_config.yaml if present.""")
    environment_group.add_argument(
        '--nav-results-root', type=str, default=None,
        help="""The root directory of the navigation results; overrides the NAV_RESULTS_ROOT
        environment variable and the nav_results_root configuration variable""")

    worker = Worker(process_task, args=sys.argv[1:], argparser=argparser)
    await worker.start()


def main() -> None:  # Required for setuptools entry points
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
