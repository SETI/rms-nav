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
from typing import Any

from cloud_tasks.worker import Worker, WorkerData
from filecache import FileCache, FCPath

from nav.dataset.dataset import ImageFile, ImageFiles
from nav.dataset import dataset_name_to_inst_name
from nav.config import DEFAULT_CONFIG
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

    # Read the default configuration file and then any override files provided
    # on the command line
    DEFAULT_CONFIG.read_config()
    if worker_data.args.config_file:
        for config_file in worker_data.args.config_file:
            DEFAULT_CONFIG.update_config(config_file)

    results_root_str = worker_data.args.results_root
    if results_root_str is None:
        try:
            results_root_str = DEFAULT_CONFIG.environment.results_root
        except AttributeError:
            pass
    if results_root_str is None:
        results_root_str = os.getenv('NAV_RESULTS_ROOT')
    if results_root_str is None:
        return False, (f'{task_id}: One of configuration variable "results_root" or '
                       'NAV_RESULTS_ROOT environment variable must be set')
    results_root = FileCache('nav_results').new_path(results_root_str)

    nav_models = task_data.get('arguments', {}).get('nav_models', None)
    nav_techniques = task_data.get('arguments', {}).get('nav_techniques', None)
    dataset_name = task_data.get('dataset_name', None)
    if dataset_name is None:
        return False, f'{task_id}: "dataset_name" field is required'
    try:
        inst_name = dataset_name_to_inst_name(dataset_name)
    except KeyError:
        return False, f'{task_id}: Unknown dataset "{dataset_name}"'
    obs_class = inst_name_to_obs_class(inst_name)
    files = task_data.get('files', [])
    if files is None:
        return False, f'{task_id}: "files" field is required'
    image_files = []
    for file in files:
        image_file_url = file.get('image_file_url', None)
        label_file_url = file.get('label_file_url', None)
        results_path_stub = file.get('results_path_stub', None)
        index_file_row = file.get('index_file_row', None)
        if image_file_url is None:
            return False, f'{task_id}: "image_file_url" field is required'
        if label_file_url is None:
            return False, f'{task_id}: "label_file_url" field is required'
        if results_path_stub is None:
            return False, f'{task_id}: "results_path_stub" field is required'
        image_file = ImageFile(
            image_file_url=FCPath(image_file_url),
            label_file_url=FCPath(label_file_url),
            results_path_stub=results_path_stub,
            index_file_row=index_file_row,
        )
        image_files.append(image_file)

    result, metadata = navigate_image_files(obs_class,
                                            ImageFiles(image_files=image_files),
                                            results_root=results_root,
                                            nav_models=nav_models,
                                            nav_techniques=nav_techniques)

    if result:
        return False, 'Success'

    return


async def main():
    argparser = argparse.ArgumentParser(
        description='Navigation & Backplane Main Interface for Offsets '
                    'Cloud Tasks version)')

    # Arguments about the environment
    environment_group = argparser.add_argument_group('Environment')
    environment_group.add_argument(
        '--config-file', action='append', default=None,
        help="""The configuration file(s) to use to override settings in the
        the default configuration file; may be specified multiple times""")
    environment_group.add_argument(
        '--results-root', type=str, default=None,
        help="""The root directory of the results; overrides the NAV_RESULTS_ROOT
        environment variable and the results_root configuration variable""")

    worker = Worker(process_task, args=sys.argv[1:], argparser=argparser)
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
