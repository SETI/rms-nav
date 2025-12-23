#!/usr/bin/env python3
################################################################################
# nav_main_backplanes_cloud_tasks.py
#
# Backplanes generator when image batches are provided by cloud_tasks.
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
from nav.config import DEFAULT_CONFIG
from nav.obs import inst_name_to_obs_class
from backplanes.backplanes import generate_backplanes_image_files


def process_task(
    task_id: str, task_data: dict[str, Any], worker_data: WorkerData
) -> tuple[bool, Any]:
    """Generate backplanes for a single batch of image files."""

    DEFAULT_CONFIG.read_config()
    arguments = cast(argparse.Namespace, worker_data.args)
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
        return False, {
            'status': 'error',
            'status_error': 'no_nav_root'
        }
    nav_results_root = FileCache().new_path(nav_results_root_str)

    backplane_results_root_str = arguments.backplane_results_root
    if backplane_results_root_str is None:
        try:
            backplane_results_root_str = DEFAULT_CONFIG.environment.backplane_results_root
        except AttributeError:
            pass
    if backplane_results_root_str is None:
        backplane_results_root_str = os.getenv('BACKPLANE_RESULTS_ROOT')
    if backplane_results_root_str is None:
        return False, {
            'status': 'error',
            'status_error': 'no_backplane_root'
        }
    backplane_results_root = FileCache().new_path(backplane_results_root_str)

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
        )
        image_files.append(image_file)

    generate_backplanes_image_files(
        obs_class,
        ImageFiles(image_files=image_files),
        nav_results_root=nav_results_root,
        backplane_results_root=backplane_results_root,
        write_output_files=True
    )

    return False, None  # No retry under any circumstances


async def main() -> None:
    argparser = argparse.ArgumentParser(
        description='Backplanes Main Interface (Cloud Tasks version)')

    environment_group = argparser.add_argument_group('Environment')
    environment_group.add_argument(
        '--config-file', action='append', default=None,
        help="""The configuration file(s) to use to override default settings;
        may be specified multiple times. If not provided, attempts to load
        ./nav_default_config.yaml if present.""")
    environment_group.add_argument(
        '--backplane-results-root', type=str, default=None,
        help='Root directory for backplane results; overrides NAV_RESULTS_ROOT or config')
    environment_group.add_argument(
        '--nav-results-root', type=str, default=None,
        help='Root directory for prior navigation results (metadata, offsets)')

    worker = Worker(process_task, args=sys.argv[1:], argparser=argparser)
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
