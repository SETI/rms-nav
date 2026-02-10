import argparse
from collections.abc import Iterator
from typing import Any, Optional

from filecache import FCPath

from .dataset import DataSet, ImageFile, ImageFiles
from nav.config import Config
from nav.support.misc import flatten_list


class DataSetSim(DataSet):
    """Dataset that yields a series of JSON-described simulated images.

    Each dataset value is a direct file path to the JSON file; selection simply
    returns that one file.
    """

    @staticmethod
    def _img_name_valid(img_name: str) -> bool:
        # Accept any path ending in .json
        return str(img_name).lower().endswith('.json')

    # Public methods

    def __init__(self, *, config: Optional[Config] = None) -> None:
        """Initializes a simulated dataset.

        Parameters:
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """
        super().__init__(config=config)

    @staticmethod
    def add_selection_arguments(
        cmdparser: argparse.ArgumentParser,
        group: Optional[argparse._ArgumentGroup] = None,
    ) -> None:
        if group is None:
            group = cmdparser.add_argument_group('Image selection (sim-specific)')
        group.add_argument(
            'img_path',
            action='append',
            nargs='*',
            type=str,
            help='Paths to JSON files containing simulated images to process',
        )

    def yield_image_files_from_arguments(
        self, arguments: argparse.Namespace
    ) -> Iterator[ImageFiles]:
        img_path_list = flatten_list(arguments.img_path)
        for img_path in img_path_list:
            json_fcpath = FCPath(img_path)
            imagefile = ImageFile(
                image_file_url=json_fcpath,
                label_file_url=json_fcpath,
                results_path_stub=str(json_fcpath.with_suffix('').name),
            )
            yield ImageFiles(image_files=[imagefile])

    def yield_image_files_index(self, **kwargs: Any) -> Iterator[ImageFiles]:
        raise NotImplementedError(
            'yield_image_files_index is not implemented for sim dataset'
        )

    @staticmethod
    def supported_grouping() -> list[str]:
        return []

    def pds4_bundle_template_dir(self) -> str:
        """Returns absolute path to template directory for PDS4 bundle generation."""
        raise NotImplementedError('Bundle generation not supported for sim dataset')

    def pds4_bundle_name(self) -> str:
        """Returns bundle name for PDS4 bundle generation."""
        raise NotImplementedError('Bundle generation not supported for sim dataset')

    @staticmethod
    def pds4_bundle_path_for_image(image_name: str) -> str:
        """Maps image name to bundle directory path."""
        raise NotImplementedError('Bundle generation not supported for sim dataset')

    def pds4_path_stub(self, image_file: ImageFile) -> str:
        """Returns PDS4 path stub for bundle directory structure."""
        raise NotImplementedError('Bundle generation not supported for sim dataset')

    def pds4_template_variables(
        self,
        image_file: ImageFile,
        nav_metadata: dict[str, Any],
        backplane_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Returns template variables for PDS4 label generation."""
        raise NotImplementedError('Bundle generation not supported for sim dataset')
