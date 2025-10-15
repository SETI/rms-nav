from abc import ABC, abstractmethod
import argparse
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, cast

from filecache import FCPath

from nav.config import Config
from nav.support.nav_base import NavBase


@dataclass
class ImageFile:
    image_file_url: FCPath
    label_file_url: FCPath
    results_path_stub: Path
    # Convert this to use a default factory
    index_file_row: dict[str, Any] = field(default_factory=dict)
    _image_file_path: Optional[Path] = None
    _label_file_path: Optional[Path] = None

    @property
    def image_file_name(self) -> str:
        return self.image_file_url.name

    @property
    def label_file_name(self) -> str:
        return self.label_file_url.name

    @property
    def image_file_path(self) -> Path:
        if self._image_file_path is None:
            self._image_file_path = cast(Path, self.image_file_url.retrieve())
        return self._image_file_path

    @property
    def label_file_path(self) -> Path:
        if self._label_file_path is None:
            self._label_file_path = cast(Path, self.label_file_url.retrieve())
        return self._label_file_path


@dataclass
class ImageFiles:
    image_files: list[ImageFile]


class DataSet(ABC, NavBase):
    def __init__(self,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:
        """Initializes a dataset with configuration and logging options.

        Parameters:
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
            logger_name: Name for the logger. If None, uses class name.
        """

        super().__init__(config=config, logger_name=logger_name)

    @staticmethod
    @abstractmethod
    def _img_name_valid(img_name: str) -> bool:
        """Validates whether the provided image name follows the dataset's naming convention.

        Parameters:
            img_name: The image name to validate.

        Returns:
            True if the image name is valid for this dataset, False otherwise.
        """
        ...

    @staticmethod
    @abstractmethod
    def add_selection_arguments(cmdparser: argparse.ArgumentParser,
                                group: Optional[argparse._ArgumentGroup] = None) -> None:
        """Adds dataset-specific command-line arguments for image selection.

        Parameters:
            cmdparser: The argument parser to add arguments to.
            group: Optional argument group to add arguments to. If None, creates a new group.
        """
        ...

    @abstractmethod
    def yield_image_files_from_arguments(self,
                                         arguments: argparse.Namespace
                                         ) -> Iterator[ImageFiles]:
        """Yields image filenames based on provided command-line arguments.

        Parameters:
            arguments: The parsed arguments structure.

        Yields:
            Paths to the selected files as (label, image) tuples.
        """
        ...

    @abstractmethod
    def yield_image_files_index(self,
                                **kwargs: Any) -> Iterator[ImageFiles]:
        """Yields image filenames based on index information.

        Parameters:
            **kwargs: Arbitrary keyword arguments, usually used to restrict the search.

        Yields:
            Paths to the selected files as (label, image) tuples.
        """
        ...

    @staticmethod
    @abstractmethod
    def supported_grouping() -> list[str]:
        """Returns the list of supported grouping types.

        Returns:
            The list of supported grouping types.
        """
        ...
