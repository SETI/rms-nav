from abc import ABC, abstractmethod
import argparse
from pathlib import Path
from typing import Any, Iterator, Optional

from nav.config import Config
from nav.support.nav_base import NavBase


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
    def image_name_valid(name: str) -> bool:
        """Validates whether the provided image name follows the dataset's naming convention.

        Parameters:
            name: The image name to validate.

        Returns:
            True if the image name is valid for this dataset, False otherwise.
        """
        ...

    @staticmethod
    @abstractmethod
    def add_selection_arguments(*args: Any, **kwargs: Any) -> None:
        """Adds dataset-specific command-line arguments for image selection.

        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        ...

    @abstractmethod
    def yield_filenames_from_arguments(self,
                                       arguments: argparse.Namespace
                                       ) -> Iterator[tuple[Path, Path]]:
        """Yields image filenames based on provided command-line arguments.

        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Yields:
            Paths to the selected files as (label, image) tuples.
        """
        ...

    @abstractmethod
    def yield_filenames_index(**kwargs: Any) -> Iterator[tuple[Path, Path]]:
        """Yields image filenames based on index information.

        Parameters:
            **kwargs: Arbitrary keyword arguments.

        Yields:
            Paths to the selected files as (label, image) tuples.
        """
        ...
