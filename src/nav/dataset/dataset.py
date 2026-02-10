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
    """Represents a single image file with its metadata and lazy-loaded paths.

    Attributes:
        image_file_url: Remote URL for the image file.
        label_file_url: Remote URL for the label file.
        results_path_stub: Local path stub for storing results.
        index_file_row: Optional metadata from index files.
        extra_params: Optional extra parameters that will be passed to the observation
            class's from_file method when the file is read.
    """

    image_file_url: FCPath
    label_file_url: FCPath
    results_path_stub: str
    # Convert this to use a default factory
    index_file_row: dict[str, Any] = field(default_factory=dict)
    extra_params: dict[str, Any] = field(default_factory=dict)
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
            self._image_file_path = cast(Path, self.image_file_url.get_local_path())
        return self._image_file_path

    def retrieve_image_file(self) -> Path:
        return cast(Path, self.image_file_url.retrieve())

    @property
    def label_file_path(self) -> Path:
        if self._label_file_path is None:
            self._label_file_path = cast(Path, self.label_file_url.get_local_path())
        return self._label_file_path

    def retrieve_label_file(self) -> Path:
        return cast(Path, self.label_file_url.retrieve())


@dataclass
class ImageFiles:
    """A collection of ImageFile objects that behaves like a sequence.

    Supports iteration, indexing, and length operations on the wrapped image files.
    """

    image_files: list[ImageFile]

    def __iter__(self) -> Iterator[ImageFile]:
        return iter(self.image_files)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> ImageFile:
        return self.image_files[idx]


class DataSet(ABC, NavBase):
    def __init__(self, *, config: Optional[Config] = None) -> None:
        """Initializes a dataset.

        Parameters:
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """
        super().__init__(config=config)

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
    def add_selection_arguments(
        cmdparser: argparse.ArgumentParser,
        group: Optional[argparse._ArgumentGroup] = None,
    ) -> None:
        """Adds dataset-specific command-line arguments for image selection.

        Parameters:
            cmdparser: The argument parser to add arguments to.
            group: Optional argument group to add arguments to. If None, creates a new group.
        """
        ...

    @abstractmethod
    def yield_image_files_from_arguments(
        self, arguments: argparse.Namespace
    ) -> Iterator[ImageFiles]:
        """Yields image filenames based on provided command-line arguments.

        Parameters:
            arguments: The parsed arguments structure.

        Yields:
            Information about the selected files in groups as ImageFiles objects.
        """
        ...

    @abstractmethod
    def yield_image_files_index(self, **kwargs: Any) -> Iterator[ImageFiles]:
        """Yields image filenames based on index information.

        Parameters:
            **kwargs: Arbitrary keyword arguments, usually used to restrict the search.

        Yields:
            Information about the selected files in groups as ImageFiles objects.
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

    def pds4_bundle_template_dir(self) -> str:
        """Returns absolute path to template directory for PDS4 bundle generation.

        Checks config section pds4.{dataset_name}.template_dir first, then allows override.
        If just a name is given, it is relative to the pds4/templates directory.
        If a full path is given, it will be left as absolute.

        Returns:
            Absolute path to template directory
            (e.g., "/path/to/pds4/templates/cassini_iss_saturn_1.0").
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_bundle_name(self) -> str:
        """Returns bundle name for PDS4 bundle generation.

        Checks config section pds4.{dataset_name}.bundle_name first, then allows override.

        Returns:
            Bundle name (e.g., "cassini_iss_saturn_backplanes_rsfrench2027").
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    @staticmethod
    def pds4_bundle_path_for_image(image_name: str) -> str:
        """Maps image name to bundle directory path.

        Parameters:
            image_name: The image name to map.

        Returns:
            Bundle directory path relative to bundle root (e.g., "1234xxxxxx/123456xxxx").
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_path_stub(self, image_file: ImageFile) -> str:
        """Returns PDS4 path stub for bundle directory structure.

        Parameters:
            image_file: The image file to generate path stub for.

        Returns:
            Path stub relative to bundle root (e.g., "1234xxxxxx/123456xxxx/1234567890w").
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_image_name_to_browse_lid(self, image_name: str) -> str:
        """Returns the browse LID for the given image name.

        Parameters:
            image_name: The image name to convert to a browse LID.

        Returns:
            The browse LID.
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_image_name_to_browse_lidvid(self, image_name: str) -> str:
        """Returns the browse LIDVID for the given image name.

        Parameters:
            image_name: The image name to convert to a browse LID.

        Returns:
            The browse LID.
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_image_name_to_data_lid(self, image_name: str) -> str:
        """Returns the data LID for the given image name.

        Parameters:
            image_name: The image name to convert to a data LID.

        Returns:
            The data LID.
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_image_name_to_data_lidvid(self, image_name: str) -> str:
        """Returns the data LIDVID for the given image name.

        Parameters:
            image_name: The image name to convert to a data LID.

        Returns:
            The data LID.
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_template_variables(
        self,
        *,
        image_file: ImageFile,
        nav_metadata: dict[str, Any],
        backplane_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Returns template variables for PDS4 label generation.

        Parameters:
            image_file: The image file being processed.
            nav_metadata: Navigation metadata dictionary (as read from offset_metadata
                JSON file).
            backplane_metadata: Backplane metadata dictionary (created from backplane FITS
                file).

        Returns:
            Dictionary mapping variable names to values for template substitution.
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError
