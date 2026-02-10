import argparse
from abc import ABC
from collections.abc import Iterator
from typing import Any

from .dataset import DataSet, ImageFiles


class DataSetPDS4(DataSet, ABC):
    """Parent class for PDS4 datasets.

    This class provides functionality common to all PDS4 datasets.
    """

    @staticmethod
    def _img_name_valid(img_name: str) -> bool:
        raise NotImplementedError('PDS4 datasets are not yet implemented')

    @staticmethod
    def add_selection_arguments(
        cmdparser: argparse.ArgumentParser,
        group: argparse._ArgumentGroup | None = None,
    ) -> None:
        raise NotImplementedError('PDS4 datasets are not yet implemented')

    def yield_image_files_from_arguments(
        self, arguments: argparse.Namespace
    ) -> Iterator[ImageFiles]:
        raise NotImplementedError('PDS4 datasets are not yet implemented')

    def yield_image_files_index(self, **kwargs: Any) -> Iterator[ImageFiles]:
        raise NotImplementedError('PDS4 datasets are not yet implemented')

    @staticmethod
    def supported_grouping() -> list[str]:
        raise NotImplementedError('PDS4 datasets are not yet implemented')
