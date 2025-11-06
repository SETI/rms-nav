from abc import ABC
import argparse
from typing import Any, Iterator, Optional

from .dataset import DataSet, ImageFiles


class DataSetPDS4(DataSet, ABC):
    @staticmethod
    def _img_name_valid(img_name: str) -> bool:
        return False

    @staticmethod
    def add_selection_arguments(cmdparser: argparse.ArgumentParser,
                                group: Optional[argparse._ArgumentGroup] = None) -> None:
        raise NotImplementedError('PDS4 datasets are not yet implemented')

    def yield_image_files_from_arguments(self,
                                         arguments: argparse.Namespace) -> Iterator[ImageFiles]:
        raise NotImplementedError('PDS4 datasets are not yet implemented')

    def yield_image_files_index(self,
                                **kwargs: Any) -> Iterator[ImageFiles]:
        raise NotImplementedError('PDS4 datasets are not yet implemented')

    @staticmethod
    def supported_grouping() -> list[str]:
        return []
