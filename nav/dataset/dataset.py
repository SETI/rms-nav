from abc import ABC, abstractmethod
import argparse
from typing import Optional


class DataSet(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def image_name_valid(self):
        ...

    @abstractmethod
    def add_selection_arguments(self):
        ...

    @abstractmethod
    def yield_image_filenames_index(self):
        ...
