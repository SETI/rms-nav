from abc import ABC, abstractmethod


class DataSet(ABC):
    def __init__(self):
        ...

    @staticmethod
    @abstractmethod
    def image_name_valid(name: str) -> bool:
        ...

    @staticmethod
    @abstractmethod
    def add_selection_arguments(self):
        ...

    @abstractmethod
    def yield_image_filenames_index(self):
        ...
