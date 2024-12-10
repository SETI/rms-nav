from abc import ABC, abstractmethod


class Inst(ABC):
    def __init__(self):
        ...

    @staticmethod
    @abstractmethod
    def from_file(self):
        ...

    @property
    def obs(self):
        return self._obs
