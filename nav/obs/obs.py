from abc import ABC, abstractmethod

import numpy as np


class Obs(ABC):

    @property
    def clean_detector(self):
        if self.instrument == "ISS":
            return self.detector
        return self.instrument

    def make_fov_zeros(self, dtype=np.float64):
        return np.zeros(self.data.shape, dtype=dtype)

    def make_extfov_zeros(self, dtype=np.float64):
        return np.zeros(self.extdata.shape, dtype=dtype)
