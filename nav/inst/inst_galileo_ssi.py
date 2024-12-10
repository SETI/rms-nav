import oops.hosts.galileo.ssi as galileo_ssi

from .inst import Inst


class InstGalileoSSI(Inst):
    def __init__(self, obs):
        self._obs = obs

    @staticmethod
    def from_file(filename):
        obs = galileo_ssi.from_file(filename)
        return InstGalileoSSI(obs)
