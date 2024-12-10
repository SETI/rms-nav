import oops.hosts.voyager.iss as voyager_iss

from .inst import Inst


class InstVoyagerISS(Inst):
    def __init__(self, obs):
        self._obs = obs

    @staticmethod
    def from_file(filename):
        obs = voyager_iss.from_file(filename)
        return InstVoyagerISS(obs)
