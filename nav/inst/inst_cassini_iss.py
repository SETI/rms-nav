import oops.hosts.cassini.iss as cassini_iss

from .inst import Inst


class InstCassiniISS(Inst):
    def __init__(self, obs):
        self._obs = obs

    @staticmethod
    def from_file(filename):
        obs = cassini_iss.from_file(filename)
        return InstCassiniISS(obs)
