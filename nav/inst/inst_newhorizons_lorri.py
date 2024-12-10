import oops.hosts.newhorizons.lorri as nh_lorri

from .inst import Inst


class InstNewHorizonsLORRI(Inst):
    def __init__(self, obs):
        self._obs = obs

    @staticmethod
    def from_file(filename):
        obs = nh_lorri.from_file(filename)
        return InstNewHorizonsLORRI(obs)
