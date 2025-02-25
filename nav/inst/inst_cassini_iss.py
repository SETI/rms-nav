from typing import Any, Optional

from oops import Observation
import oops.hosts.cassini.iss
from psfmodel import GaussianPSF, PSF

from nav.config import Config, DEFAULT_CONFIG
from nav.obs import ObsSnapshot
from nav.support.types import PathLike

from .inst import Inst


class InstCassiniISS(Inst):
    def __init__(self,
                 obs: Observation,
                 **kwargs: Any) -> None:
        super().__init__(obs, logger_name='InstCassiniISS', **kwargs)

    @staticmethod
    def from_file(path: PathLike,
                  config: Optional[Config] = None,
                  extfov_margin_vu: tuple[int, int] | None = None) -> ObsSnapshot:

        config = config or DEFAULT_CONFIG
        obs = oops.hosts.cassini.iss.from_file(path,
                                               fast_distortion=True,
                                               return_all_planets=True)
        # TODO Calibrate
        # obs.data = obs.extended_calib.value_from_dn(obs.data).vals
        new_obs = ObsSnapshot(obs,
                              config=config,
                              extfov_margin_vu=extfov_margin_vu)
        new_obs.set_inst(InstCassiniISS(new_obs, config=config))
        return new_obs

    def star_psf(self) -> PSF:
        return GaussianPSF(sigma=0.54)
# PSF_SIGMA = {"NAC":   0.54,
#              "WAC":   0.77,
#              "LORRI": 0.5} # XXX
