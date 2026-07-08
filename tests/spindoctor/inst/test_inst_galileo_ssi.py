import math

import pytest
from tests.config import URL_GALILEO_SSI_IO_01

import spindoctor.obs.obs_inst_galileo_ssi as obstgossi
from spindoctor.obs.obs_inst_galileo_ssi import ObsGalileoSSI

# Documented anchor limiting magnitude (limiting mag at texp = 1 s).
_GALILEO_ANCHOR = 10.3


def _make_obs(texp: float) -> ObsGalileoSSI:
    """Build a bare ObsGalileoSSI carrying only texp.

    The star-magnitude gate is a pure function of ``self.texp``, so a fully
    constructed observation (and an external image fetch) is unnecessary for
    testing it.
    """
    obs = object.__new__(ObsGalileoSSI)
    obs.texp = texp
    return obs


def test_galileo_ssi_basic() -> None:
    obs = obstgossi.ObsGalileoSSI.from_file(URL_GALILEO_SSI_IO_01)
    assert obs.midtime == -110923771.01052806


def test_star_max_usable_vmag_anchor_at_unit_exposure() -> None:
    """The limiting magnitude equals the anchor at texp = 1 s."""
    obs = _make_obs(1.0)
    assert obs.star_max_usable_vmag() == pytest.approx(_GALILEO_ANCHOR, abs=1e-6)


def test_star_max_usable_vmag_gains_one_mag_per_pogson_ratio() -> None:
    """A 2.512x longer exposure deepens the limit by ~1 mag."""
    base = _make_obs(1.0).star_max_usable_vmag()
    deeper = _make_obs(2.512).star_max_usable_vmag()
    assert deeper - base == pytest.approx(1.0, abs=1e-3)


def test_star_max_usable_vmag_in_sane_range() -> None:
    """The limiting magnitude stays within a sane 3..15 range."""
    vmag = _make_obs(2.512).star_max_usable_vmag()
    assert 3.0 <= vmag <= 15.0


def test_star_max_usable_vmag_is_finite() -> None:
    """The limiting magnitude is finite."""
    vmag = _make_obs(2.512).star_max_usable_vmag()
    assert math.isfinite(vmag)


def test_star_max_usable_vmag_non_positive_exposure_returns_anchor() -> None:
    """A non-positive exposure falls back to the anchor magnitude."""
    obs = _make_obs(0.0)
    assert obs.star_max_usable_vmag() == pytest.approx(_GALILEO_ANCHOR, abs=1e-6)
