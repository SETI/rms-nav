import math

import pytest
from tests.config import URL_NEWHORIZONS_LORRI_CHARON_01

import nav.obs.obs_inst_newhorizons_lorri as obstnhlorri
from nav.obs.obs_inst_newhorizons_lorri import ObsNewHorizonsLORRI

# Documented anchor limiting magnitude (limiting mag at texp = 1 s).
_LORRI_ANCHOR = 11.7


def _make_obs(texp: float) -> ObsNewHorizonsLORRI:
    """Build a bare ObsNewHorizonsLORRI carrying only texp.

    The star-magnitude gate is a pure function of ``self.texp``, so a fully
    constructed observation (and an external image fetch) is unnecessary for
    testing it.
    """
    obs = object.__new__(ObsNewHorizonsLORRI)
    obs.texp = texp
    return obs


def test_newhorizons_lorri_basic() -> None:
    obs = obstnhlorri.ObsNewHorizonsLORRI.from_file(URL_NEWHORIZONS_LORRI_CHARON_01)
    assert obs.midtime == 490113790.9641424


def test_star_max_usable_vmag_anchor_at_unit_exposure() -> None:
    """The limiting magnitude equals the anchor at texp = 1 s."""
    obs = _make_obs(1.0)
    assert obs.star_max_usable_vmag() == pytest.approx(_LORRI_ANCHOR, abs=1e-6)


def test_star_max_usable_vmag_gains_one_mag_per_pogson_ratio() -> None:
    """A 2.512x longer exposure deepens the limit by ~1 mag."""
    base = _make_obs(1.0).star_max_usable_vmag()
    deeper = _make_obs(2.512).star_max_usable_vmag()
    assert deeper - base == pytest.approx(1.0, abs=1e-3)


def test_star_max_usable_vmag_in_sane_range() -> None:
    """The limiting magnitude stays within a sane 3..15 range."""
    vmag = _make_obs(1.0).star_max_usable_vmag()
    assert 3.0 <= vmag <= 15.0


def test_star_max_usable_vmag_is_finite() -> None:
    """The limiting magnitude is finite."""
    vmag = _make_obs(1.0).star_max_usable_vmag()
    assert math.isfinite(vmag)


def test_star_max_usable_vmag_non_positive_exposure_returns_anchor() -> None:
    """A non-positive exposure falls back to the anchor magnitude."""
    obs = _make_obs(0.0)
    assert obs.star_max_usable_vmag() == pytest.approx(_LORRI_ANCHOR, abs=1e-6)
