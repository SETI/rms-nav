"""Tests for ``spindoctor.nav_model.stars.saturation``.

The bright-end correction replaces UCAC4's saturated aperture magnitude
with the true Johnson V from YBSC wherever the two catalogs positionally
match, and flags bright records with no YBSC match so downstream
consumers do not trust their magnitude.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
from tests.shims import make_star

from spindoctor.nav_model.stars.saturation import (
    UCAC4_SATURATION_VMAG_LIMIT,
    _nearest_within,
    correct_saturated_vmags,
    correct_star_photometry,
    ybsc_reference_photometry,
)
from spindoctor.support.types import MutableStar

# Two Pleiades bright members near their catalog positions (radians).
_ETA_TAU = (math.radians(56.8713), math.radians(24.1050))
_MEROPE = (math.radians(56.5817), math.radians(23.9483))
_MATCH_RAD = math.radians(5.0 / 3600.0)
_DVMAG = 3.0
_ORDER = ('ucac4', 'tycho2', 'ybsc')


def _star(**kwargs: object) -> MutableStar:
    """Build a ``MutableStar`` record from keyword overrides."""
    return cast(MutableStar, make_star(**kwargs))


def _ucac4(ra: float, dec: float, vmag: float) -> MutableStar:
    """A UCAC4 record with proper-motion position pinned to ``(ra, dec)``."""
    return _star(catalog_name='ucac4', ra_pm=ra, dec_pm=dec, vmag=vmag, dn=1.0)


def _ybsc(ra: float, dec: float, vmag: float, *, b_v: float = 0.0) -> MutableStar:
    """A YBSC reference record with Johnson photometry populated."""
    return _star(
        catalog_name='ybsc',
        ra_pm=ra,
        dec_pm=dec,
        vmag=vmag,
        b_v=b_v,
        johnson_mag_v=vmag,
        johnson_mag_b=vmag + b_v,
    )


def test_saturation_limit_is_bright_end() -> None:
    """The documented limit sits in the bright regime UCAC4 mishandles."""
    assert UCAC4_SATURATION_VMAG_LIMIT == 8.0


def test_correct_star_photometry_replaces_saturated_vmag() -> None:
    """A saturated UCAC4 star adopts the matched YBSC magnitude."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87, b_v=-0.09)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.vmag == pytest.approx(2.87)


def test_correct_star_photometry_sets_corrected_flag() -> None:
    """A corrected star is flagged so provenance is visible downstream."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.photometry_corrected is True


def test_correct_star_photometry_recomputes_dn() -> None:
    """The flux-derived DN follows the corrected magnitude, not the old one."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.dn == pytest.approx(2.512 ** -(2.87 - 4.0))


def test_correct_star_photometry_copies_johnson_mags() -> None:
    """Johnson B follows the YBSC colour once corrected."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87, b_v=0.5)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.johnson_mag_b == pytest.approx(3.37)


def test_correct_star_photometry_keeps_bv_consistent_without_johnson_b() -> None:
    """When the YBSC ref has no Johnson B, ``b_v`` stays consistent (zero)."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87)
    ref.johnson_mag_b = None
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.b_v == pytest.approx(0.0)


def test_correct_star_photometry_uses_vmag_when_ref_johnson_v_missing() -> None:
    """A YBSC ref with no Johnson V falls back to its V-magnitude."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87)
    ref.johnson_mag_v = None
    ref.johnson_mag_b = None
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.johnson_mag_v == pytest.approx(2.87)


def test_correct_star_photometry_clears_faked_flag() -> None:
    """A corrected star no longer carries a faked-photometry mark."""
    star = _ucac4(*_ETA_TAU, 6.68)
    star.johnson_mag_faked = True
    ref = _ybsc(*_ETA_TAU, 2.87)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.johnson_mag_faked is False


def test_correct_star_photometry_flags_unmatched_bright_star() -> None:
    """A bright star absent from YBSC is flagged saturated, not corrected."""
    star = _ucac4(*_ETA_TAU, 6.68)
    far = _ybsc(math.radians(120.0), math.radians(-10.0), 3.0)
    correct_star_photometry(
        [star], [far], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.photometry_saturated is True


def test_correct_star_photometry_keeps_unmatched_bright_vmag() -> None:
    """An unmatched bright star keeps its (untrusted) catalog magnitude."""
    star = _ucac4(*_ETA_TAU, 6.68)
    far = _ybsc(math.radians(120.0), math.radians(-10.0), 3.0)
    correct_star_photometry(
        [star], [far], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.vmag == pytest.approx(6.68)


def test_correct_star_photometry_ignores_faint_star() -> None:
    """A faint star is not a candidate even next to a bright YBSC star."""
    star = _ucac4(*_ETA_TAU, 11.0)
    ref = _ybsc(*_ETA_TAU, 2.87)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.vmag == pytest.approx(11.0)


def test_correct_star_photometry_leaves_faint_star_unflagged() -> None:
    """A faint star gets neither the corrected nor the saturated flag."""
    star = _ucac4(*_ETA_TAU, 11.0)
    ref = _ybsc(*_ETA_TAU, 2.87)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.photometry_saturated is False


def test_correct_star_photometry_skips_ybsc_records() -> None:
    """A YBSC record already in the list is never treated as a candidate."""
    star = _ybsc(*_ETA_TAU, 2.87)
    correct_star_photometry(
        [star], [star], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.photometry_corrected is False


def test_correct_star_photometry_respects_limit_boundary() -> None:
    """A star exactly at the limit is not corrected (strict inequality)."""
    star = _ucac4(*_ETA_TAU, 8.0)
    ref = _ybsc(*_ETA_TAU, 3.0)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.vmag == pytest.approx(8.0)


def test_correct_star_photometry_collapses_saturated_ucac4_onto_ybsc_twin() -> None:
    """A saturated UCAC4 star collapses onto its native YBSC twin.

    The YBSC record read the star at its true magnitude, so it also placed
    it accurately: it wins and the saturated UCAC4 record is dropped.
    """
    ucac4 = _ucac4(*_ETA_TAU, 6.68)
    ybsc = _ybsc(*_ETA_TAU, 2.87)
    out = correct_star_photometry(
        [ucac4, ybsc],
        [ybsc],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert out == [ybsc]


def test_correct_star_photometry_keeps_unconsumed_ybsc_star() -> None:
    """A YBSC bright star with no UCAC4 twin stays in the merged list."""
    ucac4 = _ucac4(*_ETA_TAU, 6.68)
    ybsc_match = _ybsc(*_ETA_TAU, 2.87)
    ybsc_solo = _ybsc(*_MEROPE, 4.18)
    out = correct_star_photometry(
        [ucac4, ybsc_match, ybsc_solo],
        [ybsc_match, ybsc_solo],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert ybsc_solo in out


def test_correct_star_photometry_collapses_revealed_tycho2_duplicate() -> None:
    """Correcting a saturated star collapses its co-located Tycho-2 twin.

    UCAC4 lists the star saturated, so the merge kept it alongside the
    Tycho-2 record at the same position (magnitudes disagreed).  Once
    corrected they match; the Tycho-2 record, which never needed
    correcting, keeps its accurate bright-star astrometry and survives.
    """
    ucac4 = _ucac4(*_ETA_TAU, 6.88)
    tycho2 = _star(catalog_name='tycho2', ra_pm=_ETA_TAU[0], dec_pm=_ETA_TAU[1], vmag=3.40)
    ref = _ybsc(*_ETA_TAU, 3.40)
    out = correct_star_photometry(
        [ucac4, tycho2],
        [ref],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert out == [tycho2]


def test_correct_star_photometry_revealed_duplicate_inherits_name() -> None:
    """The surviving twin inherits the dropped record's catalog name."""
    ucac4 = _star(
        catalog_name='ucac4', ra_pm=_ETA_TAU[0], dec_pm=_ETA_TAU[1], vmag=6.88, name='27Tau', dn=1.0
    )
    tycho2 = _star(catalog_name='tycho2', ra_pm=_ETA_TAU[0], dec_pm=_ETA_TAU[1], vmag=3.40, name='')
    ref = _ybsc(*_ETA_TAU, 3.40)
    correct_star_photometry(
        [ucac4, tycho2],
        [ref],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert tycho2.name == '27Tau'


def test_correct_star_photometry_keeps_higher_precedence_when_both_corrected() -> None:
    """Between two corrected twins, the higher-precedence catalog survives."""
    ucac4 = _ucac4(*_ETA_TAU, 6.88)
    tycho2 = _star(catalog_name='tycho2', ra_pm=_ETA_TAU[0], dec_pm=_ETA_TAU[1], vmag=6.9, dn=1.0)
    ref = _ybsc(*_ETA_TAU, 3.40)
    out = correct_star_photometry(
        [ucac4, tycho2],
        [ref],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert out == [ucac4]


def test_correct_star_photometry_both_corrected_prefers_precedence_regardless_of_order() -> None:
    """A corrected higher-precedence twin survives even when listed second."""
    tycho2 = _star(catalog_name='tycho2', ra_pm=_ETA_TAU[0], dec_pm=_ETA_TAU[1], vmag=6.9, dn=1.0)
    ucac4 = _ucac4(*_ETA_TAU, 6.88)
    ref = _ybsc(*_ETA_TAU, 3.40)
    out = correct_star_photometry(
        [tycho2, ucac4],
        [ref],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert out == [ucac4]


def test_correct_star_photometry_ignores_none_vmag_neighbour() -> None:
    """A co-located record with no magnitude is never treated as a duplicate."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87)
    ghost = _star(catalog_name='ucac4', ra_pm=_ETA_TAU[0], dec_pm=_ETA_TAU[1], vmag=None)
    out = correct_star_photometry(
        [star, ghost],
        [ref],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert ghost in out


def test_correct_star_photometry_keeps_corrected_star_with_distant_neighbour() -> None:
    """A corrected star with only a distant neighbour is kept as-is."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87)
    far = _star(catalog_name='ucac4', ra_pm=_MEROPE[0], dec_pm=_MEROPE[1], vmag=9.0)
    out = correct_star_photometry(
        [star, far],
        [ref],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert star in out
    assert far in out


def test_correct_star_photometry_returns_same_list_when_nothing_corrected() -> None:
    """With no correction the original list object is returned unchanged."""
    star = _ucac4(*_ETA_TAU, 11.0)
    stars = [star]
    out = correct_star_photometry(
        stars, [], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert out is stars


def test_ybsc_reference_photometry_skips_none_vmag() -> None:
    """Reference records with no magnitude are dropped from the arrays."""
    good = _ybsc(*_ETA_TAU, 2.87)
    bad = _ybsc(*_MEROPE, 4.18)
    bad.vmag = None
    ref_ra, _, ref_stars = ybsc_reference_photometry([good, bad])
    assert ref_ra.shape == (1,)
    assert ref_stars == [good]


def test_nearest_within_returns_none_for_empty_reference() -> None:
    """An empty reference set yields no match."""
    empty = np.asarray([], dtype=np.float64)
    assert _nearest_within(0.0, 0.0, empty, empty, _MATCH_RAD) is None


def test_nearest_within_returns_none_beyond_radius() -> None:
    """A reference star just outside the radius is not matched."""
    ref_ra = np.asarray([math.radians(1.0)], dtype=np.float64)
    ref_dec = np.asarray([0.0], dtype=np.float64)
    assert _nearest_within(0.0, 0.0, ref_ra, ref_dec, _MATCH_RAD) is None


def test_nearest_within_matches_across_ra_seam() -> None:
    """A reference just across the RA 0/2*pi seam is matched, not missed."""
    ref_ra = np.asarray([2.0 * math.pi - 1e-6], dtype=np.float64)
    ref_dec = np.asarray([0.0], dtype=np.float64)
    assert _nearest_within(1e-6, 0.0, ref_ra, ref_dec, _MATCH_RAD) == 0


def test_correct_star_photometry_matches_across_ra_seam() -> None:
    """A saturated star is corrected against a YBSC twin across the RA seam."""
    star = _star(catalog_name='ucac4', ra_pm=1e-6, dec_pm=0.0, vmag=6.68, dn=1.0)
    ref = _star(
        catalog_name='ybsc',
        ra_pm=2.0 * math.pi - 1e-6,
        dec_pm=0.0,
        vmag=2.87,
        johnson_mag_v=2.87,
        johnson_mag_b=2.87,
    )
    correct_star_photometry(
        [star],
        [ref],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert star.photometry_corrected is True


def test_correct_saturated_vmags_adopts_ybsc_for_match() -> None:
    """A saturated catalog star reports the YBSC magnitude after correction."""
    catalog = [(*_ETA_TAU, 6.68)]
    ybsc = [(*_ETA_TAU, 2.87)]
    out = correct_saturated_vmags(catalog, ybsc, match_radius_rad=_MATCH_RAD)
    assert out == [pytest.approx(2.87)]


def test_correct_saturated_vmags_keeps_faint_catalog_star() -> None:
    """A faint catalog star is never corrected, even beside a bright YBSC star.

    The co-located YBSC record is not re-added (it has a catalog
    counterpart), so the faint reading is the only surviving magnitude.
    """
    catalog = [(*_ETA_TAU, 11.0)]
    ybsc = [(*_ETA_TAU, 2.87)]
    out = correct_saturated_vmags(catalog, ybsc, match_radius_rad=_MATCH_RAD)
    assert out == [pytest.approx(11.0)]


def test_correct_saturated_vmags_keeps_unmatched_bright_star() -> None:
    """A bright catalog star with no YBSC match keeps its magnitude."""
    catalog = [(*_ETA_TAU, 6.68)]
    ybsc = [(math.radians(120.0), math.radians(-10.0), 3.0)]
    out = correct_saturated_vmags(catalog, ybsc, match_radius_rad=_MATCH_RAD)
    assert out == sorted([6.68, 3.0])


def test_correct_saturated_vmags_appends_ybsc_only_star() -> None:
    """A bright YBSC star the catalog missed is added to the result."""
    catalog = [(*_ETA_TAU, 6.68)]
    ybsc = [(*_ETA_TAU, 2.87), (*_MEROPE, 4.18)]
    out = correct_saturated_vmags(catalog, ybsc, match_radius_rad=_MATCH_RAD)
    assert out == [pytest.approx(2.87), pytest.approx(4.18)]


def test_correct_saturated_vmags_does_not_double_count_uncorrected_match() -> None:
    """A YBSC star co-located with an at-limit catalog star is not re-added.

    The catalog value sits at/above the limit (so it is left uncorrected),
    but its YBSC counterpart must not be appended a second time.
    """
    catalog = [(*_ETA_TAU, 8.5)]
    ybsc = [(*_ETA_TAU, 6.4)]
    out = correct_saturated_vmags(catalog, ybsc, match_radius_rad=_MATCH_RAD)
    assert out == [pytest.approx(8.5)]


def test_correct_saturated_vmags_is_sorted() -> None:
    """The returned magnitudes are sorted ascending (brightest first)."""
    catalog = [(*_MEROPE, 6.9), (*_ETA_TAU, 6.68), (math.radians(57.0), math.radians(24.0), 10.2)]
    ybsc = [(*_ETA_TAU, 2.87), (*_MEROPE, 4.18)]
    out = correct_saturated_vmags(catalog, ybsc, match_radius_rad=_MATCH_RAD)
    assert out == sorted(out)
