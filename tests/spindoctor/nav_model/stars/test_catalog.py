"""Tests for ``spindoctor.nav_model.stars.catalog``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from spindoctor.nav_model.stars.catalog import (
    CATALOG_MAGNITUDE_BINS,
    _find_stars_in_one_catalog,
    _mark_visual_overlaps,
    _merge_catalogs,
    aberrate_star,
    select_radec_list,
)
from spindoctor.support.types import MutableStar


@dataclass
class _FakeStar:
    """Mutable star stand-in used by the catalog dedup / overlap helpers."""

    ra: float | None = None
    dec: float | None = None
    ra_pm: float = 0.0
    dec_pm: float = 0.0
    vmag: float | None = 5.0
    name: str = ''
    pretty_name: str = ''
    catalog_name: str = ''
    psf_size: tuple[int, int] = (3, 3)
    u: float = 0.0
    v: float = 0.0
    conflicts: str = ''

    def ra_dec_with_pm(self, tdb: float) -> tuple[float, float]:
        """Return the catalog ``(ra, dec)`` unchanged (no proper motion)."""
        del tdb
        return float(self.ra or 0.0), float(self.dec or 0.0)


def _as_stars(values: list[_FakeStar]) -> list[MutableStar]:
    """Cast a list of fake stars to the ``MutableStar`` protocol type."""
    return cast(list[MutableStar], values)


def test_catalog_magnitude_bins_monotonic() -> None:
    """The bin edges are strictly increasing — required by ``itertools.pairwise``."""
    diffs = np.diff(np.array(CATALOG_MAGNITUDE_BINS))
    assert (diffs > 0.0).all()


def test_catalog_magnitude_bins_first_and_last() -> None:
    """The bin edges span the configured magnitude range."""
    assert CATALOG_MAGNITUDE_BINS[0] == 0.0
    assert CATALOG_MAGNITUDE_BINS[-1] == 17.0


def test_select_radec_list_with_proper_motion() -> None:
    """When ``use_proper_motion`` is True, ``ra_dec_with_pm`` is consulted."""
    out = select_radec_list(
        _as_stars([_FakeStar(ra=1.0, dec=2.0)]), use_proper_motion=True, midtime=0.0
    )
    assert out == [(1.0, 2.0)]


def test_select_radec_list_without_proper_motion() -> None:
    """Without proper motion, the helper returns the catalog ``(ra, dec)`` pair."""
    out = select_radec_list(
        _as_stars([_FakeStar(ra=0.5, dec=1.5)]), use_proper_motion=False, midtime=0.0
    )
    assert out == [(0.5, 1.5)]


def test_aberrate_star_rejects_a_record_with_no_ra() -> None:
    """A star carrying no RA is a caller error, not silent garbage."""
    star = cast(MutableStar, _FakeStar(ra=None, dec=2.0))
    obs = cast(Any, SimpleNamespace(midtime=0.0, path=None, frame=None))
    with pytest.raises(ValueError, match='requires a star with RA and DEC'):
        aberrate_star(obs, star)


def test_aberrate_star_rejects_a_record_with_no_dec() -> None:
    """A star carrying no DEC is a caller error, not silent garbage."""
    star = cast(MutableStar, _FakeStar(ra=1.0, dec=None))
    obs = cast(Any, SimpleNamespace(midtime=0.0, path=None, frame=None))
    with pytest.raises(ValueError, match='requires a star with RA and DEC'):
        aberrate_star(obs, star)


def test_merge_catalogs_drops_exact_duplicates() -> None:
    """A duplicate within both thresholds is excluded from the merged list."""
    earlier = _as_stars([_FakeStar(ra_pm=0.1, dec_pm=0.2, vmag=5.0, pretty_name='AAA')])
    later = _as_stars([_FakeStar(ra_pm=0.1, dec_pm=0.2, vmag=5.0, pretty_name='BBB')])
    merged = _merge_catalogs(
        earlier,
        later,
        duplicate_radec=math.radians(10 / 3600),
        duplicate_vmag=2.0,
    )
    assert len(merged) == 1
    assert merged[0].pretty_name == 'AAA'


def test_merge_catalogs_keeps_distant_star() -> None:
    """A star far enough away in DEC is appended even when other fields match."""
    earlier = _as_stars([_FakeStar(ra_pm=0.0, dec_pm=0.0, vmag=5.0)])
    later = _as_stars([_FakeStar(ra_pm=0.0, dec_pm=0.05, vmag=5.0)])
    merged = _merge_catalogs(
        earlier,
        later,
        duplicate_radec=math.radians(10 / 3600),
        duplicate_vmag=2.0,
    )
    assert len(merged) == 2


def test_merge_catalogs_upgrades_pretty_name_from_named_duplicate() -> None:
    """When ``earlier`` lacks a name and ``later`` has one, the later name wins."""
    earlier = _as_stars([_FakeStar(ra_pm=0.0, dec_pm=0.0, vmag=5.0, name='', pretty_name='123')])
    later = _as_stars(
        [_FakeStar(ra_pm=0.0, dec_pm=0.0, vmag=5.0, name='Sirius', pretty_name='Sirius')]
    )
    merged = _merge_catalogs(
        earlier,
        later,
        duplicate_radec=math.radians(10 / 3600),
        duplicate_vmag=2.0,
    )
    assert len(merged) == 1
    assert merged[0].pretty_name == 'Sirius'


def test_merge_catalogs_returns_later_when_earlier_empty() -> None:
    """An empty ``earlier`` list yields exactly ``later``."""
    later = _as_stars([_FakeStar(ra_pm=0.1, dec_pm=0.2, vmag=5.0)])
    merged = _merge_catalogs([], later, duplicate_radec=1e-3, duplicate_vmag=1.0)
    assert merged == later


def test_mark_visual_overlaps_tags_both_for_similar_magnitudes() -> None:
    """Overlap with similar V-magnitudes tags both stars with conflict ``'STAR'``."""
    s1 = _FakeStar(u=10.0, v=10.0, psf_size=(5, 5), vmag=5.0)
    s2 = _FakeStar(u=10.5, v=10.5, psf_size=(5, 5), vmag=5.5)
    _mark_visual_overlaps(_as_stars([s1, s2]), overlap_vmag=2.0)
    assert s1.conflicts == 'STAR'
    assert s2.conflicts == 'STAR'


def test_mark_visual_overlaps_tags_only_fainter_for_dominant_pair() -> None:
    """Overlap with a 5-mag delta tags only the fainter star."""
    bright = _FakeStar(u=5.0, v=5.0, psf_size=(5, 5), vmag=2.0)
    faint = _FakeStar(u=5.5, v=5.5, psf_size=(5, 5), vmag=8.0)
    _mark_visual_overlaps(_as_stars([bright, faint]), overlap_vmag=2.0)
    assert bright.conflicts == ''
    assert faint.conflicts == 'STAR'


def test_mark_visual_overlaps_no_op_on_singletons() -> None:
    """A list of one star is unchanged."""
    star = _FakeStar(u=0.0, v=0.0)
    _mark_visual_overlaps(_as_stars([star]), overlap_vmag=2.0)
    assert star.conflicts == ''


def test_mark_visual_overlaps_skips_separated_stars() -> None:
    """Stars whose PSF supports do not overlap are left unmarked."""
    s1 = _FakeStar(u=0.0, v=0.0, psf_size=(3, 3))
    s2 = _FakeStar(u=20.0, v=20.0, psf_size=(3, 3))
    _mark_visual_overlaps(_as_stars([s1, s2]), overlap_vmag=2.0)
    assert s1.conflicts == ''
    assert s2.conflicts == ''


def test_find_stars_in_one_catalog_rejects_invalid_catalog_name() -> None:
    """An unknown catalog name raises ``ValueError`` naming the bad string."""
    obs: Any = object()
    config: Any = object()
    with pytest.raises(ValueError, match=r"Invalid star catalog: 'klingon'"):
        _find_stars_in_one_catalog(
            obs=obs,
            config=config,
            catalog_name='klingon',
            ra_min=0.0,
            ra_max=1.0,
            dec_min=0.0,
            dec_max=1.0,
            mag_min=0.0,
            mag_max=20.0,
            radec_movement=None,
        )
