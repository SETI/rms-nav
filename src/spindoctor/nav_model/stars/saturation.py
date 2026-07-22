"""UCAC4 bright-end photometric saturation correction against YBSC.

UCAC4 aperture V-magnitudes saturate at the bright end: the reported
magnitude reads systematically too faint for bright stars.  A UCAC4-vs-
YBSC cross-match shows the residual (UCAC4 minus the true Johnson V) is a
few tenths of a magnitude near V8 and climbs to several magnitudes by
V3.  For the Pleiades, UCAC4 lists Eta Tau near V6.7 when the true value
is V2.9, and 27 Tau near V6.7 against a true V3.6.

Both the per-star detectability prediction and the navigable-content
screen read the catalog V-magnitude, so a field of two genuinely bright
stars plus a faint remainder is mistaken for a dozen comparably faint
stars: the two dominant anchors are dragged down into the faint
population and the pipeline cannot tell they are there.

YBSC (the Yale Bright Star Catalog) carries real Johnson V photometry
and is complete through the saturated regime, so it is the authoritative
reference.  Wherever a UCAC4 (or Tycho-2) star positionally matches a
YBSC star, the YBSC magnitude replaces the saturated catalog value while
the more precise UCAC4 astrometry is retained.  Bright stars with no
YBSC match are flagged so downstream consumers treat their magnitude as
unreliable and potentially too faint rather than a trustworthy reading.

The correction is exposed both as an in-place pass over ``MutableStar``
records (used by the catalog reduction, which feeds the detectability
model) and as a plain-magnitude helper (used by the navigable-content
screen), so every consumer of catalog brightness shares one policy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from collections.abc import Sequence

    from spindoctor.support.types import MutableStar, NDArrayFloatType

__all__ = [
    'UCAC4_SATURATION_VMAG_LIMIT',
    'correct_saturated_vmags',
    'correct_star_photometry',
    'ybsc_reference_photometry',
]


UCAC4_SATURATION_VMAG_LIMIT: float = 8.0
"""V-magnitude brighter than which UCAC4 aperture photometry is untrusted.

Chosen from a UCAC4-vs-YBSC cross-match: the residual (UCAC4 minus the
true Johnson V) is a few tenths of a magnitude near V8 and climbs to
about 4 magnitudes by V3, and UCAC4 itself documents its aperture
photometry as reliable only over roughly V9 to V14.  8.0 is a
conservative onset that captures every saturated Pleiades member
observed (reported by UCAC4 between V6.5 and V7.6) without reaching into
the regime where the aperture magnitude is trustworthy.
"""


SATURATION_CORRECTION_MIN_MAG: float = 0.5
"""Smallest catalog-versus-YBSC magnitude gap treated as real saturation.

A bright record whose magnitude already agrees with YBSC to within this
tolerance is not saturated -- its photometry (and, for a bright star, its
astrometry) is trustworthy -- so it is left untouched.  Only a larger gap
marks a genuinely saturated record to correct and flag.  The value sits
just above the UCAC4-vs-YBSC scatter for well-behaved bright stars and
well below the multi-magnitude error of a truly saturated one.
"""


def _wrap_to_pi(delta: float) -> float:
    """Wrap an angle difference (radians) into ``[-pi, pi]``.

    Keeps RA separations correct across the ``0``/``2*pi`` seam; the wrap is
    the identity for ``|delta| < pi``.

    Parameters:
        delta: Angle difference in radians.

    Returns:
        The equivalent difference in ``[-pi, pi]``.
    """
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def _nearest_within(
    ra: float,
    dec: float,
    ref_ra: NDArrayFloatType,
    ref_dec: NDArrayFloatType,
    radius_rad: float,
) -> int | None:
    """Return the index of the nearest reference star within ``radius_rad``.

    Distance is the small-angle separation on the sky (RA scaled by
    ``cos(dec)``), which is accurate for the few-arcsecond match radius
    used here.  The RA difference is wrapped into ``[-pi, pi]`` so a field
    straddling the ``0``/``2*pi`` seam still matches (the wrap is the
    identity away from the seam).

    Parameters:
        ra: Target right ascension in radians.
        dec: Target declination in radians.
        ref_ra: Reference right ascensions in radians.
        ref_dec: Reference declinations in radians.
        radius_rad: Match radius in radians.

    Returns:
        Index into the reference arrays of the closest star within the
        radius, or ``None`` when the arrays are empty or nothing is close
        enough.
    """
    if ref_ra.size == 0:
        return None
    d_dec = ref_dec - dec
    # Wrap the RA difference into [-pi, pi] (identity away from the seam).
    d_ra = (np.remainder(ref_ra - ra + math.pi, 2.0 * math.pi) - math.pi) * math.cos(dec)
    d2 = d_ra * d_ra + d_dec * d_dec
    idx = int(np.argmin(d2))
    if float(d2[idx]) <= radius_rad * radius_rad:
        return idx
    return None


def ybsc_reference_photometry(
    ybsc_stars: Sequence[MutableStar],
) -> tuple[NDArrayFloatType, NDArrayFloatType, list[MutableStar]]:
    """Build position arrays and the star list for a YBSC reference set.

    Only YBSC records with a usable V-magnitude are retained.

    Parameters:
        ybsc_stars: YBSC records already reduced and FOV-projected (so
            ``ra_pm`` / ``dec_pm`` / ``vmag`` are populated).

    Returns:
        A ``(ref_ra, ref_dec, ref_stars)`` triple whose arrays are
        parallel to ``ref_stars``.
    """
    ref_ra: list[float] = []
    ref_dec: list[float] = []
    ref_stars: list[MutableStar] = []
    for star in ybsc_stars:
        if star.vmag is None:
            continue
        ref_ra.append(float(star.ra_pm))
        ref_dec.append(float(star.dec_pm))
        ref_stars.append(star)
    return np.asarray(ref_ra, dtype=np.float64), np.asarray(ref_dec, dtype=np.float64), ref_stars


def correct_star_photometry(
    stars: list[MutableStar],
    ybsc_reference: Sequence[MutableStar],
    *,
    match_radius_rad: float,
    duplicate_vmag: float,
    catalog_order: Sequence[str],
    saturation_limit: float = UCAC4_SATURATION_VMAG_LIMIT,
    min_correction_mag: float = SATURATION_CORRECTION_MIN_MAG,
) -> list[MutableStar]:
    """Correct UCAC4/Tycho-2 bright-end saturation against a YBSC reference.

    Each non-YBSC record brighter than ``saturation_limit`` that
    positionally matches a YBSC reference star adopts that star's Johnson
    photometry (``vmag``, ``johnson_mag_v``, ``johnson_mag_b``, ``b_v``)
    and has its flux-derived ``dn`` recomputed, while keeping its own
    astrometry.  A matched record is flagged ``photometry_corrected``; a
    bright record with no match is flagged ``photometry_saturated`` so
    downstream consumers know its magnitude is unreliable and potentially
    too faint.

    Correcting a saturated magnitude can reveal a duplicate the catalog
    merge missed: the same physical star from two catalogs is only deduped
    when their magnitudes agree, so a star whose UCAC4 value was several
    magnitudes too faint survived alongside its Tycho-2 or YBSC twin.  Once
    corrected the two magnitudes match, so a final pass collapses any
    position-and-magnitude duplicate a correction exposes.  The twin that
    did NOT need correcting wins, because a catalog that reads a bright star
    at its true magnitude also places it accurately, whereas the same
    saturation that corrupted UCAC4's photometry displaces its astrometry;
    the survivor inherits a name from the dropped record.  A corrected
    record with no surviving twin (its brighter catalog counterpart was
    already dropped by the merge) is retained with its own astrometry,
    since no more accurate position is available.

    Parameters:
        stars: The merged, deduplicated star list.  Records are mutated in
            place.
        ybsc_reference: The full set of YBSC records in the field, used as
            the photometric reference (need not be a subset of ``stars``).
        match_radius_rad: Position match radius in radians.
        duplicate_vmag: V-magnitude tolerance for judging two co-located
            records the same star once photometry is corrected.
        catalog_order: Catalog names in precedence order (most precise
            first); decides which record of a revealed duplicate survives.
        saturation_limit: V-magnitude brighter than which a non-YBSC
            record is a saturation-correction candidate.
        min_correction_mag: Smallest catalog-versus-YBSC magnitude gap
            treated as saturation; a smaller gap is left untouched.

    Returns:
        The corrected star list with correction-revealed duplicates removed.
    """
    ref_ra, ref_dec, ref_stars = ybsc_reference_photometry(ybsc_reference)
    corrected: list[MutableStar] = []
    for star in stars:
        if star.catalog_name == 'ybsc':
            continue
        if star.vmag is None or float(star.vmag) >= saturation_limit:
            continue
        idx = _nearest_within(
            float(star.ra_pm), float(star.dec_pm), ref_ra, ref_dec, match_radius_rad
        )
        if idx is None:
            # Bright per UCAC4 but absent from YBSC: the aperture magnitude
            # is unreliable and possibly too faint, so flag it.
            star.photometry_saturated = True
            continue
        ref = ref_stars[idx]
        if abs(float(star.vmag) - cast(float, ref.vmag)) < min_correction_mag:
            # Already agrees with YBSC: not saturated, so leave it be.  A
            # record like this keeps its accurate astrometry and wins the
            # duplicate collapse against a genuinely saturated twin.
            continue
        _apply_ybsc_photometry(star, ref)
        corrected.append(star)
    if not corrected:
        return stars
    return _drop_revealed_duplicates(
        stars,
        corrected,
        match_radius_rad=match_radius_rad,
        duplicate_vmag=duplicate_vmag,
        catalog_order=catalog_order,
    )


def _drop_revealed_duplicates(
    stars: list[MutableStar],
    corrected: list[MutableStar],
    *,
    match_radius_rad: float,
    duplicate_vmag: float,
    catalog_order: Sequence[str],
) -> list[MutableStar]:
    """Remove duplicates that a photometry correction exposed.

    For every corrected record, any other record within
    ``match_radius_rad`` and ``duplicate_vmag`` is the same physical star.
    A twin that was not itself corrected wins (its native photometry and,
    for a bright star, its astrometry are trustworthy where UCAC4's
    saturated readings are not); between two corrected records the
    higher-precedence catalog wins.  The survivor inherits a name from the
    dropped record when it has none.  Only pairs involving a corrected
    record are considered, so records untouched by the correction keep the
    merge's original deduplication.

    Parameters:
        stars: The full star list (original order is preserved on return).
        corrected: The records whose photometry was just corrected.
        match_radius_rad: Position match radius in radians.
        duplicate_vmag: V-magnitude tolerance for a duplicate.
        catalog_order: Catalog names in precedence order (most precise
            first).

    Returns:
        ``stars`` with the dropped duplicates removed, order preserved.
    """
    priority = {name.lower(): i for i, name in enumerate(catalog_order)}
    fallback = len(catalog_order)

    def _rank(star: MutableStar) -> int:
        return priority.get(star.catalog_name.lower(), fallback)

    removed: set[int] = set()
    for star in corrected:
        if id(star) in removed:
            continue
        for other in stars:
            if other is star or id(other) in removed:
                continue
            if other.vmag is None or star.vmag is None:
                continue
            # Same seam-safe small-angle separation the YBSC match uses, so
            # any record matched for correction also collapses against its twin.
            d_dec = float(other.dec_pm) - float(star.dec_pm)
            d_ra = _wrap_to_pi(float(other.ra_pm) - float(star.ra_pm)) * math.cos(
                float(star.dec_pm)
            )
            if (
                d_ra * d_ra + d_dec * d_dec <= match_radius_rad * match_radius_rad
                and abs(float(other.vmag) - float(star.vmag)) < duplicate_vmag
            ):
                if not other.photometry_corrected:
                    # ``other`` reads this bright star natively, so its
                    # astrometry is trustworthy where ``star``'s saturated
                    # UCAC4 position is not: keep ``other``.
                    winner, loser = other, star
                elif _rank(star) <= _rank(other):
                    winner, loser = star, other
                else:
                    winner, loser = other, star
                if (not winner.name) and loser.name:
                    winner.name = loser.name
                    winner.pretty_name = loser.pretty_name
                removed.add(id(loser))
                if loser is star:
                    break
    return [s for s in stars if id(s) not in removed]


def _apply_ybsc_photometry(star: MutableStar, ref: MutableStar) -> None:
    """Overwrite a star's photometry with a YBSC reference star's values.

    Parameters:
        star: Record to correct in place.
        ref: YBSC reference star supplying the trusted photometry.
    """
    ref_vmag = cast(float, ref.vmag)
    star.vmag = ref_vmag
    star.johnson_mag_v = ref.johnson_mag_v if ref.johnson_mag_v is not None else ref_vmag
    if ref.johnson_mag_b is not None:
        star.johnson_mag_b = ref.johnson_mag_b
        star.b_v = ref.johnson_mag_b - star.johnson_mag_v
    else:
        # No Johnson B from YBSC: treat the star as its own B (colour 0) so
        # johnson_mag_b and b_v stay mutually consistent.
        star.johnson_mag_b = star.johnson_mag_v
        star.b_v = 0.0
    star.johnson_mag_faked = False
    # Standard flux-to-DN scaling, matching the catalog reduction's anchor.
    star.dn = float(2.512 ** -(ref_vmag - 4.0))
    star.photometry_corrected = True
    star.photometry_saturated = False


def correct_saturated_vmags(
    catalog_positions: Sequence[tuple[float, float, float]],
    ybsc_positions: Sequence[tuple[float, float, float]],
    *,
    match_radius_rad: float,
    saturation_limit: float = UCAC4_SATURATION_VMAG_LIMIT,
) -> list[float]:
    """Return catalog V-magnitudes with UCAC4 bright-end saturation corrected.

    Each catalog star brighter than ``saturation_limit`` that positionally
    matches a YBSC bright star adopts the YBSC magnitude; a matched YBSC
    star is not double counted.  Bright YBSC stars the catalog missed
    entirely are appended so the returned set reflects the field's true
    bright content.  Faint catalog stars and unmatched bright catalog
    stars keep their own magnitude.

    This is the magnitude-only counterpart of :func:`correct_star_photometry`
    for consumers (such as the navigable-content screen) that work with
    plain ``(ra, dec, vmag)`` triples rather than star records.

    Parameters:
        catalog_positions: ``(ra, dec, vmag)`` triples (radians, radians,
            magnitude) for the catalog under correction (typically UCAC4).
        ybsc_positions: ``(ra, dec, vmag)`` triples for the YBSC bright
            stars in the same field.
        match_radius_rad: Position match radius in radians.
        saturation_limit: V-magnitude brighter than which a catalog star is
            a saturation-correction candidate.

    Returns:
        Corrected V-magnitudes sorted ascending (brightest first).
    """
    ref_ra = np.asarray([p[0] for p in ybsc_positions], dtype=np.float64)
    ref_dec = np.asarray([p[1] for p in ybsc_positions], dtype=np.float64)
    ref_vmag = [p[2] for p in ybsc_positions]
    cat_ra = np.asarray([p[0] for p in catalog_positions], dtype=np.float64)
    cat_dec = np.asarray([p[1] for p in catalog_positions], dtype=np.float64)
    out: list[float] = []
    for ra, dec, vmag in catalog_positions:
        if vmag >= saturation_limit:
            out.append(float(vmag))
            continue
        idx = _nearest_within(ra, dec, ref_ra, ref_dec, match_radius_rad)
        if idx is None:
            out.append(float(vmag))
            continue
        out.append(float(ref_vmag[idx]))
    # Add bright YBSC stars the catalog missed entirely.  A YBSC star that
    # already has any catalog counterpart (whether corrected above or left
    # uncorrected because the catalog value sits at/above the limit) is
    # skipped so it is never counted twice.  A catalog value at/above the
    # limit therefore keeps its faint reading here rather than the YBSC
    # value -- unlike the record path, which retains the YBSC record.  That
    # only matters if UCAC4 saturates a star past the limit (a multi-mag
    # error where its documented error is tenths of a mag), so it does not
    # arise for real UCAC4 photometry.
    for yra, ydec, yvmag in ybsc_positions:
        if _nearest_within(yra, ydec, cat_ra, cat_dec, match_radius_rad) is None:
            out.append(float(yvmag))
    return sorted(out)
