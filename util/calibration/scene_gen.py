"""Randomized sim-scene generation for the confidence calibration.

Generates seeded, randomized sim_params dicts (the same flat mapping the
sim-sweep harness feeds ObsSim) spanning each NavTechnique's operating
regime, from clean well-resolved frames through the failure cliff:

- disc        : resolved regular/irregular body, low-moderate phase
                (BodyDiscCorrelateNav; mesh-vs-ellipsoid shape mismatch)
- limb        : large body partially off-frame, low phase (BodyLimbNav;
                pose-disagreement and shape-mismatch slices)
- terminator  : high-phase crescent (BodyTerminatorNav)
- blob        : small / distant body (BodyBlobNav; irregular slice)
- ring        : 1-2 ringlets, curved through near-flat arcs (RingEdgeNav
                and, via the always-emitted template, RingAnnulusNav)
- star_field  : 3-15 stars, bright through sub-detection (the star
                techniques incl. the pass-2 StarRefineNav); a controlled
                fraction plant catalog-position scatter and non-navigable
                confounders (the astrometric-error and star-clutter regimes)
- star_unique : 1-2 star scenes with varying brightness margin
                (StarUniqueMatchNav one- and two-star paths); a controlled
                fraction plant a per-star catalog error or an unresolved
                companion (astrometric and photocenter-bias regimes)

Every scene carries a planted (offset_v, offset_u) ground truth. The noise
level, feature sizes/counts, and the model-error axes (mesh lumpiness vs an
ellipsoidal nav prediction, predicted-pose error) are drawn from ranges wide
enough that each technique's diagnostics span healthy to broken -- the spread
a logistic calibration fit needs on both label classes.

All randomness comes from a caller-seeded ``random.Random`` so a campaign is
reproducible from its seed.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

# The body families deliberately do not yet draw the newer body truth axes
# (photometric_law / opposition_surge, limb_relief_*, albedo_texture,
# disc_texture / transits, mesh shading / pose_scatter): those axes join
# the sweep when the calibration campaign is re-collected on this renderer
# and every sim-anchored coefficient is refit, so no coefficient ships
# ahead of the renderer it was fit on.
FAMILIES = (
    'disc',
    'limb',
    'terminator',
    'blob',
    'ring',
    'star_field',
    'star_unique',
)

# Body identities the campaign draws from, with mean radius and published
# ellipsoid RMS residual pulled live from config_220_body_shape.yaml (so
# the campaign always matches the shipped table).  Every rendered body is
# a polyhedral mesh whose relief AMPLITUDE is ~3x the body's fractional
# RMS residual (amplitude ~ 3 sigma of relief), while the navigator
# predicts the smooth (zero-relief) limit -- so the rendered shape error,
# and hence the recovered-offset error, physically tracks the
# max_phase_irregularity_factor the confidence formula consumes.  The
# scene also carries km_per_pixel = mean_radius_km / radius_px, without
# which the sim blob feature reports a zero irregularity factor.
_REGULAR_BODIES = ('MIMAS', 'ENCELADUS', 'TETHYS', 'DIONE', 'RHEA')
_IRREGULAR_BODIES = ('HYPERION', 'PHOEBE', 'JANUS', 'EPIMETHEUS')


def _body_catalog() -> dict[str, tuple[float, float]]:
    """Return ``{name: (mean_radius_km, rms_residual_km)}`` from config_220."""
    from spindoctor.config import DEFAULT_CONFIG

    catalog: dict[str, tuple[float, float]] = {}
    for name in _REGULAR_BODIES + _IRREGULAR_BODIES:
        entry = DEFAULT_CONFIG.body_shape[name]
        radii = entry['radii_km']
        catalog[name] = (
            sum(float(r) for r in radii) / len(radii),
            float(entry['ellipsoid_rms_residual_km']),
        )
    return catalog


_CATALOG: dict[str, tuple[float, float]] | None = None


def _catalog() -> dict[str, tuple[float, float]]:
    """Lazily built body catalog (config load deferred past import)."""
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _body_catalog()
    return _CATALOG


def _read_noise(rng: random.Random, *, lo: float = 1.0, hi: float = 48.0) -> float:
    """Log-uniform read-noise draw (DN) between ``lo`` and ``hi``."""
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def _planted_offset(rng: random.Random) -> tuple[float, float]:
    """Planted (dv, du): mostly a few px, a tail out to +/-10 px."""
    span = 4.0 if rng.random() < 0.7 else 10.0
    return rng.uniform(-span, span), rng.uniform(-span, span)


def _base(rng: random.Random, *, size: int) -> dict[str, Any]:
    """Common sim_params skeleton every family builds on."""
    dv, du = _planted_offset(rng)
    return {
        'instrument': 'coiss_nac',
        'size_v': size,
        'size_u': size,
        'random_seed': rng.randrange(2**31),
        'exposure_sec': 1.0,
        'bodies': [],
        'rings': [],
        'noise': {'poisson': True, 'read_noise_dn': _read_noise(rng)},
        'offset_v': dv,
        'offset_u': du,
        'offset_rotation_deg': 0.0,
    }


def _catalog_body(
    rng: random.Random,
    *,
    center_v: float,
    center_u: float,
    radius: float,
    phase: float,
    illumination: float,
    irregular_fraction: float,
    pose_error_deg: float = 0.0,
) -> dict[str, Any]:
    """A named body whose rendered relief tracks its published shape residual.

    Draws the identity from the config_220-backed catalog (regular vs
    irregular per ``irregular_fraction``), renders a polyhedral mesh whose
    relief amplitude is ~3x the body's fractional RMS residual, and has
    the navigator predict the smooth (zero-relief) limit -- so the shape
    error the navigator cannot model scales with the same
    residual-over-radius ratio the ``max_phase_irregularity_factor``
    confidence term consumes.  ``km_per_pixel`` gives the sim blob feature
    the physical scale that factor needs (it reports 0.0 without it).

    Parameters:
        rng: Scene-local random generator.
        center_v: Body center row in image coordinates (px).
        center_u: Body center column in image coordinates (px).
        radius: Apparent body radius (px).
        phase: Phase angle (deg).
        illumination: Image-plane illumination direction (deg).
        irregular_fraction: Probability of drawing an irregular-catalog body.
        pose_error_deg: Pose error applied to the predicted (nav) mesh (deg).

    Returns:
        A sim ``bodies`` entry dict ready for the scene parameter file.
    """
    if rng.random() < irregular_fraction:
        name = rng.choice(_IRREGULAR_BODIES)
    else:
        name = rng.choice(_REGULAR_BODIES)
    mean_radius_km, rms_residual_km = _catalog()[name]
    residual_fraction = rms_residual_km / mean_radius_km
    lumpiness = 3.0 * residual_fraction
    pose = [rng.uniform(0.0, 60.0), rng.uniform(0.0, 60.0), 0.0]
    nav_override: dict[str, Any] = {'mesh_lumpiness': 0.0}
    if pose_error_deg != 0.0:
        nav_override['pose_euler_deg'] = [pose[0], pose[1] + pose_error_deg, pose[2]]
    # Irregular bodies get genuinely tri-axial base ellipsoids; the
    # near-spherical regulars keep modest axis spreads.
    if name in _IRREGULAR_BODIES:
        axis2 = radius * rng.uniform(0.75, 0.95)
        axis3 = radius * rng.uniform(0.6, 0.85)
    else:
        axis2 = radius * rng.uniform(0.97, 1.0)
        axis3 = radius * rng.uniform(0.95, 1.0)
    return {
        'name': name,
        'shape_model': 'polyhedral_mesh',
        'mesh_lumpiness': lumpiness,
        'mesh_seed': rng.randrange(2**31),
        'pose_euler_deg': pose,
        'center_v': center_v,
        'center_u': center_u,
        'axis1': radius,
        'axis2': axis2,
        'axis3': axis3,
        'illumination_angle': illumination,
        'phase_angle': phase,
        'km_per_pixel': mean_radius_km / radius,
        'nav_override': nav_override,
    }


def gen_disc(rng: random.Random) -> dict[str, Any]:
    """Resolved body at low-moderate phase (BodyDiscCorrelateNav regime).

    Parameters:
        rng: Scene-local random generator.

    Returns:
        A complete sim scene parameter dict.
    """
    size = 200
    params = _base(rng, size=size)
    radius = rng.uniform(18.0, 95.0)
    phase = rng.uniform(5.0, 80.0)
    illumination = rng.uniform(0.0, 40.0)
    center_v = size / 2 + rng.uniform(-0.15, 0.15) * size
    center_u = size / 2 + rng.uniform(-0.15, 0.15) * size
    body = _catalog_body(
        rng,
        center_v=center_v,
        center_u=center_u,
        radius=radius,
        phase=phase,
        illumination=illumination,
        irregular_fraction=0.35,
    )
    params['bodies'] = [body]
    return params


def gen_limb(rng: random.Random) -> dict[str, Any]:
    """Large body, often partially off-frame, low phase (BodyLimbNav regime).

    Parameters:
        rng: Scene-local random generator.

    Returns:
        A complete sim scene parameter dict.
    """
    size = 220
    params = _base(rng, size=size)
    radius = rng.uniform(90.0, 190.0)
    phase = rng.uniform(8.0, 55.0)
    illumination = rng.uniform(0.0, 35.0)
    # Slide the center along a random direction so the visible limb arc
    # fraction spans "fully in frame" to "small arc in one corner".
    theta = rng.uniform(0.0, 2.0 * math.pi)
    displacement = rng.uniform(0.0, radius * 0.9)
    center_v = size / 2 + displacement * math.sin(theta)
    center_u = size / 2 + displacement * math.cos(theta)
    # A pose-error slice keeps the "confidently wrong limb" failure mode
    # in the cohort (the trained-away B7-3 scenario).
    pose_error = rng.uniform(2.0, 35.0) if rng.random() < 0.2 else 0.0
    body = _catalog_body(
        rng,
        center_v=center_v,
        center_u=center_u,
        radius=radius,
        phase=phase,
        illumination=illumination,
        irregular_fraction=0.4,
        pose_error_deg=pose_error,
    )
    params['bodies'] = [body]
    return params


def gen_terminator(rng: random.Random) -> dict[str, Any]:
    """High-phase crescent (BodyTerminatorNav regime).

    Parameters:
        rng: Scene-local random generator.

    Returns:
        A complete sim scene parameter dict.
    """
    size = 200
    params = _base(rng, size=size)
    radius = rng.uniform(35.0, 90.0)
    phase = rng.uniform(95.0, 150.0)
    illumination = rng.uniform(5.0, 35.0)
    center_v = size / 2 + rng.uniform(-0.1, 0.1) * size
    center_u = size / 2 + rng.uniform(-0.1, 0.1) * size
    body = _catalog_body(
        rng,
        center_v=center_v,
        center_u=center_u,
        radius=radius,
        phase=phase,
        illumination=illumination,
        irregular_fraction=0.4,
    )
    params['bodies'] = [body]
    return params


def gen_blob(rng: random.Random) -> dict[str, Any]:
    """Small-to-mid body (BodyBlobNav regime).

    The radius range spans from below the 5 px BODY_BLOB emission floor
    (those scenes exercise the fused failure path) through the sizes where
    the disc correlation takes over, so the fit sees the technique's whole
    design regime including the small-blob band the detection-SNR gate
    admits.

    Parameters:
        rng: Scene-local random generator.

    Returns:
        A complete sim scene parameter dict.
    """
    size = 128
    params = _base(rng, size=size)
    radius = rng.uniform(2.5, 30.0)
    phase = rng.uniform(10.0, 130.0)
    illumination = rng.uniform(0.0, 35.0)
    center_v = size / 2 + rng.uniform(-0.2, 0.2) * size
    center_u = size / 2 + rng.uniform(-0.2, 0.2) * size
    body = _catalog_body(
        rng,
        center_v=center_v,
        center_u=center_u,
        radius=radius,
        phase=phase,
        illumination=illumination,
        irregular_fraction=0.5,
    )
    params['bodies'] = [body]
    return params


def _ringlet(
    rng: random.Random, *, center_v: float, center_u: float, a_inner: float, a_outer: float
) -> dict[str, Any]:
    """One circular RINGLET entry in renderer form."""
    return {
        'name': 'SATURN',
        'feature_type': 'RINGLET',
        'center_v': center_v,
        'center_u': center_u,
        'inner_data': [
            {'mode': 1, 'a': a_inner, 'rms': 1.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
        ],
        'outer_data': [
            {'mode': 1, 'a': a_outer, 'rms': 1.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
        ],
    }


def gen_ring(rng: random.Random) -> dict[str, Any]:
    """Ringlet scene: curved through near-flat arcs, 1-2 ringlets, narrow to wide.

    Curvature is controlled by pushing the ring center off-frame: an arc from
    a ring of radius R passing through a frame R away from its center is
    nearly straight (the rank-1 regime), while an in-frame center gives
    fully-curved closed edges.
    """
    size = 220
    params = _base(rng, size=size)
    rings: list[dict[str, Any]] = []
    n_ringlets = 1 if rng.random() < 0.7 else 2
    flat = rng.random() < 0.3
    if flat:
        # Distant center: edges cross the frame as gentle arcs.
        big_r = rng.uniform(300.0, 900.0)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        center_v = size / 2 + big_r * math.sin(theta)
        center_u = size / 2 + big_r * math.cos(theta)
        base_a = big_r + rng.uniform(-30.0, 30.0)
    else:
        center_v = size / 2 + rng.uniform(-40.0, 40.0)
        center_u = size / 2 + rng.uniform(-40.0, 40.0)
        base_a = rng.uniform(40.0, 80.0)
    for _ in range(n_ringlets):
        width = rng.uniform(2.0, 35.0)
        rings.append(
            _ringlet(
                rng,
                center_v=center_v,
                center_u=center_u,
                a_inner=base_a,
                a_outer=base_a + width,
            )
        )
        base_a += width + rng.uniform(10.0, 30.0)
    params['rings'] = rings
    return params


# Controlled fractions of the star families that exercise the star
# information-asymmetry regimes.  They are deliberately a minority so the
# clean-lock label class stays dominant, and they are drawn from the family's
# own RNG so a campaign stays reproducible.
_STAR_FIELD_CATALOG_SCATTER_FRAC = 0.3  # scenes planting a small catalog scatter
_STAR_FIELD_CONFOUNDER_FRAC = 0.25  # scenes marking some field stars non-navigable
_STAR_UNIQUE_CATALOG_ERROR_FRAC = 0.3  # scenes planting a per-star catalog error
_STAR_UNIQUE_COMPANION_FRAC = 0.2  # scenes planting an unresolved companion


def gen_star_field(rng: random.Random) -> dict[str, Any]:
    """3-15 stars, bright through sub-detection (star-technique regimes)."""
    size = 128
    params = _base(rng, size=size)
    # Stars deposit as flux-normalized point masses; the navigator-matched PSF
    # gives them the floor-form profile (no 1-pixel spike).
    params['optics'] = {'psf': {'match_navigator': True}}
    # The flux-normalized coiss_nac limiting magnitude is ~10.9 at read noise 4
    # (ObsSim.star_max_usable_vmag), ~5 mag deeper than the prior peak-normalized
    # model; the magnitude ranges shift with it so the brightness regime still
    # spans "well above the limit" to "straddling it".  Cap the star-scene noise
    # a little lower than the body/ring families.
    params['noise']['read_noise_dn'] = _read_noise(rng, hi=16.0)
    n_stars = rng.randint(3, 15)
    # Per-scene brightness regime so whole frames span easy to marginal
    # (a per-star-only draw would almost always leave >= 3 bright stars).
    mag_lo = rng.uniform(7.0, 11.5)
    mag_hi = mag_lo + rng.uniform(0.5, 2.5)
    margin = 12.0
    stars = []
    for index in range(n_stars):
        stars.append(
            {
                'name': f'C{index + 1}',
                'v': rng.uniform(margin, size - margin),
                'u': rng.uniform(margin, size - margin),
                'vmag': rng.uniform(mag_lo, mag_hi),
            }
        )
    params['stars'] = stars
    # Astrometric-error regime: a small scene-level catalog scatter displaces
    # every rendered star off its predicted position, sweeping the tolerance the
    # star fit absorbs before the match degrades.
    if rng.random() < _STAR_FIELD_CATALOG_SCATTER_FRAC:
        params['star_catalog_scatter_px'] = rng.uniform(0.2, 1.0)
    # Star-clutter regime: mark a fraction of the field non-navigable so it
    # renders as a confounder the navigator has no knowledge of, while keeping at
    # least three navigable stars (the pattern matcher's minimum).
    max_confounders = max(0, n_stars - 3)
    if max_confounders > 0 and rng.random() < _STAR_FIELD_CONFOUNDER_FRAC:
        n_confounders = rng.randint(1, max_confounders)
        for star in rng.sample(stars, n_confounders):
            star['navigable'] = False
    if rng.random() < 0.25:
        params['optics']['stray_light'] = {
            'amplitude': rng.uniform(0.05, 0.5),
            'direction_deg': rng.uniform(0.0, 360.0),
            'model': 'linear',
        }
    return params


def gen_star_unique(rng: random.Random) -> dict[str, Any]:
    """1-2 star scenes with varying brightness margin (StarUniqueMatchNav)."""
    size = 128
    params = _base(rng, size=size)
    params['optics'] = {'psf': {'match_navigator': True}}
    params['noise']['read_noise_dn'] = _read_noise(rng, hi=16.0)
    margin = 16.0
    primary_mag = rng.uniform(7.0, 11.5)
    primary = {
        'name': 'U1',
        'v': rng.uniform(margin, size - margin),
        'u': rng.uniform(margin, size - margin),
        'vmag': primary_mag,
    }
    # Physical or planted catalog error on the primary: either an unresolved
    # companion (a magnitude-weighted photocenter shift) or a small planted
    # per-star position error.  These are mutually exclusive so the drawn regime
    # is unambiguous, and the companion is kept faint and close so the biased
    # centroid stays a sub-pixel model error rather than a resolved second star.
    roll = rng.random()
    if roll < _STAR_UNIQUE_COMPANION_FRAC:
        primary['companion'] = {
            'sep_px': rng.uniform(1.5, 3.0),
            'delta_mag': rng.uniform(1.5, 3.0),
            'angle_deg': rng.uniform(0.0, 360.0),
        }
    elif roll < _STAR_UNIQUE_COMPANION_FRAC + _STAR_UNIQUE_CATALOG_ERROR_FRAC:
        primary['catalog_error_v'] = rng.uniform(-1.0, 1.0)
        primary['catalog_error_u'] = rng.uniform(-1.0, 1.0)
    stars = [primary]
    if rng.random() < 0.55:
        # Two-star scene; the pairwise brightness gap varies the
        # assignment ambiguity the two-star path must resolve.
        stars.append(
            {
                'name': 'U2',
                'v': rng.uniform(margin, size - margin),
                'u': rng.uniform(margin, size - margin),
                'vmag': primary_mag + rng.uniform(0.1, 3.0),
            }
        )
    params['stars'] = stars
    return params


_GENERATORS: dict[str, Callable[[random.Random], dict[str, Any]]] = {
    'disc': gen_disc,
    'limb': gen_limb,
    'terminator': gen_terminator,
    'blob': gen_blob,
    'ring': gen_ring,
    'star_field': gen_star_field,
    'star_unique': gen_star_unique,
}


def generate_scenes(
    family: str, count: int, *, campaign_seed: int
) -> list[tuple[str, dict[str, Any]]]:
    """Return ``count`` seeded scenes for ``family`` as (scene_id, sim_params).

    Each scene draws from its own ``random.Random`` seeded by
    ``(campaign_seed, family, index)`` so any single scene can be regenerated
    without replaying the whole campaign.

    Parameters:
        family: One of :data:`FAMILIES`.
        count: Number of scenes to generate.
        campaign_seed: Campaign-level seed recorded in the output manifest.

    Returns:
        List of ``(scene_id, sim_params)`` pairs.
    """
    # Local import keeps this module importable without the package installed
    # (mirrors the deferred config import above).
    from spindoctor.sim.scene import validate_sim_params

    if family not in _GENERATORS:
        raise ValueError(f'unknown scene family {family!r}; valid: {sorted(_GENERATORS)}')
    generator = _GENERATORS[family]
    scenes = []
    for index in range(count):
        rng = random.Random(f'{campaign_seed}/{family}/{index}')
        scene_id = f'{family}_{index:05d}'
        # Campaign scenes are dict-authored (never files), so they validate
        # here; a generator drifting from the schema fails at generation.
        scenes.append((scene_id, validate_sim_params(generator(rng), source=scene_id)))
    return scenes
