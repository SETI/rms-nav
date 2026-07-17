"""Matched sim-scene construction for the Section 7 realism runner.

Every real cohort frame gets one matched simulated frame: same instrument
signal chain (``artifacts.instrument_defaults``), same exposure, and scene
content of the same class -- a star field for a star frame, an ellipsoid
whose apparent diameter and phase angle mirror the real body for a limb
frame, a multi-feature ring system for a ring frame, a bare sky for the
negative / scattered-light frames.  Matching content class and exposure is
what makes the FOM 5 strata and the FOM 3 bins compare the forward model
rather than what the spacecraft happened to point at.

Scenes are deterministic: the random seed derives from the real frame's
image_id, so re-running the runner re-renders identical frames.

The matched scenes plant **zero offset**: realism statistics measure image
texture, not recovery, so predicted feature positions equal actual ones and
no shifting is needed on the sim side.
"""

from __future__ import annotations

import zlib
from typing import Any

from spindoctor.sim.scene import validate_sim_params

__all__ = [
    'limb_scene',
    'matched_scene',
    'ring_scene',
    'sky_scene',
    'star_field_scene',
]

# Matched-frame size: large enough for 16x16 sky patches of 32 px and for
# the PSD's 64-px tiles, small enough to keep 69 renders in minutes.
_SCENE_SIZE = 512

# Star classes place this many stars; magnitudes ladder from bright to the
# detection edge so the FOM 2 sample spans the usable brightness range.
_STAR_COUNT = 25
_STAR_VMAG_BRIGHT = 7.0
_STAR_VMAG_STEP = 0.2

# Scene classes mapped to matched content builders (sidecar primary tags).
_STAR_CLASSES = frozenset(
    {
        'star_dominated',
        'stars_plus_body',
        'faint_stars',
        'one_bright_star_no_body',
        'two_bright_stars_no_body',
    }
)
_LIMB_CLASSES = frozenset(
    {'body_full_fov', 'body_partial_overflow', 'high_phase_terminator', 'multi_body'}
)
_RING_CLASSES = frozenset({'ring_only_flat', 'ring_only_curved', 'ring_plus_body'})


def _seed_for(image_id: str) -> int:
    """Deterministic scene seed from the matched real frame's image_id."""
    return int(zlib.crc32(image_id.encode('utf-8')) & 0x7FFFFFFF)


def _base_scene(scene_name: str, instrument: str, exposure_sec: float, seed: int) -> dict[str, Any]:
    """The shared scene skeleton: instrument chain on, zero offset."""
    return {
        'schema_version': 2,
        'scene_name': scene_name,
        'instrument': instrument,
        'size_v': _SCENE_SIZE,
        'size_u': _SCENE_SIZE,
        'random_seed': seed,
        'exposure_sec': float(exposure_sec),
        'offset_v': 0.0,
        'offset_u': 0.0,
        'artifacts': {'instrument_defaults': True},
        'bodies': [],
    }


def _star_list(
    seed: int, count: int, *, vmag_bright: float, vmag_step: float
) -> list[dict[str, Any]]:
    """A deterministic scattered star field with a magnitude ladder."""
    # A small linear-congruential walk keeps this independent of numpy so
    # scene construction (not rendering) never draws from a global RNG.
    state = seed or 1
    stars: list[dict[str, Any]] = []
    margin = 24.0
    span = _SCENE_SIZE - 2 * margin
    for i in range(count):
        state = (1103515245 * state + 12345) % (1 << 31)
        v = margin + span * (state / float(1 << 31))
        state = (1103515245 * state + 12345) % (1 << 31)
        u = margin + span * (state / float(1 << 31))
        stars.append(
            {
                'name': f'R{i}',
                'v': round(v, 2),
                'u': round(u, 2),
                'vmag': round(vmag_bright + vmag_step * i, 2),
            }
        )
    return stars


def sky_scene(image_id: str, instrument: str, exposure_sec: float) -> dict[str, Any]:
    """A matched frame for negative / scattered-light / offscreen classes.

    A sparse dim star field keeps the frame from being pathologically
    empty while leaving nearly every patch pure detector output.
    """
    seed = _seed_for(image_id)
    scene = _base_scene(f'realism_sky_{image_id}', instrument, exposure_sec, seed)
    scene['stars'] = _star_list(seed, 5, vmag_bright=10.0, vmag_step=0.5)
    return scene


def star_field_scene(
    image_id: str, instrument: str, exposure_sec: float, *, count: int = _STAR_COUNT
) -> dict[str, Any]:
    """A matched star-field frame for the star scene classes."""
    seed = _seed_for(image_id)
    scene = _base_scene(f'realism_stars_{image_id}', instrument, exposure_sec, seed)
    scene['stars'] = _star_list(
        seed, count, vmag_bright=_STAR_VMAG_BRIGHT, vmag_step=_STAR_VMAG_STEP
    )
    return scene


def limb_scene(
    image_id: str,
    instrument: str,
    exposure_sec: float,
    *,
    diameter_px: float,
    phase_angle_deg: float,
) -> dict[str, Any]:
    """A matched limb frame: one ellipsoid at the real body's scale and phase.

    Parameters:
        image_id: The matched real frame's image_id (drives the seed).
        instrument: Sim instrument name.
        exposure_sec: The real frame's exposure.
        diameter_px: The real body's predicted apparent diameter; clamped
            so the rendered disc always presents a limb inside the frame.
        phase_angle_deg: The real frame's phase angle.
    """
    seed = _seed_for(image_id)
    scene = _base_scene(f'realism_limb_{image_id}', instrument, exposure_sec, seed)
    # The scene 'axis' values are apparent extents in pixels (a body with
    # axis1 = D renders a D-px-diameter silhouette); clamp so a
    # frame-filling real body still presents its limb inside the matched
    # frame while staying in the same FOM 3 resolution bin.
    axis = max(40.0, min(float(diameter_px), 0.85 * _SCENE_SIZE))
    center = _SCENE_SIZE / 2.0
    scene['bodies'] = [
        {
            'name': 'REALISM_BODY',
            'center_v': center,
            'center_u': center,
            'axis1': axis,
            'axis2': axis,
            'axis3': axis,
            'illumination_angle': 0.0,
            'phase_angle': float(min(max(phase_angle_deg, 0.0), 179.0)),
        }
    ]
    scene['stars'] = _star_list(seed, 5, vmag_bright=9.0, vmag_step=0.5)
    return scene


def ring_scene(
    image_id: str,
    instrument: str,
    exposure_sec: float,
    *,
    curved: bool,
    with_body: bool,
) -> dict[str, Any]:
    """A matched ring frame: three sharp-edged features around an annulus.

    Parameters:
        image_id: The matched real frame's image_id (drives the seed).
        instrument: Sim instrument name.
        exposure_sec: The real frame's exposure.
        curved: True for an open (curved-edge) geometry, False for a
            shallow opening whose edges cross the frame nearly straight.
        with_body: Add a central body (the ring_plus_body class).
    """
    seed = _seed_for(image_id)
    scene = _base_scene(f'realism_ring_{image_id}', instrument, exposure_sec, seed)
    opening = 65.0 if curved else 12.0
    center_v = _SCENE_SIZE / 2.0
    scene['ring_system'] = {
        'geometry': {
            'center_v': center_v,
            'center_u': _SCENE_SIZE / 2.0,
            'opening_deg_obs': opening,
            'opening_deg_sun': opening,
            'node_deg': 0.0,
        },
        'features': [
            {
                'name': 'REALISM_RING_A',
                'kind': 'ringlet',
                'tau': 2.0,
                'width': 30.0,
                'navigable': True,
                'orbit': {'a': 150.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0},
            },
            {
                'name': 'REALISM_RING_B',
                'kind': 'ringlet',
                'tau': 0.8,
                'width': 18.0,
                'navigable': True,
                'orbit': {'a': 205.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0},
            },
            {
                'name': 'REALISM_GAP',
                'kind': 'gap',
                'tau': 0.05,
                'width': 12.0,
                'navigable': True,
                'orbit': {'a': 160.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0},
            },
        ],
    }
    if with_body:
        scene['bodies'] = [
            {
                'name': 'REALISM_PLANET',
                'center_v': center_v,
                'center_u': _SCENE_SIZE / 2.0,
                'axis1': 90.0,
                'axis2': 90.0,
                'axis3': 90.0,
                'illumination_angle': 0.0,
                'phase_angle': 30.0,
            }
        ]
    scene['stars'] = _star_list(seed, 5, vmag_bright=9.5, vmag_step=0.5)
    return scene


def matched_scene(
    image_id: str,
    scene_class: str,
    instrument: str,
    exposure_sec: float,
    *,
    diameter_px: float = 150.0,
    phase_angle_deg: float = 45.0,
) -> dict[str, Any]:
    """The matched sim scene for one real cohort frame.

    Parameters:
        image_id: The real frame's image_id.
        scene_class: The sidecar's primary scene tag.
        instrument: Sim instrument name matching the cohort's signal chain
            and units (e.g. ``coiss_calib_nac`` for the CALIB cohort).
        exposure_sec: The real frame's exposure (seconds).
        diameter_px: Real body's apparent diameter for limb classes (from
            the navigator's model metadata; the default covers frames
            where no body model could be built).
        phase_angle_deg: Real frame's phase angle for limb classes.

    Returns:
        A validated scene mapping.
    """
    if scene_class in _STAR_CLASSES:
        count = {
            'one_bright_star_no_body': 1,
            'two_bright_stars_no_body': 2,
        }.get(scene_class, _STAR_COUNT)
        scene = star_field_scene(image_id, instrument, exposure_sec, count=count)
    elif scene_class in _LIMB_CLASSES:
        scene = limb_scene(
            image_id,
            instrument,
            exposure_sec,
            diameter_px=diameter_px,
            phase_angle_deg=phase_angle_deg,
        )
    elif scene_class in _RING_CLASSES:
        scene = ring_scene(
            image_id,
            instrument,
            exposure_sec,
            curved=scene_class != 'ring_only_flat',
            with_body=scene_class == 'ring_plus_body',
        )
    else:
        scene = sky_scene(image_id, instrument, exposure_sec)
    validate_sim_params(scene)
    return scene
