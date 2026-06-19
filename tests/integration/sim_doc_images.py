"""Render the documentation example images for the simulator chapters.

Run as ``python -m tests.integration.sim_doc_images`` (no holdings or SPICE
needed -- everything is rendered in-process).  It writes two galleries of
viewable PNGs:

* ``docs/dev_guide/_sim_images/`` -- illustrative scenes for the simulator
  developer-guide chapter (one per scene ingredient: ellipsoid vs mesh body,
  craters, crescent, rings, stars, multi-body, detector noise, stray light, and
  a composite frame).  These are hand-built ``sim_params`` dicts chosen to show
  one feature each.

* ``docs/simulator_report/_scene_images/`` -- the *actual* catalog scenes the
  sensitivity report discusses (rendered from their YAML), so a reader sees the
  frames behind each measurement.

Each gallery also gets a ``NOTES.md`` describing how to regenerate it, so the
images can be rebuilt if the renderer changes.  The PNGs are committed assets
(Sphinx embeds them directly); rerun this module and review the diff after any
change that alters rendering.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from nav.sim.png_export import render_scene_png
from nav.sim.scene import load_sim_scene

_DOCS = Path(__file__).parent.parent.parent / 'docs'
_GUI_DIR = _DOCS / 'dev_guide' / '_sim_images'
_SCENES_ROOT = Path(__file__).parent / 'sim_scenes'
_REPORT_DIR = _DOCS / 'simulator_report' / '_scene_images'

_COISS = 'coiss_nac'


def _scene(bodies: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """Build a minimal coiss_nac sim_params scene around a body list."""
    params: dict[str, Any] = {
        'size_v': 220,
        'size_u': 220,
        'instrument': _COISS,
        'random_seed': 7,
        'offset_v': 0.0,
        'offset_u': 0.0,
        'noise': {'poisson': True, 'read_noise_dn': 3.0},
        'bodies': bodies,
    }
    params.update(extra)
    return params


def _ellipsoid(**over: Any) -> dict[str, Any]:
    body = {
        'name': 'BODY',
        'center_v': 110.0,
        'center_u': 110.0,
        'axis1': 160.0,
        'axis2': 120.0,
        'axis3': 110.0,
        'illumination_angle': 25.0,
        'phase_angle': 35.0,
    }
    body.update(over)
    return body


def _mesh(**over: Any) -> dict[str, Any]:
    body = _ellipsoid()
    body.update(
        {
            'shape_model': 'polyhedral_mesh',
            'mesh_lumpiness': 0.4,
            'mesh_seed': 7,
            'pose_euler_deg': [10.0, 35.0, 0.0],
        }
    )
    body.update(over)
    return body


# (filename, sim_params, render kwargs) for the developer-guide gallery.
_GUI_GALLERY: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    ('ellipsoid_body', _scene([_ellipsoid()]), {'gamma': 1.1}),
    ('mesh_body', _scene([_mesh()]), {'gamma': 1.1}),
    (
        'body_craters',
        _scene([_ellipsoid(axis2=150.0, crater_fill=0.5, crater_max_radius=0.3)]),
        {'gamma': 1.1},
    ),
    ('crescent_body', _scene([_mesh(phase_angle=130.0)]), {'gamma': 1.4}),
    (
        'rings',
        _scene(
            [],
            rings=[
                {
                    'name': 'RINGLET_INNER',
                    'feature_type': 'RINGLET',
                    'center_v': 110.0,
                    'center_u': 110.0,
                    'inner_data': [{'mode': 1, 'a': 56.0}],
                    'outer_data': [{'mode': 1, 'a': 66.0}],
                    'shading_distance': 10.0,
                    'range': 1000.0,
                },
                {
                    'name': 'RINGLET_OUTER',
                    'feature_type': 'RINGLET',
                    'center_v': 110.0,
                    'center_u': 110.0,
                    'inner_data': [{'mode': 1, 'a': 90.0, 'ae': 6.0}],
                    'outer_data': [{'mode': 1, 'a': 98.0, 'ae': 6.0}],
                    'shading_distance': 10.0,
                    'range': 1000.0,
                },
            ],
        ),
        {'gamma': 1.2},
    ),
    (
        'star_field',
        _scene([], background_stars_num=70, background_stars_psf_sigma=1.0),
        {'gamma': 1.9, 'high_percentile': 99.9},
    ),
    (
        'body_and_stars',
        _scene([_ellipsoid(axis1=120.0, axis2=95.0, axis3=85.0)], background_stars_num=60),
        {'gamma': 1.6, 'high_percentile': 99.9},
    ),
    (
        'multi_body',
        _scene(
            [
                _ellipsoid(
                    name='A', center_v=70.0, center_u=70.0, axis1=90.0, axis2=80.0, axis3=75.0
                ),
                _mesh(name='B', center_v=150.0, center_u=150.0, axis1=70.0, axis2=55.0, axis3=50.0),
                _ellipsoid(
                    name='C', center_v=60.0, center_u=160.0, axis1=40.0, axis2=36.0, axis3=34.0
                ),
            ]
        ),
        {'gamma': 1.2},
    ),
    (
        'detector_noise',
        _scene(
            [_ellipsoid(axis1=120.0, axis2=95.0, axis3=85.0)],
            noise={
                'poisson': True,
                'read_noise_dn': 8.0,
                'cosmic_ray_rate_per_sec': 0.0006,
                'missing_data_rate': 0.01,
            },
            exposure_sec=2.0,
        ),
        {'gamma': 1.3, 'high_percentile': 99.7},
    ),
    (
        'stray_light_gradient',
        _scene(
            [_ellipsoid(axis1=120.0, axis2=95.0, axis3=85.0)],
            stray_light={'amplitude': 0.5, 'direction_deg': 35.0, 'model': 'linear'},
        ),
        {'gamma': 1.2},
    ),
    (
        'composite_scene',
        _scene(
            [_mesh(name='MOON', center_v=140.0, center_u=95.0, axis1=90.0, axis2=72.0, axis3=66.0)],
            background_stars_num=40,
            rings=[
                {
                    'name': 'RINGLET',
                    'feature_type': 'RINGLET',
                    'center_v': 110.0,
                    'center_u': 110.0,
                    'inner_data': [{'mode': 1, 'a': 150.0, 'ae': 8.0}],
                    'outer_data': [{'mode': 1, 'a': 170.0, 'ae': 8.0}],
                    'shading_distance': 6.0,
                    'range': 5000.0,
                },
            ],
            noise={'poisson': True, 'read_noise_dn': 4.0},
        ),
        {'gamma': 1.5, 'high_percentile': 99.9},
    ),
]

# (filename, scene-relative path) for the report scene gallery: the actual
# catalog scenes the report measures, rendered from their YAML.
_REPORT_SCENES: list[tuple[str, str, dict[str, Any]]] = [
    ('regular_sphere_base', 'phase_sweep_regular_body/regular_sphere_base.yaml', {'gamma': 1.1}),
    ('disc', 'algorithmic_invariants/planted_offset_disc.yaml', {'gamma': 1.1}),
    ('blob_crescent', 'algorithmic_invariants/planted_offset_blob_crescent.yaml', {'gamma': 1.4}),
    ('mesh_disc', 'algorithmic_invariants/planted_offset_irregular.yaml', {'gamma': 1.1}),
    ('limb_mesh', 'algorithmic_invariants/planted_offset_limb_mesh.yaml', {'gamma': 1.1}),
    ('shape_mismatch', 'algorithmic_invariants/planted_offset_shapemismatch.yaml', {'gamma': 1.1}),
    (
        'mesh_crescent',
        'algorithmic_invariants/planted_offset_blob_mesh_crescent.yaml',
        {'gamma': 1.4},
    ),
    ('pose_disagree', 'phase_sweep_irregular_body/hyperion_pose_disagree.yaml', {'gamma': 1.1}),
    ('ring', 'algorithmic_invariants/planted_offset_ring.yaml', {'gamma': 1.3}),
    (
        'star_field',
        'algorithmic_invariants/planted_offset_star_field.yaml',
        {'gamma': 1.9, 'high_percentile': 99.9},
    ),
]

_GUI_NOTES = """\
# Developer-guide simulator gallery

These PNGs illustrate the simulator's scene ingredients for
`dev_guide_simulator.rst`. They are rendered in-process by
`tests/integration/sim_doc_images.py` from hand-built `sim_params` dicts (one
feature per image) and committed as Sphinx assets.

Regenerate (and review the diff) after any change that alters rendering:

    python -m tests.integration.sim_doc_images

Each image uses `nav.sim.png_export.render_scene_png`, which stretches detector
counts to visible grayscale with a percentile clip plus a per-image gamma (dim
features such as a crescent or a faint star field use a higher gamma). The scene
definitions live in `_GUI_GALLERY` in the generator; edit there to change a
panel.
"""

_REPORT_NOTES = """\
# Simulator-report scene gallery

These PNGs are the actual catalog scenes the sensitivity report measures,
rendered from their YAML so a reader sees the frame behind each result. They are
produced in-process by `tests/integration/sim_doc_images.py` and committed as
Sphinx assets.

Regenerate (and review the diff) after a scene's geometry changes:

    python -m tests.integration.sim_doc_images

The mapping from PNG to scene file is `_REPORT_SCENES` in the generator. Each
image is rendered with `nav.sim.png_export.render_scene_png` (percentile stretch
plus a per-image gamma).
"""


def generate() -> list[Path]:
    """Render both galleries and write their NOTES files; return all paths."""
    written: list[Path] = []
    _GUI_DIR.mkdir(parents=True, exist_ok=True)
    for name, params, kwargs in _GUI_GALLERY:
        written.append(render_scene_png(params, _GUI_DIR / f'{name}.png', upscale=2, **kwargs))
    (_GUI_DIR / 'NOTES.md').write_text(_GUI_NOTES)

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rel_path, kwargs in _REPORT_SCENES:
        scene_path = _SCENES_ROOT / rel_path
        if not scene_path.is_file():
            print(f'skip {name}: {scene_path} not found', file=sys.stderr)
            continue
        params = load_sim_scene(scene_path)
        written.append(
            render_scene_png(
                params, _REPORT_DIR / f'{name}.png', ignore_offset=False, upscale=2, **kwargs
            )
        )
    (_REPORT_DIR / 'NOTES.md').write_text(_REPORT_NOTES)
    return written


def main() -> int:
    """Entry point: render every documentation image."""
    paths = generate()
    print(f'Wrote {len(paths)} documentation image(s):')
    for path in paths:
        print(f'  {path.relative_to(_DOCS.parent)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
