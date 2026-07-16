"""Key inventory and information-boundary classification for sim scenes.

Every top-level and per-object key a scene may carry is inventoried here
(unknown keys fail validation so typos do not silently render the default
scene), and every inventory key is classified as either idealized
(information the production pipeline could know from catalogs, SPICE, labels,
or config: exposed to the navigator through ``obs.nav_params``) or truth
(nature's values, planted errors, variance knobs, and contaminants: readable
only by the image-side renderer).  :data:`TRUTH_KEYS` is the machine-readable
truth set the ObsSim boundary filter strips and the structural boundary test
iterates.  The import-time completeness assertion below keeps the
classification complete and disjoint, so a key added to the schema without a
classification fails everything loudly, not just one test.

:mod:`spindoctor.sim.scene` is the public entry point: it consumes this
inventory for validation, builds the filtered navigator view from the
classification, and re-exports the boundary names.
"""


class SimSceneValidationError(ValueError):
    """Raised when a sim scene YAML is missing or malformed."""


# ---------------------------------------------------------------------------
# Key inventory and information-boundary classification.
#
# _ALLOWED_KEYS / _*_KEYS are the complete inventory (unknown keys fail
# validation so typos do not silently render the default scene).  The
# *_IDEALIZED_KEYS / *_TRUTH_KEYS sets classify every inventory key for the
# information boundary; the import-time assertion below keeps the
# classification complete and disjoint.
# ---------------------------------------------------------------------------

# Every top-level key a scene may carry.  These are the flat runtime sim_params
# names the renderer / ObsSim consume directly, plus the schema_version /
# scene_name metadata the renderer ignores.
_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        'schema_version',
        'scene_name',
        'instrument',
        'size_v',
        'size_u',
        'random_seed',
        'exposure_sec',
        'offset_v',
        'offset_u',
        'offset_rotation_deg',
        'midtime_utc',
        'closest_planet',
        'time',
        'ring_epoch',
        'shade_solid_rings',
        'oversample',
        'optics',
        'detector',
        'artifacts',
        'spk_error',
        'bodies',
        'rings',
        'stars',
        'background_stars_num',
        'background_stars_psf_sigma',
        'background_stars_distribution_exponent',
        'noise',
        'instrument_config',
        'fit_camera_rotation',
    }
)

# Top-level idealized keys: frame identity, emulated-instrument configuration,
# and epoch/timing values the production pipeline reads from labels and
# published models.  'ring_epoch' is deliberately idealized: the precessing
# ring model's epoch is catalog knowledge the navigator-side ring model reads.
TOP_LEVEL_IDEALIZED_KEYS: frozenset[str] = frozenset(
    {
        'schema_version',
        'scene_name',
        'instrument',
        'size_v',
        'size_u',
        'exposure_sec',
        'midtime_utc',
        'closest_planet',
        'time',
        'ring_epoch',
        'bodies',
        'rings',
        'stars',
        'instrument_config',
        'fit_camera_rotation',
    }
)

# Top-level truth keys: the planted pointing error the navigator must recover,
# the RNG realization, and the contaminant / noise fields.  The renderer's
# appearance knob 'shade_solid_rings' is image-side only (the navigator's
# ring template is always solid-shaded by its own convention).
TOP_LEVEL_TRUTH_KEYS: frozenset[str] = frozenset(
    {
        'random_seed',
        'offset_v',
        'offset_u',
        'offset_rotation_deg',
        'shade_solid_rings',
        'oversample',
        'optics',
        'detector',
        'artifacts',
        'spk_error',
        'background_stars_num',
        'background_stars_psf_sigma',
        'background_stars_distribution_exponent',
        'noise',
    }
)

# Per-body idealized keys: the ellipsoid/mesh geometry, pose, lighting, and
# physical scale the production pipeline knows from SPICE and shape catalogs.
# The mesh keys are idealized because the published shape model of an
# irregular body is catalog knowledge; a scene plants shape error through
# 'nav_override', not by hiding the mesh.
_BODY_IDEALIZED_KEYS: frozenset[str] = frozenset(
    {
        'name',
        'shape_model',
        'center_v',
        'center_u',
        'axis1',
        'axis2',
        'axis3',
        'rotation_z',
        'rotation_tilt',
        'illumination_angle',
        'phase_angle',
        'range_km',
        'km_per_pixel',
        'mesh_lumpiness',
        'mesh_n_lat',
        'mesh_n_lon',
        'mesh_seed',
        'pose_euler_deg',
    }
)

# Per-body truth keys: surface texture (craters) is nature's terrain, 'seed'
# is its realization, and 'anti_aliasing' is an image-side rendering-fidelity
# knob (the navigator's template always renders at full anti-aliasing).
# 'nav_override' is special: its VALUES are what the navigator believes
# (idealized), so build_nav_params overlays them onto the body and drops the
# key; the underlying overridden true values never cross.
_BODY_TRUTH_KEYS: frozenset[str] = frozenset(
    {
        'crater_fill',
        'crater_min_radius',
        'crater_max_radius',
        'crater_power_law_exponent',
        'crater_relief_scale',
        'seed',
        'anti_aliasing',
        'nav_override',
    }
)

_BODY_KEYS: frozenset[str] = _BODY_IDEALIZED_KEYS | _BODY_TRUTH_KEYS

# Per-star idealized keys: catalog identity, position, magnitude, spectral
# class, the predicted smear vector (the pipeline computes it from attitude
# telemetry), and the PSF fitting-window size (instrument configuration).
_STAR_IDEALIZED_KEYS: frozenset[str] = frozenset(
    {
        'name',
        'catalog_name',
        'v',
        'u',
        'vmag',
        'spectral_class',
        'move_v',
        'move_u',
        'psf_size',
    }
)

# Per-star truth keys: a per-star PSF width override is an anomaly of the
# rendered image (the navigator only knows the instrument's published PSF).
_STAR_TRUTH_KEYS: frozenset[str] = frozenset({'psf_sigma'})

_STAR_KEYS: frozenset[str] = _STAR_IDEALIZED_KEYS | _STAR_TRUTH_KEYS

# Per-ring keys, all idealized at present fidelity: the mode-1 orbits ARE the
# catalog orbits, with no planted per-feature error.  'range' is the
# z-order/depth hint of the rings list; the list and its keys remain valid
# until a ring-system block with plantable per-feature error replaces them.
_RING_IDEALIZED_KEYS: frozenset[str] = frozenset(
    {
        'name',
        'feature_type',
        'center_v',
        'center_u',
        'shading_distance',
        'inner_data',
        'outer_data',
        'range',
        'range_km',
    }
)

_RING_TRUTH_KEYS: frozenset[str] = frozenset()

_RING_KEYS: frozenset[str] = _RING_IDEALIZED_KEYS | _RING_TRUTH_KEYS

# The object blocks of the schema: block name -> (allowed, idealized, truth).
_OBJECT_BLOCKS: dict[str, tuple[frozenset[str], frozenset[str], frozenset[str]]] = {
    'bodies': (_BODY_KEYS, _BODY_IDEALIZED_KEYS, _BODY_TRUTH_KEYS),
    'stars': (_STAR_KEYS, _STAR_IDEALIZED_KEYS, _STAR_TRUTH_KEYS),
    'rings': (_RING_KEYS, _RING_IDEALIZED_KEYS, _RING_TRUTH_KEYS),
}

# The machine-readable truth-key set the ObsSim boundary filter strips and
# the structural boundary test iterates.  Per-object-block entries use dotted
# '<block>.<key>' paths; top-level entries are bare key names.
TRUTH_KEYS: frozenset[str] = frozenset(TOP_LEVEL_TRUTH_KEYS) | frozenset(
    f'{block}.{key}'
    for block, (_allowed, _idealized, truth) in _OBJECT_BLOCKS.items()
    for key in truth
)


def _assert_boundary_classification_complete() -> None:
    """Every schema key must be classified idealized or truth, never both.

    Runs at import so a schema change that adds a key without classifying it
    fails everything loudly, not just one test.
    """
    overlap = TOP_LEVEL_IDEALIZED_KEYS & TOP_LEVEL_TRUTH_KEYS
    assert not overlap, f'top-level keys classified both idealized and truth: {sorted(overlap)}'
    unclassified = _ALLOWED_KEYS - (TOP_LEVEL_IDEALIZED_KEYS | TOP_LEVEL_TRUTH_KEYS)
    assert not unclassified, f'top-level keys with no boundary class: {sorted(unclassified)}'
    unknown = (TOP_LEVEL_IDEALIZED_KEYS | TOP_LEVEL_TRUTH_KEYS) - _ALLOWED_KEYS
    assert not unknown, f'classified top-level keys not in the inventory: {sorted(unknown)}'
    for block, (allowed, idealized, truth) in _OBJECT_BLOCKS.items():
        overlap = idealized & truth
        assert not overlap, f'{block} keys classified both idealized and truth: {sorted(overlap)}'
        unclassified = allowed - (idealized | truth)
        assert not unclassified, f'{block} keys with no boundary class: {sorted(unclassified)}'


_assert_boundary_classification_complete()
