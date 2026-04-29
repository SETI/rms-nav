"""Save-as-library-entry helper for the manual-navigation dialog.

The manual-nav dialog drives library curation: an operator picks an
offset by hand, then clicks "Save as Library Entry..." to drop a sidecar
into ``tests/integration/image_library/images/<class>/<image_id>.yaml``.
This module owns the YAML template and the obs-introspection helper so
the dialog (which is otherwise PyQt-only) keeps a thin UI surface.

The emitted YAML carries ``TODO``-style placeholders for every field the
dialog cannot infer from the observation (scene class, expected
technique, etc.).  The library's own structural-invariants test
(:mod:`tests.integration.test_image_library`) refuses to load a sidecar
that still has placeholder enum values, so the operator must edit the
file before committing it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any

__all__ = [
    'LibraryEntryDraft',
    'build_sidecar_yaml',
    'infer_obs_metadata',
]


# Mission-string mapping — keys are class names of the supported obs subclasses.
# Add a new entry here when adding a new instrument; the new entry must use
# the mission codes from :data:`tests.integration.sidecar.ALLOWED_MISSIONS`.
_OBS_CLASS_TO_MISSION: dict[str, str] = {
    'ObsCassiniISS': 'CASSINI_ISS',
    'ObsVoyagerISS': 'VOYAGER_ISS',
    'ObsGalileoSSI': 'GOSSI',
    'ObsNewHorizonsLORRI': 'NHLORRI',
}


@dataclass(frozen=True)
class LibraryEntryDraft:
    """Auto-inferred fields the dialog seeds into a fresh sidecar."""

    image_id: str
    mission: str
    camera: str
    filter_combo: str


def infer_obs_metadata(obs: Any) -> LibraryEntryDraft:
    """Pull the sidecar's auto-fillable fields off an observation snapshot.

    Missing or unknown fields are returned as empty strings so the
    operator sees them in the YAML and can decide what to write; an
    empty mission/camera trips :func:`tests.integration.sidecar.load_sidecar`
    on validation, which is the right error.
    """
    image_id = ''
    abspath = getattr(obs, 'abspath', None)
    if abspath is not None and getattr(abspath, 'stem', None):
        image_id = str(abspath.stem)

    mission = _OBS_CLASS_TO_MISSION.get(type(obs).__name__, '')

    camera = ''
    detector = getattr(obs, 'detector', None)
    if detector:
        camera = str(detector)

    f1 = getattr(obs, 'filter1', None)
    f2 = getattr(obs, 'filter2', None)
    filters = [str(f) for f in (f1, f2) if f]
    filter_combo = '+'.join(sorted(filters)) if filters else ''

    return LibraryEntryDraft(
        image_id=image_id,
        mission=mission,
        camera=camera,
        filter_combo=filter_combo,
    )


def build_sidecar_yaml(
    *,
    draft: LibraryEntryDraft,
    image_url: str,
    offset_dv_px: float,
    offset_du_px: float,
    ui_version: str,
    operator: str | None = None,
    today: date | None = None,
) -> str:
    """Render the sidecar as a YAML string with placeholders for the rest.

    The output is deliberately not an opaque blob: every placeholder is
    a clear ``TODO_REPLACE_*`` string so the operator knows exactly what
    to edit before committing.  Fields the dialog can infer
    (``offset_*_px``, ``mission``, ``camera``, ``filter_combo``,
    ``image_id``) are filled in directly.
    """
    op_name = operator or os.environ.get('USER') or 'unknown'
    on_date = (today or date.today()).isoformat()
    return (
        'schema_version: 1\n'
        f'image_id: {draft.image_id or "TODO_REPLACE_IMAGE_ID"}\n'
        f'mission: {draft.mission or "TODO_REPLACE_MISSION"}'
        '            # CASSINI_ISS | VOYAGER_ISS | GOSSI | NHLORRI\n'
        f'camera: {draft.camera or "TODO_REPLACE_CAMERA"}'
        '              # NAC | WAC | SSI | NA | WA | LORRI\n'
        f"filter_combo: '{draft.filter_combo}'"
        "          # canonicalized: filters sorted, '+'-joined\n"
        f"image_url: '{image_url}'\n"
        '\n'
        'scene_tags:\n'
        '  - TODO_REPLACE_PRIMARY_CLASS         # First tag is the primary class;\n'
        '                                       # must match the directory the\n'
        '                                       # sidecar lives in.\n'
        '\n'
        'ground_truth:\n'
        f'  offset_dv_px: {offset_dv_px:.4f}\n'
        f'  offset_du_px: {offset_du_px:.4f}\n'
        '  offset_uncertainty_px: 1.0           # 1sigma marginal; tighten\n'
        '                                       # for bright stars / sharp limbs.\n'
        '  source: operator_verified\n'
        f'  operator: {op_name}\n'
        f'  verified_date: {on_date}\n'
        f"  ui_version: 'rms-nav {ui_version}'\n"
        '  notes: |\n'
        '    TODO: describe the scene and any caveats.\n'
        '\n'
        'expected:\n'
        '  status: ok                           # ok | failed | conflicted\n'
        '  confidence_tier: high                # high | medium | low | failed\n'
        '  primary_technique: TODO_REPLACE_TECHNIQUE  # e.g. BodyLimbNav\n'
        '  techniques_must_run: []\n'
        '  techniques_must_skip: []\n'
    )
