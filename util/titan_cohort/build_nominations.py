"""Draft image-library nominations for the Titan cohort (pending operator votes).

Navigates the calibrated counterpart of each nominated frame -- the library
holds ``_CALIB`` products, not the raw ones the cohort campaign runs -- and
writes, under ``nominations/``, one draft sidecar per frame plus a manifest
naming why each was chosen.

These are DRAFTS and they deliberately do not live under
``tests/integration/image_library/images/``.  A library sidecar's ground
truth must carry ``source: operator_verified``; what a draft can carry is the
autonomous fix, which is a proposal.  Promoting one means an operator
verifying the offset, choosing the scene class, and moving the file.

Run::

    source /seti/newnav/setup.sh
    python util/titan_cohort/build_nominations.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'src'))

from cohort import CohortFrame, resolved_cohort  # noqa: E402

OUT_DIR = HERE / 'nominations'

# The nominated frames and why each is in the set.  Chosen to span camera,
# filter, and phase while keeping every frame one the annotation called
# clean and the campaign navigated with an attributable outcome.
NOMINATIONS: tuple[tuple[str, str], ...] = (
    (
        'W1822132529_1',
        'Mid-phase WAC Titan with nothing else navigable in the frame: the '
        'plain case, and the frame the technique has been exercised on since '
        'its first integration test.',
    ),
    (
        'W1552216540_1',
        'Red-filter half of a cross-filter pair 106 s apart, with an '
        'independent star lock on the same frame.  Together with its violet '
        'twin it is the library entry that would pin the filter-independence '
        'claim.',
    ),
    (
        'W1552216646_1',
        'Violet half of the same pair.  Its haze fit and the frame star lock '
        'agree to 0.9 px, and its fitted haze radius sits 66 km above the red '
        'frame - the wavelength-dependent haze top, in one matched pair.',
    ),
    (
        'W1643376091_1',
        'High phase (135 deg), where the sunward limb has shrunk toward a '
        'crescent and the arc fit has least support.  The upper end of the '
        'working range.',
    ),
    (
        'W1883905091_1',
        'The methane surface window (CB3), the one Cassini filter that sees '
        'through the haze to the ground.  The method must not depend on the '
        'haze being opaque, and this is the frame that says so.',
    ),
    (
        'N1702239331_1',
        'NAC frame carrying a companion moon and a star lock alongside Titan: '
        'the multi-technique composition, where the haze result has to fuse '
        'with others rather than stand alone.',
    ),
)


def _calib_url(image_id: str) -> str | None:
    """Holdings URL of a frame's calibrated product, or None when absent."""
    root = os.environ.get('PDS3_HOLDINGS_DIR')
    if not root:
        raise RuntimeError('PDS3_HOLDINGS_DIR is not set; source /seti/newnav/setup.sh')
    matches = glob.glob(f'{root.rstrip("/")}/calibrated/COISS_?xxx/*/data/*/{image_id}_CALIB.IMG')
    return matches[0] if matches else None


def _navigate(url: str, image_id: str) -> dict[str, Any]:
    """Run the full pipeline on one calibrated frame and return its metadata."""
    import tempfile

    from filecache import FCPath

    from spindoctor.dataset.dataset import ImageFile, ImageFiles
    from spindoctor.navigate_image_files import navigate_image_files
    from spindoctor.obs import ObsCassiniISS

    image_files = ImageFiles(
        image_files=[
            ImageFile(
                image_file_url=FCPath(url),
                label_file_url=FCPath(url),
                results_path_stub=image_id,
            )
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        _success, metadata = navigate_image_files(
            ObsCassiniISS, image_files, FCPath(tmp), write_output_files=False
        )
    return metadata


def _iso_datetime(stamp: str | None) -> str:
    """Convert a PDS3 day-of-year timestamp to the sidecar's ISO form."""
    if not stamp:
        return ''
    try:
        return datetime.strptime(stamp, '%Y-%jT%H:%M:%S.%f').isoformat(timespec='milliseconds')
    except ValueError:
        return ''


def _draft_sidecar(frame: CohortFrame, url: str, metadata: dict[str, Any], rationale: str) -> str:
    """Render one draft sidecar as YAML text, marked pending."""
    nav = metadata.get('navigation_result') or {}
    offset = metadata.get('offset') or [float('nan'), float('nan')]
    sigma = nav.get('sigma_px') or [1.0, 1.0]
    per_technique = [t for t in nav.get('per_technique', []) if not t.get('spurious')]
    ordered = sorted(
        per_technique,
        key=lambda entry: (-float(entry.get('confidence', 0.0)), str(entry.get('technique_name'))),
    )
    primary = ordered[0]['technique_name'] if ordered else 'none'
    haze = next(
        (t for t in nav.get('per_technique', []) if t['technique_name'] == 'TitanHazeNav'), None
    )
    diagnostics = (haze or {}).get('diagnostics') or {}
    haze_offset = (haze or {}).get('offset_px') or ['?', '?']
    library_url = 'pds3://' + url.split('/holdings/', 1)[-1]
    uncertainty = round(max(float(sigma[0]), float(sigma[1])), 2)
    note = (
        f'Autonomous fix from the 2026-07-26 Titan cohort campaign '
        f'(util/titan_cohort/CAMPAIGN_20260726.md). TitanHazeNav reports '
        f'({haze_offset[0]}, {haze_offset[1]}) px at confidence '
        f'{(haze or {}).get("confidence", "?")} from a '
        f'{diagnostics.get("envelope_diameter_px", "?")} px envelope at phase '
        f'{diagnostics.get("phase_deg", "?")} deg, with '
        f'{diagnostics.get("arc_rays_inlier", "?")} of '
        f'{diagnostics.get("arc_rays_total", "?")} limb rays inlier and an arc '
        f'residual of {diagnostics.get("arc_residual_rms_px", "?")} px. '
        f'The fused offset above is the ensemble answer, which on a frame with '
        f'other committed techniques is not the haze fit alone.'
    )
    note_block = '\n'.join('    ' + line for line in textwrap.wrap(note, 68))
    return f"""# DRAFT NOMINATION -- PENDING OPERATOR VOTES.  Not a library sidecar.
#
# ground_truth below is the AUTONOMOUS fix, not a verified one.  Promoting
# this frame means an operator confirming the offset against the overlay,
# choosing the scene class (see the campaign record's recommendation), and
# moving the file into tests/integration/image_library/images/<class>/ with
# source, operator, and verified_date filled in.
#
# Why this frame:
{chr(10).join('#   ' + line for line in textwrap.wrap(rationale, 70))}
schema_version: 1
image_id: {frame.image_id}_CALIB
mission: COISS
camera: {frame.camera}
image_datetime_utc: '{_iso_datetime(frame.image_time)}'
exposure_time_sec: {frame.exposure_sec if frame.exposure_sec is not None else 0.0}
filter_combo: '{'+'.join(sorted(f for f in (frame.filter1, frame.filter2) if f))}'
image_url: '{library_url}'

# PENDING: the campaign recommends a new titan_haze class; until that is
# decided this frame belongs to no declared class.
scene_tags:
  - PENDING_SCENE_CLASS

ground_truth:
  offset_dv_px: {round(float(offset[0]), 4)}
  offset_du_px: {round(float(offset[1]), 4)}
  offset_uncertainty_px: {uncertainty}
  source: PENDING_OPERATOR_VERIFICATION
  operator: PENDING
  verified_date: PENDING
  ui_version: PENDING
  notes: |
{note_block}

expected:
  status: {nav.get('status', 'PENDING')}
  confidence_tier: {nav.get('confidence_rank', 'PENDING')}
  primary_technique: {primary}
  techniques_must_run: []
  techniques_must_skip: []
"""


def main(argv: list[str] | None = None) -> int:
    """Navigate every nominated frame and write its draft sidecar."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import pdslogger

    from spindoctor.config import IMAGE_LOGGER, MAIN_LOGGER

    for logger in (IMAGE_LOGGER, MAIN_LOGGER):
        logger.remove_all_handlers()
        logger.add_handler(pdslogger.NULL_HANDLER)

    frames = {frame.image_id: frame for frame in resolved_cohort()}
    manifest: list[dict[str, Any]] = []
    for image_id, rationale in NOMINATIONS:
        url = _calib_url(image_id)
        if url is None:
            print(f'{image_id}: no calibrated product in holdings; skipped')
            continue
        metadata = _navigate(url, image_id)
        nav = metadata.get('navigation_result') or {}
        (args.out_dir / f'{image_id}_CALIB.yaml').write_text(
            _draft_sidecar(frames[image_id], url, metadata, rationale)
        )
        haze = next(
            (t for t in nav.get('per_technique', []) if t['technique_name'] == 'TitanHazeNav'),
            None,
        )
        manifest.append(
            {
                'image_id': f'{image_id}_CALIB',
                'rationale': rationale,
                'status': nav.get('status'),
                'confidence_tier': nav.get('confidence_rank'),
                'offset_px': metadata.get('offset'),
                'titan_haze_confidence': (haze or {}).get('confidence'),
                'titan_haze_spurious': (haze or {}).get('spurious'),
            }
        )
        print(f'{image_id}: {nav.get("status")} / {nav.get("confidence_rank")}')
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{len(manifest)} draft nominations -> {args.out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
