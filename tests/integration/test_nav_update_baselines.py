"""Unit tests for the ``nav_update_baselines`` CLI helpers.

The end-to-end path that runs the orchestrator against real holdings is
exercised by ``test_baselines.test_regression_baseline_exact_match``;
this module covers the parts that work without ``PDS3_HOLDINGS_DIR``:
argument parsing, sidecar selection / filtering, the diff and
write-decision logic, and the dry-run guarantee.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path
from unittest import mock

import pytest

# Make the CLI module importable from a source-tree checkout (mirrors the
# sys.path manipulation the script itself performs at top-level).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from main import nav_update_baselines  # noqa: E402  (path-dependent import)
from tests.integration.baseline import Baseline, baseline_path  # noqa: E402
from tests.integration.sidecar import (  # noqa: E402
    Expected,
    GroundTruth,
    LibraryRoot,
    Sidecar,
)


def _make_sidecar(
    image_id: str,
    *,
    image_url: str = 'pds3://volumes/COISS_2xxx/COISS_2021/data/x.IMG',
    primary_class: str = 'star_dominated',
) -> Sidecar:
    """Build a minimally valid Sidecar for tests that don't go to disk."""
    return Sidecar(
        path=Path(f'/tmp/{image_id}.yaml'),
        schema_version=1,
        image_id=image_id,
        mission='COISS',
        camera='NAC',
        filter_combo='CL1+CL2',
        image_url=image_url,
        scene_tags=(primary_class,),
        ground_truth=GroundTruth(
            offset_dv_px=1.0,
            offset_du_px=2.0,
            offset_uncertainty_px=1.0,
            source='operator_verified',
            operator='tester',
            verified_date=date(2026, 5, 1),
            ui_version='rms-nav 0.0',
            notes=None,
        ),
        expected=Expected(
            status='ok',
            confidence_tier='high',
            primary_technique='BodyLimbNav',
        ),
    )


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_requires_selection() -> None:
    """Neither ``--all`` nor ``--image-id`` is a usage error."""
    with pytest.raises(SystemExit) as exc:
        nav_update_baselines.parse_args([])
    assert exc.value.code == 2


def test_parse_args_all_and_image_id_are_mutually_exclusive() -> None:
    """``--all`` and ``--image-id`` cannot be combined."""
    with pytest.raises(SystemExit) as exc:
        nav_update_baselines.parse_args(['--all', '--image-id', 'X'])
    assert exc.value.code == 2


def test_parse_args_all_alone_is_accepted() -> None:
    """``--all`` alone parses to ``Namespace(all=True, image_id=[])``."""
    args = nav_update_baselines.parse_args(['--all'])
    assert args.all is True
    assert args.image_id == []
    assert args.dry_run is False


def test_parse_args_repeatable_image_id() -> None:
    """``--image-id`` is repeatable; values accumulate in order."""
    args = nav_update_baselines.parse_args(['--image-id', 'A', '--image-id', 'B'])
    assert args.all is False
    assert args.image_id == ['A', 'B']


def test_parse_args_dry_run_flag() -> None:
    """``--dry-run`` flips the corresponding boolean."""
    args = nav_update_baselines.parse_args(['--all', '--dry-run'])
    assert args.dry_run is True


# ---------------------------------------------------------------------------
# select_sidecars
# ---------------------------------------------------------------------------


def test_select_sidecars_all_returns_every_discovered_sidecar() -> None:
    """``--all`` returns every loaded sidecar; missing-list is empty."""
    fake_a = _make_sidecar('A')
    fake_b = _make_sidecar('B')
    with (
        mock.patch.object(LibraryRoot, 'discover_sidecar_paths', return_value=['/p/A', '/p/B']),
        mock.patch('tests.integration.sidecar.load_sidecar', side_effect=[fake_a, fake_b]),
    ):
        selected, missing = nav_update_baselines.select_sidecars(
            LibraryRoot(), use_all=True, image_ids=[]
        )
    assert [s.image_id for s in selected] == ['A', 'B']
    assert missing == []


def test_select_sidecars_filters_by_image_id() -> None:
    """``--image-id`` selects exactly the named sidecar(s)."""
    fake_a = _make_sidecar('A')
    fake_b = _make_sidecar('B')
    fake_c = _make_sidecar('C')
    with (
        mock.patch.object(
            LibraryRoot, 'discover_sidecar_paths', return_value=['/p/A', '/p/B', '/p/C']
        ),
        mock.patch(
            'tests.integration.sidecar.load_sidecar',
            side_effect=[fake_a, fake_b, fake_c],
        ),
    ):
        selected, missing = nav_update_baselines.select_sidecars(
            LibraryRoot(), use_all=False, image_ids=['B', 'C']
        )
    assert [s.image_id for s in selected] == ['B', 'C']
    assert missing == []


def test_select_sidecars_reports_missing_image_ids() -> None:
    """Unknown ``--image-id`` values land in the missing list, not selected."""
    fake_a = _make_sidecar('A')
    with (
        mock.patch.object(LibraryRoot, 'discover_sidecar_paths', return_value=['/p/A']),
        mock.patch('tests.integration.sidecar.load_sidecar', side_effect=[fake_a]),
    ):
        selected, missing = nav_update_baselines.select_sidecars(
            LibraryRoot(), use_all=False, image_ids=['A', 'NOT_THERE']
        )
    assert [s.image_id for s in selected] == ['A']
    assert missing == ['NOT_THERE']


# ---------------------------------------------------------------------------
# update_one (mocked navigate_image_files)
# ---------------------------------------------------------------------------


def test_update_one_creates_new_baseline_when_missing(tmp_path: Path) -> None:
    """A sidecar with no on-disk baseline yields CREATE and writes the file."""
    sidecar = _make_sidecar('NEW_001')
    with mock.patch(
        'nav.navigate_image_files.navigate_image_files',
        return_value=(True, {'offset': [12.34567, -7.89012], 'confidence': 0.876}),
    ):
        outcome = nav_update_baselines.update_one(sidecar, baselines_dir=tmp_path, dry_run=False)
    assert outcome.kind == 'CREATE'
    target = baseline_path(tmp_path, 'NEW_001')
    assert target.is_file()
    written = json.loads(target.read_text())
    assert written['offset_dv_px'] == 12.3457  # rounded to 4 decimals
    assert written['confidence'] == 0.876  # rounded to 3


def test_update_one_unchanged_when_baseline_already_matches(tmp_path: Path) -> None:
    """Identical inputs yield UNCHANGED and leave the file untouched."""
    sidecar = _make_sidecar('STABLE_001')
    existing = Baseline(
        image_id='STABLE_001',
        offset_dv_px=1.0,
        offset_du_px=2.0,
        confidence=0.5,
    )
    target = baseline_path(tmp_path, 'STABLE_001')
    target.write_text(existing.to_json())
    mtime_before = target.stat().st_mtime_ns
    with mock.patch(
        'nav.navigate_image_files.navigate_image_files',
        return_value=(True, {'offset': [1.0, 2.0], 'confidence': 0.5}),
    ):
        outcome = nav_update_baselines.update_one(sidecar, baselines_dir=tmp_path, dry_run=False)
    assert outcome.kind == 'UNCHANGED'
    # The matching write would still have produced the same bytes, but a
    # caller relying on mtime to detect baseline churn deserves a no-op.
    assert target.stat().st_mtime_ns == mtime_before


def test_update_one_update_includes_field_diff(tmp_path: Path) -> None:
    """Different inputs yield UPDATE with a per-field diff in the detail."""
    sidecar = _make_sidecar('DRIFT_001')
    existing = Baseline(
        image_id='DRIFT_001',
        offset_dv_px=1.0,
        offset_du_px=2.0,
        confidence=0.500,
    )
    target = baseline_path(tmp_path, 'DRIFT_001')
    target.write_text(existing.to_json())
    with mock.patch(
        'nav.navigate_image_files.navigate_image_files',
        return_value=(True, {'offset': [1.5, 2.0], 'confidence': 0.555}),
    ):
        outcome = nav_update_baselines.update_one(sidecar, baselines_dir=tmp_path, dry_run=False)
    assert outcome.kind == 'UPDATE'
    assert 'dv +1.0000 -> +1.5000' in outcome.detail
    assert 'conf 0.500 -> 0.555' in outcome.detail
    assert 'du' not in outcome.detail  # du didn't change
    written = Baseline(
        image_id='DRIFT_001',
        offset_dv_px=1.5,
        offset_du_px=2.0,
        confidence=0.555,
    )
    assert json.loads(target.read_text()) == json.loads(written.to_json())


def test_update_one_dry_run_does_not_write(tmp_path: Path) -> None:
    """``--dry-run`` reports the would-action without touching disk."""
    sidecar = _make_sidecar('DRY_001')
    target = baseline_path(tmp_path, 'DRY_001')
    with mock.patch(
        'nav.navigate_image_files.navigate_image_files',
        return_value=(True, {'offset': [1.5, 2.0], 'confidence': 0.555}),
    ):
        outcome = nav_update_baselines.update_one(sidecar, baselines_dir=tmp_path, dry_run=True)
    assert outcome.kind == 'CREATE'
    assert not target.exists()


def test_update_one_failed_when_orchestrator_returns_no_offset(tmp_path: Path) -> None:
    """A run that produces no offset yields FAILED and writes nothing."""
    sidecar = _make_sidecar('FAIL_001')
    with mock.patch(
        'nav.navigate_image_files.navigate_image_files',
        return_value=(False, {'status': 'failed'}),
    ):
        outcome = nav_update_baselines.update_one(sidecar, baselines_dir=tmp_path, dry_run=False)
    assert outcome.kind == 'FAILED'
    assert 'no offset' in outcome.detail
    assert not baseline_path(tmp_path, 'FAIL_001').exists()


# ---------------------------------------------------------------------------
# main precondition: PDS3_HOLDINGS_DIR
# ---------------------------------------------------------------------------


def test_main_refuses_to_run_without_holdings() -> None:
    """``main`` exits with code 2 when ``PDS3_HOLDINGS_DIR`` is unset."""
    with mock.patch.dict(os.environ, {}, clear=True):
        rc = nav_update_baselines.main(['--all'])
    assert rc == 2
