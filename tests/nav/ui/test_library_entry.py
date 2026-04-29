"""Tests for ``nav.ui.library_entry`` — the YAML helper behind the dialog's
"Save as Library Entry..." button.

These tests do not need PyQt; they exercise the pure-Python helpers in
isolation so the (heavier) dialog tests stay focused on UI wiring.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
from tests.integration.sidecar import load_sidecar

from nav.ui.library_entry import (
    LibraryEntryDraft,
    build_sidecar_yaml,
    infer_obs_metadata,
)


def _fake_obs(
    *,
    cls_name: str = 'ObsCassiniISS',
    abspath: Path | None = None,
    detector: str | None = 'NAC',
    filter1: str | None = 'CL1',
    filter2: str | None = 'CL2',
) -> Any:
    """Build a tiny class with the named class name and the given attributes."""
    cls = type(
        cls_name,
        (object,),
        {
            'abspath': abspath,
            'detector': detector,
            'filter1': filter1,
            'filter2': filter2,
        },
    )
    return cls()


def test_infer_obs_metadata_reads_cassini_iss_fields(tmp_path: Path) -> None:
    """A Cassini-ISS-shaped obs yields the right mission / camera / filter combo."""
    img_path = tmp_path / 'W1521598221_1_CALIB.IMG'
    img_path.write_bytes(b'')
    draft = infer_obs_metadata(_fake_obs(abspath=img_path))
    assert draft.image_id == 'W1521598221_1_CALIB'
    assert draft.mission == 'CASSINI_ISS'
    assert draft.camera == 'NAC'
    assert draft.filter_combo == 'CL1+CL2'  # sorted, '+'-joined


def test_infer_obs_metadata_returns_blanks_for_unknown_obs() -> None:
    """An unrecognised obs class defaults to empty mission / camera fields."""
    obs = _fake_obs(cls_name='ObsMystery', detector=None, filter1=None, filter2=None)
    draft = infer_obs_metadata(obs)
    assert draft.mission == ''
    assert draft.camera == ''
    assert draft.filter_combo == ''


def test_infer_obs_metadata_canonicalizes_filter_order(tmp_path: Path) -> None:
    """``filter_combo`` is sorted lexically so ``CL+IR`` and ``IR+CL`` are identical."""
    img_path = tmp_path / 'X.IMG'
    img_path.write_bytes(b'')
    a = infer_obs_metadata(_fake_obs(abspath=img_path, filter1='IR', filter2='CL'))
    b = infer_obs_metadata(_fake_obs(abspath=img_path, filter1='CL', filter2='IR'))
    assert a.filter_combo == b.filter_combo == 'CL+IR'


def test_build_sidecar_yaml_round_trips_through_validator(tmp_path: Path) -> None:
    """A sidecar with TODO placeholders fails validation but with a clear error.

    The operator workflow is: save -> edit TODOs -> commit.  This test
    exercises the *post-edit* state by replacing every placeholder with
    a valid value.
    """
    draft = LibraryEntryDraft(
        image_id='W1521598221_1_CALIB',
        mission='CASSINI_ISS',
        camera='NAC',
        filter_combo='CL1+CL2',
    )
    yaml_text = build_sidecar_yaml(
        draft=draft,
        image_url='pds3://volumes/COISS_2xxx/COISS_2021/.../W1521598221_1_CALIB.IMG',
        offset_dv_px=12.345,
        offset_du_px=-6.789,
        ui_version='0.1.dev0',
        operator='rfrench',
        today=date(2026, 4, 28),
    )
    edited = (
        yaml_text.replace('TODO_REPLACE_PRIMARY_CLASS', 'body_mostly_offscreen')
        .replace(
            'TODO_REPLACE_TECHNIQUE',
            'BodyLimbNav',
        )
        .replace(
            'TODO: describe the scene and any caveats.',
            'Limb fit verified by overlay.',
        )
    )
    p = tmp_path / 'W1521598221_1_CALIB.yaml'
    p.write_text(edited)
    sidecar = load_sidecar(p)
    assert sidecar.image_id == 'W1521598221_1_CALIB'
    assert sidecar.mission == 'CASSINI_ISS'
    assert sidecar.camera == 'NAC'
    assert sidecar.filter_combo == 'CL1+CL2'
    assert sidecar.scene_tags == ('body_mostly_offscreen',)
    assert sidecar.ground_truth.offset_dv_px == pytest.approx(12.345)
    assert sidecar.ground_truth.offset_du_px == pytest.approx(-6.789)
    assert sidecar.ground_truth.operator == 'rfrench'
    assert sidecar.ground_truth.verified_date == date(2026, 4, 28)
    assert sidecar.expected.primary_technique == 'BodyLimbNav'


def test_build_sidecar_yaml_unedited_fails_validation(tmp_path: Path) -> None:
    """The unedited template trips the validator with a useful message.

    This is the safety net: an operator who forgets to edit TODOs gets a
    loud error from CI rather than a silently-broken library entry.
    """
    draft = LibraryEntryDraft(
        image_id='X', mission='CASSINI_ISS', camera='NAC', filter_combo='CL+CL'
    )
    yaml_text = build_sidecar_yaml(
        draft=draft,
        image_url='pds3://x/X.IMG',
        offset_dv_px=0.0,
        offset_du_px=0.0,
        ui_version='0.0.0',
        today=date(2026, 4, 28),
    )
    p = tmp_path / 'X.yaml'
    p.write_text(yaml_text)
    with pytest.raises(Exception, match=r'TODO_REPLACE_PRIMARY_CLASS|primary scene_tag'):
        load_sidecar(p)
