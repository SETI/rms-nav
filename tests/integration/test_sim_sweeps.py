"""Single-variable sweep invariants (Phase T3).

Each sweep drives one catalog scene by varying a single parameter and navigates
every step.  These tests assert how a navigation diagnostic *responds* to the
controlled change -- the verification layer a calibrated confidence formula
relies on.  Assertions are trends and bounds (a technique transition, a
degradation to failure, recovery within tolerance), not exact values, because the
underlying navigation carries sub-millipixel cross-process jitter.

Run in-process; heavier than the algorithmic invariants (each sweep navigates
several frames), so the module is ``@pytest.mark.integration`` -- the deliberate
tier, alongside the baselines.
"""

from pathlib import Path

import pytest

from tests.integration.sim_sweep import (
    SweepRow,
    iter_sweep_paths,
    load_sweep,
    run_sweep,
)

pytestmark = pytest.mark.integration

_SWEEPS_ROOT = Path(__file__).parent / 'sim_sweeps'
_RECOVERY_TOLERANCE_PX = 0.5


def _rows(sweep_name: str) -> list[SweepRow]:
    """Load and run a sweep by name."""
    return run_sweep(load_sweep(_SWEEPS_ROOT / f'{sweep_name}.yaml'))


def test_every_sweep_validates() -> None:
    """Every sweep spec parses and validates."""
    paths = iter_sweep_paths(_SWEEPS_ROOT)
    assert paths
    for path in paths:
        load_sweep(path)


def test_noise_sweep_starts_clean() -> None:
    """At the lowest read noise the scene recovers the planted offset."""
    rows = _rows('noise_read_noise')
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_noise_sweep_degrades_to_failure() -> None:
    """At the highest read noise navigation fails -- the navigability cliff."""
    rows = _rows('noise_read_noise')
    assert rows[-1].status == 'failed'


def test_noise_sweep_low_noise_all_succeed() -> None:
    """Every below-cliff step recovers within tolerance."""
    rows = _rows('noise_read_noise')
    for row in rows:
        if row.status != 'success':
            continue
        assert row.offset_error_px is not None
        assert row.offset_error_px < _RECOVERY_TOLERANCE_PX


def test_phase_sweep_navigates_every_phase() -> None:
    """The resolved body navigates to success across the full phase range."""
    rows = _rows('phase_regular_body')
    for row in rows:
        assert row.status == 'success'


def test_phase_sweep_recovers_every_phase() -> None:
    """The recovered offset stays within tolerance across the full phase range."""
    rows = _rows('phase_regular_body')
    for row in rows:
        assert row.offset_error_px is not None
        assert row.offset_error_px < _RECOVERY_TOLERANCE_PX


def test_range_sweep_largest_body_uses_resolved_body_technique() -> None:
    """The largest (well-resolved) body navigates by a resolved-body technique.

    Disc correlation and the limb DT fit both recover a clean resolved body;
    which of the two ranks primary is a confidence-ordering question the
    WS-5 calibration owns (the sim-anchored fit ranks the disc correlation's
    ~0.03 px recovery above the limb's ~0.1 px gradient-peak floor), so the
    assertion accepts either rather than pinning the ordering.
    """
    rows = _rows('range_body_size')
    assert rows[0].primary_technique in ('BodyDiscCorrelateNav', 'BodyLimbNav')


def test_range_sweep_smallest_body_fails() -> None:
    """The smallest body is unnavigable."""
    rows = _rows('range_body_size')
    assert rows[-1].status == 'failed'


def test_range_sweep_reaches_blob_regime() -> None:
    """A small-but-navigable body falls to the orientation-free BodyBlobNav."""
    rows = _rows('range_body_size')
    primaries = [row.primary_technique for row in rows]
    assert 'BodyBlobNav' in primaries


def test_range_sweep_transitions_technique() -> None:
    """The primary technique is not constant -- the range ladder transitions."""
    rows = _rows('range_body_size')
    distinct = {row.primary_technique for row in rows if row.primary_technique is not None}
    assert len(distinct) >= 2


_ROLL_TOLERANCE_DEG = 0.3


def test_star_rotation_sweep_recovers_roll() -> None:
    """The star field recovers every planted roll in the working window.

    The recovered roll is read from whichever technique reports it (the two-star
    path at +/-1 deg, the field matcher at the larger rolls), so the assertion is
    on the roll error, not the fused status.
    """
    rows = _rows('star_rotation')
    for row in rows:
        assert row.rotation_error_deg is not None
        assert row.rotation_error_deg < _ROLL_TOLERANCE_DEG


def test_irregularity_sweep_starts_matched() -> None:
    """At zero relief the predicted smooth body equals the rendered one."""
    rows = _rows('irregularity_shape_mismatch')
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < 0.2


def test_irregularity_sweep_bias_grows() -> None:
    """Shape mismatch grows the recovered centroid bias as relief increases."""
    rows = _rows('irregularity_shape_mismatch')
    assert rows[-1].offset_error_px is not None
    assert rows[0].offset_error_px is not None
    assert rows[-1].offset_error_px > 2.0
    assert rows[-1].offset_error_px > rows[0].offset_error_px + 1.0


def test_irregularity_sweep_confidence_drops() -> None:
    """The fused confidence falls as the predicted shape mismatch widens."""
    rows = _rows('irregularity_shape_mismatch')
    assert rows[-1].confidence < rows[0].confidence


def test_pose_disagreement_starts_clean() -> None:
    """With the predicted pose agreeing, the limb recovers the planted offset."""
    rows = _rows('pose_disagreement')
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < 1.0


def test_pose_disagreement_limb_error_grows() -> None:
    """The limb fit error grows monotonically as the predicted pose drifts."""
    rows = _rows('pose_disagreement')
    first = rows[0].offset_error_px
    second = rows[1].offset_error_px
    last = rows[-1].offset_error_px
    assert first is not None
    assert second is not None
    assert last is not None
    assert second > first
    assert last > first + 1.0
    assert last > 2.0


def test_pose_disagreement_limb_is_confidently_wrong() -> None:
    """At the largest disagreement the limb is several pixels off yet still 'success'.

    The pinned limb does not self-flag across this tumble range -- it returns a
    confidently-wrong fix (a multi-pixel error at unchanged confidence), which is
    why the per-technique demote decision (test_sim_irregular_pose) cannot rely on
    the limb's own status and compares it against the pose-free blob instead.
    """
    rows = _rows('pose_disagreement')
    assert rows[-1].status == 'success'
    assert rows[-1].offset_error_px is not None
    assert rows[-1].offset_error_px > 2.0


# The per-technique dense and wide offset sweeps (``*_offset_fine`` /
# ``*_offset_wide``) are characterization runs, not assertions: they are executed
# by ``sim_sweep_runner`` to produce the report's figures, and the specific defect
# they expose (the disc gradient-NCC sub-pixel bias) is guarded by the fast
# ``test_sim_regression`` case rather than by re-running the full sweep here.
