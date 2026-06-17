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


def test_range_sweep_largest_body_uses_limb() -> None:
    """The largest (well-resolved) body navigates by BodyLimbNav."""
    rows = _rows('range_body_size')
    assert rows[0].primary_technique == 'BodyLimbNav'


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
_BLOB_OFFSET_TOLERANCE_PX = 0.1
_DISC_OFFSET_TOLERANCE_PX = 0.6


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


def test_blob_offset_sweep_is_quantization_free() -> None:
    """The blob centroid recovers every offset -- whole, near-boundary, fractional.

    This is the check that nothing snaps to a pixel boundary: a small body's
    lit-weighted centroid recovers offsets like 0.12783 and 0.99 px as accurately
    as a whole-pixel offset.
    """
    rows = _rows('offset_fractional_blob')
    for row in rows:
        assert row.status == 'success'
        assert row.offset_error_px is not None
        assert row.offset_error_px < _BLOB_OFFSET_TOLERANCE_PX


def test_disc_offset_sweep_stays_subpixel() -> None:
    """The disc correlation recovers every offset to within a fraction of a pixel.

    Looser than the blob: the NCC sub-pixel refinement carries a fraction-
    dependent bias, but it stays well inside a pixel across the offset range.
    """
    rows = _rows('offset_fractional_disc')
    for row in rows:
        assert row.status == 'success'
        assert row.offset_error_px is not None
        assert row.offset_error_px < _DISC_OFFSET_TOLERANCE_PX


def test_disc_offset_sweep_shows_pixel_locking() -> None:
    """The disc is most accurate at the half-pixel and biased at integer offsets.

    A direct measurement of NCC pixel-locking: the recovered offset error at a
    half-pixel planted offset (0.5) is markedly smaller than at a whole-pixel
    offset (12.0), independent of magnitude. The blob, by contrast, is uniform
    across both (previous test).
    """
    by_value = {row.value: row for row in _rows('offset_fractional_disc')}
    half = by_value[0.5]
    whole = by_value[12.0]
    assert half.offset_error_px is not None
    assert whole.offset_error_px is not None
    assert half.offset_error_px < whole.offset_error_px
