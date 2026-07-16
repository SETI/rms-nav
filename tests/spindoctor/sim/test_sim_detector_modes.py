"""Geometry and disabled-floor tests for the detector-electronics artifact modes.

Every registered detector-stage mode gets a geometry test here -- the structure
it plants lands where commanded and is stable per seed -- plus a disabled test
(a zero amplitude / incidence is a no-op) and, for the stochastic modes, an
adversarial-placement and determinism check.  The mechanics are exercised
directly on synthetic planes where the shape is easiest to assert, and the
registry routing (an explicit mode wins over the generic knob; LORRI's
frame-transfer smear under instrument_defaults) is exercised end to end through
the detector resolver and a whole-scene render.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.detector.electronics_stages import (
    add_bright_dark_pairs,
    add_coherent_banding,
    add_dark_ramp,
    add_fixed_pattern_dn,
    add_fixed_pattern_response,
    add_residual_image,
    add_serial_tail,
    apply_beam_bend,
    apply_exposure_shading,
    apply_frame_transfer_smear,
)
from spindoctor.sim.forward.detector.params import resolve_detector_params
from spindoctor.sim.render import render_combined_model

_SIZE = 64


def _rng(seed: int = 0) -> np.random.Generator:
    """A seeded generator for the mechanics under test."""
    return np.random.default_rng(seed)


# --- banding_coherent -------------------------------------------------------


def test_banding_horizontal_is_row_correlated() -> None:
    """Horizontal banding is constant along a row and varies between rows."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    add_coherent_banding(
        electrons,
        amplitude_e=100.0,
        period_px=16.0,
        orientation='horizontal',
        freq_step_factor=1.0,
        dark_step_dn=0.0,
        gain_e_per_dn=1.0,
        rng=_rng(5),
    )
    assert float(np.ptp(electrons, axis=1).max()) < 1e-9
    assert float(np.ptp(electrons, axis=0).max()) > 1.0


def test_banding_vertical_is_column_correlated() -> None:
    """Vertical banding is constant down a column and varies between columns."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    add_coherent_banding(
        electrons,
        amplitude_e=100.0,
        period_px=8.0,
        orientation='vertical',
        freq_step_factor=1.0,
        dark_step_dn=0.0,
        gain_e_per_dn=1.0,
        rng=_rng(5),
    )
    assert float(np.ptp(electrons, axis=0).max()) < 1e-9
    assert float(np.ptp(electrons, axis=1).max()) > 1.0


def test_banding_dark_step_lifts_the_lower_half() -> None:
    """A dark-level step raises the mean of the rows below the mid-line."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    add_coherent_banding(
        electrons,
        amplitude_e=10.0,
        period_px=16.0,
        orientation='horizontal',
        freq_step_factor=1.0,
        dark_step_dn=50.0,
        gain_e_per_dn=2.0,
        rng=_rng(1),
    )
    top = float(electrons[: _SIZE // 2].mean())
    bottom = float(electrons[_SIZE // 2 :].mean())
    assert bottom - top == pytest.approx(100.0, abs=5.0)


def test_banding_disabled_at_zero_amplitude() -> None:
    """A zero banding amplitude is a no-op."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    record = add_coherent_banding(
        electrons,
        amplitude_e=0.0,
        period_px=16.0,
        orientation='both',
        freq_step_factor=1.0,
        dark_step_dn=0.0,
        gain_e_per_dn=1.0,
        rng=_rng(5),
    )
    assert record['active'] is False
    assert np.all(electrons == 0.0)


# --- dark_ramp --------------------------------------------------------------


def test_dark_ramp_grows_with_line_number() -> None:
    """The additive dark ramp is zero at the top and peaks at the last line."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    add_dark_ramp(
        electrons, amplitude_e=100.0, nonlinear=1.0, rbi_column_factor=0.0, hot_columns=None
    )
    assert float(electrons[0].mean()) == pytest.approx(0.0)
    assert float(electrons[-1].mean()) == pytest.approx(100.0)
    assert float(electrons[_SIZE // 2].mean()) < float(electrons[-1].mean())


def test_dark_ramp_rbi_enhances_named_columns() -> None:
    """The RBI flavor adds an extra ramp on the enhanced columns."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    add_dark_ramp(
        electrons,
        amplitude_e=100.0,
        nonlinear=1.0,
        rbi_column_factor=0.5,
        hot_columns=np.array([10, 20], dtype=np.int64),
    )
    assert float(electrons[-1, 10]) > float(electrons[-1, 5])
    assert float(electrons[-1, 10]) == pytest.approx(150.0)


def test_dark_ramp_disabled_at_zero_amplitude() -> None:
    """A zero ramp amplitude is a no-op."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    record = add_dark_ramp(
        electrons, amplitude_e=0.0, nonlinear=1.0, rbi_column_factor=0.0, hot_columns=None
    )
    assert record['active'] is False
    assert np.all(electrons == 0.0)


def test_exposure_shading_scales_by_line() -> None:
    """The shutter shading scales line 0 by 1 and the last line by the ratio."""
    electrons = np.full((_SIZE, _SIZE), 10.0, dtype=np.float64)
    apply_exposure_shading(electrons, top_factor=1.5, bottom_factor=1.05)
    assert float(electrons[0].mean()) == pytest.approx(10.0)
    assert float(electrons[-1].mean()) == pytest.approx(10.0 * 1.05 / 1.5)


# --- frame_transfer_smear ---------------------------------------------------


def test_frame_transfer_smear_adds_a_column_pedestal() -> None:
    """A bright column gains a full-column pedestal; a dark column stays dark."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    electrons[20:30, 32] = 1000.0
    apply_frame_transfer_smear(electrons, t_scrub_sec=0.012, t_transfer_sec=0.011, exposure_sec=0.1)
    assert float(electrons[0, 32]) > 0.0
    assert float(electrons[0, 5]) == pytest.approx(0.0)


def test_frame_transfer_smear_disabled_at_zero_transfer() -> None:
    """A zero scrub and transfer time is a no-op."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    electrons[20:30, 32] = 1000.0
    before = electrons.copy()
    apply_frame_transfer_smear(electrons, t_scrub_sec=0.0, t_transfer_sec=0.0, exposure_sec=0.1)
    assert np.array_equal(electrons, before)


def test_lorri_instrument_defaults_injects_frame_transfer_smear() -> None:
    """LORRI turns frame_transfer_smear on under instrument_defaults (15.7)."""
    dp = resolve_detector_params(
        {
            'instrument': 'nhlorri',
            'random_seed': 1,
            'exposure_sec': 0.1,
            'artifacts': {'instrument_defaults': True},
        }
    )
    assert 'frame_transfer_smear' in dp.detector_modes
    assert dp.detector_modes['frame_transfer_smear']['t_transfer_sec'] == 0.011


def test_lorri_floor_does_not_inject_frame_transfer_smear() -> None:
    """Without instrument_defaults LORRI carries no injected smear."""
    dp = resolve_detector_params({'instrument': 'nhlorri', 'random_seed': 1, 'exposure_sec': 0.1})
    assert 'frame_transfer_smear' not in dp.detector_modes


# --- serial_tail ------------------------------------------------------------


def test_serial_tail_writes_a_horizontal_tail_off_saturation() -> None:
    """A saturated pixel seeds a bright-then-dark tail in the readout direction."""
    dn = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    dn[32, 20] = 4095.0
    add_serial_tail(
        dn,
        saturation_dn=4095.0,
        saturation_frac=0.99,
        amplitude_dn=12.0,
        length_px=6,
        direction='right',
    )
    assert float(dn[32, 21]) > 0.0  # bright overshoot just right of the source
    assert float(dn[32, 24]) < 0.0  # dark undershoot further along
    assert float(dn[32, 19]) == pytest.approx(0.0)  # nothing to the left


def test_serial_tail_disabled_without_saturation() -> None:
    """With no saturated pixel the serial tail is a no-op."""
    dn = np.full((_SIZE, _SIZE), 100.0, dtype=np.float64)
    before = dn.copy()
    record = add_serial_tail(
        dn,
        saturation_dn=4095.0,
        saturation_frac=0.99,
        amplitude_dn=12.0,
        length_px=6,
        direction='right',
    )
    assert record['sources'] == 0
    assert np.array_equal(dn, before)


# --- beam_bend --------------------------------------------------------------


def test_beam_bend_displaces_a_bright_edge() -> None:
    """The beam bend warps the image near a bright horizontal limb (v shift)."""
    signal = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    signal[32:, :] = 1.0
    before = signal.copy()
    apply_beam_bend(signal, amplitude_px=2.0)
    # The vertical displacement field moves the limb; the frame changes near it.
    assert not np.array_equal(signal, before)
    assert float(np.abs(signal - before)[28:36, :].sum()) > 0.0


def test_beam_bend_disabled_at_zero_amplitude() -> None:
    """A zero bend amplitude leaves the image untouched."""
    signal = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    signal[32:, :] = 1.0
    before = signal.copy()
    apply_beam_bend(signal, amplitude_px=0.0)
    assert np.array_equal(signal, before)


# --- residual_image ---------------------------------------------------------


def test_residual_image_adds_a_displaced_ghost() -> None:
    """The self_offset ghost adds a shifted, scaled copy of the frame."""
    signal = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    signal[30, 30] = 100.0
    add_residual_image(signal, amplitude=0.1, prior='self_offset', offset_v=5, offset_u=5)
    assert float(signal[35, 35]) == pytest.approx(10.0)  # ghost of the bright pixel
    assert float(signal[30, 30]) == pytest.approx(100.0)  # original intact


def test_residual_image_disabled_at_zero_amplitude() -> None:
    """A zero ghost amplitude is a no-op."""
    signal = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    signal[30, 30] = 100.0
    before = signal.copy()
    add_residual_image(signal, amplitude=0.0, prior='self_offset', offset_v=5, offset_u=5)
    assert np.array_equal(signal, before)


# --- fixed_pattern ----------------------------------------------------------


def test_fixed_pattern_response_is_stable_per_seed() -> None:
    """The multiplicative fixed pattern is identical for equal seeds."""
    a = np.full((_SIZE, _SIZE), 1000.0, dtype=np.float64)
    b = np.full((_SIZE, _SIZE), 1000.0, dtype=np.float64)
    add_fixed_pattern_response(
        a, prnu_rms=0.01, vignetting_frac=0.04, dust_donut_count=2, rng=_rng(9)
    )
    add_fixed_pattern_response(
        b, prnu_rms=0.01, vignetting_frac=0.04, dust_donut_count=2, rng=_rng(9)
    )
    assert np.array_equal(a, b)


def test_fixed_pattern_vignetting_darkens_the_corner() -> None:
    """Corner vignetting lowers the corner response below the centre."""
    electrons = np.full((_SIZE, _SIZE), 1000.0, dtype=np.float64)
    add_fixed_pattern_response(
        electrons, prnu_rms=0.0, vignetting_frac=0.1, dust_donut_count=0, rng=_rng(1)
    )
    centre = float(electrons[_SIZE // 2, _SIZE // 2])
    corner = float(electrons[0, 0])
    assert corner < centre


def test_fixed_pattern_stitch_comb_raises_periodic_columns() -> None:
    """The stitch comb raises columns on the stitch period by the amplitude."""
    dn = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    add_fixed_pattern_dn(
        dn, stitch_period_px=16, stitch_amplitude_dn=5.0, jail_bar_dn=0.0, rng=_rng(2)
    )
    assert float(dn[10, 0]) == pytest.approx(5.0)
    assert float(dn[10, 16]) == pytest.approx(5.0)
    assert float(dn[10, 1]) == pytest.approx(0.0)


def test_fixed_pattern_jail_bars_offset_even_odd_columns() -> None:
    """Jail bars offset even and odd columns in opposite directions."""
    dn = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    add_fixed_pattern_dn(
        dn, stitch_period_px=0, stitch_amplitude_dn=0.0, jail_bar_dn=0.5, rng=_rng(2)
    )
    assert float(dn[5, 0]) == pytest.approx(-float(dn[5, 1]))
    assert abs(float(dn[5, 0])) == pytest.approx(0.5)


def test_fixed_pattern_dn_disabled_at_zero() -> None:
    """A zeroed stitch comb and jail bar is a no-op."""
    dn = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    record = add_fixed_pattern_dn(
        dn, stitch_period_px=0, stitch_amplitude_dn=0.0, jail_bar_dn=0.0, rng=_rng(2)
    )
    assert record['active'] is False
    assert np.all(dn == 0.0)


# --- bright_dark_pairs ------------------------------------------------------


def test_bright_dark_pairs_are_vertical_and_charge_conserving() -> None:
    """Each pair raises one pixel and lowers the pixel below it by the same charge."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    record = add_bright_dark_pairs(electrons, count=5, amplitude_e=1000.0, rng=_rng(3))
    assert record['pairs']
    for v, u in record['pairs']:
        assert float(electrons[v, u]) == pytest.approx(1000.0)
        assert float(electrons[v + 1, u]) == pytest.approx(-1000.0)


def test_bright_dark_pairs_disabled_at_zero_count() -> None:
    """A zero pair count is a no-op."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    record = add_bright_dark_pairs(electrons, count=0, amplitude_e=1000.0, rng=_rng(3))
    assert record['pairs'] == []
    assert np.all(electrons == 0.0)


def test_bright_dark_pairs_adversarial_land_on_the_pool() -> None:
    """An adversarial pool confines the pairs to the feature region."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    pool_v = np.arange(28, 36, dtype=np.int64)
    pool_u = np.full(pool_v.size, 40, dtype=np.int64)
    record = add_bright_dark_pairs(
        electrons, count=6, amplitude_e=1000.0, rng=_rng(3), candidate_pool=(pool_v, pool_u)
    )
    for v, _u in record['pairs']:
        assert 28 <= v <= 35


# --- routing and truth bookkeeping through the resolver ---------------------


def _disc(instrument: str, **extra: Any) -> dict[str, Any]:
    """A centered lit disc scene for the given instrument."""
    scene: dict[str, Any] = {
        'size_v': 96,
        'size_u': 96,
        'random_seed': 3,
        'instrument': instrument,
        'exposure_sec': 1.0,
        'bodies': [
            {'name': 'B', 'center_v': 48, 'center_u': 48, 'axis1': 30, 'axis2': 28, 'axis3': 28}
        ],
    }
    scene.update(extra)
    return scene


def test_bloom_mode_overrides_bloom_length() -> None:
    """An explicit bloom mode sets the bloom length over the generic knob."""
    dp = resolve_detector_params(
        _disc('coiss_nac', artifacts={'bloom': {'incidence': 1.0, 'bloom_length': 9}})
    )
    assert dp.bloom_length == 9


def test_quantization_ls8b_mode_routes_to_the_ls8b_submode() -> None:
    """The quantization_ls8b mode selects the LS8B ADC sub-mode."""
    dp = resolve_detector_params(
        _disc('coiss_nac', artifacts={'quantization_ls8b': {'incidence': 1.0}})
    )
    assert dp.quantization == 'ls8b'


def test_banding_mode_silences_the_generic_banding() -> None:
    """banding_coherent yields the generic banding to the explicit mode."""
    dp = resolve_detector_params(
        _disc(
            'coiss_nac',
            artifacts={'instrument_defaults': True, 'banding_coherent': {'incidence': 1.0}},
        )
    )
    assert dp.banding_amplitude_e == 0.0
    assert 'banding_coherent' in dp.detector_modes


def test_detector_mode_records_truth() -> None:
    """A rendered detector mode records its realized geometry in the frame truth."""
    _img, truth = render_combined_model(
        _disc('coiss_nac', artifacts={'fixed_pattern': {'incidence': 1.0, 'dust_donut_count': 3}})
    )
    artifacts = truth.get('artifacts', {})
    assert artifacts.get('fixed_pattern', {}).get('active') is True


def test_floor_scene_records_no_detector_artifacts() -> None:
    """A scene with no artifact modes records no detector-artifact truth."""
    _img, truth = render_combined_model(_disc('coiss_nac'))
    assert 'artifacts' not in truth or not truth['artifacts']
