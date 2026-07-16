"""End-to-end detector paths: calibrated I/F, vidicon, and instrument_defaults.

These render whole scenes to check the calibrated round-trip, the Voyager
vidicon DN path, and the ``instrument_defaults`` physical-chain opt-in against the
disabled floor (a scene with neither an artifacts nor a noise block).
"""

from typing import Any

import numpy as np

from spindoctor.sim.render import render_combined_model


def _disc(instrument: str, *, size: int = 64, **extra: Any) -> dict[str, Any]:
    """A centered lit disc scene for the given instrument."""
    scene: dict[str, Any] = {
        'size_v': size,
        'size_u': size,
        'random_seed': 3,
        'instrument': instrument,
        'exposure_sec': 1.0,
        'bodies': [
            {
                'name': 'B',
                'center_v': size / 2,
                'center_u': size / 2,
                'axis1': size * 0.4,
                'axis2': size * 0.35,
                'axis3': size * 0.35,
            }
        ],
    }
    scene.update(extra)
    return scene


def test_calibrated_disc_round_trips_to_if_unity() -> None:
    """A noise-free calibrated disc round-trips its fully-lit peak to I/F ~1."""
    img, _ = render_combined_model(_disc('coiss_calib_nac'))
    assert abs(float(img.max()) - 1.0) < 0.02


def test_calibrated_path_carries_propagated_noise() -> None:
    """A calibrated scene with a noise block carries noise texture in I/F units."""
    clean, _ = render_combined_model(_disc('coiss_calib_nac'))
    noisy, _ = render_combined_model(
        _disc('coiss_calib_nac', noise={'poisson': True, 'read_noise_dn': 4.0})
    )
    assert not np.array_equal(clean, noisy)


def test_calibrated_sky_sits_near_zero_if() -> None:
    """Bias and dark are subtracted before the divide, so sky sits near I/F 0."""
    img, _ = render_combined_model(_disc('coiss_calib_nac'))
    corner = float(img[0, 0])
    assert abs(corner) < 0.05


def test_vidicon_floor_is_clean() -> None:
    """A vgiss scene with no noise block renders without vidicon noise (floor)."""
    plain, _ = render_combined_model(_disc('vgiss'))
    corner = plain[:4, :4]
    assert float(corner.std()) == 0.0


def test_vidicon_defaults_add_dn_domain_noise() -> None:
    """instrument_defaults turns on the vidicon DN-domain noise model."""
    plain, _ = render_combined_model(_disc('vgiss'))
    noisy, _ = render_combined_model(_disc('vgiss', artifacts={'instrument_defaults': True}))
    assert not np.array_equal(plain, noisy)


def test_floor_scene_has_no_noise() -> None:
    """A scene with neither artifacts nor a noise block renders a clean DN frame."""
    img, _ = render_combined_model(_disc('coiss_nac'))
    corner = img[:8, :8]
    assert float(corner.std()) == 0.0


def test_instrument_defaults_activates_the_physical_chain() -> None:
    """instrument_defaults injects detector noise the floor scene does not carry."""
    floor, _ = render_combined_model(_disc('coiss_nac'))
    physical, _ = render_combined_model(_disc('coiss_nac', artifacts={'instrument_defaults': True}))
    assert not np.array_equal(floor, physical)
    # The physical chain lifts the sky off the flat bias with dark + banding.
    assert float(physical[:8, :8].std()) > 0.0


def test_instrument_defaults_defaults_oversample_when_psf_active() -> None:
    """instrument_defaults implies a PSF, which oversamples the radiance grid."""
    from spindoctor.sim.render import resolve_oversample

    assert resolve_oversample(_disc('coiss_nac', artifacts={'instrument_defaults': True})) == 4
    assert resolve_oversample(_disc('coiss_nac')) == 1


def test_gain_state_selection_changes_dn_scale() -> None:
    """Selecting a lower-gain state raises the DN a given signal reaches."""
    state2, _ = render_combined_model(_disc('coiss_nac', detector={'gain_state': 2}))
    state3, _ = render_combined_model(_disc('coiss_nac', detector={'gain_state': 3}))
    # Gain state 3 (~13 e-/DN) yields more DN per electron than state 2 (~30).
    assert float(state3.max()) > float(state2.max())
