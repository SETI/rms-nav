"""Detector-noise model for the simulator (Poisson, read floor, cosmics, dropouts).

These tests drive ``apply_detector_noise`` on flat synthetic fields so the
statistical properties of each noise term can be asserted directly in DN.
"""

import numpy as np

from nav.sim.render import apply_detector_noise

_FULL_SCALE = 2048.0
_SATURATION = 4095.0


def _flat(value: float, *, size: int = 200) -> np.ndarray:
    """A flat normalized signal field of the given value."""
    return np.full((size, size), value, dtype=np.float64)


def test_signal_maps_to_dn() -> None:
    """A flat normalized signal maps to its DN level on average."""
    img = _flat(0.5)
    apply_detector_noise(
        img,
        signal_full_scale_dn=_FULL_SCALE,
        read_noise_dn=0.0,
        saturation_dn=_SATURATION,
        noise_seed=1,
        cosmic_ray_seed=2,
        missing_data_seed=3,
    )
    assert abs(float(img.mean()) - 1024.0) < 15.0


def test_poisson_variance_tracks_mean() -> None:
    """With Poisson on and no read noise, variance approximates the mean."""
    img = _flat(0.5)
    apply_detector_noise(
        img,
        signal_full_scale_dn=_FULL_SCALE,
        read_noise_dn=0.0,
        saturation_dn=_SATURATION,
        noise_seed=1,
        cosmic_ray_seed=2,
        missing_data_seed=3,
    )
    assert abs(float(img.var()) - 1024.0) < 150.0


def test_poisson_off_is_exact_without_read_noise() -> None:
    """Disabling Poisson and read noise yields the exact DN signal."""
    img = _flat(0.5)
    apply_detector_noise(
        img,
        signal_full_scale_dn=_FULL_SCALE,
        read_noise_dn=0.0,
        saturation_dn=_SATURATION,
        poisson=False,
        noise_seed=1,
        cosmic_ray_seed=2,
        missing_data_seed=3,
    )
    assert np.all(img == 1024.0)


def test_read_noise_sets_floor_spread() -> None:
    """Read noise adds a Gaussian spread of the configured sigma."""
    img = _flat(0.5)
    apply_detector_noise(
        img,
        signal_full_scale_dn=_FULL_SCALE,
        read_noise_dn=10.0,
        saturation_dn=_SATURATION,
        poisson=False,
        noise_seed=1,
        cosmic_ray_seed=2,
        missing_data_seed=3,
    )
    assert abs(float(img.std()) - 10.0) < 1.0


def test_cosmic_rays_exceed_saturation() -> None:
    """A nonzero cosmic-ray rate plants spikes above the full well."""
    img = _flat(0.1, size=100)
    apply_detector_noise(
        img,
        signal_full_scale_dn=_FULL_SCALE,
        read_noise_dn=0.0,
        saturation_dn=_SATURATION,
        poisson=False,
        cosmic_ray_rate_per_sec=0.001,
        exposure_sec=1.0,
        pixel_area_cm2=1.0,
        noise_seed=1,
        cosmic_ray_seed=2,
        missing_data_seed=3,
    )
    assert int((img > _SATURATION).sum()) >= 1


def test_missing_data_marks_pixels() -> None:
    """A missing-data rate marks roughly that fraction of pixels."""
    img = _flat(0.5, size=200)
    apply_detector_noise(
        img,
        signal_full_scale_dn=_FULL_SCALE,
        read_noise_dn=0.0,
        saturation_dn=_SATURATION,
        poisson=False,
        missing_data_marker_dn=0.0,
        missing_data_rate=0.1,
        noise_seed=1,
        cosmic_ray_seed=2,
        missing_data_seed=3,
    )
    marked_frac = float((img == 0.0).mean())
    assert abs(marked_frac - 0.1) < 0.02


def test_noise_is_deterministic_for_equal_seeds() -> None:
    """Equal seeds and inputs produce byte-identical noisy output."""
    img_a = _flat(0.5, size=64)
    img_b = _flat(0.5, size=64)
    for img in (img_a, img_b):
        apply_detector_noise(
            img,
            signal_full_scale_dn=_FULL_SCALE,
            read_noise_dn=5.0,
            saturation_dn=_SATURATION,
            noise_seed=11,
            cosmic_ray_seed=12,
            missing_data_seed=13,
        )
    assert np.array_equal(img_a, img_b)


def test_noise_differs_for_different_seed() -> None:
    """Changing the noise seed changes the realized noise field."""
    img_a = _flat(0.5, size=64)
    img_b = _flat(0.5, size=64)
    apply_detector_noise(
        img_a,
        signal_full_scale_dn=_FULL_SCALE,
        read_noise_dn=5.0,
        saturation_dn=_SATURATION,
        noise_seed=11,
        cosmic_ray_seed=12,
        missing_data_seed=13,
    )
    apply_detector_noise(
        img_b,
        signal_full_scale_dn=_FULL_SCALE,
        read_noise_dn=5.0,
        saturation_dn=_SATURATION,
        noise_seed=99,
        cosmic_ray_seed=12,
        missing_data_seed=13,
    )
    assert not np.array_equal(img_a, img_b)
