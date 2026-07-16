"""Detector-noise model for the simulator (Poisson, read floor, cosmics).

These tests drive ``apply_detector_noise`` on flat synthetic fields so the
statistical properties of each noise term can be asserted directly in DN.
The missing-data markers are telemetry-stage scope and are tested through
``apply_telemetry`` on a rendered-DN frame.
"""

import numpy as np

from spindoctor.sim.forward.detector import apply_detector_noise
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.forward.telemetry import apply_telemetry

_FULL_SCALE = 2048.0
_SATURATION = 4095.0


def _flat(value: float, *, size: int = 200) -> np.ndarray:
    """A flat normalized signal field of the given value."""
    return np.full((size, size), value, dtype=np.float64)


def _rngs(
    noise_seed: int = 1, cosmic_seed: int = 2
) -> tuple[np.random.Generator, np.random.Generator]:
    """Seeded (noise, cosmic-ray) generator pair for one noise application."""
    return np.random.default_rng(noise_seed), np.random.default_rng(cosmic_seed)


def test_signal_maps_to_dn() -> None:
    """A flat normalized signal maps to its DN level on average."""
    img = _flat(0.5)
    rng, cosmic_rng = _rngs()
    apply_detector_noise(
        img,
        signal_full_scale_dn=_FULL_SCALE,
        read_noise_dn=0.0,
        saturation_dn=_SATURATION,
        rng=rng,
        cosmic_ray_rng=cosmic_rng,
    )
    assert abs(float(img.mean()) - 1024.0) < 15.0


def test_poisson_variance_tracks_mean() -> None:
    """With Poisson on and no read noise, variance approximates the mean."""
    img = _flat(0.5)
    rng, cosmic_rng = _rngs()
    apply_detector_noise(
        img,
        signal_full_scale_dn=_FULL_SCALE,
        read_noise_dn=0.0,
        saturation_dn=_SATURATION,
        rng=rng,
        cosmic_ray_rng=cosmic_rng,
    )
    assert abs(float(img.var()) - 1024.0) < 150.0


def test_poisson_off_is_exact_without_read_noise() -> None:
    """Disabling Poisson and read noise yields the exact DN signal."""
    img = _flat(0.5)
    rng, cosmic_rng = _rngs()
    apply_detector_noise(
        img,
        signal_full_scale_dn=_FULL_SCALE,
        read_noise_dn=0.0,
        saturation_dn=_SATURATION,
        poisson=False,
        rng=rng,
        cosmic_ray_rng=cosmic_rng,
    )
    assert np.all(img == 1024.0)


def test_read_noise_sets_floor_spread() -> None:
    """Read noise adds a Gaussian spread of the configured sigma."""
    img = _flat(0.5)
    rng, cosmic_rng = _rngs()
    apply_detector_noise(
        img,
        signal_full_scale_dn=_FULL_SCALE,
        read_noise_dn=10.0,
        saturation_dn=_SATURATION,
        poisson=False,
        rng=rng,
        cosmic_ray_rng=cosmic_rng,
    )
    assert abs(float(img.std()) - 10.0) < 1.0


def test_cosmic_rays_exceed_saturation() -> None:
    """A nonzero cosmic-ray rate plants spikes above the full well."""
    img = _flat(0.1, size=100)
    rng, cosmic_rng = _rngs()
    apply_detector_noise(
        img,
        signal_full_scale_dn=_FULL_SCALE,
        read_noise_dn=0.0,
        saturation_dn=_SATURATION,
        poisson=False,
        cosmic_ray_rate_per_sec=0.001,
        exposure_sec=1.0,
        pixel_area_cm2=1.0,
        rng=rng,
        cosmic_ray_rng=cosmic_rng,
    )
    assert int((img > _SATURATION).sum()) >= 1


def test_cosmic_rays_independent_of_poisson_toggle() -> None:
    """The cosmic-ray realization does not move when Poisson is toggled."""
    img_poisson = _flat(0.1, size=100)
    img_clean = _flat(0.1, size=100)
    for img, poisson in ((img_poisson, True), (img_clean, False)):
        rng, cosmic_rng = _rngs()
        apply_detector_noise(
            img,
            signal_full_scale_dn=_FULL_SCALE,
            read_noise_dn=0.0,
            saturation_dn=_SATURATION,
            poisson=poisson,
            cosmic_ray_rate_per_sec=0.001,
            rng=rng,
            cosmic_ray_rng=cosmic_rng,
        )
    hits_poisson = img_poisson > _SATURATION
    hits_clean = img_clean > _SATURATION
    assert np.array_equal(hits_poisson, hits_clean)


def test_missing_data_marks_pixels() -> None:
    """The telemetry stage marks roughly the configured fraction of pixels."""
    frame = SimFrame(
        signal=_flat(100.0, size=200),
        point_e=np.zeros((200, 200), dtype=np.float64),
    )
    params = {'noise': {'missing_data_rate': 0.1}}
    apply_telemetry(frame, params=params, rng=np.random.default_rng(3))
    marked_frac = float((frame.signal != 100.0).mean())
    assert abs(marked_frac - 0.1) < 0.02


def test_missing_data_disabled_without_rate() -> None:
    """With no missing-data rate configured the telemetry stage is a no-op."""
    frame = SimFrame(
        signal=_flat(100.0, size=64),
        point_e=np.zeros((64, 64), dtype=np.float64),
    )
    apply_telemetry(frame, params={}, rng=np.random.default_rng(3))
    assert np.all(frame.signal == 100.0)


def test_noise_is_deterministic_for_equal_seeds() -> None:
    """Equal seeds and inputs produce byte-identical noisy output."""
    img_a = _flat(0.5, size=64)
    img_b = _flat(0.5, size=64)
    for img in (img_a, img_b):
        rng, cosmic_rng = _rngs(noise_seed=11, cosmic_seed=12)
        apply_detector_noise(
            img,
            signal_full_scale_dn=_FULL_SCALE,
            read_noise_dn=5.0,
            saturation_dn=_SATURATION,
            rng=rng,
            cosmic_ray_rng=cosmic_rng,
        )
    assert np.array_equal(img_a, img_b)


def test_noise_differs_for_different_seed() -> None:
    """Changing the noise seed changes the realized noise field."""
    img_a = _flat(0.5, size=64)
    img_b = _flat(0.5, size=64)
    rng_a, cosmic_a = _rngs(noise_seed=11, cosmic_seed=12)
    apply_detector_noise(
        img_a,
        signal_full_scale_dn=_FULL_SCALE,
        read_noise_dn=5.0,
        saturation_dn=_SATURATION,
        rng=rng_a,
        cosmic_ray_rng=cosmic_a,
    )
    rng_b, cosmic_b = _rngs(noise_seed=99, cosmic_seed=12)
    apply_detector_noise(
        img_b,
        signal_full_scale_dn=_FULL_SCALE,
        read_noise_dn=5.0,
        saturation_dn=_SATURATION,
        rng=rng_b,
        cosmic_ray_rng=cosmic_b,
    )
    assert not np.array_equal(img_a, img_b)
