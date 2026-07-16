"""Image-side detector stage: normalized signal to noisy detector counts.

Phase-A port at present fidelity: the composed [0, 1] signal maps straight
to DN (Poisson on DN, Gaussian read noise, flat bias, cosmic-ray spikes,
column bloom, full-well clip).  Phase B replaces the unit chain with the
exposure-referenced electron conversion (gain states, dark current, hot
pixels, banding, quantization, electron-domain bloom) and gives the
calibrated (I/F) path propagated noise; until then ``calibrated_if`` scenes
render noise-free.

Physical noise parameters delegate to the emulated instrument's config
block; sim-only knobs (signal full-scale fraction, cosmic-ray rate) come
from the scene's ``noise`` block with ``sim`` config defaults.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.instruments import resolve_sim_inst_config
from spindoctor.sim.seeds import derive_effect_seed
from spindoctor.support.types import NDArrayFloatType

__all__ = ['apply_detector', 'apply_detector_noise', 'apply_saturation']


def apply_detector_noise(
    img: NDArrayFloatType,
    *,
    signal_full_scale_dn: float,
    read_noise_dn: float,
    saturation_dn: float,
    poisson: bool = True,
    bias_dn: float = 0.0,
    cosmic_ray_rate_per_sec: float = 0.0,
    exposure_sec: float = 1.0,
    pixel_area_cm2: float = 1.0,
    rng: np.random.Generator,
    cosmic_ray_rng: np.random.Generator,
) -> None:
    """Convert a normalized signal image to DN and add detector noise in place.

    The composed bodies/rings/stars/sky signal arrives normalized to [0, 1].
    This stage maps it to detector counts (DN) at ``signal_full_scale_dn`` per
    unit signal, then applies the real camera-noise structure: signal-dependent
    Poisson shot noise, a Gaussian read-noise floor, and sparse cosmic-ray
    spikes.  Saturation clipping at the full well is left to a later stage so
    cosmic-ray spikes can exceed it; only a zero floor (raw frames carry no
    negative DN) is enforced here.  Missing-data markers are telemetry-stage
    scope (:mod:`spindoctor.sim.forward.telemetry`), not detector physics.

    Parameters:
        img: Normalized [0, 1] signal image, overwritten in place with DN.
        signal_full_scale_dn: DN that a normalized signal of 1.0 maps to.
        read_noise_dn: Standard deviation of the Gaussian read-noise floor, DN.
        saturation_dn: Full-well DN; cosmic-ray spikes are scaled to exceed it.
        poisson: Whether to apply Poisson shot noise (usually on).
        bias_dn: Additive bias pedestal in DN.  Real raw frames sit on a bias
            level, so signal-free sky is never exactly zero; without it the dark
            sky collides with the missing-data marker (0) and is misclassified.
        cosmic_ray_rate_per_sec: Cosmic-ray fluence in events / cm^2 / sec.
        exposure_sec: Exposure time in seconds (scales cosmic-ray count).
        pixel_area_cm2: Detector pixel area in cm^2 (scales cosmic-ray count).
        rng: Generator for the shot + read-noise stream.
        cosmic_ray_rng: Generator for cosmic-ray placement and intensity, kept
            separate from ``rng`` so toggling the shot-noise term does not
            reshuffle the cosmic-ray realization.
    """
    size_v, size_u = img.shape

    signal_dn = np.clip(img, 0.0, 1.0) * signal_full_scale_dn

    if poisson:
        # Poisson mean equals the noise-free signal, so noise grows with
        # brightness the way a real detector's shot noise does.
        out_dn = rng.poisson(np.maximum(signal_dn, 0.0)).astype(np.float64)
    else:
        out_dn = signal_dn.copy()
    if read_noise_dn > 0:
        out_dn += rng.normal(0.0, read_noise_dn, size=(size_v, size_u))
    # Bias pedestal lifts signal-free sky off zero so it is not confused with
    # the missing-data marker; telemetry-stage markers overwrite it back.
    out_dn += bias_dn

    expected_hits = cosmic_ray_rate_per_sec * exposure_sec * pixel_area_cm2 * size_v * size_u
    if expected_hits > 0:
        n_hits = int(cosmic_ray_rng.poisson(expected_hits))
        if n_hits > 0:
            hit_v = cosmic_ray_rng.integers(0, size_v, size=n_hits)
            hit_u = cosmic_ray_rng.integers(0, size_u, size=n_hits)
            # Long-tailed spike intensities that exceed the full well by design,
            # so the cosmic-ray mask in the orchestrator has something to catch.
            spikes = saturation_dn * (1.0 + cosmic_ray_rng.lognormal(0.0, 1.0, size=n_hits))
            np.add.at(out_dn, (hit_v, hit_u), spikes)

    # Raw frames carry no negative DN; the upper (saturation) clip is deferred
    # to apply_saturation so cosmic-ray spikes can bloom before being clipped.
    img[:] = np.maximum(out_dn, 0.0)


def apply_saturation(
    img: NDArrayFloatType,
    *,
    saturation_dn: float,
    bloom_length: int = 0,
) -> None:
    """Clip pixels at the full-well DN, optionally blooming excess along columns.

    Pixels above ``saturation_dn`` are clipped to it, so they land on the
    orchestrator's saturation mask (``image >= full_well_dn``).  When
    ``bloom_length`` is positive, the charge above full well is first spread
    along the column (the v axis) up to that many pixels in each direction --
    conserving the total excess -- so a saturated star blooms into a vertical
    streak the way it does on cameras with column bloom (e.g. Cassini NAC).

    Parameters:
        img: DN image, modified in place.
        saturation_dn: Full-well DN ceiling.
        bloom_length: Column-bloom half-length in pixels; 0 disables bloom.
    """
    if bloom_length > 0:
        excess = np.maximum(img - saturation_dn, 0.0)
        if float(excess.max()) > 0.0:
            width = 2 * bloom_length + 1
            # The box mean over the column window conserves the summed excess
            # while spreading it onto neighbours, which may saturate in turn.
            spread = ndimage.uniform_filter1d(excess, size=width, axis=0, mode='constant')
            np.minimum(img, saturation_dn, out=img)
            img += spread
    np.minimum(img, saturation_dn, out=img)


def apply_detector(
    frame: SimFrame,
    *,
    params: Mapping[str, Any],
    rng: np.random.Generator,
) -> None:
    """Detector stage: convert the composed signal to noisy DN in place.

    All signal is composed before this stage runs, so the Poisson shot term
    sees the noise-free signal it should grow with.  Feature masks in
    ``frame.truth`` were derived from the noise-free signal and are
    unaffected.

    Parameters:
        frame: The frame whose signal plane is converted in place.
        params: The full scene mapping; reads ``instrument``,
            ``instrument_config``, ``noise``, ``exposure_sec``, and
            ``random_seed`` (for the independent cosmic-ray stream).
        rng: The stage generator, used for shot and read noise.
    """
    inst_config = resolve_sim_inst_config(
        DEFAULT_CONFIG, params.get('instrument'), params.get('instrument_config')
    )
    if inst_config.get('data_units', 'raw_dn') != 'raw_dn':
        # calibrated_if: realistic I/F noise is deferred (phase B scope).  The
        # DN detector model (Poisson counts, full-well saturation, cosmic-ray
        # spikes) does not map onto I/F, so the composed signal is left as I/F
        # in [0, 1] with no detector noise applied.
        np.clip(frame.signal, 0.0, 1.0, out=frame.signal)
        return

    sim_noise = DEFAULT_CONFIG.category('sim')['noise']
    scene_noise = params.get('noise') or {}
    inst_noise = inst_config.get('noise') or {}
    # Map a normalized signal of 1.0 to a fraction of the camera's full well,
    # so a scene's brightness scales with the selected instrument's DN depth.
    full_scale_frac = float(
        scene_noise.get('signal_full_scale_frac', sim_noise['signal_full_scale_frac'])
    )
    signal_full_scale_dn = float(
        scene_noise.get('signal_full_scale_dn', full_scale_frac * float(inst_noise['full_well_dn']))
    )
    # The cosmic-ray stream is seeded independently of the stage's shot/read
    # stream so toggling Poisson or read noise cannot reshuffle which pixels
    # take hits (single-variable sweeps depend on that isolation).
    cosmic_ray_rng = np.random.default_rng(
        derive_effect_seed(int(params.get('random_seed', 42)), 'detector/cosmic_rays')
    )
    apply_detector_noise(
        frame.signal,
        signal_full_scale_dn=signal_full_scale_dn,
        read_noise_dn=float(scene_noise.get('read_noise_dn', inst_noise['read_noise_dn'])),
        saturation_dn=float(inst_noise['saturation_dn']),
        poisson=bool(scene_noise.get('poisson', True)),
        bias_dn=float(scene_noise.get('bias_dn', sim_noise.get('bias_dn', 0.0))),
        cosmic_ray_rate_per_sec=float(
            scene_noise.get(
                'cosmic_ray_rate_per_sec', sim_noise.get('cosmic_ray_rate_per_sec', 0.0)
            )
        ),
        exposure_sec=float(params.get('exposure_sec', 1.0)),
        pixel_area_cm2=float(
            scene_noise.get('pixel_area_cm2', sim_noise.get('pixel_area_cm2', 1.0))
        ),
        rng=rng,
        cosmic_ray_rng=cosmic_ray_rng,
    )
    # Clip at the camera's full well (with optional column bloom) so
    # saturated pixels land on the orchestrator's saturation mask.
    apply_saturation(
        frame.signal,
        saturation_dn=float(inst_noise['saturation_dn']),
        bloom_length=int(scene_noise.get('bloom_length', sim_noise.get('bloom_length', 0))),
    )
