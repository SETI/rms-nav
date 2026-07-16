"""Image-side star rendering: scene stars and the background star field.

Phase-A port of the star renderer at present fidelity: each star is drawn
peak-normalized (`2.512 ** -(vmag - 4)` at the PSF peak) and PSF-spread
directly into the normalized signal plane.  The flux-normalized point-mass
deposition into ``SimFrame.point_e`` (with the whole-scene optics PSF as the
only PSF application) is phase B/D scope and replaces this module's spread.

The rendered :class:`~spindoctor.support.types.MutableStar` records and the
``star_info`` hit-test entries are renderer *output* metadata (they carry the
planted offset and rendered DN); they stay on the image side of the
information boundary and are never handed to the navigator-side models.
"""

import json
from functools import lru_cache
from typing import Any, cast

import numpy as np
from psfmodel import GaussianPSF
from starcat import Star

from spindoctor.support.types import MutableStar, NDArrayFloatType

__all__ = ['render_background_stars', 'render_stars']


@lru_cache(maxsize=1)
def _render_stars_cached(
    size_v: int,
    size_u: int,
    stars_params_json: str,
    *,
    offset_v: float,
    offset_u: float,
    default_psf_sigma: float,
    rotation_deg: float,
) -> tuple[Any, ...]:
    """Internal cached function to compute star rendering."""
    stars_params = json.loads(stars_params_json)
    img = np.zeros((size_v, size_u), dtype=np.float64)
    sim_star_list: list[MutableStar] = []
    star_info: list[dict[str, Any]] = []

    # A camera roll rotates the whole frame about the boresight (image centre).
    # The rendered star position is therefore the catalog position rotated by
    # ``rotation_deg`` about the centre, then translated by the planted offset.
    # The star record keeps its unrotated catalog (v, u) so the NavModel predicts
    # the unshifted geometry and a star technique recovers BOTH the rotation and
    # the translation.  The rotation matrix matches the navigator's
    # ``similarity_transform_fit`` convention (maps catalog -> detection in
    # ``(v, u)`` order), so the fitted angle equals ``rotation_deg``.
    theta = np.radians(rotation_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    roll_center_v = size_v / 2.0
    roll_center_u = size_u / 2.0

    for i, star_params in enumerate(stars_params):
        star = cast(MutableStar, Star())
        star.unique_number = i + 1
        star.catalog_name = str(star_params.get('catalog_name', 'SIM'))
        star.pretty_name = str(star_params.get('name', f'SIM-{i + 1}'))
        star.name = star.pretty_name
        star.v = float(star_params.get('v', size_v / 2))
        star.u = float(star_params.get('u', size_u / 2))
        star.move_v = float(star_params.get('move_v', 0.0))
        star.move_u = float(star_params.get('move_u', 0.0))
        star.vmag = float(star_params.get('vmag', 8.0))
        star.spectral_class = str(star_params.get('spectral_class', 'G2'))
        star.temperature = Star.temperature_from_sclass(star.spectral_class)
        star.temperature_faked = star.temperature is None
        if star.temperature is None:
            star.temperature = 5780.0
        star.johnson_mag_v = star.vmag
        bmv = Star.bmv_from_sclass(star.spectral_class or 'G2') or 0.63
        star.johnson_mag_b = star.johnson_mag_v + bmv
        star.johnson_mag_faked = False
        star.ra_pm = 0.0
        star.dec_pm = 0.0
        star.conflicts = ''
        star.psf_size = tuple(star_params.get('psf_size', (11, 11)))
        star.dn = 2.512 ** -(star.vmag - 4.0)
        sim_star_list.append(star)

        rel_v = star.v - roll_center_v
        rel_u = star.u - roll_center_u
        rot_v = cos_t * rel_v - sin_t * rel_u
        rot_u = sin_t * rel_v + cos_t * rel_u
        star_offset_v = roll_center_v + rot_v + offset_v
        star_offset_u = roll_center_u + rot_u + offset_u
        v_int = int(star_offset_v)
        u_int = int(star_offset_u)
        v_frac = star_offset_v - v_int
        u_frac = star_offset_u - u_int

        psf_size_half_u = int(star.psf_size[1] + np.round(abs(star.move_u))) // 2
        psf_size_half_v = int(star.psf_size[0] + np.round(abs(star.move_v))) // 2

        max_move_steps = 1  # TODO configurable
        move_gran = max(abs(star.move_u) / max_move_steps, abs(star.move_v) / max_move_steps)
        move_gran = np.clip(move_gran, 0.1, 1.0)

        sigma = star_params.get('psf_sigma', default_psf_sigma)
        psf = GaussianPSF(sigma=sigma)

        # Stars where any part of the PSF would be off the edge of the image are ignored.
        # This is because PSF fitting will not work in these cases.
        if (
            u_int < psf_size_half_u
            or u_int >= img.shape[1] - psf_size_half_u
            or v_int < psf_size_half_v
            or v_int >= img.shape[0] - psf_size_half_v
        ):
            # Still collect info for hit-testing
            star_info.append(
                {
                    'name': star.name,
                    'center_v': star_offset_v,
                    'center_u': star_offset_u,
                    'sigma': sigma,
                    'psf_half_v': psf_size_half_v,
                    'psf_half_u': psf_size_half_u,
                }
            )
            continue

        # Evaluate PSF with scale=1.0 first to get unnormalized PSF.
        #
        # ``eval_rect`` centres the PSF half a pixel low for ``offset=0`` (its
        # offset is measured from the pixel's lower edge, so ``offset=0.5`` lands
        # on the pixel centre).  The navigator's detection centroid and the star
        # NavModel's predicted position both use the pixel-centre convention
        # (integer index ``i`` *is* coordinate ``i``).  Adding 0.5 to the eval
        # offset renders the star centroid at exactly ``star.v + offset_v``, so a
        # star the model predicts at ``(v, u)`` lands there in the image and a
        # technique recovers the planted offset without a half-pixel bias.
        star_psf = psf.eval_rect(
            (psf_size_half_v * 2 + 1, psf_size_half_u * 2 + 1),
            offset=(v_frac + 0.5, u_frac + 0.5),
            scale=1.0,
            movement=(star.move_v, star.move_u),
            movement_granularity=move_gran,
        )

        # Normalize PSF so peak is 1.0, then scale by magnitude
        psf_max = np.max(star_psf)
        if psf_max > 0:
            star_psf = star_psf / psf_max
        # Scale so that vmag=0 results in peak=1.0
        # star.dn = 2.512^-(vmag - 4.0), so for vmag=0: star.dn = 2.512^4
        # We want vmag=0 -> peak=1, so scale by star.dn / (2.512^4)
        scale_factor = star.dn / (2.512**4.0)
        star_psf = star_psf * scale_factor

        img[
            v_int - psf_size_half_v : v_int + psf_size_half_v + 1,
            u_int - psf_size_half_u : u_int + psf_size_half_u + 1,
        ] += star_psf

        star_info.append(
            {
                'name': star.name,
                'center_v': star_offset_v,
                'center_u': star_offset_u,
                'sigma': sigma,
                'psf_half_v': psf_size_half_v,
                'psf_half_u': psf_size_half_u,
            }
        )

    return (img, sim_star_list, star_info)


def render_stars(
    img: NDArrayFloatType,
    stars_params: list[dict[str, Any]],
    offset_v: float,
    *,
    offset_u: float,
    default_psf_sigma: float = 3.0,
    rotation_deg: float = 0.0,
) -> tuple[NDArrayFloatType, list[MutableStar], list[dict[str, Any]]]:
    """Render stars into img. Returns (img, sim_star_list, star_render_info).

    Parameters:
        img: Image array to render stars into.
        stars_params: Per-star parameter dictionaries.
        offset_v: V offset to apply to every star.
        offset_u: U offset to apply to every star.
        default_psf_sigma: PSF sigma used for stars that do not specify their
            own ``psf_sigma`` (the selected instrument's value).
        rotation_deg: Camera-roll angle (degrees) applied about the image centre
            before the translation offset, modelling a pointing rotation the star
            techniques recover.
    """
    size_v, size_u = img.shape
    stars_params_json = json.dumps(stars_params, sort_keys=True)
    cached_img, cached_star_list, cached_star_info = _render_stars_cached(
        size_v,
        size_u,
        stars_params_json,
        offset_v=offset_v,
        offset_u=offset_u,
        default_psf_sigma=default_psf_sigma,
        rotation_deg=rotation_deg,
    )
    # Add cached stars to input image (don't overwrite background noise/stars)
    img[:] = np.clip(img + cached_img, 0.0, 1.0)
    return img, cached_star_list, cached_star_info


@lru_cache(maxsize=1)
def _render_background_stars_cached(
    size_v: int,
    size_u: int,
    n_stars: int,
    *,
    seed: int,
    psf_sigma: float,
    distribution_exponent: float,
) -> NDArrayFloatType:
    """Internal cached function to compute background star additions."""
    rng = np.random.default_rng(seed)
    star_additions = np.zeros((size_v, size_u), dtype=np.float64)

    # Power law for intensity: weight toward dimmer stars
    # intensity = uniform^power where power > 1 makes dimmer stars more common
    uniform_samples = rng.uniform(0.0, 1.0, size=n_stars)
    intensities = uniform_samples**distribution_exponent

    # PSF size: at least 11x11, but scale with sigma
    # Use at least 3*sigma pixels on each side, minimum 6 for 11x11
    psf_size_half = max(6, int(np.ceil(3.0 * psf_sigma)))

    psf = GaussianPSF(sigma=psf_sigma)

    for i in range(n_stars):
        # Random position
        v = rng.uniform(0.0, float(size_v))
        u = rng.uniform(0.0, float(size_u))

        v_int = int(v)
        u_int = int(u)
        v_frac = v - v_int
        u_frac = u - u_int

        # Skip if too close to edge
        if (
            u_int < psf_size_half
            or u_int >= size_u - psf_size_half
            or v_int < psf_size_half
            or v_int >= size_v - psf_size_half
        ):
            continue

        # Generate PSF (normalized so peak is 1.0)
        star_psf = psf.eval_rect(
            (psf_size_half * 2 + 1, psf_size_half * 2 + 1),
            offset=(v_frac, u_frac),
            scale=1.0,  # Use scale=1.0 to get normalized PSF
            movement=(0.0, 0.0),
            movement_granularity=1.0,
        )

        # Normalize PSF to have peak value of 1.0, then scale by intensity
        # This ensures stars are bright (peak brightness = intensity, not distributed)
        psf_max = np.max(star_psf)
        if psf_max > 0:
            star_psf = star_psf / psf_max * intensities[i]
        else:
            star_psf = star_psf * intensities[i]

        # Add to star additions accumulator
        star_additions[
            v_int - psf_size_half : v_int + psf_size_half + 1,
            u_int - psf_size_half : u_int + psf_size_half + 1,
        ] += star_psf

    return star_additions


def render_background_stars(
    img: NDArrayFloatType,
    n_stars: int,
    seed: int,
    *,
    psf_sigma: float = 0.9,
    distribution_exponent: float = 2.5,
) -> None:
    """Add random background stars to the image.

    Parameters:
        img: Image array to modify in-place (stars are added, not overwritten).
        n_stars: Number of stars to add (0-1000).
        seed: Random seed for reproducibility.
        psf_sigma: PSF sigma value for star rendering (default 0.9).
        distribution_exponent: Power law exponent for intensity distribution (default 2.5).
            Higher values make dimmer stars more common.
    """
    if n_stars <= 0:
        return
    size_v, size_u = img.shape
    star_additions = _render_background_stars_cached(
        size_v,
        size_u,
        n_stars,
        seed=seed,
        psf_sigma=psf_sigma,
        distribution_exponent=distribution_exponent,
    )
    img[:] = np.clip(img + star_additions, 0.0, 1.0)
