"""Pure mathematical functions for ring edge rendering.

This module contains standalone functions for computing ring edge fade gradients
and anti-aliasing. Extracting these from the monolithic ``NavModelRings`` class
serves several purposes:

1. **Testability**: Pure functions with no rendering context are trivially unit-tested
   with numpy arrays. The backplane-integration tests in ``NavModelRings`` are
   eliminated; math correctness is verified here.

2. **Reuse**: Both the real ring model (backplane rendering) and the simulated ring
   model share the same anti-aliasing logic. A single implementation in this module
   avoids the current duplication between ``NavModelRingsBase._compute_antialiasing``
   and ``SimRing._compute_antialiasing_shade``.

3. **Single responsibility**: Rendering orchestration (choosing which edges to render,
   building result objects) stays in ``RingFeature.render()``; math stays here.

Design notes
------------
**Per-pixel fade width**: ``compute_edge_fade`` accepts ``fade_width_pix`` (a scalar
pixel count) and a per-pixel ``resolutions`` array, computing
``fade_width_km = fade_width_pix * resolutions`` element-wise. This ensures the
fade spans exactly ``fade_width_pix`` pixels everywhere in the image, regardless
of the local radial resolution. The integration bounds therefore vary per pixel.

**Unified shade direction**: The two historically duplicated code paths
(``shade_above=True`` and ``shade_above=False``) are unified via an internal
``shade_sign`` parameter (+1 or -1) in ``compute_fade_integral``. The two
formulas differ only in the sign of two terms, which eliminates ~80 lines of
near-duplicate code.

**Conflict detection vs exclusion**: ``compute_edge_fade`` handles *width reduction*
when a neighboring feature's edge falls within the fade zone (halving the fade at
the conflict boundary). Exclusion of edges whose adjusted width falls below the
minimum is handled upstream by ``RingFeatureFilter`` before rendering. This
function therefore always produces a valid result.
"""

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from nav.support.types import NDArrayFloatType

# Smallest positive float used to avoid division by zero in per-pixel formulas.
_FLOAT64_MIN_POS = float(np.finfo(np.float64).tiny)

_logger = logging.getLogger(__name__)


def _as_finite_float64(name: str, arr: NDArrayFloatType) -> np.ndarray:
    """Convert to float64 ndarray and require all elements finite."""
    out = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f'{name} must contain only finite values')
    return out


def _require_matching_shapes(
    *pairs: tuple[str, np.ndarray],
) -> None:
    """Raise ValueError if arrays do not share the same shape."""
    if len(pairs) < 2:
        return
    ref_shape = pairs[0][1].shape
    for label, arr in pairs[1:]:
        if arr.shape != ref_shape:
            names = ', '.join(f'{n} {a.shape}' for n, a in pairs)
            raise ValueError(
                'compute_edge_fade / compute_antialiasing: arrays must have identical shapes '
                f'({names})'
            )


def compute_antialiasing(
    *,
    radii: NDArrayFloatType,
    edge_radius: float,
    shade_above: bool,
    resolutions: NDArrayFloatType,
    max_value: float = 1.0,
) -> NDArrayFloatType:
    """Compute anti-aliasing shade at pixel boundaries near a ring edge.

    Creates smooth sub-pixel transitions at the pixel boundary where the ring
    edge crosses. The shade value represents the fraction of the pixel that is
    covered by the ring, linearly interpolated between 0.0 and ``max_value``
    as the edge moves from one side of the pixel to the other.

    When the pixel center is exactly at the edge, shade = 0.5 * max_value.
    When the edge is half a resolution unit past the pixel center (in the
    shade direction), shade = max_value (full coverage). When the edge is
    half a resolution unit in the opposite direction, shade = 0.0.

    Parameters:
        radii: Array of ring radii at pixel centers (km).
        edge_radius: Target edge radius (km).
        shade_above: If True, shading is applied on the low-radius side of the
            edge (the object is above the edge, anti-aliasing goes below). If
            False, shading is applied on the high-radius side.
        resolutions: Array of radial resolutions at each pixel (km/pixel).
        max_value: Maximum shade value (default 1.0). Use values < 1.0 for
            partial-opacity rendering.

    Returns:
        Array of shade values in [0, max_value], same shape as ``radii``.

    Zero entries in ``resolutions`` are replaced with a tiny positive value before
    dividing so shades stay finite; results are clipped to [0, ``max_value``].

    Raises:
        ValueError: If ``radii`` and ``resolutions`` differ in shape or contain
            non-finite values, or ``max_value`` is not finite and non-negative.
    """
    if isinstance(max_value, bool) or not isinstance(max_value, (int, float)):
        raise TypeError('max_value must be int or float')
    if not math.isfinite(float(max_value)) or float(max_value) < 0.0:
        raise ValueError(f'max_value must be finite and non-negative, got {max_value!r}')
    if not math.isfinite(float(edge_radius)):
        raise ValueError(f'edge_radius must be finite, got {edge_radius!r}')

    rad = _as_finite_float64('radii', radii)
    res = _as_finite_float64('resolutions', resolutions)
    _require_matching_shapes(('radii', rad), ('resolutions', res))

    shade_sign = 1.0 if shade_above else -1.0
    res_safe = np.maximum(res, _FLOAT64_MIN_POS)

    shade = 1.0 - shade_sign * (rad - edge_radius) / res_safe
    shade -= 0.5

    shade_arr = np.clip(np.asarray(shade, dtype=np.float64), 0.0, 1.0)
    shade_arr *= max_value

    return shade_arr


def compute_fade_integral(
    a0: NDArrayFloatType,
    a1: NDArrayFloatType,
    *,
    edge_radius: float,
    width: NDArrayFloatType,
    resolutions: NDArrayFloatType,
    shade_sign: float,
) -> NDArrayFloatType:
    """Compute the definite integral of the linear fade function over a pixel.

    The fade function is a linear gradient from 1.0 at the edge to 0.0 at
    ``edge_radius + shade_sign * width``. The integral gives the average shade
    value for the portion of the pixel that overlaps the fade zone, which is
    what a properly anti-aliased renderer should compute.

    The two historically separate integration formulas (``int_func`` for
    ``shade_above=True`` and ``int_func2`` for ``shade_above=False``) are
    unified here via ``shade_sign`` (+1 or -1). They differ only in the sign
    of two terms:

    .. code-block:: text

        result = ((1 + shade_sign * edge_radius / width) * (a1 - a0)
                  + shade_sign * (a0² - a1²) / (2 * width)) / resolutions

    Parameters:
        a0: Lower integration bounds per pixel (km).
        a1: Upper integration bounds per pixel (km).
        edge_radius: Fixed edge radius (km).
        width: Per-pixel fade width in km. Varies per pixel because
            ``fade_width_km = fade_width_pix * resolutions``.
        resolutions: Per-pixel radial resolution (km/pixel).
        shade_sign: +1.0 for shade_above, -1.0 for shade_below.

    Returns:
        Per-pixel integral values, same shape as ``a0``.

    Elements of ``width`` and ``resolutions`` must be non-negative; each is clamped
    to a tiny positive value before dividing so ``compute_edge_fade`` (which may
    supply ``half_dist == 0`` or zero resolution) does not produce NaNs or
    infinities.
    """
    w = np.maximum(np.asarray(width, dtype=np.float64), _FLOAT64_MIN_POS)
    res_c = np.maximum(np.asarray(resolutions, dtype=np.float64), _FLOAT64_MIN_POS)
    result = (
        (1.0 + shade_sign * edge_radius / w) * (a1 - a0) + shade_sign * (a0**2 - a1**2) / (2.0 * w)
    ) / res_c
    return np.asarray(result, dtype=np.float64)


def compute_edge_fade(
    *,
    model: NDArrayFloatType,
    radii: NDArrayFloatType,
    edge_radius: float,
    shade_above: bool,
    fade_width_pix: float,
    resolutions: NDArrayFloatType,
    all_edge_radii: Sequence[tuple[float, str]],
) -> NDArrayFloatType:
    """Compute a linear fade from a single ring edge with per-pixel fade width.

    This function produces a linear gradient from full brightness at a known
    ring edge to zero over a configurable distance. The fade is necessary when
    a ring feature has only one known edge -- without it, the model image would
    show a false sharp boundary where the ring ceases to be defined. The
    gradient provides a smooth signal that works well for correlation-based
    navigation.

    **Per-pixel fade width**: The fade width in km varies per pixel:
    ``fade_width_km = fade_width_pix * resolutions``. This ensures the fade
    always spans exactly ``fade_width_pix`` pixels at every location in the
    image, regardless of the local radial resolution. At the ansae (fine
    resolution) the fade covers fewer km; at foreshortened regions (coarse
    resolution) it covers more km.

    **Conflict detection and width reduction**: When a neighboring feature's
    edge falls within the fade zone, the fade width is reduced per pixel to
    half the distance to the neighbor, matching current behavior. The
    ``RingFeatureFilter`` has already excluded edges where this reduction falls
    below ``min_allowed_fade_width_pix``, so this function always produces a
    result.

    **Shade direction unified**: Shade direction is determined by ``shade_above``
    and internally represented as ``shade_sign`` (+1 or -1). This eliminates
    the historical duplication of the two integration code paths.

    The integration uses four cases for pixel coverage (matching the historical
    implementation):

    - Case 1: Both edge and fade end within the pixel.
    - Case 2: Edge within pixel, fade end extends beyond.
    - Case 3: Edge before pixel, fade end within pixel.
    - Case 4: Full coverage (edge before pixel, fade end after pixel).

    Parameters:
        model: Current model image array. The fade is added to this.
        radii: Per-pixel ring radius array from the backplane (km).
        edge_radius: Nominal radius of the ring edge (km).
        shade_above: If True, shade toward larger radii (away from planet);
            if False, shade toward smaller radii (toward planet).
        fade_width_pix: Desired fade extent in pixels (from config).
        resolutions: Per-pixel radial resolution (km/pixel).
        all_edge_radii: Sorted sequence of (radius, label) pairs for all
            surviving feature edges. Used to detect conflict and reduce fade
            width when a neighboring edge falls within the fade zone.

    Returns:
        Updated model image with the fade added (values are clipped to
        [0, 1] before adding, so the result may exceed 1.0 if the model
        already has non-zero values).

    Raises:
        ValueError: If array shapes differ, values are non-finite, or scalar
            parameters are out of range.
        TypeError: If ``fade_width_pix`` has an invalid type.
    """
    if isinstance(fade_width_pix, bool) or not isinstance(fade_width_pix, (int, float)):
        raise TypeError(f'fade_width_pix must be int or float, got {type(fade_width_pix).__name__}')
    fwp = float(fade_width_pix)
    if not math.isfinite(fwp) or fwp < 0.0:
        raise ValueError(f'fade_width_pix must be finite and non-negative, got {fade_width_pix!r}')
    if not math.isfinite(float(edge_radius)):
        raise ValueError(f'edge_radius must be finite, got {edge_radius!r}')

    m = _as_finite_float64('model', model)
    r = _as_finite_float64('radii', radii)
    res = _as_finite_float64('resolutions', resolutions)
    _require_matching_shapes(('model', m), ('radii', r), ('resolutions', res))

    for j, pair in enumerate(all_edge_radii):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(
                f'all_edge_radii[{j}] must be a (radius, label) pair, got {type(pair).__name__!r}'
            )
        rad_km, label = pair
        if isinstance(rad_km, bool) or not isinstance(rad_km, (int, float)):
            raise TypeError(f'all_edge_radii[{j}][0] must be numeric')
        if not math.isfinite(float(rad_km)):
            raise ValueError(f'all_edge_radii[{j}][0] must be finite, got {rad_km!r}')
        if not isinstance(label, str):
            raise TypeError(f'all_edge_radii[{j}][1] must be str, got {type(label).__name__}')

    shade_sign = 1.0 if shade_above else -1.0

    # Per-pixel fade width in km
    fade_width_km: Any = (fwp * res).astype(np.float64)
    requested_fade_km = fade_width_km.copy()

    # Conflict detection: reduce fade_width_km when a neighbor is in the shade
    # direction. np.minimum handles all neighbors correctly -- a very distant
    # neighbor has a large half_dist that won't reduce anything; only a close
    # neighbor produces a meaningful reduction. Processing all neighbors and
    # taking the minimum is equivalent to the original sequential approach but
    # correctly handles multiple neighbors regardless of processing order.
    for other_a, _ in all_edge_radii:
        signed_dist = shade_sign * (other_a - edge_radius)
        if signed_dist > 0:
            half_dist = signed_dist / 2.0
            fade_width_km = np.minimum(fade_width_km, half_dist)

    narrowed = fade_width_km < requested_fade_km
    if np.any(narrowed):
        n_narrowed = int(np.count_nonzero(narrowed))
        _logger.debug(
            'compute_edge_fade: neighbor edges narrowed fade width at %d / %d pixels '
            '(edge_radius=%.3f km, shade_above=%s)',
            n_narrowed,
            narrowed.size,
            edge_radius,
            shade_above,
        )

    # Per-pixel fade zone boundaries
    pixel_lower = r - res / 2.0
    pixel_upper = r + res / 2.0

    if shade_above:
        fade_end = edge_radius + fade_width_km  # per-pixel
        eq2 = (pixel_lower <= edge_radius) & (edge_radius < pixel_upper)
        eq3 = (pixel_lower <= fade_end) & (fade_end < pixel_upper)
    else:
        fade_end = edge_radius - fade_width_km  # per-pixel
        eq2 = (pixel_lower < edge_radius) & (edge_radius <= pixel_upper)
        eq3 = (pixel_lower < fade_end) & (fade_end <= pixel_upper)

    eq_case1 = eq2 & eq3
    eq_case4 = ~eq2 & ~eq3
    if shade_above:
        eq_case4 = eq_case4 & (edge_radius < pixel_lower) & (fade_end > pixel_upper)
    else:
        eq_case4 = eq_case4 & (edge_radius > pixel_upper) & (fade_end < pixel_lower)

    eq_case2 = eq2 & ~eq_case1
    eq_case3 = eq3 & ~eq_case1

    shade = np.zeros(r.shape, dtype=np.float64)
    edge_arr = np.full_like(r, edge_radius)

    if shade_above:
        if np.any(eq_case1):
            m = eq_case1
            shade[m] = compute_fade_integral(
                edge_arr[m],
                fade_end[m],
                edge_radius=edge_radius,
                width=fade_width_km[m],
                resolutions=res[m],
                shade_sign=1.0,
            )
        if np.any(eq_case4):
            m = eq_case4
            shade[m] = compute_fade_integral(
                pixel_lower[m],
                pixel_upper[m],
                edge_radius=edge_radius,
                width=fade_width_km[m],
                resolutions=res[m],
                shade_sign=1.0,
            )
        if np.any(eq_case2):
            m = eq_case2
            shade[m] = compute_fade_integral(
                edge_arr[m],
                pixel_upper[m],
                edge_radius=edge_radius,
                width=fade_width_km[m],
                resolutions=res[m],
                shade_sign=1.0,
            )
        if np.any(eq_case3):
            m = eq_case3
            shade[m] = compute_fade_integral(
                pixel_lower[m],
                fade_end[m],
                edge_radius=edge_radius,
                width=fade_width_km[m],
                resolutions=res[m],
                shade_sign=1.0,
            )
    else:
        if np.any(eq_case1):
            m = eq_case1
            shade[m] = compute_fade_integral(
                fade_end[m],
                edge_arr[m],
                edge_radius=edge_radius,
                width=fade_width_km[m],
                resolutions=res[m],
                shade_sign=-1.0,
            )
        if np.any(eq_case4):
            m = eq_case4
            shade[m] = compute_fade_integral(
                pixel_lower[m],
                pixel_upper[m],
                edge_radius=edge_radius,
                width=fade_width_km[m],
                resolutions=res[m],
                shade_sign=-1.0,
            )
        if np.any(eq_case2):
            m = eq_case2
            shade[m] = compute_fade_integral(
                pixel_lower[m],
                edge_arr[m],
                edge_radius=edge_radius,
                width=fade_width_km[m],
                resolutions=res[m],
                shade_sign=-1.0,
            )
        if np.any(eq_case3):
            m = eq_case3
            shade[m] = compute_fade_integral(
                fade_end[m],
                pixel_upper[m],
                edge_radius=edge_radius,
                width=fade_width_km[m],
                resolutions=res[m],
                shade_sign=-1.0,
            )

    shade = np.clip(shade, 0.0, 1.0)
    return np.asarray(m + shade, dtype=np.float64)
