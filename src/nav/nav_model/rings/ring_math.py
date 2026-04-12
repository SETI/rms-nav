"""Pure mathematical functions for ring edge rendering.

This module provides standalone functions for ring edge fade gradients and
anti-aliasing. Keeping the numerics here separate from orchestration gives:

1. **Testability**: Pure functions are exercised with numpy arrays without
   backplane-heavy integration tests.

2. **Reuse**: Backplane-based ring rendering and ``nav.sim.sim_ring`` both rely on
   the same anti-aliasing and fade logic defined here.

3. **Single responsibility**: ``RingFeature.render()`` chooses what to render and
   assembles results; this module performs the mathematical work.

Design notes
------------
**Per-pixel fade width**: ``compute_edge_fade`` accepts ``fade_width_pix`` (a scalar
pixel count) and a per-pixel ``resolutions`` array, computing
``fade_width_km = fade_width_pix * resolutions`` element-wise. This ensures the
fade spans exactly ``fade_width_pix`` pixels everywhere in the image, regardless
of the local radial resolution. The integration bounds therefore vary per pixel.

**Shade direction**: ``shade_above=True`` and ``shade_above=False`` share one
implementation through an internal ``shade_sign`` (+1 or -1) in
``compute_fade_integral``. The two directions differ only in the sign of two
terms in the closed form.

**Conflict detection vs exclusion**: ``compute_edge_fade`` handles *width reduction*
when a neighboring feature's edge falls within the fade zone (halving the fade at
the conflict boundary). Exclusion of edges whose adjusted width falls below the
minimum is handled upstream by ``RingFeatureFilter`` before rendering. This
function therefore always produces a valid result.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from nav.support.types import NDArrayFloatType


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
    for _label, arr in pairs[1:]:
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
        Array of shade values in [0, max_value], same shape as ``radii``. Results
        are clipped to [0, ``max_value``].

    Raises:
        ValueError: If ``radii`` and ``resolutions`` differ in shape, contain
            non-finite values, or any resolution is not strictly positive; or if
            ``max_value`` is not finite and non-negative.
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
    if np.any(res <= 0.0):
        raise ValueError(
            'compute_antialiasing: resolutions must be strictly positive at every pixel'
        )

    shade_sign = 1.0 if shade_above else -1.0
    res_safe = res

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

    ``shade_sign`` is +1.0 or -1.0 according to fade direction. The closed form
    depends on ``shade_sign`` only through the sign of two terms:

    .. code-block:: text

        result = ((1 + shade_sign * edge_radius / width) * (a1 - a0)
                  + shade_sign * (a0^2 - a1^2) / (2 * width)) / resolutions

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

    Raises:
        ValueError: If ``a0`` or ``a1`` contain a non-finite value, if array shapes
            differ, or if ``width`` or ``resolutions`` contain a non-finite value or
            any element that is not strictly positive.
    """
    w = np.asarray(width, dtype=np.float64)
    res_c = np.asarray(resolutions, dtype=np.float64)
    if not np.all(np.isfinite(w)):
        raise ValueError('compute_fade_integral: width must contain only finite values')
    if not np.all(np.isfinite(res_c)):
        raise ValueError('compute_fade_integral: resolutions must contain only finite values')
    if np.any(w <= 0.0):
        raise ValueError('compute_fade_integral: width must be strictly positive at every element')
    if np.any(res_c <= 0.0):
        raise ValueError(
            'compute_fade_integral: resolutions must be strictly positive at every element'
        )

    a0_a = np.asarray(a0, dtype=np.float64)
    a1_a = np.asarray(a1, dtype=np.float64)
    if not np.all(np.isfinite(a0_a)):
        raise ValueError('compute_fade_integral: a0 must contain only finite values')
    if not np.all(np.isfinite(a1_a)):
        raise ValueError('compute_fade_integral: a1 must contain only finite values')
    _require_matching_shapes(
        ('a0', a0_a),
        ('a1', a1_a),
        ('width', w),
        ('resolutions', res_c),
    )

    result = (
        (1.0 + shade_sign * edge_radius / w) * (a1_a - a0_a)
        + shade_sign * (a0_a**2 - a1_a**2) / (2.0 * w)
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
    logger: Any,
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
    half the distance to the neighbor. The ``RingFeatureFilter`` has already
    excluded edges where this reduction falls below
    ``min_allowed_fade_width_pix``, so this function always produces a result.

    **Shade direction**: ``shade_above`` maps to an internal ``shade_sign``
    (+1 or -1) used by ``compute_fade_integral``.

    The integration uses four cases for pixel coverage:

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
        fade_width_pix: Desired fade extent in pixels (from config); must be
            strictly positive (zero would yield zero per-pixel width and break
            integration).
        resolutions: Per-pixel radial resolution (km/pixel).
        all_edge_radii: Sorted sequence of (radius, label) pairs for all
            surviving feature edges. Used to detect conflict and reduce fade
            width when a neighboring edge falls within the fade zone.
        logger: ``PdsLogger`` from the ring ``NavModel`` (same as
            ``RingsRenderContext.logger``) for optional debug output when fade
            width is narrowed by neighbor edges.

    Returns:
        Updated model image: per-pixel fade contribution is clipped to
        ``[0, 1]``, then added to the input ``model``. The result may exceed
        ``1.0`` if the model already had large values.

    Raises:
        ValueError: If array shapes differ, values are non-finite, any resolution
            is not strictly positive, ``fade_width_pix`` is not finite or is not
            strictly positive, or ``edge_radius`` is not finite.
        TypeError: If ``fade_width_pix`` has an invalid type.
    """
    if isinstance(fade_width_pix, bool) or not isinstance(fade_width_pix, (int, float)):
        raise TypeError(f'fade_width_pix must be int or float, got {type(fade_width_pix).__name__}')
    fwp = float(fade_width_pix)
    if not math.isfinite(fwp) or fwp <= 0.0:
        raise ValueError(f'fade_width_pix must be finite and > 0, got {fade_width_pix!r}')
    if not math.isfinite(float(edge_radius)):
        raise ValueError(f'edge_radius must be finite, got {edge_radius!r}')
    if logger is None:
        raise ValueError('logger must not be None')

    model_f64 = _as_finite_float64('model', model)
    r = _as_finite_float64('radii', radii)
    res = _as_finite_float64('resolutions', resolutions)
    _require_matching_shapes(('model', model_f64), ('radii', r), ('resolutions', res))
    if np.any(res <= 0.0):
        raise ValueError('compute_edge_fade: resolutions must be strictly positive at every pixel')

    shade_sign = 1.0 if shade_above else -1.0

    # Per-pixel fade width in km
    fade_width_km: Any = (fwp * res).astype(np.float64)
    requested_fade_km = fade_width_km.copy()

    # Conflict detection: reduce fade_width_km when a neighbor is in the shade
    # direction. np.minimum handles all neighbors correctly -- a very distant
    # neighbor has a large half_dist that won't reduce anything; only a close
    # neighbor produces a meaningful reduction. Taking the element-wise minimum
    # over all neighbors yields the correct narrowed width regardless of neighbor
    # order when several edges conflict.
    for other_a, _ in all_edge_radii:
        signed_dist = shade_sign * (other_a - edge_radius)
        if signed_dist > 0:
            half_dist = signed_dist / 2.0
            fade_width_km = np.minimum(fade_width_km, half_dist)

    narrowed = fade_width_km < requested_fade_km
    if np.any(narrowed):
        n_narrowed = int(np.count_nonzero(narrowed))
        logger.debug(
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
        eq_case4 = eq_case4 & (edge_radius <= pixel_lower) & (fade_end >= pixel_upper)
    else:
        eq_case4 = eq_case4 & (edge_radius >= pixel_upper) & (fade_end <= pixel_lower)

    eq_case2 = eq2 & ~eq_case1
    eq_case3 = eq3 & ~eq_case1

    shade = np.zeros(r.shape, dtype=np.float64)
    edge_arr = np.full_like(r, edge_radius)

    if shade_above:
        if np.any(eq_case1):
            msk = eq_case1
            shade[msk] = compute_fade_integral(
                edge_arr[msk],
                fade_end[msk],
                edge_radius=edge_radius,
                width=fade_width_km[msk],
                resolutions=res[msk],
                shade_sign=1.0,
            )
        if np.any(eq_case4):
            msk = eq_case4
            shade[msk] = compute_fade_integral(
                pixel_lower[msk],
                pixel_upper[msk],
                edge_radius=edge_radius,
                width=fade_width_km[msk],
                resolutions=res[msk],
                shade_sign=1.0,
            )
        if np.any(eq_case2):
            msk = eq_case2
            shade[msk] = compute_fade_integral(
                edge_arr[msk],
                pixel_upper[msk],
                edge_radius=edge_radius,
                width=fade_width_km[msk],
                resolutions=res[msk],
                shade_sign=1.0,
            )
        if np.any(eq_case3):
            msk = eq_case3
            shade[msk] = compute_fade_integral(
                pixel_lower[msk],
                fade_end[msk],
                edge_radius=edge_radius,
                width=fade_width_km[msk],
                resolutions=res[msk],
                shade_sign=1.0,
            )
    else:
        if np.any(eq_case1):
            msk = eq_case1
            shade[msk] = compute_fade_integral(
                fade_end[msk],
                edge_arr[msk],
                edge_radius=edge_radius,
                width=fade_width_km[msk],
                resolutions=res[msk],
                shade_sign=-1.0,
            )
        if np.any(eq_case4):
            msk = eq_case4
            shade[msk] = compute_fade_integral(
                pixel_lower[msk],
                pixel_upper[msk],
                edge_radius=edge_radius,
                width=fade_width_km[msk],
                resolutions=res[msk],
                shade_sign=-1.0,
            )
        if np.any(eq_case2):
            msk = eq_case2
            shade[msk] = compute_fade_integral(
                pixel_lower[msk],
                edge_arr[msk],
                edge_radius=edge_radius,
                width=fade_width_km[msk],
                resolutions=res[msk],
                shade_sign=-1.0,
            )
        if np.any(eq_case3):
            msk = eq_case3
            shade[msk] = compute_fade_integral(
                fade_end[msk],
                pixel_upper[msk],
                edge_radius=edge_radius,
                width=fade_width_km[msk],
                resolutions=res[msk],
                shade_sign=-1.0,
            )

    shade = np.clip(shade, 0.0, 1.0)
    return np.asarray(model_f64 + shade, dtype=np.float64)
