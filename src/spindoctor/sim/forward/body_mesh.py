"""Image-side polyhedral-mesh body rendering.

Meshes render with flat per-face shading and a polygon silhouette; smooth
shading and higher-frequency relief are deliberately not implemented.  The
mesh primitives themselves live in the shared
:mod:`spindoctor.sim.mesh_geometry` module so the rendered mesh and the
navigator's predicted mesh are the same shape by construction.
"""

from functools import lru_cache
from typing import Any

from spindoctor.sim.forward.body import finish_single_body
from spindoctor.sim.mesh_geometry import (
    MeshBodySpec,
    mesh_spec_from_params,
    render_mesh_body_image,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['render_single_mesh_body']


@lru_cache(maxsize=30)
def _render_mesh_shape_cached(
    size_v: int,
    size_u: int,
    axis1: float,
    *,
    axis2: float,
    axis3: float,
    spec: MeshBodySpec,
    illumination_angle: float,
    phase_angle: float,
    anti_aliasing: float,
) -> NDArrayFloatType:
    """Cache an irregular mesh body shape at the reference (image) centre."""
    return render_mesh_body_image(
        size=(size_v, size_u),
        center=(size_v / 2.0, size_u / 2.0),
        semi_axes_px=(axis1 / 2.0, axis2 / 2.0, axis3 / 2.0),
        spec=spec,
        illumination_angle=illumination_angle,
        phase_angle=phase_angle,
        anti_aliasing=anti_aliasing,
    )


def render_single_mesh_body(
    img: NDArrayFloatType,
    body_params: dict[str, Any],
    body_name: str,
    *,
    center_v: float,
    center_u: float,
    axis1: float,
    axis2: float,
    axis3: float,
    illumination_angle: float,
    phase_angle: float,
    anti_aliasing: float,
    ref_center_v: float,
    ref_center_u: float,
) -> tuple[NDArrayBoolType, dict[str, Any]]:
    """Render one polyhedral-mesh body into the image in place.

    Parameters:
        img: Image array to modify in-place.
        body_params: Body parameters dictionary (mesh keys are parsed by
            ``mesh_spec_from_params``).
        body_name: Upper-cased body name for the inventory keys.
        center_v: Body center V in the image (offset already applied).
        center_u: Body center U in the image (offset already applied).
        axis1: Full width of ellipsoidal-envelope axis 1 in pixels.
        axis2: Full width of ellipsoidal-envelope axis 2 in pixels.
        axis3: Full width of ellipsoidal-envelope axis 3 in pixels.
        illumination_angle: Image-plane light azimuth in radians.
        phase_angle: Phase angle in radians.
        anti_aliasing: Limb supersampling control.
        ref_center_v: Reference center V for shape caching.
        ref_center_u: Reference center U for shape caching.

    Returns:
        Tuple of (body_mask, body_info_dict) matching the ellipsoid path.
    """
    size_v, size_u = img.shape
    body_shape = _render_mesh_shape_cached(
        size_v,
        size_u,
        axis1,
        axis2=axis2,
        axis3=axis3,
        spec=mesh_spec_from_params(body_params),
        illumination_angle=illumination_angle,
        phase_angle=phase_angle,
        anti_aliasing=anti_aliasing,
    )
    # A mesh body's pose is an arbitrary 3D rotation, so fall back to the
    # bounding sphere of the ellipsoidal envelope for both image axes.
    mesh_half_extent = max(axis1, axis2, axis3) / 2.0
    return finish_single_body(
        img,
        body_shape,
        body_params,
        body_name=body_name,
        center_v=center_v,
        center_u=center_u,
        half_extent_v=mesh_half_extent,
        half_extent_u=mesh_half_extent,
        ref_center_v=ref_center_v,
        ref_center_u=ref_center_u,
    )
