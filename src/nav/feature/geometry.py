"""Sum-type variants describing the geometry of a NavFeature.

Each ``NavFeature`` carries a ``geometry`` payload whose concrete dataclass
type matches the ``feature_type``.  The payload holds whatever the consuming
technique needs to know about *where in the image* the feature lives —
image-side operations remain global, so no payload describes a per-feature
image crop.

Coordinates are in extended-FOV (extfov) image coordinates (v, u).  Bounding
boxes are half-open in the numpy slicing sense: ``v_min, u_min, v_max,
u_max`` with ``arr[v_min:v_max, u_min:u_max]`` covering the box.
"""

from dataclasses import dataclass

from nav.support.types import NDArrayFloatType

__all__ = [
    'BodyBlobGeometry',
    'BodyDiscGeometry',
    'CartographicModelGeometry',
    'LimbPolyline',
    'NavFeatureGeometry',
    'RingAnnulusGeometry',
    'RingEdgePolyline',
    'StarGeometry',
    'TerminatorPolyline',
]


@dataclass(frozen=True, eq=False)
class StarGeometry:
    """Geometry payload for a STAR feature.

    Single (v, u) point in extended-FOV coordinates plus the catalog-derived
    prediction from which it was generated.  The two are equal at extraction
    time and may differ after a refinement step records the matched
    detection.

    Parameters:
        predicted_vu: Predicted star (v, u) in extfov coordinates.
        catalog_vu: Catalog-aberrated star (v, u) in extfov coordinates.
            Equal to ``predicted_vu`` at extraction; may differ after
            refinement.
        bbox_extfov_vu: Half-open bounding box ``(v_min, u_min, v_max,
            u_max)`` covering the postage stamp around the predicted
            position.
    """

    predicted_vu: tuple[float, float]
    catalog_vu: tuple[float, float]
    bbox_extfov_vu: tuple[int, int, int, int]


@dataclass(frozen=True, eq=False)
class LimbPolyline:
    """Geometry payload for a LIMB_ARC feature.

    A polyline of vertices along a body's predicted limb, after
    extraction-time cropping for occlusion / off-FOV / shadow.  Each vertex
    carries its own normal direction and per-vertex anisotropic uncertainty.

    Parameters:
        vertices_vu: ``(N, 2)`` array of (v, u) per surviving vertex.
        normals_vu: ``(N, 2)`` array of outward limb normal per vertex.
        sigma_normal_per_vertex_px: ``(N,)`` per-vertex sigma along normal.
        sigma_tangent_per_vertex_px: ``(N,)`` per-vertex sigma along tangent;
            typically a small constant (~0.5 px) reflecting polyline
            sampling resolution.
        bbox_extfov_vu: Half-open bounding box of the polyline.
    """

    vertices_vu: NDArrayFloatType
    normals_vu: NDArrayFloatType
    sigma_normal_per_vertex_px: NDArrayFloatType
    sigma_tangent_per_vertex_px: NDArrayFloatType
    bbox_extfov_vu: tuple[int, int, int, int]


@dataclass(frozen=True, eq=False)
class TerminatorPolyline:
    """Geometry payload for a TERMINATOR_ARC feature.

    Mirrors ``LimbPolyline`` with terminator-specific semantics — the vertices
    lie along the terminator (where ``cos(incidence) == 0``) rather than the
    silhouette.  Per-vertex sigma_normal is generally larger than the
    matching limb because albedo variation softens the photometric edge.

    Parameters: see ``LimbPolyline`` (identical field set).
    """

    vertices_vu: NDArrayFloatType
    normals_vu: NDArrayFloatType
    sigma_normal_per_vertex_px: NDArrayFloatType
    sigma_tangent_per_vertex_px: NDArrayFloatType
    bbox_extfov_vu: tuple[int, int, int, int]


@dataclass(frozen=True, eq=False)
class RingEdgePolyline:
    """Geometry payload for a RING_EDGE feature.

    Polyline of vertices along one named ring edge.  Each vertex's per-axis
    uncertainty is along the radial direction (across the edge) and along
    the edge tangent.  The straight-line flag is set when the projected
    polyline's deviation from a best-fit straight line is below threshold;
    in that case its rank-1 covariance must be combined with another feature
    to resolve a 2-D offset.

    Parameters:
        vertices_vu: ``(N, 2)`` (v, u) per vertex.
        normals_vu: ``(N, 2)`` radially outward per vertex.
        sigma_radial_per_vertex_px: ``(N,)`` sigma across the edge (radial).
        sigma_along_edge_per_vertex_px: ``(N,)`` sigma along the edge.
        is_straight_line: ``True`` if the polyline's max-deviation from a
            best-fit straight line is below the curvature threshold.
        bbox_extfov_vu: Half-open bounding box of the polyline.
    """

    vertices_vu: NDArrayFloatType
    normals_vu: NDArrayFloatType
    sigma_radial_per_vertex_px: NDArrayFloatType
    sigma_along_edge_per_vertex_px: NDArrayFloatType
    is_straight_line: bool
    bbox_extfov_vu: tuple[int, int, int, int]


@dataclass(frozen=True, eq=False)
class BodyDiscGeometry:
    """Geometry payload for a BODY_DISC feature.

    The body's full-disc rendering is carried on ``NavFeature.template_img``;
    the geometry payload only records the position of that template within
    the extfov image, the predicted body-center pixel, and the fraction of
    the predicted disc area that falls outside the sensor.

    Parameters:
        bbox_extfov_vu: Half-open bounding box where the template sits.
        predicted_center_vu: Predicted body center in extfov coordinates.
        overflow_fraction: Fraction of the disc area outside the sensor
            ``[0, 1]``; ``0`` means fully in-FOV.
    """

    bbox_extfov_vu: tuple[int, int, int, int]
    predicted_center_vu: tuple[float, float]
    overflow_fraction: float


@dataclass(frozen=True, eq=False)
class BodyBlobGeometry:
    """Geometry payload for a BODY_BLOB feature.

    Carries only the predicted centroid and bounding extent of an under-
    resolved or irregular body.  No template is rendered.

    Parameters:
        predicted_center_vu: Predicted body center in extfov coordinates.
        bbox_extfov_vu: Half-open bounding box around the predicted body.
        predicted_diameter_px: Predicted disc diameter in pixels (longer
            axis of the predicted ellipse silhouette).
    """

    predicted_center_vu: tuple[float, float]
    bbox_extfov_vu: tuple[int, int, int, int]
    predicted_diameter_px: float


@dataclass(frozen=True, eq=False)
class RingAnnulusGeometry:
    """Geometry payload for a RING_ANNULUS feature.

    Multi-ring composite template carried on ``NavFeature.template_img``;
    this payload records only the template's location in the extfov image
    and the predicted ring-system center.

    Parameters:
        bbox_extfov_vu: Half-open bounding box where the template sits.
        predicted_center_vu: Predicted planet center for the ring system.
    """

    bbox_extfov_vu: tuple[int, int, int, int]
    predicted_center_vu: tuple[float, float]


@dataclass(frozen=True, eq=False)
class CartographicModelGeometry:
    """Geometry payload for a CARTOGRAPHIC_MODEL feature.

    A pre-built cartographic mosaic of a body, reprojected into the predicted
    body silhouette and stored on ``NavFeature.template_img``.  Same shape
    as ``BodyDiscGeometry`` — the difference is that the template carries
    surface detail rather than smooth Lambert shading.

    Parameters:
        bbox_extfov_vu: Half-open bounding box where the template sits.
        predicted_center_vu: Predicted body center in extfov coordinates.
        overflow_fraction: Fraction of the disc area outside the sensor.
    """

    bbox_extfov_vu: tuple[int, int, int, int]
    predicted_center_vu: tuple[float, float]
    overflow_fraction: float


NavFeatureGeometry = (
    StarGeometry
    | LimbPolyline
    | TerminatorPolyline
    | RingEdgePolyline
    | BodyDiscGeometry
    | BodyBlobGeometry
    | RingAnnulusGeometry
    | CartographicModelGeometry
)
"""Sum type spanning every NavFeatureType's geometry payload."""
