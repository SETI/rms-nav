"""Enum of feature types produced by NavFeatureExtractors.

A NavFeature carries one of these tags identifying which kind of
independently-navigable scene element it represents.

Modules:

    The single canonical list of NavFeatureType values.  New types are added
    by adding an enum value here and providing an extractor that emits it.
"""

from enum import Enum

__all__ = ['NavFeatureType']


class NavFeatureType(Enum):
    """The kind of scene element a NavFeature represents.

    Each value names a category of independently-navigable scene element with
    its own geometry, uncertainty model, and consuming techniques.

    - ``STAR``: a single catalog star (one ``(v, u)`` point).
    - ``LIMB_ARC``: one body's lit limb expressed as a polyline of vertices.
    - ``TERMINATOR_ARC``: one body's terminator polyline.
    - ``BODY_DISC``: a body disc rendered as a pixel template for correlation.
    - ``BODY_BLOB``: a body that is too irregular or under-resolved to fit a
      limb arc; carries only a predicted centroid and bounding box.
    - ``RING_EDGE``: a single named ring edge expressed as a polyline.
    - ``RING_ANNULUS``: a multi-ring composite rendered as a pixel template,
      used when individual edges cannot be separated at the image resolution.
    - ``TITAN_LIMB``: reserved for Titan navigation; never emitted
      by the current extractor set (the algorithm is unimplemented).
    - ``CARTOGRAPHIC_MODEL``: a pre-built cartographic mosaic of a body
      reprojected into the predicted body silhouette for high-detail
      correlation.
    """

    STAR = 'STAR'
    LIMB_ARC = 'LIMB_ARC'
    TERMINATOR_ARC = 'TERMINATOR_ARC'
    BODY_DISC = 'BODY_DISC'
    BODY_BLOB = 'BODY_BLOB'
    RING_EDGE = 'RING_EDGE'
    RING_ANNULUS = 'RING_ANNULUS'
    TITAN_LIMB = 'TITAN_LIMB'
    CARTOGRAPHIC_MODEL = 'CARTOGRAPHIC_MODEL'
