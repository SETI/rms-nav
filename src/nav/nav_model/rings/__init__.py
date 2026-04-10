"""Ring feature domain model for planetary navigation.

This subpackage implements typed domain objects for ring feature data, rendering,
and filtering. It replaces the untyped ``dict[str, Any]`` approach in the legacy
``nav_model_rings.py`` with immutable dataclasses that carry validation at
construction time and rendering intelligence as methods.

Architecture overview:

- ``ring_types``: Pure frozen dataclasses for orbital parameters. No rendering
  dependencies; safe for lightweight import.
- ``ring_render_context``: Immutable bundle of rendering dependencies passed to
  ``RingFeature.render()``.
- ``ring_render_result``: Lightweight result object returned by rendering.
- ``ring_feature``: Core domain object. Owns backplane-based rendering and
  cross-feature date-overlap validation.
- ``ring_filter``: Four-pass filter pipeline deciding which features to render.
- ``ring_math``: Pure mathematical functions for fade and anti-aliasing.
"""

from .ring_feature import RingFeature, validate_no_date_overlaps
from .ring_filter import RingFeatureFilter
from .ring_math import compute_antialiasing, compute_edge_fade, compute_fade_integral
from .ring_render_context import RingsRenderContext
from .ring_render_result import RingRenderResult
from .ring_types import (
    RingBaseOrbitMode,
    RingEdgeData,
    RingFeatureType,
    RingPerturbationMode,
)

__all__ = [
    'RingBaseOrbitMode',
    'RingEdgeData',
    'RingFeature',
    'RingFeatureFilter',
    'RingFeatureType',
    'RingPerturbationMode',
    'RingRenderResult',
    'RingsRenderContext',
    'compute_antialiasing',
    'compute_edge_fade',
    'compute_fade_integral',
    'validate_no_date_overlaps',
]
