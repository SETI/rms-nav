"""Result object for ring feature backplane rendering.

This module defines ``RingRenderResult``, the structured output of
``RingFeature.render()``. Returning a typed dataclass instead of a tuple of
arrays makes the return value self-documenting and lets the orchestrator access
the uncertainty and annotation data without relying on positional unpacking.

The ``edge_info_list`` is computed during rendering rather than in a separate
annotation pass. This avoids recomputing the edge radius backplanes a second
time: the render method already has the computed backplane results in scope
when it creates the edge masks for ``border_atop``.
"""

from dataclasses import dataclass, field

from nav.support.types import NDArrayBoolType, NDArrayFloatType


@dataclass(slots=True)
class RingRenderResult:
    """Result of rendering a single ring feature edge or band.

    Returned by ``RingFeature.render()``. Contains the rendered model image
    and mask, the feature uncertainty (for ``NavModelResult``), and pre-computed
    annotation edge data.

    ``edge_info_list`` contains ``(edge_mask, label_text, edge_label)`` tuples
    for annotation creation. ``render()`` computes these during rendering to
    avoid recomputing the edge radius backplanes a second time. The orchestrator
    passes this list to ``NavModelRingsBase._create_edge_annotations()``.

    Parameters:
        model_img: Float64 array of rendered ring brightness values. Shape
            matches the extended FOV.
        model_mask: Boolean mask array where True indicates pixels with
            non-zero ring model contribution.
        uncertainty: Maximum RMS across all rendered edges (km). Sourced from
            ``RingEdgeData.rms`` via ``RingFeature.uncertainty``.
        edge_info_list: Pre-computed annotation data: list of
            ``(edge_mask, label_text, edge_label)`` tuples. ``edge_mask`` is a
            boolean array in extended FOV coordinates.
    """

    model_img: NDArrayFloatType
    model_mask: NDArrayBoolType
    uncertainty: float
    edge_info_list: list[tuple[NDArrayBoolType, str, str]] = field(default_factory=list)
