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

import math
from dataclasses import dataclass, field

import numpy as np

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType


@dataclass(slots=True)
class RingRenderResult:
    """Result of rendering a single ring feature edge or band.

    Returned by ``RingFeature.render()``. Contains the rendered model image
    and mask, the feature uncertainty (km), and pre-computed annotation edge
    data.

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

    def __post_init__(self) -> None:
        """Validate image/mask shape, uncertainty, and edge annotation tuples."""
        if not isinstance(self.model_img, np.ndarray):
            raise TypeError('RingRenderResult.model_img must be a numpy ndarray')
        if not isinstance(self.model_mask, np.ndarray):
            raise TypeError('RingRenderResult.model_mask must be a numpy ndarray')
        if self.model_img.shape != self.model_mask.shape:
            raise ValueError(
                'RingRenderResult.model_img and model_mask must have the same shape, got '
                f'{self.model_img.shape} and {self.model_mask.shape}'
            )
        if isinstance(self.uncertainty, bool) or not isinstance(self.uncertainty, (int, float)):
            raise TypeError('RingRenderResult.uncertainty must be int or float')
        u = float(self.uncertainty)
        if not math.isfinite(u) or u < 0.0:
            raise ValueError(
                f'RingRenderResult.uncertainty must be finite and non-negative (km RMS), got '
                f'{self.uncertainty!r}'
            )

        for k, entry in enumerate(self.edge_info_list):
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise ValueError(
                    f'RingRenderResult.edge_info_list[{k}] must be '
                    f'(edge_mask, label_text, edge_label)'
                )
            em, t1, t2 = entry
            if not isinstance(em, np.ndarray):
                raise TypeError(f'RingRenderResult.edge_info_list[{k}][0] must be ndarray')
            if not np.issubdtype(em.dtype, np.bool_):
                raise ValueError(f'RingRenderResult.edge_info_list[{k}][0] must have boolean dtype')
            if em.shape != self.model_mask.shape:
                raise ValueError(
                    f'RingRenderResult.edge_info_list[{k}][0] shape {em.shape} must match '
                    f'model_mask shape {self.model_mask.shape}'
                )
            if not isinstance(t1, str) or not isinstance(t2, str):
                raise TypeError(f'RingRenderResult.edge_info_list[{k}][1] and [2] must be str')
