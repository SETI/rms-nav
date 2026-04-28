"""Orchestrator layer — turns observations into navigation results.

The orchestrator extracts features, gates them by reliability, runs each
feasible technique, and reconciles the per-technique results into one
``NavResult``.

Modules:

    ``nav_context``
        ``NavContext`` — per-image global state.
    ``feature_summary``
        ``NavFeatureSummary`` — per-feature post-mortem entry.
    ``image_classifier``
        ``NavImageClassifier`` — quick-fail image-quality classifier.
    ``image_classifier_result``
        ``NavImageClassifierResult`` — output of the classifier.
    ``image_derivatives``
        ``build_image_edge_dt`` and ``ImageDerivativesConfig`` — shared
        gradient and edge-distance-transform computation.
    ``provenance``
        ``Provenance`` — reproducibility metadata.
    ``ensemble``
        ``ensemble`` — free function that reconciles per-technique results.
    ``nav_result``
        ``NavResult`` — final output dataclass.
    ``curator``
        ``build_metadata_dict`` — JSON output curation.
    ``status_reason_info``
        ``STATUS_REASON_INFO_TEMPLATE`` — per-``status_reason`` operator log
        templates.
"""

from nav.nav_orchestrator.curator import (
    assert_diagnostic_fields_present,
    build_metadata_dict,
)
from nav.nav_orchestrator.ensemble import (
    EnsembleConfig,
    derive_confidence_rank,
    ensemble,
)
from nav.nav_orchestrator.feature_summary import NavFeatureSummary
from nav.nav_orchestrator.image_classifier import (
    ImageQualityThresholds,
    NavImageClassifier,
)
from nav.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from nav.nav_orchestrator.image_derivatives import (
    ImageDerivativesConfig,
    build_image_edge_dt,
)
from nav.nav_orchestrator.nav_context import NavContext
from nav.nav_orchestrator.nav_result import NavResult
from nav.nav_orchestrator.orchestrator import NavOrchestrator
from nav.nav_orchestrator.provenance import Provenance
from nav.nav_orchestrator.status_reason_info import STATUS_REASON_INFO_TEMPLATE

__all__ = [
    'STATUS_REASON_INFO_TEMPLATE',
    'EnsembleConfig',
    'ImageDerivativesConfig',
    'ImageQualityThresholds',
    'NavContext',
    'NavFeatureSummary',
    'NavImageClassifier',
    'NavImageClassifierResult',
    'NavOrchestrator',
    'NavResult',
    'Provenance',
    'assert_diagnostic_fields_present',
    'build_image_edge_dt',
    'build_metadata_dict',
    'derive_confidence_rank',
    'ensemble',
]
