"""Navigation techniques.

Each technique is a subclass of ``NavTechnique`` that consumes a subset of
``NavFeature`` instances and produces a ``NavTechniqueResult`` (translation
offset, covariance, calibrated confidence, plus per-technique diagnostics).

Modules:

    ``nav_technique``
        ``NavTechnique`` ABC and the registry of registered subclasses.
    ``feasibility``
        ``NavFeasibilityReport`` returned by ``NavTechnique.is_feasible``.
    ``technique_result``
        ``NavTechniqueResult`` dataclass.
    ``diagnostics``
        Per-technique typed diagnostics dataclasses.
    ``confidence``
        ``evaluate_sigmoid_combination`` and the supporting ``ConfidenceSpec``
        / ``ConfidenceTerm`` dataclasses.
"""

import logging

from nav.nav_technique.confidence import (
    ConfidenceSpec,
    ConfidenceTerm,
    evaluate_sigmoid_combination,
)
from nav.nav_technique.diagnostics import (
    BodyBlobDiagnostics,
    BodyDiscDiagnostics,
    BodyLimbDiagnostics,
    BodyTerminatorDiagnostics,
    CartographicDiagnostics,
    NavTechniqueDiagnostics,
    RingAnnulusDiagnostics,
    RingEdgeDiagnostics,
    StarFieldDiagnostics,
    StarRefineDiagnostics,
    StarUniqueMatchDiagnostics,
)
from nav.nav_technique.feasibility import NavFeasibilityReport
from nav.nav_technique.nav_technique import NavTechnique, filter_technique_names
from nav.nav_technique.nav_technique_manual import NavTechniqueManual
from nav.nav_technique.technique_result import NavTechniqueResult

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    'BodyBlobDiagnostics',
    'BodyDiscDiagnostics',
    'BodyLimbDiagnostics',
    'BodyTerminatorDiagnostics',
    'CartographicDiagnostics',
    'ConfidenceSpec',
    'ConfidenceTerm',
    'NavFeasibilityReport',
    'NavTechnique',
    'NavTechniqueDiagnostics',
    'NavTechniqueManual',
    'NavTechniqueResult',
    'RingAnnulusDiagnostics',
    'RingEdgeDiagnostics',
    'StarFieldDiagnostics',
    'StarRefineDiagnostics',
    'StarUniqueMatchDiagnostics',
    'evaluate_sigmoid_combination',
    'filter_technique_names',
]
