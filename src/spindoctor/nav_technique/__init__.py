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

from spindoctor.nav_technique.confidence import (
    ConfidenceSpec,
    ConfidenceTerm,
    evaluate_sigmoid_combination,
)
from spindoctor.nav_technique.diagnostics import (
    BodyBlobDiagnostics,
    BodyDiscDiagnostics,
    BodyLimbDiagnostics,
    BodyTerminatorDiagnostics,
    ManualNavDiagnostics,
    NavTechniqueDiagnostics,
    RingAnnulusDiagnostics,
    RingEdgeDiagnostics,
    StarFieldDiagnostics,
    StarRefineDiagnostics,
    StarUniqueMatchDiagnostics,
)
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import NavTechnique, filter_technique_names
from spindoctor.nav_technique.nav_technique_body_blob import BodyBlobNav
from spindoctor.nav_technique.nav_technique_body_disc import BodyDiscCorrelateNav
from spindoctor.nav_technique.nav_technique_body_limb import BodyLimbNav
from spindoctor.nav_technique.nav_technique_body_terminator import BodyTerminatorNav
from spindoctor.nav_technique.nav_technique_manual import NavTechniqueManual, run_manual_nav
from spindoctor.nav_technique.nav_technique_ring_annulus import RingAnnulusNav
from spindoctor.nav_technique.nav_technique_ring_edge import RingEdgeNav
from spindoctor.nav_technique.nav_technique_star_field import StarFieldFromCatalogNav
from spindoctor.nav_technique.nav_technique_star_refine import StarRefineNav
from spindoctor.nav_technique.nav_technique_star_unique_match import StarUniqueMatchNav
from spindoctor.nav_technique.technique_result import NavTechniqueResult

__all__ = [
    'BodyBlobDiagnostics',
    'BodyBlobNav',
    'BodyDiscCorrelateNav',
    'BodyDiscDiagnostics',
    'BodyLimbDiagnostics',
    'BodyLimbNav',
    'BodyTerminatorDiagnostics',
    'BodyTerminatorNav',
    'ConfidenceSpec',
    'ConfidenceTerm',
    'ManualNavDiagnostics',
    'NavFeasibilityReport',
    'NavTechnique',
    'NavTechniqueDiagnostics',
    'NavTechniqueManual',
    'NavTechniqueResult',
    'RingAnnulusDiagnostics',
    'RingAnnulusNav',
    'RingEdgeDiagnostics',
    'RingEdgeNav',
    'StarFieldDiagnostics',
    'StarFieldFromCatalogNav',
    'StarRefineDiagnostics',
    'StarRefineNav',
    'StarUniqueMatchDiagnostics',
    'StarUniqueMatchNav',
    'evaluate_sigmoid_combination',
    'filter_technique_names',
    'run_manual_nav',
]
