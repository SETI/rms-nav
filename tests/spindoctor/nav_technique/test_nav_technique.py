"""Tests for ``spindoctor.nav_technique.nav_technique.NavTechnique``."""

from typing import Any

import numpy as np
import pytest

# Side-effect imports — register the shipped techniques in
# ``NavTechnique._registry`` so the validator has a non-empty input.
import spindoctor.nav_technique.nav_technique_body_limb
import spindoctor.nav_technique.nav_technique_body_terminator
import spindoctor.nav_technique.nav_technique_ring_edge  # noqa: F401
from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_technique.confidence import ConfidenceSpec, ConfidenceTerm
from spindoctor.nav_technique.diagnostics import BodyLimbDiagnostics
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import (
    NavTechnique,
    filter_technique_names,
    validate_registered_confidence_specs,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult


class _ConcreteTechniqueForTest(NavTechnique):
    """Concrete subclass for registry testing."""

    name = '_ConcreteTechniqueForTest'
    accepts_feature_types = frozenset({NavFeatureType.STAR})

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        return NavFeasibilityReport(feasible=False, reason='test_only')

    def navigate(self, features: list[NavFeature], context: Any) -> NavTechniqueResult:
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=(),
            offset_px=(0.0, 0.0),
            covariance_px2=np.eye(2, dtype=np.float64),
            confidence=0.0,
            spurious=False,
            at_edge=False,
            diagnostics=BodyLimbDiagnostics(),
        )


def test_navtechnique_registry_records_subclass() -> None:
    """Concrete subclasses self-register via __init_subclass__."""
    assert _ConcreteTechniqueForTest in NavTechnique._registry


def test_navtechnique_can_invoke_navigate() -> None:
    """A concrete subclass's navigate returns a NavTechniqueResult."""
    technique = _ConcreteTechniqueForTest()
    result = technique.navigate([], context=None)
    assert isinstance(result, NavTechniqueResult)


def test_filter_technique_names_inclusion() -> None:
    """Inclusion patterns admit matching names only."""
    names = ['BodyLimbNav', 'BodyDiscCorrelateNav', 'StarRefineNav']
    out = filter_technique_names(names, ['Body*'])
    assert out == ['BodyLimbNav', 'BodyDiscCorrelateNav']


def test_filter_technique_names_exclusion() -> None:
    """Leading-bang patterns exclude matches."""
    names = ['BodyLimbNav', 'BodyDiscCorrelateNav', 'StarRefineNav']
    out = filter_technique_names(names, ['*', '!Body*'])
    assert out == ['StarRefineNav']


def test_filter_technique_names_exclusion_only_implies_star_include() -> None:
    """A pure-exclusion pattern list implies '*' as the include."""
    names = ['BodyLimbNav', 'StarRefineNav']
    out = filter_technique_names(names, ['!StarRefineNav'])
    assert out == ['BodyLimbNav']


def test_filter_technique_names_default_star() -> None:
    """Default '*' pattern keeps every name."""
    names = ['A', 'B', 'C']
    out = filter_technique_names(names, '*')
    assert out == ['A', 'B', 'C']


def test_filter_technique_names_rejects_empty_patterns() -> None:
    """An empty list raises ValueError."""
    with pytest.raises(ValueError, match='at least one'):
        filter_technique_names(['A'], [])


def test_validate_registered_confidence_specs_passes_for_shipped_techniques() -> None:
    """Every shipped technique's spec only references declared attributes."""
    validate_registered_confidence_specs()


def test_validate_registered_confidence_specs_rejects_unknown_attribute() -> None:
    """A spec referencing an attribute outside confidence_attributes raises."""

    class _BadConfidenceTechnique(NavTechnique):
        name = '_BadConfidenceTechnique'
        accepts_feature_types = frozenset({NavFeatureType.STAR})
        confidence_spec = ConfidenceSpec(
            alpha0=0.0,
            terms=(ConfidenceTerm(feature='nope_undeclared', alpha=1.0),),
        )
        confidence_attributes = frozenset({'at_edge'})

        def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
            return NavFeasibilityReport(feasible=False, reason='test_only')

        def navigate(self, features: list[NavFeature], context: Any) -> NavTechniqueResult:
            return NavTechniqueResult(
                technique_name=self.name,
                feature_ids=(),
                offset_px=(0.0, 0.0),
                covariance_px2=np.eye(2, dtype=np.float64),
                confidence=0.0,
                spurious=False,
                at_edge=False,
                diagnostics=BodyLimbDiagnostics(),
            )

    try:
        with pytest.raises(ValueError, match='nope_undeclared'):
            validate_registered_confidence_specs()
    finally:
        NavTechnique._registry.remove(_BadConfidenceTechnique)
