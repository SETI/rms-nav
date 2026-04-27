"""Tests for ``nav.nav_technique.nav_technique.NavTechnique``."""

from typing import Any

import numpy as np
import pytest

from nav.feature.feature import NavFeature
from nav.feature.feature_type import NavFeatureType
from nav.nav_technique.diagnostics import BodyLimbDiagnostics
from nav.nav_technique.feasibility import NavFeasibilityReport
from nav.nav_technique.nav_technique import NavTechnique, filter_technique_names
from nav.nav_technique.technique_result import NavTechniqueResult


class _ConcreteTechniqueForTest(NavTechnique):
    """Concrete subclass for registry testing."""

    name = '_ConcreteTechniqueForTest'
    accepts_feature_types = frozenset({NavFeatureType.STAR})

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        return NavFeasibilityReport(feasible=False, reason='test_only')

    def navigate(self, features: list[NavFeature], context: Any) -> NavTechniqueResult:
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=[],
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
