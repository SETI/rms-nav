"""Tests for ``spindoctor.nav_technique.diagnostics`` per-technique dataclasses."""

import dataclasses

import pytest

from spindoctor.nav_technique.diagnostics import (
    BodyBlobDiagnostics,
    BodyDiscDiagnostics,
    BodyLimbDiagnostics,
    BodyTerminatorDiagnostics,
    RingAnnulusDiagnostics,
    RingEdgeDiagnostics,
    StarFieldDiagnostics,
    StarRefineDiagnostics,
    StarUniqueMatchDiagnostics,
    TitanHazeDiagnostics,
)


@pytest.mark.parametrize(
    'cls',
    [
        BodyDiscDiagnostics,
        BodyLimbDiagnostics,
        BodyTerminatorDiagnostics,
        BodyBlobDiagnostics,
        RingEdgeDiagnostics,
        RingAnnulusDiagnostics,
        StarFieldDiagnostics,
        StarUniqueMatchDiagnostics,
        StarRefineDiagnostics,
        TitanHazeDiagnostics,
    ],
)
def test_diagnostic_dataclasses_construct_with_defaults(cls: type) -> None:
    """Every diagnostics dataclass can be instantiated with no arguments."""
    diag = cls()
    assert hasattr(diag, 'CURATOR_FIELDS')


@pytest.mark.parametrize(
    'cls',
    [
        BodyDiscDiagnostics,
        BodyLimbDiagnostics,
        BodyTerminatorDiagnostics,
        BodyBlobDiagnostics,
        RingEdgeDiagnostics,
        RingAnnulusDiagnostics,
        StarFieldDiagnostics,
        StarUniqueMatchDiagnostics,
        StarRefineDiagnostics,
        TitanHazeDiagnostics,
    ],
)
def test_curator_fields_lists_every_attribute(cls: type) -> None:
    """CURATOR_FIELDS keys cover every dataclass field."""
    field_names = {f.name for f in dataclasses.fields(cls)}
    curator_fields: dict[str, str | None] = cls.CURATOR_FIELDS  # type: ignore[attr-defined]
    declared = set(curator_fields.keys())
    assert field_names == declared
