"""Tests for ``spindoctor.nav_model.nav_model_titan.NavModelTitan``."""

from __future__ import annotations

from typing import Any

from spindoctor.annotation import Annotations
from spindoctor.nav_model.nav_model_titan import NavModelTitan


class _FakeObs:
    """Minimal obs stand-in for the Titan stub."""

    midtime: float = 0.0


def test_navmodeltitan_create_model_records_stub_metadata() -> None:
    """``create_model`` records the ``stub`` flag in metadata."""
    obs: Any = _FakeObs()
    model = NavModelTitan('titan', obs)
    model.create_model()
    assert model.metadata == {'stub': True}


def test_navmodeltitan_to_features_returns_empty() -> None:
    """``to_features`` returns an empty list — Titan navigation is unsupported."""
    obs: Any = _FakeObs()
    model = NavModelTitan('titan', obs)
    model.create_model()
    assert model.to_features(context=None) == []  # type: ignore[arg-type]


def test_navmodeltitan_to_annotations_returns_empty_collection() -> None:
    """``to_annotations`` returns an empty Annotations collection."""
    obs: Any = _FakeObs()
    model = NavModelTitan('titan', obs)
    model.create_model()
    annotations = model.to_annotations(context=None)  # type: ignore[arg-type]
    assert isinstance(annotations, Annotations)
    assert len(annotations.annotations) == 0
