"""Stub Titan NavModel — atmospheric-body navigation deferred.

Titan and other bodies with thick opaque atmospheres need a different
algorithm than ellipsoid-limb fitting: the visible "limb" is the haze top,
varies with wavelength, and the surface inside is invisible.  This module
exists as a registered placeholder; it never emits features or annotations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from oops import Observation

from spindoctor.annotation import Annotations
from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature
from spindoctor.nav_model.nav_model import NavModel

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['NavModelTitan']


class NavModelTitan(NavModel):
    """Placeholder Titan NavModel that produces no features or annotations.

    Concrete atmospheric-body navigation requires a haze-aware limb-fit
    technique with per-filter haze profiles; that algorithm is out of
    scope for this pipeline today.

    Parameters:
        name: Model name (e.g. ``'titan'``).
        obs: Observation snapshot.
        config: Optional ``Config`` override.
    """

    def __init__(self, name: str, obs: Observation, *, config: Config | None = None) -> None:
        super().__init__(name, obs, config=config)

    def create_model(self) -> None:
        """No-op: the Titan model has no internal state to populate."""
        self._metadata['stub'] = True

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Return an empty feature list — Titan navigation is unsupported."""
        return []

    def to_annotations(self, context: NavContext) -> Annotations:
        """Return an empty annotation collection."""
        return Annotations()
