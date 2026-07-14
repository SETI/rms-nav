"""Atmospheric-body NavModel -- records a no-result for thick-haze bodies.

Titan and other bodies with thick opaque atmospheres need a different
algorithm than ellipsoid-limb fitting: the visible "limb" is the haze top,
varies with wavelength, and the surface inside is invisible.  At high phase
such a body is not even a circle, so disc / limb / terminator navigation is
systematically wrong rather than merely noisy.

This model is built and active whenever a member of
``bodies.atmospheric_bodies`` is in the field of view (the shape-based
``NavModelBody`` skips those bodies).  It emits no features, so no technique
navigates it; instead it records, per image, *why* an atmospheric-body scene
cannot be navigated.  The orchestrator reads the atmospheric-body name it
exposes and fails such a frame with
:attr:`~spindoctor.support.status_reason.NavStatusReason.ATMOSPHERIC_BODY_UNSUPPORTED`
rather than a silent empty failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from oops import Observation

from spindoctor.annotation import Annotations
from spindoctor.config import DEFAULT_CONFIG, Config
from spindoctor.feature.feature import NavFeature
from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.nav_model_body import atmospheric_body_set, bodies_in_extfov

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['NavModelTitan']


class NavModelTitan(NavModel):
    """Atmospheric-body NavModel that declines to navigate.

    Concrete atmospheric-body navigation requires a haze-aware limb-fit
    technique with per-filter haze profiles; that algorithm is out of scope
    for this pipeline today.  The model exists so a thick-atmosphere body in
    the FOV is recorded as an explicit no-result rather than vanishing into a
    generic empty-scene failure.

    Parameters:
        name: Model name (e.g. ``'atmospheric:TITAN'``).
        obs: Observation snapshot.
        body_name: SPICE body name of the atmospheric body.
        config: Optional ``Config`` override.
    """

    def __init__(
        self,
        name: str,
        obs: Observation,
        body_name: str,
        *,
        config: Config | None = None,
    ) -> None:
        super().__init__(name, obs, config=config)
        self._body_name = body_name.upper()

    @property
    def atmospheric_body_name(self) -> str:
        """Upper-case SPICE name of the atmospheric body this model covers.

        The orchestrator reads this to attribute an otherwise-empty frame to
        atmospheric-body non-support.
        """
        return self._body_name

    @classmethod
    def instances_for_obs(cls, obs: Observation, *, config: Config | None = None) -> list[NavModel]:
        """Return one instance per thick-atmosphere body inside the extfov.

        Parameters:
            obs: Observation snapshot.
            config: Configuration whose ``bodies.atmospheric_bodies`` list and
                satellite catalog decide which bodies qualify.  ``None`` uses
                ``DEFAULT_CONFIG``.

        Returns:
            One ``NavModelTitan`` per atmospheric body present in the extfov.
        """
        # Simulated obs drive model selection from operator parameters, not the
        # SPICE inventory; mirror NavModelBody and build nothing here.
        if getattr(obs, 'is_simulated', False):
            return []
        if config is None:
            config = DEFAULT_CONFIG
        atmospheric = atmospheric_body_set(config)
        if not atmospheric:
            return []
        out: list[NavModel] = []
        for body_name, _entry in bodies_in_extfov(obs, config=config):
            if body_name.upper() in atmospheric:
                out.append(cls(f'atmospheric:{body_name}', obs, body_name, config=config))
        return out

    def create_model(self) -> None:
        """Record the atmospheric body and log why it cannot be navigated."""
        self._metadata.clear()
        self._metadata['atmospheric_body'] = self._body_name
        self._metadata['navigable'] = False
        with self._logger.open(f'ATMOSPHERIC BODY MODEL: {self._body_name}'):
            self._logger.info(
                'atmospheric body %s in FOV: navigation not supported '
                '(opaque haze hides the surface; no shape or haze-limb model)',
                self._body_name,
            )

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Return an empty feature list -- atmospheric-body navigation is unsupported."""
        del context
        return []

    def to_annotations(self, context: NavContext) -> Annotations:
        """Return an empty annotation collection."""
        del context
        return Annotations()
