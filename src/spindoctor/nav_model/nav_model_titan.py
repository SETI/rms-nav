"""Titan NavModel -- records a no-result for Titan's opaque haze.

Titan needs a different algorithm than ellipsoid-limb fitting: its visible
"limb" is the haze top, varies with wavelength, and the surface inside is
invisible.  At high phase Titan is not even a circle, so disc / limb /
terminator navigation is systematically wrong rather than merely noisy.

Titan is handled as a deliberate special case: its atmosphere is unique
(transparent at some wavelengths), so its handling does not generalize to
other thick-atmosphere bodies such as Venus.  This model is built and active
whenever Titan is in the field of view (the shape-based ``NavModelBody``
skips Titan).  It
emits no features, so no technique navigates it; instead it records, per
image, *why* a Titan scene cannot be navigated.  The orchestrator reads the
marker it exposes and fails such a frame with
:attr:`~spindoctor.support.status_reason.NavStatusReason.TITAN_UNSUPPORTED`
rather than a silent empty failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from oops import Observation

from spindoctor.annotation import Annotations
from spindoctor.config import DEFAULT_CONFIG, Config
from spindoctor.feature.feature import NavFeature
from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.nav_model_body import TITAN_BODY_NAME, bodies_in_extfov

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['NavModelTitan']


class NavModelTitan(NavModel):
    """Titan NavModel that declines to navigate.

    Concrete Titan navigation requires a haze-aware limb-fit technique with
    per-filter haze profiles; that algorithm is out of scope for this
    pipeline today.  The model exists so Titan in the FOV is recorded as an
    explicit no-result rather than vanishing into a generic empty-scene
    failure.

    Parameters:
        name: Model name (``'titan:TITAN'``).
        obs: Observation snapshot.
        config: Optional ``Config`` override.
    """

    def __init__(
        self,
        name: str,
        obs: Observation,
        *,
        config: Config | None = None,
    ) -> None:
        super().__init__(name, obs, config=config)

    @property
    def titan_in_fov(self) -> bool:
        """True: this model is only built when Titan is in the field of view.

        The orchestrator reads this to attribute an otherwise-empty frame to
        Titan non-support.
        """
        return True

    @classmethod
    def instances_for_obs(cls, obs: Observation, *, config: Config | None = None) -> list[NavModel]:
        """Return one instance when Titan is inside the extfov, else none.

        Parameters:
            obs: Observation snapshot.
            config: Configuration whose satellite catalog decides whether
                Titan is in the mission set.  ``None`` uses ``DEFAULT_CONFIG``.

        Returns:
            A single ``NavModelTitan`` when Titan is present in the extfov,
            otherwise an empty list.
        """
        # Simulated obs drive model selection from operator parameters, not the
        # SPICE inventory; mirror NavModelBody and build nothing here.
        if getattr(obs, 'is_simulated', False):
            return []
        if config is None:
            config = DEFAULT_CONFIG
        out: list[NavModel] = []
        for body_name, _entry in bodies_in_extfov(obs, config=config):
            if body_name.upper() == TITAN_BODY_NAME:
                out.append(cls('titan:TITAN', obs, config=config))
        return out

    def create_model(self) -> None:
        """Record Titan and log why it cannot be navigated."""
        self._metadata.clear()
        self._metadata['body'] = TITAN_BODY_NAME
        self._metadata['navigable'] = False
        with self._logger.open('TITAN MODEL'):
            self._logger.info(
                'Titan in FOV: navigation not supported '
                '(opaque haze hides the surface; no shape or haze-limb model)',
            )

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Return an empty feature list -- Titan navigation is unsupported."""
        del context
        return []

    def to_annotations(self, context: NavContext) -> Annotations:
        """Return an empty annotation collection."""
        del context
        return Annotations()
