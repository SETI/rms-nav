"""Simulated-scene star NavModel.

Emits STAR ``NavFeature`` instances for a simulated frame exactly the way
:class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars` does for a real
frame -- same predicted-SNR, covariance, reliability, and annotation
machinery -- but sources the star list from the sim renderer's output
(carried on ``obs.sim_star_list``) rather than reducing real catalogs.

The renderer builds each star at its *unshifted* predicted ``(v, u)`` and
draws it into the image shifted by the scene's planted offset.  This model
emits the unshifted prediction, so a star technique that detects the shifted
peak recovers the planted offset -- the same prediction/observation split a
real navigation has, which is why the recovery transfers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from oops import Observation

from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.stars.nav_model_stars import (
    NavModelStars,
    _star_short_info,
    _star_summary,
)
from spindoctor.support.time import now_dt

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.support.types import MutableStar

__all__ = ['NavModelStarsSimulated']


class NavModelStarsSimulated(NavModelStars):
    """Star NavModel populated from the sim renderer's star list.

    Inherits feature emission (``to_features``), annotations, and the
    extfov-coordinate plumbing from :class:`NavModelStars`; only the model
    construction differs -- the stars come from the rendered scene rather
    than from a catalog reduction.

    Parameters:
        name: Model name (typically ``'stars'``).
        obs: Simulated observation snapshot carrying ``sim_star_list``.
        config: Optional ``Config`` override.
    """

    @classmethod
    def instances_for_obs(cls, obs: Observation) -> list[NavModel]:
        """Return one star model for a simulated obs that rendered stars.

        Returns an empty list for a real obs (the catalog-driven
        :class:`NavModelStars` handles those) and for a simulated obs that
        rendered no stars (so the orchestrator builds no empty star model).

        Parameters:
            obs: Observation snapshot.

        Returns:
            ``[NavModelStarsSimulated('stars', obs)]`` for a simulated obs
            with at least one rendered star, else ``[]``.
        """
        if not getattr(obs, 'is_simulated', False):
            return []
        if not getattr(obs, 'sim_star_list', None):
            return []
        return [cls('stars', obs)]

    def create_model(self) -> None:
        """Populate the star list from the rendered scene.

        The sim renderer builds the per-star :class:`MutableStar` objects at
        their unshifted predicted positions and stashes them on
        ``obs.sim_star_list``; this model adopts that list directly.  Smear
        is zero (the sim renders no per-image attitude rate) and there are no
        body/ring occlusion conflicts to mark -- a simulated star field is
        clean by construction.
        """
        start_time = now_dt()
        self._metadata.clear()
        self._metadata['start_time'] = start_time.isoformat()
        self._metadata['end_time'] = None
        self._metadata['elapsed_time_sec'] = None
        with self._logger.open('CREATE SIMULATED STARS MODEL'):
            stars: list[MutableStar] = list(getattr(self.obs, 'sim_star_list', []) or [])
            self._stars = stars
            self._smear_vu = (0.0, 0.0)
            self._metadata['star_count'] = len(stars)
            self._metadata['stars'] = [_star_summary(star) for star in stars]
            self._logger.info('Using %d simulated star(s) from the rendered scene', len(stars))
            for star in stars:
                self._logger.info('  %s', _star_short_info(star))
            end_time = now_dt()
            self._metadata['end_time'] = end_time.isoformat()
            self._metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()
