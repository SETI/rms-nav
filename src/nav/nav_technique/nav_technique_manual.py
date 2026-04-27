"""Manual navigation technique — interactive PyQt6 dialog.

Renders the composite predicted scene from every NavModel's template
features and lets the operator pick a (dv, du) offset by hand.  An
``Auto`` button inside the dialog runs the same masked-NCC pyramid that
correlation-based techniques use, so the operator can either accept the
auto-pick or override it manually.

This technique is not part of the autonomous pipeline; it does not appear
in the ``NavTechnique._registry`` and is invoked directly by an
interactive driver.  The plan calls for replacing the dialog's single
``Auto`` button with a per-technique side-by-side panel showing each
registered technique's proposal; that redesign is deferred and tracked
separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nav.config import Config
from nav.feature.composition import compose_template_features
from nav.feature.feature import NavFeature
from nav.feature.feature_type import NavFeatureType
from nav.nav_technique.diagnostics import BodyDiscDiagnostics
from nav.nav_technique.feasibility import NavFeasibilityReport
from nav.nav_technique.nav_technique import NavTechnique
from nav.nav_technique.technique_result import NavTechniqueResult

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from nav.nav_orchestrator.nav_context import NavContext

__all__ = ['NavTechniqueManual']


# Ext-FOV pixel covariance assigned to a manual-pick offset.  Operator
# precision in this dialog is limited by zoom, eye, and screen pixels;
# 1 px sigma per axis is a reasonable conservative value.  The ensemble
# never combines a manual result with auto results, so this number only
# determines what shows up in JSON metadata.
_MANUAL_OFFSET_SIGMA_PX = 1.0


class NavTechniqueManual(NavTechnique):
    """Interactive manual navigation.

    Composes every template-bearing feature into a single ext-FOV image
    plus mask, hands the result plus the observation to the
    ``ManualNavDialog``, and packages the operator's choice into a
    ``NavTechniqueResult``.

    Class attributes:
        _abstract: ``True`` — kept out of the auto-discovery registry so
            the orchestrator does not invoke the dialog during background
            navigation runs.
        accepts_feature_types: every feature type — manual navigation
            looks at whatever the scene has rendered.
    """

    _abstract = True

    name = 'NavTechniqueManual'
    accepts_feature_types = frozenset(NavFeatureType)
    requires_prior = False

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Manual navigation runs whenever there is anything to render.

        At least one feature must carry a template (``BODY_DISC``,
        ``RING_ANNULUS`` or ``CARTOGRAPHIC_MODEL``); without one the
        dialog has no overlay to display.
        """
        template_count = sum(
            1 for f in features if f.template_img is not None and f.template_mask is not None
        )
        if template_count == 0:
            return NavFeasibilityReport(
                feasible=False,
                reason='no_template_features_for_manual_nav',
            )
        return NavFeasibilityReport(
            feasible=True,
            reason='ok',
            consumed_feature_count=template_count,
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Run the dialog and convert the operator's choice to a result.

        Cancelling the dialog yields a spurious result with zero
        confidence so the ensemble drops it; accepting the dialog yields
        a result with ``_MANUAL_OFFSET_SIGMA_PX`` per-axis covariance.
        """
        # Local imports keep PyQt6 out of import-time graphs; the dialog
        # is only ever loaded when manual navigation is invoked.
        from PyQt6.QtWidgets import QApplication

        from nav.ui.manual_nav_dialog import ManualNavDialog

        with self.logger.open('NAVIGATION PASS: MANUAL'):
            obs = context.obs
            shape = context.image_ext.shape
            model_img, model_mask = compose_template_features(
                features,
                (int(shape[0]), int(shape[1])),
            )
            app_created = False
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
                app_created = True
            dialog = ManualNavDialog(
                obs=obs,  # type: ignore[arg-type]
                model_img_ext=model_img,
                model_mask_ext=model_mask,
                config=self.config,
                parent=None,
            )
            accepted, chosen_offset, _last_corr = dialog.run_modal()
            if app_created:
                app.quit()
        if not accepted or chosen_offset is None:
            self.logger.info('Manual navigation canceled by user')
            return NavTechniqueResult(
                technique_name=self.name,
                feature_ids=tuple(f.feature_id for f in features),
                offset_px=(0.0, 0.0),
                covariance_px2=np.eye(2, dtype=np.float64),
                confidence=0.0,
                spurious=True,
                at_edge=False,
                diagnostics=BodyDiscDiagnostics(),
            )
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=tuple(f.feature_id for f in features),
            offset_px=chosen_offset,
            covariance_px2=np.eye(2, dtype=np.float64) * (_MANUAL_OFFSET_SIGMA_PX**2),
            confidence=1.0,
            spurious=False,
            at_edge=False,
            diagnostics=BodyDiscDiagnostics(),
        )
