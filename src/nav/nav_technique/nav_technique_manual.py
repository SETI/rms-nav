from typing import TYPE_CHECKING

from nav.config import Config
from nav.nav_model import NavModelCombined

from .nav_technique import NavTechnique

if TYPE_CHECKING:
    from nav.nav_master import NavMaster


class NavTechniqueManual(NavTechnique):
    """Manual navigation technique using an interactive PyQt6 dialog.

    Builds the same combined model as NavTechniqueCorrelateAll, but lets the user
    manually specify the (dv, du) offset using a GUI with pan/zoom. The dialog also
    supports an Auto button that calls the same KPeaks correlation used by the
    correlate_all technique.
    """

    def __init__(self, nav_master: 'NavMaster', *, config: Config | None = None) -> None:
        super().__init__(nav_master, config=config)
        self._combined_model: NavModelCombined | None = None

    def combined_model(self) -> NavModelCombined | None:
        """Returns the combined model created for this technique."""
        return self._combined_model

    def navigate(self) -> None:
        """Run the manual navigation dialog and return the chosen offset."""
        with self.logger.open('NAVIGATION PASS: MANUAL'):
            # Build combined model from all available models
            combined_model = self._combine_models(['*'])
            self._combined_model = combined_model
            if combined_model is None:
                self.logger.info('Manual navigation technique failed - no models available')
                return

            if (
                len(combined_model.models) == 0
                or combined_model.models[0].model_img is None
                or combined_model.models[0].model_mask is None
            ):
                raise ValueError('Combined model has no result or missing image/mask')

            # Import here to avoid importing PyQt6 unless needed
            from PyQt6.QtWidgets import QApplication

            from nav.ui.manual_nav_dialog import ManualNavDialog

            obs = self.nav_master.obs

            # Ensure a QApplication exists before creating any QWidget
            app_created = False
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
                app_created = True

            # Launch the dialog, allowing the user to select offset
            dialog = ManualNavDialog(
                obs=obs, combined_model=combined_model, config=self.config, parent=None
            )

            accepted, chosen_offset, _last_corr = dialog.run_modal()

            # If we created the QApplication in this method, shut it down
            if app_created:
                app.quit()

            if not accepted or chosen_offset is None:
                self.logger.info('Manual navigation canceled by user')
                self._offset = None
                self._uncertainty = None
                self._confidence = None
                return

            # chosen_offset is (dv, du)
            self._offset = chosen_offset
            self._uncertainty = None
            self._confidence = 1.0

            # Record metadata
            self._metadata['offset'] = self._offset
            self._metadata['uncertainty'] = None
            self._metadata['confidence'] = 1.0
