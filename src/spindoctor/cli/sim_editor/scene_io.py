"""Scene file I/O glue: save, load, and apply a validated scene dict.

Wraps :func:`spindoctor.sim.scene.save_sim_scene` /
:func:`spindoctor.sim.scene.load_sim_scene` behind file dialogs and rebuilds
the whole editor state from a loaded ``sim_params`` mapping, syncing every
General-tab widget and rebuilding the object tabs.
"""

from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from spindoctor.cli.sim_editor.base import SimEditorBase
from spindoctor.sim.scene import load_sim_scene, save_sim_scene


class SceneIoMixin(SimEditorBase):
    """Save / load scenes and apply a loaded scene dict to the editor."""

    def _save_scene(self) -> None:
        """Prompt for a path and write the current scene as YAML."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save Scene',
            'scene.yaml',
            'YAML Files (*.yaml *.yml)',
        )
        if not filename:
            return
        try:
            save_sim_scene(self.sim_params, Path(filename))
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save scene:\n{e!s}')

    def _load_scene(self) -> None:
        """Prompt for a path and load a scene YAML into the editor."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            'Load Scene',
            '',
            'YAML Files (*.yaml *.yml)',
        )
        if not filename:
            return
        try:
            self._apply_params_dict(load_sim_scene(Path(filename)))
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load scene:\n{e!s}')

    def _apply_params_dict(self, params: dict[str, Any]) -> None:
        """Rebuild sim_params from a loaded params dict and sync every widget."""
        background_stars_val = params.get('background_stars_num', 0)
        self.sim_params = {
            'size_v': int(params.get('size_v', 512)),
            'size_u': int(params.get('size_u', 512)),
            'offset_v': float(params.get('offset_v', 0.0)),
            'offset_u': float(params.get('offset_u', 0.0)),
            'offset_rotation_deg': float(params.get('offset_rotation_deg', 0.0)),
            'exposure_sec': float(params.get('exposure_sec', 1.0)),
            'random_seed': int(params.get('random_seed', 42)),
            'instrument': str(params.get('instrument', 'generic')),
            'background_stars_num': int(background_stars_val),
            'background_stars_psf_sigma': float(params.get('background_stars_psf_sigma', 0.9)),
            'background_stars_distribution_exponent': float(
                params.get('background_stars_distribution_exponent', 2.5)
            ),
            'time': float(params.get('time', 0.0)),
            'ring_epoch': float(params.get('ring_epoch', 0.0)),
            'closest_planet': params.get('closest_planet') or 'SATURN',
            'shade_solid_rings': bool(params.get('shade_solid_rings', False)),
            'bodies': list(params.get('bodies', [])),
            'stars': list(params.get('stars', [])),
            'rings': list(params.get('rings', [])),
        }
        # Preserve catalog-only blocks the General tab does not yet edit
        # (noise model, optics including stray light, exposure, instrument-config
        # overrides) so loading a scene round-trips them instead of dropping them.
        for passthrough_key in (
            'noise',
            'optics',
            'spk_error',
            'oversample',
            'instrument_config',
            'midtime_utc',
            'fit_camera_rotation',
        ):
            if passthrough_key in params:
                self.sim_params[passthrough_key] = params[passthrough_key]
        # Sync the shade-solid-rings checkbox
        self._shade_solid_rings_check.blockSignals(True)
        self._shade_solid_rings_check.setChecked(bool(self.sim_params['shade_solid_rings']))
        self._shade_solid_rings_check.blockSignals(False)
        # Update general UI
        self._size_v_spin.setValue(self.sim_params['size_v'])
        self._size_u_spin.setValue(self.sim_params['size_u'])
        self._offset_v_spin.setValue(self.sim_params['offset_v'])
        self._offset_u_spin.setValue(self.sim_params['offset_u'])
        self._offset_rotation_spin.blockSignals(True)
        self._offset_rotation_spin.setValue(self.sim_params['offset_rotation_deg'])
        self._offset_rotation_spin.blockSignals(False)
        self._exposure_spin.blockSignals(True)
        self._exposure_spin.setValue(float(self.sim_params['exposure_sec']))
        self._exposure_spin.blockSignals(False)
        self._random_seed_spin.setValue(self.sim_params['random_seed'])
        # Update instrument selector
        self._instrument_combo.blockSignals(True)
        instrument_index = self._instrument_combo.findText(str(self.sim_params['instrument']))
        if instrument_index >= 0:
            self._instrument_combo.setCurrentIndex(instrument_index)
        self._instrument_combo.blockSignals(False)
        # Camera-rotation fit override and midtime
        fit_rotation = self.sim_params.get('fit_camera_rotation')
        fit_text = 'on' if fit_rotation is True else 'off' if fit_rotation is False else '(inherit)'
        self._fit_rotation_combo.blockSignals(True)
        fit_index = self._fit_rotation_combo.findText(fit_text)
        if fit_index >= 0:
            self._fit_rotation_combo.setCurrentIndex(fit_index)
        self._fit_rotation_combo.blockSignals(False)
        self._midtime_edit.blockSignals(True)
        self._midtime_edit.setText(str(self.sim_params.get('midtime_utc') or ''))
        self._midtime_edit.blockSignals(False)
        self._update_psf_preview()
        # Update time and epoch
        self._time_spin.setValue(self.sim_params.get('time', 0.0))
        self._epoch_spin.setValue(self.sim_params.get('ring_epoch', 0.0))
        # Update closest planet
        closest_planet = self.sim_params.get('closest_planet') or 'SATURN'
        index = self._closest_planet_combo.findText(closest_planet)
        if index >= 0:
            self._closest_planet_combo.setCurrentIndex(index)
        else:
            self._closest_planet_combo.setCurrentText(closest_planet)
        # Update detector-noise controls from the loaded noise block.
        self._poisson_check.blockSignals(True)
        self._poisson_check.setChecked(bool(self._noise_value('poisson', True)))
        self._poisson_check.blockSignals(False)
        self._read_noise_spin.blockSignals(True)
        self._read_noise_spin.setValue(float(self._noise_value('read_noise_dn', 4.0)))
        self._read_noise_spin.blockSignals(False)
        self._cosmic_ray_spin.blockSignals(True)
        self._cosmic_ray_spin.setValue(float(self._noise_value('cosmic_ray_rate_per_sec', 0.0)))
        self._cosmic_ray_spin.blockSignals(False)
        self._missing_data_spin.blockSignals(True)
        self._missing_data_spin.setValue(float(self._noise_value('missing_data_rate', 0.0)))
        self._missing_data_spin.blockSignals(False)
        self._bias_spin.blockSignals(True)
        self._bias_spin.setValue(float(self._noise_value('bias_dn', 20.0)))
        self._bias_spin.blockSignals(False)
        self._bloom_spin.blockSignals(True)
        self._bloom_spin.setValue(int(self._noise_value('bloom_length', 0)))
        self._bloom_spin.blockSignals(False)
        self._signal_frac_spin.blockSignals(True)
        self._signal_frac_spin.setValue(float(self._noise_value('signal_full_scale_frac', 0.5)))
        self._signal_frac_spin.blockSignals(False)
        self._pixel_area_spin.blockSignals(True)
        self._pixel_area_spin.setValue(float(self._noise_value('pixel_area_cm2', 1.0)))
        self._pixel_area_spin.blockSignals(False)
        # Update stray-light controls from the loaded stray_light block.
        self._stray_amplitude_spin.blockSignals(True)
        self._stray_amplitude_spin.setValue(float(self._stray_value('amplitude', 0.0)))
        self._stray_amplitude_spin.blockSignals(False)
        self._stray_direction_spin.blockSignals(True)
        self._stray_direction_spin.setValue(float(self._stray_value('direction_deg', 0.0)))
        self._stray_direction_spin.blockSignals(False)
        self._stray_model_combo.blockSignals(True)
        stray_model_index = self._stray_model_combo.findText(
            str(self._stray_value('model', 'linear'))
        )
        if stray_model_index >= 0:
            self._stray_model_combo.setCurrentIndex(stray_model_index)
        self._stray_model_combo.blockSignals(False)
        self._stray_center_v_spin.blockSignals(True)
        self._stray_center_v_spin.setValue(float(self._stray_value('center_v', 0.0)))
        self._stray_center_v_spin.blockSignals(False)
        self._stray_center_u_spin.blockSignals(True)
        self._stray_center_u_spin.setValue(float(self._stray_value('center_u', 0.0)))
        self._stray_center_u_spin.blockSignals(False)
        # Update background stars controls
        self._background_stars_slider.blockSignals(True)
        self._background_stars_slider.setValue(self.sim_params['background_stars_num'])
        self._background_stars_slider.blockSignals(False)
        self._background_stars_spin.blockSignals(True)
        self._background_stars_spin.setValue(self.sim_params['background_stars_num'])
        self._background_stars_spin.blockSignals(False)
        # Update background stars PSF sigma controls
        self._background_stars_psf_sigma_slider.blockSignals(True)
        psf_sigma_val = int(self.sim_params['background_stars_psf_sigma'] * 100)
        self._background_stars_psf_sigma_slider.setValue(psf_sigma_val)
        self._background_stars_psf_sigma_slider.blockSignals(False)
        self._background_stars_psf_sigma_spin.blockSignals(True)
        psf_sigma_spin_val = self.sim_params['background_stars_psf_sigma']
        self._background_stars_psf_sigma_spin.setValue(psf_sigma_spin_val)
        self._background_stars_psf_sigma_spin.blockSignals(False)
        # Update background stars distribution exponent controls
        self._background_stars_dist_exp_slider.blockSignals(True)
        dist_exp_slider_val = int(self.sim_params['background_stars_distribution_exponent'] * 100)
        self._background_stars_dist_exp_slider.setValue(dist_exp_slider_val)
        self._background_stars_dist_exp_slider.blockSignals(False)
        self._background_stars_dist_exp_spin.blockSignals(True)
        dist_exp_spin_val = self.sim_params['background_stars_distribution_exponent']
        self._background_stars_dist_exp_spin.setValue(dist_exp_spin_val)
        self._background_stars_dist_exp_spin.blockSignals(False)
        # Rebuild tabs
        self._rebuild_dynamic_tabs()
        self._update_tab_titles()
        self._validate_ranges()
        self._updater.immediate_update()
