"""BodyMosaicWindow: PyQt6 window for browsing body reprojections and mosaics."""

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.ma as ma
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontMetrics, QResizeEvent
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from nav.ui.common import build_stretch_controls
from nav.ui.mosaic_viewer.common import BodyDisplayData, load_body_file
from nav.ui.mosaic_viewer.photometric_display import compute_body_display_image
from nav.ui.mosaic_viewer.tiled_image_widget import TiledImageWidget, slider_to_zoom, zoom_to_slider

logger = logging.getLogger(__name__)

STATUS_HINT = (
    'Mouse wheel zooms both axes (Shift+wheel: X only, Ctrl+wheel: Y only). '
    'Shift+Left to zoom to region, Left drag to pan.'
)


def _percentile_stretch(
    image_ma: ma.MaskedArray, lo_pct: float = 0.0, hi_pct: float = 98.0
) -> tuple[float, float]:
    valid = image_ma.compressed()
    if valid.size == 0:
        return 0.0, 1.0
    black = float(np.percentile(valid, lo_pct))
    white = float(np.percentile(valid, hi_pct))
    if white <= black:
        white = black + 1e-6
    return black, white


def _colorby_column(
    data: np.ndarray | ma.MaskedArray,
    n_cols: int,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if isinstance(data, ma.MaskedArray):
        mask = ma.getmaskarray(data)
        arr = np.where(mask, np.nan, arr)
    col_means = np.nanmean(arr, axis=0)
    if col_means.size < n_cols:
        col_means = np.resize(col_means, n_cols)
    lo = float(np.nanmin(col_means)) if vmin is None else vmin
    hi = float(np.nanmax(col_means)) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1e-6
    t = np.clip((col_means - lo) / (hi - lo), 0.0, 1.0)
    r = np.clip(1.5 * t - 0.5, 0.0, 1.0).astype(np.float32)
    g = np.clip(np.where(t < 0.5, 2.0 * t, 2.0 - 2.0 * t), 0.0, 1.0).astype(np.float32)
    b = np.clip(0.5 - 1.5 * (t - 0.5 / 1.5), 0.0, 1.0).astype(np.float32)
    rgb = np.stack([r, g, b], axis=1)
    rgb[np.isnan(col_means)] = 0.5
    return rgb


class _SyncedSlider:
    """Keeps a QLineEdit and QSlider in sync for a single numeric parameter."""

    def __init__(
        self,
        line_edit: QLineEdit,
        slider: QSlider,
        lo: float,
        hi: float,
        fmt: str = '%.4f',
        on_change: Any = None,
    ) -> None:
        self._le = line_edit
        self._sl = slider
        self._lo = lo
        self._hi = hi
        self._fmt = fmt
        self._on_change = on_change
        self._updating = False
        self._sl.valueChanged.connect(self._slider_moved)
        self._le.editingFinished.connect(self._edit_done)

    def _to_slider(self, val: float) -> int:
        if self._hi <= self._lo:
            return 0
        pos = (val - self._lo) / (self._hi - self._lo) * 1000.0
        return round(float(np.clip(pos, 0, 1000)))

    def _from_slider(self, pos: int) -> float:
        return self._lo + (self._hi - self._lo) * pos / 1000.0

    def _slider_moved(self, pos: int) -> None:
        if self._updating:
            return
        val = self._from_slider(pos)
        self._updating = True
        self._le.setText(self._fmt % val)
        self._updating = False
        if self._on_change:
            self._on_change(val)

    def _edit_done(self) -> None:
        if self._updating:
            return
        try:
            val = float(self._le.text())
        except ValueError:
            return
        val = max(self._lo, min(self._hi, val))
        self._updating = True
        self._sl.setValue(self._to_slider(val))
        self._le.setText(self._fmt % val)
        self._updating = False
        if self._on_change:
            self._on_change(val)

    def set_range(self, lo: float, hi: float) -> None:
        self._lo = lo
        self._hi = hi

    def set_value(self, val: float) -> None:
        self._updating = True
        self._sl.setValue(self._to_slider(val))
        self._le.setText(self._fmt % val)
        self._updating = False

    def get_value(self) -> float:
        try:
            return float(self._le.text())
        except ValueError:
            return self._from_slider(self._sl.value())


class BodyMosaicWindow(QMainWindow):
    """Viewer window for a list of body reprojection / mosaic files."""

    def __init__(
        self,
        file_paths: list[str],
        *,
        initial_black: float | None = None,
        initial_white: float | None = None,
        initial_gamma: float = 0.5,
        show_parallels: bool = False,
        show_meridians: bool = False,
        show_lat_ticks: bool = False,
        show_lon_ticks: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not file_paths:
            raise ValueError('file_paths must contain at least one path')
        self._file_paths = file_paths
        self._current_idx = 0
        self._display_data: BodyDisplayData | None = None
        self._initial_black = initial_black
        self._initial_white = initial_white
        self._initial_gamma = initial_gamma
        self._default_gamma = initial_gamma
        self._show_parallels = show_parallels
        self._show_meridians = show_meridians
        self._show_lat_ticks = show_lat_ticks
        self._show_lon_ticks = show_lon_ticks
        self._stretch_controls: dict[str, Any] = {}
        self._pending_fit = False
        self._body_view_ma: ma.MaskedArray | None = None
        self._setup_ui()
        self._chk_lat_ticks.setChecked(show_lat_ticks)
        self._chk_lon_ticks.setChecked(show_lon_ticks)
        self._load_file(0)

    def statusBar(self) -> QStatusBar:
        bar = super().statusBar()
        assert bar is not None
        return bar

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        if getattr(self, '_pending_fit', False):
            self._fit_zoom_to_window()

    # ------------------------------------------------------------------
    #  UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowTitle('Body Mosaic Viewer')
        self.resize(1400, 900)

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)

        header = QWidget()
        hh = QHBoxLayout(header)
        hh.setContentsMargins(4, 2, 4, 0)
        hh.setSpacing(10)
        self._chk_lat_ticks = QCheckBox('Latitude axis ticks')
        self._chk_lon_ticks = QCheckBox('Longitude axis ticks')
        self._chk_lat_ticks.toggled.connect(self._update_axis_ticks)
        self._chk_lon_ticks.toggled.connect(self._update_axis_ticks)
        hh.addWidget(self._chk_lat_ticks)
        hh.addWidget(self._chk_lon_ticks)
        hh.addStretch()
        left_layout.addWidget(header)

        self._image_widget = TiledImageWidget()
        self._image_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image_widget.mouse_moved.connect(self._on_mouse_moved)
        self._image_widget.zoom_changed.connect(self._on_zoom_changed)
        left_layout.addWidget(self._image_widget, stretch=1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([1150, 250])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        main_layout.addWidget(splitter, stretch=1)
        main_layout.addWidget(self._build_control_panel())

        self.setCentralWidget(central)

        self._cursor_status_lbl = QLabel('')
        self._cursor_status_lbl.setStyleSheet('font-family: monospace;')
        self.statusBar().addPermanentWidget(self._cursor_status_lbl)
        self.statusBar().showMessage(STATUS_HINT)

    def _build_right_panel(self) -> QWidget:
        right = QWidget()
        right.setMinimumWidth(200)
        right.setMaximumWidth(300)
        v = QVBoxLayout(right)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        nav_row = QHBoxLayout()
        self._btn_prev = QPushButton('< Prev')
        self._btn_next = QPushButton('Next >')
        self._btn_prev.clicked.connect(self._on_prev)
        self._btn_next.clicked.connect(self._on_next)
        nav_row.addWidget(self._btn_prev)
        nav_row.addWidget(self._btn_next)
        v.addLayout(nav_row)

        self._file_lbl = QLabel('')
        self._file_lbl.setWordWrap(True)
        v.addWidget(self._file_lbl)
        v.addWidget(QLabel('Files'))
        self._file_list = QListWidget()
        self._file_list.setMinimumHeight(140)
        self._file_list.currentRowChanged.connect(self._on_file_list_row_changed)
        v.addWidget(self._file_list, stretch=1)
        self._populate_file_list()

        overlay_box = QGroupBox('Overlays')
        overlay_v = QVBoxLayout(overlay_box)
        self._chk_parallels = QCheckBox('Show parallels (lat)')
        self._chk_parallels.setChecked(self._show_parallels)
        self._chk_parallels.toggled.connect(self._update_overlays)
        self._chk_meridians = QCheckBox('Show meridians (lon)')
        self._chk_meridians.setChecked(self._show_meridians)
        self._chk_meridians.toggled.connect(self._update_overlays)
        overlay_v.addWidget(self._chk_parallels)
        overlay_v.addWidget(self._chk_meridians)
        v.addWidget(overlay_box)
        return right

    def _populate_file_list(self) -> None:
        self._file_list.clear()
        for i, p in enumerate(self._file_paths):
            name = Path(p).name
            item = QListWidgetItem(f'{i + 1}. {name}')
            item.setToolTip(str(p))
            self._file_list.addItem(item)

    def _refresh_file_list_selection(self) -> None:
        self._file_list.blockSignals(True)
        self._file_list.setCurrentRow(self._current_idx)
        cur = self._file_list.currentItem()
        if cur is not None:
            self._file_list.scrollToItem(cur)
        self._file_list.blockSignals(False)

    def _on_file_list_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._file_paths):
            return
        if row == self._current_idx:
            return
        self._load_file(row)

    def _build_control_panel(self) -> QWidget:
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(4, 2, 4, 2)
        ctrl_layout.setSpacing(2)

        upper = QWidget()
        upper_h = QHBoxLayout(upper)
        upper_h.setContentsMargins(0, 0, 0, 0)
        upper_h.setSpacing(6)

        stretch_box = QGroupBox('Stretch')
        stretch_form = QFormLayout()
        stretch_form.setHorizontalSpacing(4)
        self._stretch_controls = build_stretch_controls(
            stretch_form,
            img_min=0.0,
            img_max=1.0,
            black_init=0.0,
            white_init=1.0,
            gamma_init=self._initial_gamma,
            on_black_changed=lambda _v: self._apply_stretch(),
            on_white_changed=lambda _v: self._apply_stretch(),
            on_gamma_changed=lambda _v: self._apply_stretch(),
            slider_horizontal_stretch=1,
        )
        stretch_btn_col = QVBoxLayout()
        stretch_btn_col.setSpacing(4)
        btn_stretch_reset = QPushButton('Reset')
        btn_stretch_full = QPushButton('Full')
        btn_stretch_bright = QPushButton('Bright')
        btn_stretch_reset.clicked.connect(self._on_stretch_preset_reset)
        btn_stretch_full.clicked.connect(self._on_stretch_preset_full)
        btn_stretch_bright.clicked.connect(self._on_stretch_preset_bright)
        for b in (btn_stretch_reset, btn_stretch_full, btn_stretch_bright):
            b.setMaximumWidth(72)
            stretch_btn_col.addWidget(b)
        stretch_btn_col.addStretch()
        stretch_outer = QHBoxLayout()
        stretch_outer.setContentsMargins(4, 4, 4, 4)
        stretch_outer.setSpacing(8)
        stretch_outer.addLayout(stretch_form, stretch=1)
        stretch_outer.addLayout(stretch_btn_col)
        stretch_box.setLayout(stretch_outer)
        upper_h.addWidget(stretch_box, stretch=2)

        zoom_box = QGroupBox('Zoom')
        zoom_layout = QVBoxLayout(zoom_box)
        zoom_layout.setSpacing(2)

        def _make_zoom_row(label: str) -> tuple[QLineEdit, QSlider, QWidget]:
            le = QLineEdit()
            le.setMaximumWidth(55)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(1, 1000)
            row = QWidget()
            rh = QHBoxLayout(row)
            rh.setContentsMargins(0, 0, 0, 0)
            rh.addWidget(QLabel(label))
            rh.addWidget(le)
            rh.addWidget(sl, stretch=1)
            return le, sl, row

        self._xzoom_le, self._xzoom_sl, xz_row = _make_zoom_row('X:')
        self._yzoom_le, self._yzoom_sl, yz_row = _make_zoom_row('Y:')
        zoom_layout.addWidget(xz_row)
        zoom_layout.addWidget(yz_row)
        self._xzoom_sync = self._make_zoom_sync(self._xzoom_le, self._xzoom_sl, axis='x')
        self._yzoom_sync = self._make_zoom_sync(self._yzoom_le, self._yzoom_sl, axis='y')

        btn_row = QHBoxLayout()
        self._zoom_info_lbl = QLabel('1.00x / 1.00x')
        btn_zi = QPushButton('+')
        btn_zo = QPushButton('\N{MINUS SIGN}')  # U+2212 for symmetric glyph next to '+'
        btn_zr = QPushButton('Reset')
        btn_sf = QPushButton('Save FOV')
        btn_zi.setMaximumWidth(28)
        btn_zo.setMaximumWidth(28)
        btn_zi.clicked.connect(self._on_zoom_in)
        btn_zo.clicked.connect(self._on_zoom_out)
        btn_zr.clicked.connect(self._on_zoom_reset)
        btn_sf.clicked.connect(self._on_save_fov)
        btn_row.addWidget(self._zoom_info_lbl)
        btn_row.addWidget(btn_zi)
        btn_row.addWidget(btn_zo)
        btn_row.addWidget(btn_zr)
        btn_row.addStretch()
        btn_row.addWidget(btn_sf)
        zoom_layout.addLayout(btn_row)
        upper_h.addWidget(zoom_box, stretch=1)
        ctrl_layout.addWidget(upper)

        lower = QWidget()
        lower_h = QHBoxLayout(lower)
        lower_h.setContentsMargins(0, 0, 0, 0)
        lower_h.setSpacing(6)

        info_box = QGroupBox('Cursor Info')
        info_grid_widget = QWidget()
        info_grid = QGridLayout(info_grid_widget)
        info_grid.setHorizontalSpacing(10)
        info_grid.setVerticalSpacing(0)
        info_grid.setContentsMargins(2, 0, 2, 0)
        info_box_layout = QVBoxLayout(info_box)
        info_box_layout.setContentsMargins(4, 1, 4, 1)
        info_box_layout.setSpacing(0)
        info_box_layout.addWidget(info_grid_widget)

        info_columns: list[list[tuple[str, str]]] = [
            [
                ('body', 'Body:'),
                ('lat', 'Latitude:'),
                ('lon', 'Longitude:'),
                ('latlon', 'Lat/lon mode:'),
            ],
            [
                ('phase', 'Phase angle:'),
                ('emission', 'Emission angle:'),
                ('incidence', 'Incidence angle:'),
                ('res', 'Resolution (km/px):'),
            ],
            [
                ('eff_res', 'Eff. resolution (km/px):'),
            ],
            [
                ('image', 'Source image:'),
            ],
        ]
        self._info: dict[str, QLabel] = {}
        name_w = 120
        val_w_default = 132
        val_w_image = 300
        for col_idx, col in enumerate(info_columns):
            base = col_idx * 2
            for row_idx, (key, name) in enumerate(col):
                nl = QLabel(name)
                nl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                nl.setFixedWidth(name_w)
                vl = QLabel('---')
                vl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                vw = val_w_image if col_idx == 3 else val_w_default
                vl.setFixedWidth(vw)
                info_grid.addWidget(nl, row_idx, base)
                info_grid.addWidget(vl, row_idx, base + 1)
                self._info[key] = vl
        lower_h.addWidget(info_box, stretch=1)

        colorby_box = QGroupBox('Color By')
        cb_grid = QGridLayout(colorby_box)
        cb_grid.setContentsMargins(4, 2, 4, 2)
        cb_grid.setHorizontalSpacing(12)
        cb_grid.setVerticalSpacing(2)
        self._colorby_group = QButtonGroup()
        colorby_rows: list[list[tuple[str, str]]] = [
            [('none', 'None'), ('image_no', 'Image number')],
            [('res', 'Resolution (rel)'), ('eff_res', 'Eff. resolution (rel)')],
            [('abs_phase', 'Phase (abs)'), ('rel_phase', 'Phase (rel)')],
            [('abs_emission', 'Emission (abs)'), ('rel_emission', 'Emission (rel)')],
            [('abs_incidence', 'Incidence (abs)'), ('rel_incidence', 'Incidence (rel)')],
        ]
        for row_idx, row in enumerate(colorby_rows):
            for col_idx, (key, label) in enumerate(row):
                btn = QRadioButton(label)
                btn.setProperty('colorby_key', key)
                self._colorby_group.addButton(btn)
                cb_grid.addWidget(btn, row_idx, col_idx)
                if key == 'none':
                    btn.setChecked(True)
        self._colorby_group.buttonClicked.connect(self._on_colorby_changed)
        lower_h.addWidget(colorby_box)

        photometry_box = QGroupBox('Photometric')
        ph_grid = QGridLayout(photometry_box)
        ph_grid.setContentsMargins(4, 2, 4, 2)
        ph_grid.setHorizontalSpacing(8)
        ph_grid.setVerticalSpacing(2)
        self._photometry_group = QButtonGroup()
        photometry_rows: list[list[tuple[str, str]]] = [
            [('as_saved', 'As saved'), ('intrinsic', 'Uncorrected')],
            [('lambert', 'Lambert'), ('lommel_seeliger', 'Lommel\N{EN DASH}Seeliger')],
            [('minnaert', 'Minnaert')],
        ]
        for row_idx, row in enumerate(photometry_rows):
            for col_idx, (key, label) in enumerate(row):
                btn = QRadioButton(label)
                btn.setProperty('photometry_key', key)
                self._photometry_group.addButton(btn)
                ph_grid.addWidget(btn, row_idx, col_idx)
                if key == 'as_saved':
                    btn.setChecked(True)
        self._photometry_group.buttonClicked.connect(self._on_photometry_changed)
        lower_h.addWidget(photometry_box)

        ctrl_layout.addWidget(lower)
        return ctrl

    def _make_zoom_sync(self, le: QLineEdit, sl: QSlider, axis: str) -> _SyncedSlider:
        def _on_change(zoom_val: float) -> None:
            iw = self._image_widget
            xz, yz = iw.get_zoom()
            if axis == 'x':
                iw.set_zoom(zoom_val, yz)
            else:
                iw.set_zoom(xz, zoom_val)

        class _ZoomSync(_SyncedSlider):
            def _to_slider(self, val: float) -> int:
                return zoom_to_slider(val)

            def _from_slider(self, pos: int) -> float:
                return slider_to_zoom(pos)

        sync = _ZoomSync(le, sl, 0.05, 100.0, '%.2f', on_change=_on_change)
        sync.set_value(1.0)
        return sync

    def _photometry_mode(self) -> str:
        btn = self._photometry_group.checkedButton()
        if btn is None:
            return 'as_saved'
        return str(btn.property('photometry_key'))

    def _sync_photometry_ui(self, dd: BodyDisplayData) -> None:
        """Reset photometric display to file pixels when loading a new file."""
        _ = dd
        for b in self._photometry_group.buttons():
            if b.property('photometry_key') == 'as_saved':
                b.setChecked(True)
                break

    def _apply_body_display_image(self, *, preserve_view: bool) -> None:
        dd = self._display_data
        if dd is None:
            return
        mode = self._photometry_mode()
        img = compute_body_display_image(
            mode=mode,
            image_ma=dd.image_ma,
            photometric_model_name=dd.photometric_model_name,
            phase_deg=dd.phase,
            emission_deg=dd.emission,
            incidence_deg=dd.incidence,
        )
        self._body_view_ma = img
        self._image_widget.set_image(
            img,
            x_interval=dd.lon_resolution_deg,
            y_interval=dd.lat_resolution_deg,
            x_label=f'Longitude ({dd.latlon_type}/{dd.lon_direction}) (°)',
            y_label='Latitude (°)',
            y_flip=False,
            body_full_sphere_canvas=True,
            body_lon_range_deg=dd.lon_range_deg,
            body_lat_range_deg=dd.lat_range_deg,
            preserve_view=preserve_view,
        )

    def _on_photometry_changed(self, _btn: Any = None) -> None:
        if self._display_data is None:
            return
        self._apply_body_display_image(preserve_view=True)
        dd = self._display_data
        img = self._body_view_ma if self._body_view_ma is not None else dd.image_ma
        black, white = _percentile_stretch(img, 0.0, 98.0)
        g = max(0.1, self._stretch_controls['slider_gamma'].value() / 100.0)
        self._stretch_controls['set_range'](dd.vmin, dd.vmax)
        self._stretch_controls['set_values'](black, white, g)
        self._image_widget.set_stretch(black, white, g)
        self._on_colorby_changed(self._colorby_group.checkedButton())

    # ------------------------------------------------------------------
    #  File loading
    # ------------------------------------------------------------------

    def _on_prev(self) -> None:
        self._load_file(self._current_idx - 1)

    def _on_next(self) -> None:
        self._load_file(self._current_idx + 1)

    def _load_file(self, idx: int) -> None:
        if not (0 <= idx < len(self._file_paths)):
            return
        path = self._file_paths[idx]
        try:
            dd = load_body_file(path)
        except Exception as exc:
            logger.exception('Error loading %s', path)
            self.statusBar().showMessage(f'Error loading {path}: {exc}', 5000)
            return

        self._current_idx = idx
        self._display_data = dd
        self.setWindowTitle(f'Body Mosaic Viewer - {dd.title}  ({dd.body_name})')
        self._file_lbl.setText(f'{idx + 1} / {len(self._file_paths)}\n{dd.title}')
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < len(self._file_paths) - 1)

        self._sync_photometry_ui(dd)
        self._apply_body_display_image(preserve_view=False)
        self._update_axis_ticks()

        black, white = _percentile_stretch(
            self._body_view_ma if self._body_view_ma is not None else dd.image_ma, 0.0, 98.0
        )
        if self._initial_black is not None:
            black = self._initial_black
        if self._initial_white is not None:
            white = self._initial_white
        self._stretch_controls['set_range'](dd.vmin, dd.vmax)
        self._stretch_controls['set_values'](black, white, self._initial_gamma)
        self._image_widget.set_stretch(black, white, self._initial_gamma)

        self._clear_info()
        self._update_colorby_widgets_for_mosaic(dd.is_mosaic)
        self._on_colorby_changed(self._colorby_group.checkedButton())
        self._info['body'].setText(dd.body_name)
        self._info['latlon'].setText(f'{dd.latlon_type} / {dd.lon_direction}')

        self._update_overlays(self._chk_parallels.isChecked())
        self._fit_zoom_to_window()
        self._sync_zoom_ui()
        self.statusBar().showMessage(STATUS_HINT)
        self._refresh_file_list_selection()

    # ------------------------------------------------------------------
    #  Stretch / zoom
    # ------------------------------------------------------------------

    def _apply_stretch(self) -> None:
        b = self._stretch_controls['from_slider'](self._stretch_controls['slider_black'].value())
        w = self._stretch_controls['from_slider'](self._stretch_controls['slider_white'].value())
        g = max(0.1, self._stretch_controls['slider_gamma'].value() / 100.0)
        self._image_widget.set_stretch(b, w, g)

    def _on_stretch_preset_reset(self) -> None:
        dd = self._display_data
        if dd is None:
            return
        black, white = _percentile_stretch(
            self._body_view_ma if self._body_view_ma is not None else dd.image_ma, 0.0, 98.0
        )
        self._stretch_controls['set_values'](black, white, self._default_gamma)
        self._apply_stretch()

    def _on_stretch_preset_full(self) -> None:
        dd = self._display_data
        if dd is None:
            return
        self._stretch_controls['set_values'](dd.vmin, dd.vmax, self._default_gamma)
        self._image_widget.set_stretch(dd.vmin, dd.vmax, self._default_gamma)

    def _on_stretch_preset_bright(self) -> None:
        dd = self._display_data
        if dd is None:
            return
        black, white = _percentile_stretch(
            self._body_view_ma if self._body_view_ma is not None else dd.image_ma, 2.0, 98.0
        )
        self._stretch_controls['set_values'](black, white, self._default_gamma)
        self._apply_stretch()

    def _fit_zoom_to_window(self) -> None:
        dd = self._display_data
        if dd is None:
            return
        if self._image_widget.is_body_full_sphere_canvas():
            n_rows, n_cols = self._image_widget.display_grid_shape()
        else:
            n_rows, n_cols = dd.image_ma.shape
        vw = self._image_widget.viewport().width()
        vh = self._image_widget.viewport().height()
        if vw <= 0 or vh <= 0:
            self._pending_fit = True
            return
        self._pending_fit = False
        x_zoom = min(float(vw) / max(n_cols, 1), 100.0)
        y_zoom = min(float(vh) / max(n_rows, 1), 100.0)
        self._image_widget.set_zoom(x_zoom, y_zoom)

    def _on_zoom_in(self) -> None:
        xz, yz = self._image_widget.get_zoom()
        self._image_widget.set_zoom(xz * 1.5, yz * 1.5)
        self._sync_zoom_ui()

    def _on_zoom_out(self) -> None:
        xz, yz = self._image_widget.get_zoom()
        self._image_widget.set_zoom(xz / 1.5, yz / 1.5)
        self._sync_zoom_ui()

    def _on_zoom_reset(self) -> None:
        self._fit_zoom_to_window()
        self._sync_zoom_ui()

    def _on_zoom_changed(self, xz: float, yz: float) -> None:
        self._update_zoom_slider_ranges()
        self._xzoom_sync.set_value(xz)
        self._yzoom_sync.set_value(yz)
        self._zoom_info_lbl.setText(f'{xz:.2f}x / {yz:.2f}x')

    def _update_zoom_slider_ranges(self) -> None:
        min_x, min_y = self._image_widget.get_min_zoom()
        self._xzoom_sync.set_range(min_x, 100.0)
        self._yzoom_sync.set_range(min_y, 100.0)

    def _sync_zoom_ui(self) -> None:
        self._update_zoom_slider_ranges()
        xz, yz = self._image_widget.get_zoom()
        self._xzoom_sync.set_value(xz)
        self._yzoom_sync.set_value(yz)
        self._zoom_info_lbl.setText(f'{xz:.2f}x / {yz:.2f}x')

    def _on_save_fov(self) -> None:
        dd = self._display_data
        if dd is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Field of View',
            f'{dd.title}_fov.png',
            'PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)',
        )
        if not path:
            return
        qimg = self._image_widget.render_viewport_to_image()
        if not qimg.save(path):
            QMessageBox.warning(self, 'Save FOV', f'Failed to save {path}')

    # ------------------------------------------------------------------
    #  Color-by
    # ------------------------------------------------------------------

    def _on_colorby_changed(self, btn: Any) -> None:
        if btn is None or self._display_data is None:
            self._image_widget.set_color_column(None)
            return
        key = btn.property('colorby_key')
        col = self._compute_color_column(str(key), self._display_data)
        self._image_widget.set_color_column(col)

    def _compute_color_column(self, key: str, dd: BodyDisplayData) -> np.ndarray | None:
        if key == 'none':
            return None
        n_cols = dd.image_ma.shape[1]
        if key == 'image_no':
            if dd.image_number is None:
                return None
            vals = dd.image_number.astype(float)
            valid = ma.array(vals, mask=ma.getmaskarray(dd.image_number)).compressed()
            if valid.size == 0:
                return None
            return _colorby_column(
                dd.image_number,
                n_cols,
                vmin=float(valid.min()),
                vmax=float(valid.max()),
            )
        if key == 'res':
            return _colorby_column(dd.resolution, n_cols)
        if key == 'eff_res':
            return _colorby_column(dd.eff_resolution, n_cols)
        if key == 'abs_phase':
            return _colorby_column(dd.phase, n_cols, vmin=0.0, vmax=180.0)
        if key == 'rel_phase':
            return _colorby_column(dd.phase, n_cols)
        if key == 'abs_emission':
            return _colorby_column(dd.emission, n_cols, vmin=0.0, vmax=90.0)
        if key == 'rel_emission':
            return _colorby_column(dd.emission, n_cols)
        if key == 'abs_incidence':
            return _colorby_column(dd.incidence, n_cols, vmin=0.0, vmax=90.0)
        if key == 'rel_incidence':
            return _colorby_column(dd.incidence, n_cols)
        return None

    def _update_colorby_widgets_for_mosaic(self, is_mosaic: bool) -> None:
        for btn in self._colorby_group.buttons():
            key = btn.property('colorby_key')
            if key == 'image_no':
                btn.setEnabled(is_mosaic)
                if not is_mosaic and btn.isChecked():
                    for b2 in self._colorby_group.buttons():
                        if b2.property('colorby_key') == 'none':
                            b2.setChecked(True)
                            break

    # ------------------------------------------------------------------
    #  Overlays / axis ticks
    # ------------------------------------------------------------------

    def _update_overlays(self, _checked: bool = False) -> None:
        dd = self._display_data
        if dd is None:
            self._image_widget.set_body_sphere_geo_overlays(False, False)
            self._image_widget.set_show_rows([])
            self._image_widget.set_show_cols([])
            return

        if self._image_widget.is_body_full_sphere_canvas():
            self._image_widget.set_body_sphere_geo_overlays(
                self._chk_parallels.isChecked(),
                self._chk_meridians.isChecked(),
            )
            self._image_widget.set_show_rows([])
            self._image_widget.set_show_cols([])
            return

        if self._chk_parallels.isChecked():
            lat_lo, lat_hi = -90.0, 90.0
            step = _nice_overlay_step(lat_hi - lat_lo, max_lines=8)
            row_ys = []
            lat = math.ceil(lat_lo / step) * step
            while lat <= lat_hi + 1e-9:
                py = (90.0 - lat) / dd.lat_resolution_deg
                row_ys.append(round(py))
                lat += step
            self._image_widget.set_show_rows(row_ys)
        else:
            self._image_widget.set_show_rows([])

        if self._chk_meridians.isChecked():
            lon_lo, lon_hi = -180.0, 180.0
            step = _nice_overlay_step(lon_hi - lon_lo, max_lines=12)
            col_xs = []
            lon = math.ceil(lon_lo / step) * step
            while lon <= lon_hi + 1e-9:
                px = (lon - (-180.0)) / dd.lon_resolution_deg
                col_xs.append(round(px))
                lon += step
            self._image_widget.set_show_cols(col_xs)
        else:
            self._image_widget.set_show_cols([])

    def _update_axis_ticks(self, _checked: bool = False) -> None:
        dd = self._display_data
        if dd is None:
            return
        self._image_widget.set_axis_tick_options(
            show_x=self._chk_lon_ticks.isChecked(),
            show_y=self._chk_lat_ticks.isChecked(),
            y_tick_center=0.0,
            y_tick_labels_absolute=True,
        )

    # ------------------------------------------------------------------
    #  Cursor info
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_ma(arr: ma.MaskedArray, iy: int, ix: int, fmt: str) -> str:
        v = arr[iy, ix]
        if ma.is_masked(v):
            return '---'
        return fmt % float(v)

    @staticmethod
    def _fmt_deg_label(s: str) -> str:
        return f'{s}°' if s != '---' else '---'

    def _clear_info(self) -> None:
        self._cursor_status_lbl.setText('')
        for lbl in self._info.values():
            lbl.setText('---')

    def _on_mouse_moved(self, px: float, py: float, in_bounds: bool) -> None:
        if not in_bounds or self._display_data is None:
            self._clear_info()
            return
        dd = self._display_data
        self._info['body'].setText(dd.body_name)
        self._info['latlon'].setText(f'{dd.latlon_type} / {dd.lon_direction}')

        x_phys, y_phys = self._image_widget.pixel_to_physical(px, py)
        lon_deg = float(x_phys)
        lat_deg = float(y_phys)
        lon_min, lon_max = dd.lon_range_deg
        lat_min, _lat_max = dd.lat_range_deg
        d_lon = dd.lon_resolution_deg
        n_c = dd.image_ma.shape[1]
        n_r = dd.image_ma.shape[0]
        if self._image_widget.is_body_full_sphere_canvas():
            dc, dr, inside = self._image_widget.body_sphere_data_indices(lon_deg, lat_deg)
        else:
            lm = lon_deg % 360.0
            dc = -1
            for cand in (lm, lm - 360.0, lm + 360.0):
                if lon_min - 1e-9 <= cand <= lon_max + 1e-9:
                    dc = int(np.floor((cand - lon_min) / d_lon))
                    break
            dr = int(np.floor((lat_deg - lat_min) / dd.lat_resolution_deg))
            inside = 0 <= dc < n_c and 0 <= dr < n_r
        ix = int(np.clip(dc, 0, n_c - 1))
        iy = int(np.clip(dr, 0, n_r - 1))

        raw_val = dd.image_ma[iy, ix] if inside else ma.masked
        if ma.is_masked(raw_val):
            value_str = f'{"masked":>11}'
        else:
            value_str = f'{float(raw_val):11.8f}'

        if inside:
            ph = self._fmt_ma(dd.phase, iy, ix, '%.3f')
            em = self._fmt_ma(dd.emission, iy, ix, '%.3f')
            inc = self._fmt_ma(dd.incidence, iy, ix, '%.3f')
            res = self._fmt_ma(dd.resolution, iy, ix, '%.4f')
            eff = self._fmt_ma(dd.eff_resolution, iy, ix, '%.4f')
            if dd.image_number is not None:
                img_v = dd.image_number[iy, ix]
                if not ma.is_masked(img_v):
                    idx = int(img_v)
                    names = dd.contributing_image_names
                    if idx < len(names) and names[idx]:
                        img_s = f'{names[idx]} (#{idx})'
                    else:
                        img_s = f'#{idx}'
                else:
                    img_s = '---'
            else:
                n0 = dd.contributing_image_names[0] if dd.contributing_image_names else ''
                img_s = f'{n0} (#0)' if n0 else dd.title
        else:
            ph = em = inc = res = eff = '---'
            img_s = '---'

        x_str = f'{px:8.2f}'
        y_str = f'{py:7.2f}'
        self._cursor_status_lbl.setText(f'X: {x_str}  Y: {y_str}  Value: {value_str}')
        self._info['lat'].setText(f'{lat_deg:.4f}°')
        self._info['lon'].setText(f'{lon_deg:.4f}°')
        self._info['phase'].setText(self._fmt_deg_label(ph))
        self._info['emission'].setText(self._fmt_deg_label(em))
        self._info['incidence'].setText(self._fmt_deg_label(inc))
        self._info['res'].setText(f'{res} km/px' if res != '---' else '---')
        self._info['eff_res'].setText(f'{eff} km/px' if eff != '---' else '---')
        img_lbl = self._info['image']
        iw = img_lbl.width()
        if iw > 0 and img_s != '---':
            fm = QFontMetrics(img_lbl.font())
            img_s = fm.elidedText(img_s, Qt.TextElideMode.ElideRight, iw)
        img_lbl.setText(img_s)


def _nice_overlay_step(span: float, max_lines: int = 8) -> float:
    if span <= 0:
        return 10.0
    raw = span / max_lines
    preferred = [90, 60, 45, 30, 20, 15, 10, 5, 4, 3, 2, 1, 0.5]
    for s in preferred:
        if raw <= s:
            return s
    return raw
