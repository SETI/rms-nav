"""TiledImageWidget: efficient tiled rendering of large mosaic images.

Supports ring geometry (radius versus longitude) and body geometry (latitude
versus longitude) via ``y_flip`` and axis labels.

Image pixel coordinate convention (ring mode):
    pixel_x  0 .. n_cols-1   increasing right  = increasing longitude
    pixel_y  0 .. n_rows-1   increasing DOWN   = DECREASING radius / latitude

When ``y_flip=True`` (the default for ring mosaics) the underlying numpy array
has row 0 = inner/south, which is displayed flipped.  Set ``y_flip=False`` for
body mosaics where row 0 is already the top row.

Only the viewport's visible region is ever rendered (tiled rendering), so
arbitrary zoom levels are memory-efficient.
"""

import math
from typing import cast

import numpy as np
import numpy.ma as ma
from PyQt6.QtCore import QEvent, QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QResizeEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import QAbstractScrollArea, QRubberBand, QScrollBar, QSizePolicy, QWidget

from nav.ui.common import apply_linear_gamma_stretch

# Zoom slider maps to log scale: slider 1..1000  →  zoom 0.05x..100x
_ZOOM_LOG_LO = np.log10(0.05)
_ZOOM_LOG_HI = np.log10(100.0)


def zoom_to_slider(zoom: float) -> int:
    """Convert zoom value to slider integer 1..1000."""
    log = np.log10(max(zoom, 1e-6))
    pos = (log - _ZOOM_LOG_LO) / (_ZOOM_LOG_HI - _ZOOM_LOG_LO) * 999.0 + 1.0
    return round(float(np.clip(pos, 1, 1000)))


def slider_to_zoom(pos: int) -> float:
    """Convert slider integer 1..1000 to zoom value."""
    log = _ZOOM_LOG_LO + (pos - 1) / 999.0 * (_ZOOM_LOG_HI - _ZOOM_LOG_LO)
    return float(10.0**log)


def _nice_tick_values(lo: float, hi: float, max_ticks: int) -> np.ndarray:
    """Return ~max_ticks nice tick values spanning [lo, hi]."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        return np.array([])
    span = hi - lo
    if span <= 0:
        return np.array([lo])
    raw = span / max(max_ticks - 1, 1)
    exp = 10.0 ** np.floor(np.log10(max(abs(raw), 1e-30)))
    f = raw / exp
    if f < 1.5:
        step = exp
    elif f < 3.5:
        step = 2.0 * exp
    elif f < 7.5:
        step = 5.0 * exp
    else:
        step = 10.0 * exp
    start = np.ceil(lo / step) * step
    return np.arange(start, hi + step * 0.0001, step)


def _nice_sphere_overlay_degree_step(span: float, max_lines: int = 8) -> float:
    """Pick a round step in degrees for globe parallels/meridians."""
    if span <= 0:
        return 10.0
    raw = span / max_lines
    preferred = [90, 60, 45, 30, 20, 15, 10, 5, 4, 3, 2, 1, 0.5]
    for s in preferred:
        if raw <= s:
            return s
    return raw


def _nice_latitude_tick_degrees(lo: float, hi: float, max_ticks: int = 10) -> np.ndarray:
    """Latitude ticks preferring round degree steps; generic fallback."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        return np.array([])
    span = hi - lo
    if span <= 0:
        return np.array([lo])
    preferred = (90, 60, 45, 30, 20, 15, 10, 5, 4, 3, 2, 1, 0.5, 0.25)
    cap = max_ticks + 4
    best: np.ndarray | None = None
    for step in preferred:
        start = np.ceil(lo / step) * step
        ticks = np.arange(start, hi + step * 0.0001, step)
        if 1 <= len(ticks) <= cap:
            best = ticks
    if best is not None:
        return best
    return _nice_tick_values(lo, hi, max_ticks)


def _nice_longitude_tick_degrees(lo: float, hi: float, max_ticks: int = 14) -> np.ndarray:
    """Longitude ticks preferring multiples/submultiples of 30°; generic fallback."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        return np.array([])
    span = hi - lo
    if span <= 0:
        return np.array([lo])
    preferred = (
        360,
        180,
        120,
        90,
        60,
        45,
        30,
        20,
        15,
        12,
        10,
        6,
        5,
        4,
        3,
        2,
        1,
        0.5,
        0.25,
    )
    cap = max_ticks + 6
    best: np.ndarray | None = None
    for step in preferred:
        start = np.ceil(lo / step) * step
        ticks = np.arange(start, hi + step * 0.0001, step)
        if 1 <= len(ticks) <= cap:
            best = ticks
    for n in range(1, 31):
        step = 30.0 / n
        if step < 0.15:
            break
        start = np.ceil(lo / step) * step
        ticks = np.arange(start, hi + step * 0.0001, step)
        if 1 <= len(ticks) <= cap:
            best = ticks
    if best is not None:
        return best
    return _nice_tick_values(lo, hi, max_ticks)


def _body_sphere_lon_bin_to_dc_map(
    lon_min_deg: float,
    lon_max_deg: float,
    lon_res_deg: float,
    n_cols: int,
) -> tuple[int, float, np.ndarray]:
    """Map full-grid longitude bin index → column index (``-1`` if absent).

    Matches :meth:`nav.reproj.bodies.BodyMosaic.to_bounded` logic, including
    the case ``lon_min > lon_max`` (longitude wraps through 0°/360°).
    """
    lon_res_rad = lon_res_deg * (math.pi / 180.0)
    if lon_res_rad <= 0:
        raise ValueError('longitude resolution (deg) must be positive')
    n_full_lon = max(1, int(2.0 * math.pi / lon_res_rad))
    lon_min_rad = lon_min_deg * (math.pi / 180.0)
    lon_max_rad = lon_max_deg * (math.pi / 180.0)
    lon_min_bin = round(lon_min_rad / lon_res_rad)
    lon_max_bin = round(lon_max_rad / lon_res_rad)
    if lon_min_deg <= lon_max_deg + 1e-9:
        if lon_max_bin >= lon_min_bin:
            lon_bins = np.arange(lon_min_bin, lon_max_bin + 1, dtype=np.int64)
        else:
            lon_bins = lon_min_bin + np.arange(n_cols, dtype=np.int64)
    else:
        lon_bins = np.concatenate(
            (
                np.arange(lon_min_bin, n_full_lon, dtype=np.int64),
                np.arange(0, lon_max_bin + 1, dtype=np.int64),
            )
        )
    if len(lon_bins) != n_cols:
        lon_bins = lon_min_bin + np.arange(n_cols, dtype=np.int64)
    bin_to_dc = np.full(n_full_lon, -1, dtype=np.int32)
    for dc, kb in enumerate(lon_bins):
        kk = int(kb) % n_full_lon
        if bin_to_dc[kk] < 0:
            bin_to_dc[kk] = int(dc)
    return n_full_lon, lon_res_rad, bin_to_dc


def _nice_longitude_ticks_wrapped_0_360(lo: float, hi: float, max_ticks: int = 14) -> np.ndarray:
    """Nice longitude ticks for a viewport spanning ``[lo, hi]`` deg (linear along scroll).

    ``lo``/``hi`` are unwrapped longitudes (e.g. ``px * Δλ``). When the viewport
    covers ~360°, ``hi % 360`` can equal ``lo % 360`` even though ``hi > lo``;
    without a special case that collapses to a single tick and labels vanish
    after horizontal pan.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.array([])
    linear_span = float(hi - lo)
    if linear_span <= 1e-9:
        return np.array([float(lo % 360.0)])
    if linear_span >= 359.5:
        ticks = _nice_longitude_tick_degrees(0.0, 360.0 - 1e-9, max_ticks)
        return np.unique(np.mod(ticks, 360.0))
    lo_m = float(lo % 360.0)
    hi_m = float(hi % 360.0)
    if hi_m < lo_m - 1e-9:
        hi_m += 360.0
    ticks = _nice_longitude_tick_degrees(lo_m, hi_m, max_ticks)
    return np.unique(np.mod(ticks, 360.0))


class TiledImageWidget(QAbstractScrollArea):
    """Scroll area that renders a large image in tiles.

    Only the visible viewport region is rendered on each paint, making
    arbitrary zoom levels efficient without allocating oversized QPixmaps.
    Supports independent X and Y zoom factors.

    Designed to handle both ring mosaics (radius/longitude, ``y_flip=True``)
    and body mosaics (latitude/longitude, ``y_flip=False``).
    """

    # ------------------------------------------------------------------ #
    #  Signals                                                             #
    # ------------------------------------------------------------------ #

    # (pixel_x, pixel_y, in_bounds)
    mouse_moved = pyqtSignal(float, float, bool)
    # (x_zoom, y_zoom)
    zoom_changed = pyqtSignal(float, float)
    # right-click: (pixel_x, pixel_y)
    right_clicked = pyqtSignal(float, float)
    # Ctrl+left-click: (pixel_x, pixel_y)
    ctrl_clicked = pyqtSignal(float, float)

    def horizontalScrollBar(self) -> QScrollBar:
        bar = super().horizontalScrollBar()
        assert bar is not None
        return bar

    def verticalScrollBar(self) -> QScrollBar:
        bar = super().verticalScrollBar()
        assert bar is not None
        return bar

    def viewport(self) -> QWidget:
        vp = super().viewport()
        assert vp is not None
        return vp

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setViewportMargins(0, 0, 0, 0)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.viewport().setMouseTracking(True)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Image data
        self._image_ma: ma.MaskedArray | None = None
        self._n_rows: int = 0
        self._n_cols: int = 0

        # Physical axis: x = column (longitude or longitude), y = row (radius or latitude)
        self._x_interval: float = 1.0  # degrees or km per pixel (column)
        self._y_interval: float = 1.0  # degrees or km per pixel (row)
        self._x_label: str = 'Longitude (°)'
        self._y_label: str = 'Radius (km)'
        # When True (ring mosaics): array row 0 = bottom of display; flip for display.
        self._y_flip: bool = True
        # Cap for the X axis in display units (e.g. 360 for a full ring).
        self._x_axis_max: float | None = None
        # Longitude (deg) at column 0 when columns are not anchored at 0 (ring sparse).
        self._x_origin_deg: float = 0.0
        # Body: virtual canvas covering 360 deg longitude and 180 deg latitude.
        self._body_sphere: bool = False
        self._body_data_n_rows: int = 0
        self._body_data_n_cols: int = 0
        self._body_lon_min: float = 0.0
        self._body_lon_max: float = 0.0
        self._body_lat_min: float = 0.0
        self._body_lat_max: float = 0.0
        self._body_n_full_lon: int = 0
        self._body_lon_res_rad: float = 0.0
        self._body_lon_bin_to_dc: np.ndarray | None = None

        # Stretch
        self._black: float = 0.0
        self._white: float = 1.0
        self._gamma: float = 0.5

        # Zoom
        self._x_zoom: float = 1.0
        self._y_zoom: float = 1.0

        # Color-by: (n_cols, 3) float32 in [0,1], or None for greyscale
        self._color_column: np.ndarray | None = None

        # Show-rows overlay (ring: show_radii; body full-sphere uses viewport geo lines)
        # pixel_y rows drawn as green horizontal lines
        self._show_row_pixel_ys: list[int] = []
        # Show-cols overlay (ring / bounded body: meridians in image space)
        # pixel_x cols drawn as green vertical lines
        self._show_col_pixel_xs: list[int] = []
        # Body full-sphere: draw parallels/meridians in viewport (always visible)
        self._body_geo_parallels: bool = False
        self._body_geo_meridians: bool = False

        # Axis tick options
        self._show_x_ticks: bool = False
        self._show_y_ticks: bool = False
        self._y_tick_center: float = 0.0  # baseline Y (km or deg) for offset tick labels
        self._y_tick_labels_absolute: bool = False  # ring: label Y ticks in absolute km
        self._ring_pixel_y_absolute: bool = False  # ring: pixel_to_physical Y is absolute km
        self._ring_radial_mid_km: float = 0.0  # mean radius (km) for absolute ring Y

        # Pan state
        self._drag_start_global: QPoint | None = None
        self._drag_start_scroll: tuple[int, int] = (0, 0)

        # Zoom-to-area (Shift+drag)
        self._rubber_band: QRubberBand | None = None
        self._rubber_origin: QPoint | None = None

        # Keep numpy array alive while QImage uses its buffer
        self._last_rgb: np.ndarray | None = None

        self.horizontalScrollBar().valueChanged.connect(lambda _: self.viewport().update())
        self.verticalScrollBar().valueChanged.connect(lambda _: self.viewport().update())

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def set_image(
        self,
        image_ma: ma.MaskedArray,
        x_interval: float,
        y_interval: float,
        *,
        x_label: str = 'Longitude (°)',
        y_label: str = 'Radius (km)',
        y_flip: bool = True,
        x_axis_max: float | None = None,
        x_origin_deg: float = 0.0,
        ring_radial_axis_absolute: bool = False,
        ring_radial_mid_km: float = 0.0,
        body_full_sphere_canvas: bool = False,
        body_lon_range_deg: tuple[float, float] | None = None,
        body_lat_range_deg: tuple[float, float] | None = None,
        preserve_view: bool = False,
    ) -> None:
        """Load new image data and reset scroll to origin.

        Parameters:
            image_ma: 2-D masked array (rows by columns).
            x_interval: Physical units per column pixel (e.g. degrees or km).
            y_interval: Physical units per row pixel.
            x_label: Display label for the X axis.
            y_label: Display label for the Y axis.
            y_flip: If True, row 0 is the bottom of the display (ring mosaics);
                if False, row 0 is the top (body mosaics).
            x_axis_max: Cap for the X axis in display units.  None = origin + n_cols * x_interval.
            x_origin_deg: Longitude (deg) at image column 0; physical X is
                ``x_origin_deg + column * x_interval`` (ring sparse / bounded grids).
            ring_radial_axis_absolute: For ring mosaics (``y_flip=True``), if True,
                :meth:`pixel_to_physical` returns absolute radius (km) on Y; if False,
                Y is the offset from the mean radial center (km).
            ring_radial_mid_km: Mean radius (km), i.e. ``(radius_inner + radius_outer) / 2``,
                used when ``ring_radial_axis_absolute`` is True.
            body_full_sphere_canvas: If True (with ``y_flip=False``), use a virtual image
                large enough to show longitude ``[0, 360]`` deg and latitude
                ``[+90, -90]`` deg at the same resolution as the data, so you can zoom
                out past the data extent into empty sky.
            body_lon_range_deg: Geographic longitude bounds of ``image_ma`` columns (deg).
            body_lat_range_deg: Geographic latitude bounds of ``image_ma`` rows (deg).
            preserve_view: If True, keep zoom and scroll position (same array shape
                and geometry as the previous :meth:`set_image` call).
        """
        if image_ma.ndim != 2:
            raise ValueError(
                f'image_ma must be 2-D, got ndim={image_ma.ndim}, shape={image_ma.shape}'
            )
        self._image_ma = image_ma
        self._n_rows, self._n_cols = image_ma.shape
        self._x_interval = x_interval
        self._y_interval = y_interval
        self._x_label = x_label
        self._y_label = y_label
        self._y_flip = y_flip
        self._x_axis_max = x_axis_max
        self._x_origin_deg = float(x_origin_deg)
        self._ring_pixel_y_absolute = bool(ring_radial_axis_absolute) and y_flip
        self._ring_radial_mid_km = float(ring_radial_mid_km) if self._ring_pixel_y_absolute else 0.0
        self._body_sphere = False
        self._body_n_full_lon = 0
        self._body_lon_res_rad = 0.0
        self._body_lon_bin_to_dc = None
        if body_full_sphere_canvas:
            if y_flip:
                raise ValueError('body_full_sphere_canvas is only supported with y_flip=False')
            if body_lon_range_deg is None or body_lat_range_deg is None:
                raise ValueError(
                    'body_lon_range_deg and body_lat_range_deg are required when '
                    'body_full_sphere_canvas is True'
                )
            dr, dc = image_ma.shape
            d_lon = float(x_interval)
            d_lat = float(y_interval)
            lon_min, lon_max = float(body_lon_range_deg[0]), float(body_lon_range_deg[1])
            lat_min, lat_max = float(body_lat_range_deg[0]), float(body_lat_range_deg[1])
            n_virt_w = max(dc, int(np.ceil(360.0 / d_lon)))
            n_virt_h = max(dr, int(np.ceil(180.0 / d_lat)))
            self._body_sphere = True
            self._body_data_n_rows = dr
            self._body_data_n_cols = dc
            self._body_lon_min = lon_min
            self._body_lon_max = lon_max
            self._body_lat_min = lat_min
            self._body_lat_max = lat_max
            self._body_n_full_lon, self._body_lon_res_rad, self._body_lon_bin_to_dc = (
                _body_sphere_lon_bin_to_dc_map(lon_min, lon_max, d_lon, dc)
            )
            self._n_rows, self._n_cols = n_virt_h, n_virt_w
            # Virtual longitude 0..360° (matches typical body mosaics; sampling wraps).
            self._x_origin_deg = 0.0
            self._x_axis_max = 360.0
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        if preserve_view:
            prev_xz, prev_yz = self._x_zoom, self._y_zoom
            prev_hv, prev_vv = hbar.value(), vbar.value()
        else:
            self._x_zoom = 1.0
            self._y_zoom = 1.0
            hbar.setValue(0)
            vbar.setValue(0)
        self._update_scroll_range()
        if preserve_view:
            self._x_zoom, self._y_zoom = prev_xz, prev_yz
            self._update_scroll_range()
            hbar.setValue(int(np.clip(prev_hv, 0, max(0, hbar.maximum()))))
            vbar.setValue(int(np.clip(prev_vv, 0, max(0, vbar.maximum()))))
        self.viewport().update()

    def is_body_full_sphere_canvas(self) -> bool:
        """True when ``body_full_sphere_canvas`` was used in :meth:`set_image`."""
        return self._body_sphere

    def body_sphere_data_indices(self, lon_deg: float, lat_deg: float) -> tuple[int, int, bool]:
        """Map display lon/lat (deg) to ``(data_col, data_row, inside)`` for full-sphere body.

        Returns ``(data_col, data_row, inside)``.
        """
        if not self._body_sphere or self._body_lon_bin_to_dc is None:
            return -1, -1, False
        lon_rad = math.fmod(float(lon_deg), 360.0) * (math.pi / 180.0)
        if lon_rad < 0:
            lon_rad += 2 * math.pi
        twopi = 2 * math.pi
        if lon_rad >= twopi - 1e-15:
            lon_rad = math.fmod(lon_rad, twopi)
        lr = self._body_lon_res_rad
        k = math.floor(min(lon_rad / lr, twopi / lr - 1e-15))
        k = max(0, min(k, self._body_n_full_lon - 1))
        dc = int(self._body_lon_bin_to_dc[k])
        dr = math.floor((float(lat_deg) - self._body_lat_min) / self._y_interval)
        inside = dc >= 0 and 0 <= dr < self._body_data_n_rows and 0 <= dc < self._body_data_n_cols
        return dc, dr, inside

    def _x_axis_max_val(self) -> float:
        if self._x_axis_max is not None:
            return float(self._x_axis_max)
        if self._n_cols > 0:
            return float(self._x_origin_deg + self._n_cols * self._x_interval)
        return 360.0

    def set_stretch(self, black: float, white: float, gamma: float) -> None:
        """Update contrast stretch parameters and repaint."""
        self._black = black
        self._white = max(white, black + 1e-10)
        self._gamma = max(gamma, 0.01)
        self.viewport().update()

    def set_zoom(
        self,
        x_zoom: float,
        y_zoom: float,
        anchor_vx: int | None = None,
        anchor_vy: int | None = None,
        anchor_img_x: float | None = None,
        anchor_img_y: float | None = None,
    ) -> None:
        """Set zoom, optionally anchoring a viewport point to an image coord."""
        self._apply_zoom(x_zoom, y_zoom, anchor_vx, anchor_vy, anchor_img_x, anchor_img_y)

    def get_zoom(self) -> tuple[float, float]:
        """Return current ``(x_zoom, y_zoom)``."""
        return self._x_zoom, self._y_zoom

    def get_min_zoom(self) -> tuple[float, float]:
        """Minimum X/Y zoom so the virtual image fills the viewport."""
        return self._min_zoom_xy()

    def x_fov_span_px(self) -> int:
        """Pixel width for the column span (used by EW plot to stay in sync)."""
        vp = max(1, self.viewport().width())
        vb = self.verticalScrollBar()
        if vb.isVisible():
            return max(1, vp + vb.width())
        return vp

    def _min_zoom_xy(self) -> tuple[float, float]:
        if self._image_ma is None or self._n_cols < 1 or self._n_rows < 1:
            return (0.05, 0.05)
        vw = max(1, self.viewport().width())
        vh = max(1, self.viewport().height())
        return (float(vw) / float(self._n_cols), float(vh) / float(self._n_rows))

    def set_color_column(self, color_column: np.ndarray | None) -> None:
        """Set per-column RGB tinting (n_cols, 3) float32 in [0,1], or None."""
        if color_column is None:
            self._color_column = None
            self.viewport().update()
            return
        if not isinstance(color_column, np.ndarray):
            raise ValueError(
                f'color_column must be None or numpy.ndarray, got {type(color_column)}'
            )
        if color_column.ndim != 2 or color_column.shape[1] != 3:
            raise ValueError(
                f'color_column must have shape (n_cols, 3), got shape {color_column.shape}'
            )
        n_cc = self._body_data_n_cols if self._body_sphere else self._n_cols
        if n_cc > 0 and color_column.shape[0] != n_cc:
            raise ValueError(
                f'color_column length {color_column.shape[0]} does not match n_cols={n_cc}'
            )
        cc = np.asarray(color_column, dtype=np.float64)
        if np.any(cc < 0.0) or np.any(cc > 1.0):
            raise ValueError('color_column values must lie in [0, 1]')
        self._color_column = cc.astype(np.float32)
        self.viewport().update()

    def set_show_rows(self, pixel_ys: list[int]) -> None:
        """Draw green horizontal lines at the given display pixel_y rows."""
        self._show_row_pixel_ys = pixel_ys
        self.viewport().update()

    def set_show_cols(self, pixel_xs: list[int]) -> None:
        """Draw green vertical lines at the given display pixel_x columns."""
        self._show_col_pixel_xs = pixel_xs
        self.viewport().update()

    def set_body_sphere_geo_overlays(self, show_parallels: bool, show_meridians: bool) -> None:
        """For ``body_full_sphere_canvas``, draw parallels/meridians in viewport space."""
        self._body_geo_parallels = bool(show_parallels)
        self._body_geo_meridians = bool(show_meridians)
        self.viewport().update()

    def display_grid_shape(self) -> tuple[int, int]:
        """Return ``(n_rows, n_cols)`` of the display grid (virtual size for body sphere)."""
        return (int(self._n_rows), int(self._n_cols))

    def set_axis_tick_options(
        self,
        show_x: bool,
        show_y: bool,
        y_tick_center: float = 0.0,
        *,
        y_tick_labels_absolute: bool = False,
    ) -> None:
        """Toggle axis tick overlays.

        Parameters:
            show_x: Show X-axis (longitude) tick marks at the bottom.
            show_y: Show Y-axis (radius / latitude) tick marks at the left.
            y_tick_center: Baseline for Y-axis offset tick labels (e.g. mean core radius).
            y_tick_labels_absolute: For rings, show Y tick values as absolute radius (km)
                instead of offset from ``y_tick_center``.
        """
        self._show_x_ticks = show_x
        self._show_y_ticks = show_y
        self._y_tick_center = float(y_tick_center)
        self._y_tick_labels_absolute = bool(y_tick_labels_absolute)
        self.viewport().update()

    def viewport_to_pixel(self, vx: int, vy: int) -> tuple[float, float]:
        """Convert viewport screen coords to image pixel coords."""
        hv = self.horizontalScrollBar().value()
        vv = self.verticalScrollBar().value()
        return (hv + vx) / self._x_zoom, (vv + vy) / self._y_zoom

    def pixel_to_physical(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """Return ``(x_physical, y_physical)`` from image pixel coords.

        For ring mosaics (y_flip=True): x = longitude (deg).  When
        ``ring_radial_axis_absolute`` was set in :meth:`set_image`, y is absolute
        radius (km); otherwise y is the offset from the mean radial center (km).
        For body mosaics (y_flip=False): x = longitude (deg), y = latitude from top (deg).
        """
        if self._body_sphere:
            x_phys = float(np.mod(self._x_origin_deg + pixel_x * self._x_interval, 360.0))
            y_phys = float(90.0 - pixel_y * self._y_interval)
            y_phys = float(np.clip(y_phys, -90.0, 90.0))
            return x_phys, y_phys
        x_phys = float(self._x_origin_deg + pixel_x * self._x_interval)
        x_phys = float(np.clip(x_phys, self._x_origin_deg, self._x_axis_max_val()))
        if self._y_flip:
            rel = ((self._n_rows - 1) / 2.0 - pixel_y) * self._y_interval
            if self._ring_pixel_y_absolute:
                y_phys = rel + self._ring_radial_mid_km
            else:
                y_phys = rel
        else:
            y_phys = pixel_y * self._y_interval
        return x_phys, y_phys

    def pixel_y_to_arr_row(self, pixel_y: float) -> int:
        """Convert display ``pixel_y`` (0 = outer when ``y_flip``) to array row (0 = inner).

        ``pixel_y`` is clamped to ``[0, n_rows - 1]`` before conversion. When
        ``y_flip`` is False, ``pixel_y`` maps 1:1 to the array row index.
        """
        if self._n_rows < 1:
            return 0
        cy = float(np.clip(pixel_y, 0.0, float(self._n_rows - 1)))
        if self._y_flip:
            return (self._n_rows - 1) - int(cy)
        return int(cy)

    def scroll_to_pixel(self, pixel_x: float, pixel_y: float) -> None:
        """Scroll so that the given image pixel is centred in the viewport."""
        vw = self.viewport().width()
        vh = self.viewport().height()
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        h = int(pixel_x * self._x_zoom - vw / 2)
        v = int(pixel_y * self._y_zoom - vh / 2)
        hbar.setValue(max(0, min(hbar.maximum(), h)))
        vbar.setValue(max(0, min(vbar.maximum(), v)))

    def render_viewport_to_image(self) -> QImage:
        """Render the currently-visible viewport to a QImage (for Save FOV)."""
        vw = self.viewport().width()
        vh = self.viewport().height()
        img = QImage(vw, vh, QImage.Format.Format_RGB888)
        painter = QPainter(img)
        self._do_paint(painter, vw, vh)
        painter.end()
        return img

    # ------------------------------------------------------------------ #
    #  Qt overrides                                                        #
    # ------------------------------------------------------------------ #

    def viewportEvent(self, event: QEvent | None) -> bool:
        if event is None:
            return super().viewportEvent(event)
        t = event.type()
        if t == QEvent.Type.Paint:
            painter = QPainter(self.viewport())
            self._do_paint(painter, self.viewport().width(), self.viewport().height())
            painter.end()
            return True
        if t == QEvent.Type.MouseButtonPress:
            self._mouse_press(cast(QMouseEvent, event))
            return True
        if t == QEvent.Type.MouseMove:
            self._mouse_move(cast(QMouseEvent, event))
            return True
        if t == QEvent.Type.MouseButtonRelease:
            self._mouse_release(cast(QMouseEvent, event))
            return True
        if t == QEvent.Type.MouseButtonDblClick:
            return True
        return super().viewportEvent(event)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        """Zoom at cursor: both axes, or X-only with Shift, or Y-only with Ctrl."""
        if event is None:
            super().wheelEvent(event)
            return
        if self._image_ma is None:
            event.accept()
            return
        pos = event.position().toPoint()
        vp_pos = self.viewport().mapFromParent(pos)
        vx, vy = vp_pos.x(), vp_pos.y()
        factor = 1.2 if event.angleDelta().y() > 0 else (1.0 / 1.2)
        img_x, img_y = self.viewport_to_pixel(vx, vy)
        min_x, min_y = self._min_zoom_xy()
        mods = event.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        if shift and not ctrl:
            new_xz = float(np.clip(self._x_zoom * factor, min_x, 100.0))
            new_yz = self._y_zoom
        elif ctrl and not shift:
            new_xz = self._x_zoom
            new_yz = float(np.clip(self._y_zoom * factor, min_y, 100.0))
        else:
            new_xz = float(np.clip(self._x_zoom * factor, min_x, 100.0))
            new_yz = float(np.clip(self._y_zoom * factor, min_y, 100.0))
        self._apply_zoom(new_xz, new_yz, vx, vy, img_x, img_y)
        event.accept()

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._update_scroll_range()
        if self._image_ma is not None:
            min_x, min_y = self._min_zoom_xy()
            if self._x_zoom < min_x - 1e-12 or self._y_zoom < min_y - 1e-12:
                self._apply_zoom(
                    max(self._x_zoom, min_x), max(self._y_zoom, min_y), None, None, None, None
                )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _update_scroll_range(self) -> None:
        if self._image_ma is None:
            return
        vw = self.viewport().width()
        vh = self.viewport().height()
        virtual_w = max(1, int(self._n_cols * self._x_zoom))
        virtual_h = max(1, int(self._n_rows * self._y_zoom))
        hbar = self.horizontalScrollBar()
        hbar.setRange(0, max(0, virtual_w - vw))
        hbar.setPageStep(vw)
        vbar = self.verticalScrollBar()
        vbar.setRange(0, max(0, virtual_h - vh))
        vbar.setPageStep(vh)

    def _apply_zoom(
        self,
        new_xz: float,
        new_yz: float,
        anchor_vx: int | None,
        anchor_vy: int | None,
        anchor_img_x: float | None,
        anchor_img_y: float | None,
    ) -> None:
        min_x, min_y = self._min_zoom_xy()
        new_xz = float(np.clip(new_xz, min_x, 100.0))
        new_yz = float(np.clip(new_yz, min_y, 100.0))
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        vw = self.viewport().width()
        vh = self.viewport().height()

        if anchor_vx is not None and anchor_img_x is not None:
            new_hv = int(anchor_img_x * new_xz - anchor_vx)
        else:
            cx = (hbar.value() + vw / 2) / self._x_zoom
            new_hv = int(cx * new_xz - vw / 2)

        if anchor_vy is not None and anchor_img_y is not None:
            new_vv = int(anchor_img_y * new_yz - anchor_vy)
        else:
            cy = (vbar.value() + vh / 2) / self._y_zoom
            new_vv = int(cy * new_yz - vh / 2)

        self._x_zoom = new_xz
        self._y_zoom = new_yz
        self._update_scroll_range()
        hbar.setValue(max(0, min(hbar.maximum(), new_hv)))
        vbar.setValue(max(0, min(vbar.maximum(), new_vv)))
        self.zoom_changed.emit(self._x_zoom, self._y_zoom)
        self.viewport().update()

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def _do_paint_body_sphere(
        self,
        painter: QPainter,
        vw: int,
        vh: int,
        hv: int,
        vv: int,
        xz: float,
        yz: float,
    ) -> None:
        """Paint body mode with a virtual [0,360] x [+90,-90] deg canvas."""
        px_start = max(0, int(np.floor(hv / xz)))
        px_end = min(self._n_cols - 1, int(np.ceil((hv + vw) / xz)))
        py_start = max(0, int(np.floor(vv / yz)))
        py_end = min(self._n_rows - 1, int(np.ceil((vv + vh) / yz)))
        if px_start > px_end or py_start > py_end:
            return

        gx = np.arange(px_start, px_end + 1, dtype=np.float64)
        gy = np.arange(py_start, py_end + 1, dtype=np.float64)
        gy_grid, gx_grid = np.meshgrid(gy, gx, indexing='ij')
        d_lon = self._x_interval
        d_lat = self._y_interval
        lon_m = (gx_grid + 0.5) * d_lon
        lat_m = 90.0 - (gy_grid + 0.5) * d_lat
        lon_rad = np.mod(lon_m, 360.0) * (math.pi / 180.0)
        lr = self._body_lon_res_rad
        twopi = 2 * math.pi
        k = np.floor(np.minimum(lon_rad / lr, twopi / lr - 1e-15)).astype(np.int64)
        k = np.clip(k, 0, self._body_n_full_lon - 1)
        bmap = self._body_lon_bin_to_dc
        if bmap is None:
            return
        dc = bmap[k]
        dr = np.floor((lat_m - self._body_lat_min) / d_lat).astype(np.int64)
        valid_lon = dc >= 0
        valid = (
            valid_lon & (dc < self._body_data_n_cols) & (dr >= 0) & (dr < self._body_data_n_rows)
        )

        image_data = self._image_ma
        if image_data is None:
            return
        tile_h, tile_w = lon_m.shape
        tile_data = np.zeros((tile_h, tile_w), dtype=np.float32)
        tile_mask = np.ones((tile_h, tile_w), dtype=bool)
        if np.any(valid):
            dr_v = dr[valid]
            dc_v = dc[valid]
            sub = image_data[dr_v, dc_v]
            tile_data[valid] = np.asarray(np.nan_to_num(sub.filled(0.0), nan=0.0), dtype=np.float32)
            tile_mask[valid] = ma.getmaskarray(sub)

        stretched = apply_linear_gamma_stretch(
            tile_data, black=self._black, white=self._white, gamma=self._gamma
        ).astype(np.float32)
        gray = (stretched * 255.0).astype(np.uint8)

        if self._color_column is not None and len(self._color_column) > 0:
            tint = np.ones((tile_h, tile_w, 3), dtype=np.float32)
            if np.any(valid):
                dc_safe = np.clip(dc, 0, len(self._color_column) - 1)
                tint[valid] = self._color_column[dc_safe[valid]].astype(np.float32)
            gray_f = gray[:, :, np.newaxis].astype(np.float32)
            rgb = np.clip(gray_f * tint, 0, 255).astype(np.uint8)
        else:
            rgb = np.stack([gray, gray, gray], axis=2)

        if np.any(tile_mask):
            rgb[tile_mask, 0] = 180
            rgb[tile_mask, 1] = 0
            rgb[tile_mask, 2] = 0

        dest_x = round(px_start * xz) - hv
        dest_y = round(py_start * yz) - vv
        dest_w = max(1, round((px_end + 1) * xz) - hv - dest_x)
        dest_h = max(1, round((py_end + 1) * yz) - vv - dest_y)

        self._last_rgb = np.ascontiguousarray(rgb)
        qimg = QImage(
            self._last_rgb.tobytes(),
            tile_w,
            tile_h,
            3 * tile_w,
            QImage.Format.Format_RGB888,
        )
        scaled = qimg.scaled(
            dest_w,
            dest_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        painter.drawImage(dest_x, dest_y, scaled)

    def _do_paint(self, painter: QPainter, vw: int, vh: int) -> None:
        painter.fillRect(0, 0, vw, vh, Qt.GlobalColor.black)
        if self._image_ma is None:
            return

        hv = self.horizontalScrollBar().value()
        vv = self.verticalScrollBar().value()
        xz = self._x_zoom
        yz = self._y_zoom

        if self._body_sphere:
            self._do_paint_body_sphere(painter, vw, vh, hv, vv, xz, yz)
            if self._show_x_ticks or self._show_y_ticks:
                self._draw_axis_tick_overlays(painter, vw, vh, hv, vv, xz, yz)
            if self._body_geo_parallels or self._body_geo_meridians:
                self._draw_body_sphere_geo_overlays(painter, vw, vh, hv, vv, xz, yz)
            return

        if self._n_cols == 0 or self._n_rows == 0:
            return

        # Visible range in image pixel coords
        px_start = max(0, int(np.floor(hv / xz)))
        px_end = min(self._n_cols - 1, int(np.ceil((hv + vw) / xz)))
        py_start = max(0, int(np.floor(vv / yz)))
        py_end = min(self._n_rows - 1, int(np.ceil((vv + vh) / yz)))

        if px_start > px_end or py_start > py_end:
            return

        if self._y_flip:
            # pixel_y = (n_rows-1) - arr_row
            # py_start (top of screen) → arr_row_max; py_end (bottom) → arr_row_min
            arr_row_max = min(self._n_rows - 1, (self._n_rows - 1) - py_start)
            arr_row_min = max(0, (self._n_rows - 1) - py_end)
            tile_raw = self._image_ma[arr_row_min : arr_row_max + 1, px_start : px_end + 1]
            # Flip vertically so display row 0 = py_start (outer)
            tile = tile_raw[::-1, :]
        else:
            tile = self._image_ma[py_start : py_end + 1, px_start : px_end + 1]

        tile_h, tile_w = tile.shape
        tile_mask = ma.getmaskarray(tile)
        tile_data = np.nan_to_num(tile.filled(0.0), nan=0.0).astype(np.float32)

        # Contrast stretch using shared helper
        stretched = apply_linear_gamma_stretch(
            tile_data, black=self._black, white=self._white, gamma=self._gamma
        ).astype(np.float32)
        gray = (stretched * 255.0).astype(np.uint8)

        # Build RGB (apply colour-by tinting if active)
        if self._color_column is not None and len(self._color_column) > 0:
            col_idx = np.clip(
                np.arange(px_start, px_end + 1, dtype=np.intp),
                0,
                len(self._color_column) - 1,
            )
            tint = self._color_column[col_idx].astype(np.float32)  # (w, 3)
            gray_f = gray[:, :, np.newaxis].astype(np.float32)  # (h, w, 1)
            rgb = np.clip(gray_f * tint[np.newaxis, :, :], 0, 255).astype(np.uint8)
        else:
            rgb = np.stack([gray, gray, gray], axis=2)

        # Mask overlay: masked pixels rendered as dark red
        if np.any(tile_mask):
            rgb[tile_mask, 0] = 180
            rgb[tile_mask, 1] = 0
            rgb[tile_mask, 2] = 0

        # Horizontal-line overlays (show-radii / show-parallels)
        for py in self._show_row_pixel_ys:
            if py_start <= py <= py_end:
                tile_row = py - py_start
                if 0 <= tile_row < tile_h:
                    rgb[tile_row, :, 0] = 0
                    rgb[tile_row, :, 1] = 220
                    rgb[tile_row, :, 2] = 0

        # Vertical-line overlays (show-meridians)
        for px in self._show_col_pixel_xs:
            if px_start <= px <= px_end:
                tile_col = px - px_start
                if 0 <= tile_col < tile_w:
                    rgb[:, tile_col, 0] = 0
                    rgb[:, tile_col, 1] = 220
                    rgb[:, tile_col, 2] = 0

        # Screen destination rectangle
        dest_x = round(px_start * xz) - hv
        dest_y = round(py_start * yz) - vv
        dest_w = max(1, round((px_end + 1) * xz) - hv - dest_x)
        dest_h = max(1, round((py_end + 1) * yz) - vv - dest_y)

        self._last_rgb = np.ascontiguousarray(rgb)
        qimg = QImage(
            self._last_rgb.tobytes(),
            tile_w,
            tile_h,
            3 * tile_w,
            QImage.Format.Format_RGB888,
        )
        scaled = qimg.scaled(
            dest_w,
            dest_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        painter.drawImage(dest_x, dest_y, scaled)

        if self._show_x_ticks or self._show_y_ticks:
            self._draw_axis_tick_overlays(painter, vw, vh, hv, vv, xz, yz)

    def _draw_body_sphere_geo_overlays(
        self,
        painter: QPainter,
        vw: int,
        vh: int,
        hv: int,
        vv: int,
        xz: float,
        yz: float,
    ) -> None:
        """Draw parallels and meridians in viewport coordinates (always visible)."""
        pen = QPen(QColor(0, 220, 0))
        pen.setWidth(1)
        pen.setCosmetic(True)
        painter.setPen(pen)

        if self._body_geo_parallels:
            step = _nice_sphere_overlay_degree_step(180.0, max_lines=8)
            lat = math.ceil(-90.0 / step) * step
            while lat <= 90.0 + 1e-9:
                py = (90.0 - lat) / self._y_interval
                sy = float(py * yz - vv)
                if -1.0 <= sy <= float(vh) + 1.0:
                    y = round(sy)
                    painter.drawLine(0, y, vw, y)
                lat += step

        if self._body_geo_meridians:
            step = _nice_sphere_overlay_degree_step(360.0, max_lines=12)
            lon = 0.0
            while lon < 360.0 - 1e-6:
                px = (lon - self._x_origin_deg) / self._x_interval
                sx = float(px * xz - hv)
                if -1.0 <= sx <= float(vw) + 1.0:
                    x = round(sx)
                    painter.drawLine(x, 0, x, vh)
                lon += step

    def _draw_axis_tick_overlays(
        self,
        painter: QPainter,
        vw: int,
        vh: int,
        hv: int,
        vv: int,
        xz: float,
        yz: float,
    ) -> None:
        """Draw X (bottom) and/or Y (left) axis tick overlays."""
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        pen = QPen(QColor(220, 220, 220))
        painter.setPen(pen)
        bg_col = QColor(25, 25, 28, 210)
        pad = 3
        EDGE = 8
        _BOTTOM = 30
        _LEFT = 52
        _TICK_X = _LEFT - 4
        vp_w = vw

        def _x_label_left(ix: int, tw: int) -> int:
            ideal = ix - tw // 2
            if ideal < EDGE:
                return int(min(ix + 4, max(EDGE, vp_w - EDGE - tw)))
            if ideal + tw > vp_w - EDGE:
                return int(max(ix - tw - 4, EDGE))
            return int(ideal)

        if self._show_x_ticks:
            font_x = QFont()
            font_x.setPointSize(10)
            painter.setFont(font_x)
            fm_x = painter.fontMetrics()
            vw_x = self.x_fov_span_px()
            px0 = hv / xz
            px1 = (hv + vw_x) / xz
            tick_y0 = vh - _BOTTOM
            tick_y1 = vh - EDGE
            text_baseline = vh - EDGE
            th = fm_x.ascent() + fm_x.descent()
            tty = text_baseline - fm_x.ascent()

            if self._body_sphere:
                raw0 = self._x_origin_deg + px0 * self._x_interval
                raw1 = self._x_origin_deg + px1 * self._x_interval
                tick_iter = _nice_longitude_ticks_wrapped_0_360(raw0, raw1, 14)
            else:
                hi = self._x_axis_max_val()
                lo = float(self._x_origin_deg)
                c0 = float(np.clip(self._x_origin_deg + px0 * self._x_interval, lo, hi))
                c1 = float(np.clip(self._x_origin_deg + px1 * self._x_interval, lo, hi))
                tick_iter = _nice_longitude_tick_degrees(c0, c1, 14)

            for val in tick_iter:
                img_x = (val - self._x_origin_deg) / self._x_interval
                sx = float(img_x * xz - hv)
                if not (-20 < sx < vp_w + 20):
                    continue
                ix = round(sx)
                if not (0 <= ix < vp_w):
                    continue
                txt = f'{val:.0f}°'
                tw = fm_x.horizontalAdvance(txt)
                if tw >= vp_w - 2 * EDGE:
                    continue
                tx = _x_label_left(ix, tw)
                tx = int(np.clip(tx, EDGE, vp_w - EDGE - tw))
                tick_bar = QRect(ix, tick_y0, 1, max(1, tick_y1 - tick_y0))
                text_r = QRect(tx, tty, tw, th)
                bg_x = tick_bar.united(text_r).adjusted(-pad, -pad, pad, pad)
                painter.fillRect(bg_x, bg_col)
                painter.drawLine(ix, tick_y0, ix, tick_y1)
                painter.drawText(tx, text_baseline, txt)

        if self._show_y_ticks:
            font_y = QFont()
            font_y.setPointSize(10)
            painter.setFont(font_y)
            label_h = 22
            half_h = label_h // 2
            text_left = EDGE
            text_w = max(1, _TICK_X - 4 - text_left)

            py0 = vv / yz
            py1 = (vv + vh) / yz
            if self._y_flip:
                y0_phys = self._pixel_y_to_y_physical(py1)
                y1_phys = self._pixel_y_to_y_physical(py0)
            else:
                y0_phys = self._pixel_y_to_y_physical(py0)
                y1_phys = self._pixel_y_to_y_physical(py1)
            lo, hi = min(y0_phys, y1_phys), max(y0_phys, y1_phys)
            center = self._y_tick_center
            if self._y_tick_labels_absolute:
                if self._body_sphere:
                    tick_vals = _nice_latitude_tick_degrees(lo, hi, 10)
                else:
                    tick_vals = _nice_tick_values(lo, hi, 8)
            else:
                off_lo, off_hi = lo - center, hi - center
                tick_vals = _nice_tick_values(off_lo, off_hi, 8) + center
            for abs_val in tick_vals:
                sy = self._y_physical_to_screen_y(abs_val, yz, vv)
                if not (-20 < sy < vh + 20):
                    continue
                iy = round(sy)
                if not (0 <= iy < vh):
                    continue
                if self._y_tick_labels_absolute:
                    if self._body_sphere:
                        txt = f'{abs_val:.0f}°'
                    else:
                        txt = f'{abs_val:.0f}'
                else:
                    off = abs_val - center
                    txt = f'{off:.0f}'
                tr = QRect(text_left, iy - half_h, text_w, label_h)
                if tr.top() < EDGE:
                    tr.moveTop(EDGE)
                if tr.bottom() > vh - EDGE:
                    tr.moveBottom(vh - EDGE)
                line_r = QRect(_TICK_X, iy, max(1, _LEFT - _TICK_X), 1)
                bg_r = line_r.united(tr).adjusted(-pad, -pad, pad, pad)
                painter.fillRect(bg_r, bg_col)
                painter.drawLine(_TICK_X, iy, _LEFT, iy)
                painter.drawText(
                    tr, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), txt
                )

    def _pixel_y_to_y_physical(self, pixel_y: float) -> float:
        """Convert display pixel_y to Y physical value."""
        if self._body_sphere:
            return float(90.0 - pixel_y * self._y_interval)
        if self._y_flip:
            rel = ((self._n_rows - 1) / 2.0 - pixel_y) * self._y_interval
            return rel + self._y_tick_center
        return pixel_y * self._y_interval

    def _y_physical_to_screen_y(self, y_phys: float, yz: float, vv: int) -> float:
        """Convert a physical Y value to screen y coordinate."""
        if self._body_sphere:
            pixel_y = (90.0 - y_phys) / self._y_interval
            return pixel_y * yz - vv
        if self._y_flip:
            rel = y_phys - self._y_tick_center
            pixel_y = (self._n_rows - 1) / 2.0 - rel / self._y_interval
        else:
            pixel_y = y_phys / self._y_interval
        return pixel_y * yz - vv

    # ------------------------------------------------------------------ #
    #  Mouse events                                                        #
    # ------------------------------------------------------------------ #

    def _mouse_press(self, event: QMouseEvent) -> None:
        btn = event.button()
        mods = event.modifiers()
        vx = int(event.position().x())
        vy = int(event.position().y())
        px, py = self.viewport_to_pixel(vx, vy)
        in_bounds = self._image_ma is not None and 0 <= px < self._n_cols and 0 <= py < self._n_rows

        if btn == Qt.MouseButton.RightButton:
            if in_bounds:
                self.right_clicked.emit(px, py)
            return

        if btn == Qt.MouseButton.LeftButton:
            if mods & Qt.KeyboardModifier.ControlModifier:
                if in_bounds:
                    self.ctrl_clicked.emit(px, py)
                return

            if mods & Qt.KeyboardModifier.ShiftModifier:
                origin = QPoint(vx, vy)
                self._rubber_origin = origin
                if self._rubber_band is None:
                    self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport())
                self._rubber_band.setGeometry(QRect(origin, QSize()))
                self._rubber_band.show()
                return

            # Normal left-drag: pan
            self._drag_start_global = event.globalPosition().toPoint()
            self._drag_start_scroll = (
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value(),
            )
            self.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def _mouse_move(self, event: QMouseEvent) -> None:
        vx = int(event.position().x())
        vy = int(event.position().y())
        px, py = self.viewport_to_pixel(vx, vy)
        in_bounds = self._image_ma is not None and 0 <= px < self._n_cols and 0 <= py < self._n_rows
        self.mouse_moved.emit(px, py, in_bounds)

        mods = event.modifiers()
        if (
            self._rubber_origin is not None
            and self._rubber_band is not None
            and mods & Qt.KeyboardModifier.ShiftModifier
        ):
            self._rubber_band.setGeometry(QRect(self._rubber_origin, QPoint(vx, vy)).normalized())
            return

        if self._drag_start_global is not None:
            delta = event.globalPosition().toPoint() - self._drag_start_global
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(int(np.clip(self._drag_start_scroll[0] - delta.x(), 0, hbar.maximum())))
            vbar.setValue(int(np.clip(self._drag_start_scroll[1] - delta.y(), 0, vbar.maximum())))

    def _mouse_release(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._rubber_band is not None and self._rubber_band.isVisible():
            rect = self._rubber_band.geometry()
            self._rubber_band.hide()
            self._rubber_origin = None
            self._apply_zoom_to_rect(rect)
            return
        self._drag_start_global = None
        self.viewport().setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def _apply_zoom_to_rect(self, viewport_rect: QRect) -> None:
        """Zoom so the rubber-band selection fills the viewport."""
        if viewport_rect.width() < 4 or viewport_rect.height() < 4 or self._image_ma is None:
            return
        vw = self.viewport().width()
        vh = self.viewport().height()
        hv = self.horizontalScrollBar().value()
        vv = self.verticalScrollBar().value()

        px_l = (hv + viewport_rect.left()) / self._x_zoom
        px_r = (hv + viewport_rect.right()) / self._x_zoom
        py_t = (vv + viewport_rect.top()) / self._y_zoom
        py_b = (vv + viewport_rect.bottom()) / self._y_zoom

        pix_w = max(px_r - px_l, 0.5)
        pix_h = max(py_b - py_t, 0.5)
        min_x, min_y = self._min_zoom_xy()
        new_xz = float(np.clip(vw / pix_w, min_x, 100.0))
        new_yz = float(np.clip(vh / pix_h, min_y, 100.0))

        self._x_zoom = new_xz
        self._y_zoom = new_yz
        self._update_scroll_range()
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        hx = int(np.clip(px_l * new_xz, 0, hbar.maximum()))
        hy = int(np.clip(py_t * new_yz, 0, vbar.maximum()))
        hbar.setValue(hx)
        vbar.setValue(hy)
        self.zoom_changed.emit(self._x_zoom, self._y_zoom)
        self.viewport().update()
