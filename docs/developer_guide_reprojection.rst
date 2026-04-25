==========================
Reprojection Internals
==========================

This section describes the internal design of the ``nav.reproj`` package for
developers who need to extend or debug the reprojection and mosaicing
subsystem.

Module layout
-------------

.. code-block:: text

    src/nav/reproj/
        __init__.py              # Public API re-exports and __all__
        bodies.py                # BodyMosaic, BodyMosaicMergeStrategy, BodyReprojResult, BodyMosaicData
        rings.py                 # RingMosaic, RingMosaicMergeStrategy, RingReprojResult, RingMosaicData
        cartographic_model.py    # create_cartographic_model, CartographicModelResult
        ring_orbit_model.py      # RingOrbitModel frozen dataclass
        photometric_model.py     # PhotometricModel protocol + implementations
        _context_managers.py     # _reduced_oops_precision
        _serialization.py        # Save/load helpers shared by all result dataclasses

Thread safety
-------------

``RingMosaic.reproject()`` temporarily modifies oops global precision settings
via ``_reduced_oops_precision``. Concurrent calls from different threads on the
same observation will interfere. ``BodyMosaic.reproject()`` and
``create_cartographic_model()`` create ``Backplane`` objects from the provided
observation and are likewise not safe for concurrent use with the same
observation.

If you need to call ``reproject()`` from multiple threads, give each thread
its own ``obs`` instance.

Body mosaic storage
-------------------

``BodyMosaic`` uses a *shifted circular buffer* to handle longitude
wraparound without allocating the full 0 to 2\ |pi| range.

The internal arrays (``_img``, ``_has_data``, etc.) have shape
``(n_lat, n_lon)``. A pair of integer offsets (``_lat_min_bin``,
``_lon_min_bin``) records which full-grid bin corresponds to row/column 0
of the buffer.

When new data arrives outside the current buffer extent, the buffer is
expanded by exact-fit reallocation (no padding). Latitude expansion
prepends or appends rows. Longitude expansion similarly extends the buffer;
if the data wraps around the 0/2\ |pi| boundary, the column offset is
adjusted so that column 0 always maps to ``_lon_min_bin``.

To retrieve data that spans the wraparound boundary, ``to_bounded()`` accepts
a ``lon_range`` where ``min > max`` (e.g., ``(5.9, 0.3)``) meaning from
5.9 rad through 2\ |pi| to 0.3 rad. Internally ``_extract_region`` builds
the column index list by concatenating the two disjoint ranges.

Ring sparse storage
-------------------

``RingMosaic`` stores only longitude columns that contain at least one valid
pixel. The ``_sparse_lon_mask`` boolean array (length ``n_full_lon``) marks
which full-grid longitude bins are present. The data arrays have shape
``(n_rad, n_sparse_lon)`` where ``n_sparse_lon`` equals the number of
``True`` entries in ``_sparse_lon_mask``.

When ``add()`` receives a :class:`~nav.reproj.rings.RingReprojResult`, it:

1. Identifies new longitude columns not yet in the sparse store.
2. Inserts those columns into all data arrays using a single
   ``np.insert(..., axis=1)`` call per array to avoid repeated reallocations.
3. Updates ``_sparse_lon_mask``.
4. Applies the :class:`~nav.reproj.rings.RingMosaicMergeStrategy` to resolve conflicts on existing columns.
5. If at least one valid longitude column was present, appends
   ``repro.image_name`` to ``_contributing_image_names`` and increments
   ``_image_count`` (so ``contributing_image_names[k]`` matches pixels tagged
   with ``image_number == k``).

The always-sparse design means that ``reproject()`` always returns a
:class:`~nav.reproj.rings.RingReprojResult` with only the valid longitude
columns populated. There is no ``compress_longitude`` flag; sparsity is the
invariant.

Ring radius and longitude semantics
-----------------------------------

The interpretation of ``RingMosaic.radius_inner`` / ``radius_outer`` (and the
matching fields on :class:`~nav.reproj.rings.RingReprojResult` and
:class:`~nav.reproj.rings.RingMosaicData`) depends on whether the mosaic was
constructed with an ``orbit_model``:

- ``orbit_model is None``: longitudes are inertial J2000; radii are
  **absolute km**.
- ``orbit_model`` is set: longitudes are co-rotating in that model's frame;
  radii are **signed offsets in km from the orbital radius at each
  (longitude, time)** — i.e. from
  ``orbit_model.radius_at_longitude(inertial_lon, et)``. This makes an
  eccentric ring appear as a straight line in the reprojection.

Implementation notes:

- ``_reproject_inner`` evaluates ``rad_bins_act = rad_bins * rad_res +
  radius_inner + model_r(inertial_lon, midtime)`` for each (row, column)
  when an orbit model is set, so ``radius_inner`` enters as a per-pixel
  offset added to the per-column orbital radius.
- The radius filter on ``bp_radius`` first computes per-pixel offsets
  ``bp_radius_filter = bp_radius - model_r(inertial_lon, midtime)`` so the
  ``[radius_inner, radius_outer]`` test compares offsets to offsets.
- ``RingMosaic.reproject()`` rejects a per-call ``orbit_model`` that
  differs from the constructor's, because radius semantics are tied to
  that choice.
- ``RingMosaic.add()`` validates that the reprojection's ``orbit_model``
  is value-equal to the mosaic's (``RingOrbitModel`` is a frozen dataclass
  with auto-generated ``__eq__``, so a model rebuilt from disk compares
  equal to the in-memory instance) and that ``photometric_model_name``
  matches; mismatches raise ``ValueError``.

dtype propagation
-----------------

Each ``BodyMosaic`` and ``RingMosaic`` instance holds two authoritative dtype
attributes set at construction:

- ``_image_dtype`` (default ``np.float64``) — dtype for reprojected brightness
  ``img`` arrays.
- ``_metadata_dtype`` (default ``np.float32``) — dtype for geometry arrays on
  the reprojection grid (``resolution``, ``eff_resolution``, ``phase``,
  ``emission``, ``incidence``). Detector backplane samples feeding those
  quantities (for example incidence, emission, phase, latitude, longitude)
  are cast with ``.astype(self._metadata_dtype)`` inside ``reproject()`` before
  binning and masking; they are **not** applied to observation epoch ``time``.

These propagate through the pipeline as follows:

1. ``_allocate`` / ``_expand_lat`` / ``_expand_lon_impl`` allocate internal
   arrays using these dtypes directly (mosaic ``_time`` buffers use
   ``float64``, not ``_metadata_dtype``).
2. ``reproject()`` builds per-pixel ``img`` at ``_image_dtype`` and the
   geometry fields above at ``_metadata_dtype``. Scalar ``time`` on
   ``BodyReprojResult`` is a Python ``float`` (IEEE double); ring/body mosaic
   ``time`` grids are ``numpy.float64`` arrays regardless of ``metadata_dtype``.
3. Every ``BodyReprojResult`` and ``BodyMosaicData`` (and ring equivalents)
   carries explicit ``image_dtype`` and ``metadata_dtype`` fields describing
   the stored image and geometry dtypes; ``time`` is never governed by
   ``metadata_dtype``. That contract is self-describing and survives a
   save/load round-trip.

``image_number`` is always ``uint16`` regardless of the dtype kwargs, capping a single mosaic at
65,535 contributing images. ``add()`` raises ``OverflowError`` when that
limit is exceeded.

Serialization
-------------

The ``_serialization`` module provides the format helpers used by all four
dataclass ``save()`` / ``load()`` methods. It is a private module (not
exported from ``__init__.py``).

Path arguments may be ``str``, :class:`pathlib.Path`, or
:class:`filecache.FCPath`. Each is normalized to ``FCPath`` on entry. Writes
resolve a local path with :meth:`filecache.FCPath.get_local_path`, write with
NumPy or Astropy, then call :meth:`filecache.FCPath.upload`. Reads resolve a
local path the same way (retrieving remote objects into the cache when needed)
before loading.

Supported formats
^^^^^^^^^^^^^^^^^

npz
    ``np.savez`` / ``np.savez_compressed``. Each ``MaskedArray`` is split into
    two npz entries: ``<name>__data`` (the underlying array at its declared
    dtype) and ``<name>__mask`` (a ``bool_`` array). Tuples of length 2 of
    numeric types are stored as 1-D length-2 arrays. Tuples of strings (e.g.
    ``contributing_image_names``) are stored as a 1-D Unicode string array.
    Strings, dtype names, and scalar floats/ints are stored as 0-D unicode or
    numeric arrays.

fits
    ``astropy.io.fits``. Scalar metadata (strings, numbers, dtype names) go into
    the PrimaryHDU header. Each array occupies a separate ImageHDU with
    ``EXTNAME = <FIELDNAME>``; masks are stored as a companion ImageHDU with
    ``EXTNAME = <FIELDNAME>_MASK`` (uint8, 0 = valid). Tuple-of-string
    fields (``contributing_image_names``) are stored as a 1-D ``uint8`` ImageHDU
    of UTF-8 bytes with ``NUL`` (``\\0``) separators between entries (empty
    tuple → length-0 array).

Format inference
^^^^^^^^^^^^^^^^

When ``format_=None`` (the default), the format is inferred from the file
extension:

- ``.npz`` → ``'npz'``
- ``.fits``, ``.fit``, ``.fits.gz``, ``.fz`` → ``'fits'``

An explicit ``format_='npz'`` or ``format_='fits'`` keyword overrides inference.

kind / version scheme
^^^^^^^^^^^^^^^^^^^^^

Every file includes two sentinel values:

- ``__kind__`` — a string identifying the dataclass (e.g.
  ``'BodyMosaicData'``). ``load()`` raises ``ValueError`` when this does not
  match the expected kind.
- ``__version__`` — an integer (currently ``1``). Reserved for future schema
  migrations.

To add a field in a future version: bump ``__version__`` to ``2``, write the
new field in ``save()``, and handle ``version == 1`` (missing field) in
``load()`` by supplying a sensible default.

Load-time dtype verification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After reconstructing the dataclass in ``load()``, ``verify_dtype`` checks:

- The ``img`` array dtype matches the file's declared ``image_dtype``.
- Each metadata array (``resolution``, ``eff_resolution``, ``phase``,
  ``emission``, ``incidence``, and for rings ``mean_radial_*`` etc.) dtype
  matches ``metadata_dtype``.
- The ``time`` array (mosaic data only) is ``float64``.
- ``image_number`` is ``uint16``.
- All mask arrays are ``bool_``.

A ``ValueError`` is raised on the first mismatch, naming the offending field
and both the expected and actual dtypes. This catches files produced by
external tools that may have coerced dtypes on write.

RingOrbitModel serialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``RingOrbitModel`` is not a plain array; it is serialized flat via
``orbit_model_to_dict`` / ``orbit_model_from_dict``:

- In **npz**: fields are stored as ``orbit_model__<field>`` entries.
- In **FITS**: stored as ``ORBIT_MODEL__<FIELD>`` header cards.
- When ``orbit_model=None``, a single ``is_none=True`` sentinel is written.

Photometric models
------------------

The :class:`~nav.reproj.photometric_model.PhotometricModel` protocol requires
a single method::

    def correct(
        self,
        data: NDArrayFloatType,
        *,
        incidence: NDArrayFloatType,
        emission: NDArrayFloatType,
        phase: NDArrayFloatType,
    ) -> NDArrayFloatType: ...

All three angle arrays are in radians. The correction is applied during
``reproject()`` after pixel lookup, before writing to the reprojection result.
Passing ``photometric_model=None`` (the default) bypasses the correction.

Implementations provided:

- :class:`~nav.reproj.photometric_model.LambertModel`: divides by
  ``cos(incidence)``, clamped at a minimum threshold to avoid division by
  near-zero values.
- :class:`~nav.reproj.photometric_model.LommelSeeligerModel`: divides by
  ``cos(incidence) / (cos(incidence) + cos(emission))``.
- :class:`~nav.reproj.photometric_model.MinnaertModel`: applies the
  Minnaert law ``cos(incidence)^k * cos(emission)^(k-1)`` for a
  user-specified exponent ``k``.

Context managers
----------------

One context manager in ``_context_managers.py`` ensures global state is
restored even if an exception occurs:

- ``_reduced_oops_precision(dlt=1)``: sets ``oops.config.PATH_PHOTONS`` and
  ``SURFACE_PHOTONS`` delta-time precision to ``dlt``. Restores both on exit.

Adding a new photometric model
------------------------------

Implement the :class:`~nav.reproj.photometric_model.PhotometricModel`
protocol::

    from nav.reproj.photometric_model import PhotometricModel
    from nav.support.types import NDArrayFloatType
    import numpy as np

    class MyModel:
        name = 'my_model'

        def correct(
            self,
            data: NDArrayFloatType,
            *,
            incidence: NDArrayFloatType,
            emission: NDArrayFloatType,
            phase: NDArrayFloatType,
        ) -> NDArrayFloatType:
            # custom correction logic
            return data / np.cos(incidence)

Pass the instance to ``BodyMosaic`` or ``RingMosaic`` via the
``photometric_model`` parameter.

Cartographic model projection
-------------------------------

:func:`~nav.reproj.cartographic_model.create_cartographic_model` inverts the
reprojection: for each pixel in the observation, the backplane is used to
obtain the lat/lon on the body surface, and the mosaic is sampled at that
position via bilinear interpolation (``scipy.ndimage.map_coordinates`` with
``order=1``).

Longitude wraparound is handled by the formula::

    col = ((bp_longitude - lon_min) % (2 * pi)) / lon_resolution

This is correct for both non-wrapping and wrapping ``lon_range`` values.

.. |pi| replace:: *π*

Command-line layer
-------------------

The command-line tools are composed of three layers:

**Entry-point scripts** (``src/main/``)
   - ``nav_mosaic.py`` — :func:`main` dispatches on the first positional
     argument (``rings`` or ``body``) and calls ``_run_rings`` / ``_run_body``.
     ``rings_main`` and ``body_main`` are thin wrappers that prepend the
     subcommand to ``sys.argv`` and call ``main``.
   - ``nav_mosaic_cloud_tasks.py`` — Cloud Tasks worker that runs the
     reprojection pass only. Uses the same ``rings`` / ``body`` dispatch as
     ``nav_mosaic.py`` (captured in module-level ``_MODE`` before the Worker
     starts so the right ``add_ring_args`` / ``add_body_args`` are registered).
     ``process_task`` calls the same ``reproj_cli`` helpers
     (``build_*_mosaic``, ``per_image_output_path``, ``load_offset_if_any``,
     ``apply_offset_to_obs``, ``reproject_one_*``) as the local driver.
     Mosaic combination is not performed here; run ``nav_mosaic <mode>
     --skip-reproject`` after the queue drains.
   - ``nav_mosaic_display.py`` — same pattern for the display tools.

**Shared CLI helpers** (``src/reproj_cli/``)
   This top-level package (sibling of ``nav/``, ``backplanes/``, ``pds4/``)
   contains all the reusable CLI logic:

   - ``args.py`` — ``add_common_env_args``, ``add_common_output_args``,
     ``add_ring_args``, ``add_body_args``, ``add_display_args``. Adding a new
     command-line option to rings or body mode only requires editing the
     corresponding ``add_*_args`` function here.
   - ``factories.py`` — ``build_ring_mosaic(args)`` / ``build_body_mosaic(args)``
     translate parsed ``argparse.Namespace`` into ``RingMosaic`` /
     ``BodyMosaic`` instances. Add a new constructor parameter here when the
     underlying classes gain a new option.
   - ``paths.py`` — ``per_image_output_path`` / ``mosaic_output_path`` define
     the output-file naming convention; pass-1 image logs go under
     ``<output-dir>/logs/`` (see ``nav_mosaic._reproject_image_log_handlers``).
   - ``offsets.py`` — ``load_offset_if_any`` reads the ``_metadata.json`` file
     written by ``nav_offset`` and returns ``(dv, du)`` when ``status ==
     'success'``. ``apply_offset_to_obs`` wraps the result in
     ``oops.fov.OffsetFOV``. This mirrors the same pattern used in
     ``src/backplanes/backplanes.py``.
   - ``reproject.py`` — ``reproject_one_body`` / ``reproject_one_ring`` thin
     wrappers that translate ring-specific CLI args (zoom, longitude range,
     radius range, margin) into keyword arguments for
     ``RingMosaic.reproject()`` / ``BodyMosaic.reproject()``, including the
     per-image ``image_name`` string (from the CLI or from ``args.image_name``).

**Dataset enumeration**
   Both ``nav_mosaic.py`` and the existing ``nav_offset.py`` / ``nav_backplanes.py``
   scripts enumerate images via
   :meth:`DataSet.yield_image_files_from_arguments`. The dataset class is
   instantiated from ``DATASET_NAME = sys.argv[1]`` via
   :func:`~nav.dataset.dataset_name_to_class`, which also provides
   :meth:`~nav.dataset.DataSet.add_selection_arguments` to add dataset-specific
   filtering flags to the parser.

Two-pass workflow
^^^^^^^^^^^^^^^^^

The reprojection pass loops over
``DATASET.yield_image_files_from_arguments(args)`` and for each ``ImageFile``:

1. Compute ``per_image_output_path(output_dir, prefix, image_file, fmt=…, subject_name=mosaic.body_name)``
   (body or planet name in the filename; ``fmt`` and ``subject_name`` are keyword-only).
2. Skip if the file exists and ``--overwrite`` is not set.
3. Open ``IMAGE_LOGGER`` handlers writing to ``<output-dir>/logs/…``.
4. Load the observation via ``obs_class.from_file(image_path)``.
5. Optionally apply a navigation offset via ``load_offset_if_any`` + ``apply_offset_to_obs``.
6. Call ``reproject_one_body`` / ``reproject_one_ring`` with the computed
   ``image_name`` (file stem or ``--image-name``).
7. Save the ``BodyReprojResult`` / ``RingReprojResult`` to disk.

The mosaic pass then iterates the same list a second time, loads each existing
reprojection file, calls ``mosaic.add()`` (for body mode, passing
``resolution_threshold``, ``copy_slop``, and the body ``--max-incidence`` /
``--max-emission`` / ``--max-resolution`` limits explicitly so accumulation
matches the CLI even when reprojection was skipped), and saves the final
``BodyMosaicData`` / ``RingMosaicData`` via ``mosaic_output_path`` (same
``subject_name=mosaic.body_name`` convention).

Display layer
--------------

**Package layout** — ``src/nav/ui/mosaic_viewer/``

   - ``tiled_image_widget.py`` — :class:`~nav.ui.mosaic_viewer.tiled_image_widget.TiledImageWidget`.
     A generalized :class:`QAbstractScrollArea` that:

     - Renders only the visible viewport tiles on each paint event (tile-granular
       repaint), keeping memory usage independent of zoom level.
     - Supports independent X and Y zoom factors.
     - Handles Shift+scroll (X only), Ctrl+scroll (Y only), both-axes scroll.
     - Rubber-band zoom (Shift+left-drag).
     - Left-drag pan.
     - Green horizontal/vertical line overlays (show-radii / show-parallels /
       show-meridians) via ``set_show_rows`` / ``set_show_cols``.
     - Optional axis-tick overlays (longitude at bottom, radius/latitude at left)
       via ``set_axis_tick_options``.
     - ``y_flip=True`` for ring mosaics (array row 0 = inner radius, displayed at
       bottom); ``y_flip=False`` for body mosaics (row 0 = top of display).
     - Uses :func:`nav.ui.common.apply_linear_gamma_stretch` for image contrast,
       ensuring a consistent ``data ** gamma`` convention across all viewers.
     - Does *not* use :class:`nav.ui.common.ZoomPanController` — that helper
       assumes a pre-scaled ``QLabel`` inside a ``QScrollArea``, which is
       incompatible with tile-paint independent X/Y zoom.

   - ``common.py`` — :func:`~nav.ui.mosaic_viewer.common.load_ring_file` /
     :func:`~nav.ui.mosaic_viewer.common.load_body_file`. Peeks at the
     ``__kind__`` header in an npz or FITS file, then delegates to the
     appropriate ``*.load()`` classmethod and normalises the result into a
     :class:`~nav.ui.mosaic_viewer.common.RingDisplayData` /
     :class:`~nav.ui.mosaic_viewer.common.BodyDisplayData` dataclass ready
     for the window.

   - ``ring_window.py`` — :class:`~nav.ui.mosaic_viewer.ring_window.RingMosaicWindow`.
     Includes: stretch controls (via :func:`nav.ui.common.build_stretch_controls`),
     color-by panel (radial/angular resolution, phase, emission, image number;
     ephemeris-only options are omitted), EW-profile Matplotlib panel,
     radial-slice Matplotlib panel (right-click on mosaic), show-radii overlay,
     longitude/radius axis ticks, Prev/Next file navigation, and Save-FOV.
     ``RingDisplayData.contributing_image_names`` and per-longitude observation time
     feed the cursor **Source image** line (name and UTC time combined) for mosaics
     and single reprojections.

   - ``body_window.py`` — :class:`~nav.ui.mosaic_viewer.body_window.BodyMosaicWindow`.
     Header row for latitude/longitude axis tick toggles, image area with
     parallels/meridians overlays in the sidebar, lower strip with stretch
     presets, log-style zoom controls, a four-column **Cursor Info** grid, and
     a **Color By** radio grid (resolution, effective resolution, phase, emission,
     incidence, image number when present). The cursor grid includes sub-solar
     and sub-observer longitude and latitude (degrees), resolved from
     ``BodyDisplayData`` per-image arrays populated by :func:`~nav.ui.mosaic_viewer.common.load_body_file`.
     The **Source image** line shows the contributing name (or file stem) together
     with the pixel observation time in UTC when available.

Adding a new display feature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. If the feature requires a new per-column or per-pixel metadata array, extend
   the ``RingDisplayData`` / ``BodyDisplayData`` dataclass in
   ``nav/ui/mosaic_viewer/common.py`` and populate it in
   ``load_ring_file`` / ``load_body_file`` (including fields sourced from
   ``RingMosaicData`` / ``BodyMosaicData`` such as ``contributing_image_names``,
   ``sub_solar_*_per_image``, ``sub_observer_*_per_image``, etc.).

2. If the feature needs a new image overlay (e.g. contour lines), implement it
   as a new signal-driven ``set_*`` method on ``TiledImageWidget`` and call it
   from the window.

3. If the feature adds a new UI control (e.g. checkbox or slider), add it in the
   relevant window's ``_build_right_panel``, ``_build_plot_panels`` / header row,
   or ``_build_control_panel`` method and
   wire the callback.

Gamma stretch convention
^^^^^^^^^^^^^^^^^^^^^^^^^

All viewers use the ``((clip - black) / (white - black)) ** gamma`` convention
implemented by :func:`nav.ui.common.apply_linear_gamma_stretch`. A gamma of
``1.0`` is linear; values below ``1.0`` brighten mid-tones (the common display
choice). This convention is now uniformly applied across
``TiledImageWidget``, ``manual_nav_dialog``, ``nav_backplane_viewer``, and
``nav_create_simulated_image``.
