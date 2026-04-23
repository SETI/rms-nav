=======================
Reprojection Mosaicing
=======================

The ``nav.reproj`` package provides utilities for reprojecting planetary body
and ring images onto regular grids and accumulating multiple reprojected images
into mosaics.

Overview
--------

Two main classes are provided:

- :class:`~nav.reproj.bodies.BodyMosaic` -- reprojects body images onto a
  latitude/longitude grid and accumulates them into a mosaic.
- :class:`~nav.reproj.rings.RingMosaic` -- reprojects ring images onto a
  radius/longitude grid and accumulates them with true sparse longitude storage.

A standalone utility function :func:`~nav.reproj.cartographic_model.create_cartographic_model`
projects a body mosaic back onto image coordinates for use as a navigation
correlation model.

Body reprojection and mosaicing
--------------------------------

Create a :class:`~nav.reproj.bodies.BodyMosaic` once per body, then feed it
observations::

    from nav.reproj import BodyMosaic

    mosaic = BodyMosaic(body_name='MIMAS')
    for obs in observations:
        result = mosaic.reproject(obs)
        mosaic.add(result)

    data = mosaic.to_bounded()  # BodyMosaicData

The mosaic grows automatically (``dynamic=True`` by default) to accommodate
each new reprojected image. You can pre-allocate a specific region::

    import math

    mosaic = BodyMosaic(
        body_name='MIMAS',
        lat_range=(-math.pi / 4, math.pi / 4),  # -45 to 45 degrees latitude
        lon_range=(0.0, math.pi),                # 0 to 180 degrees longitude
        dynamic=False,
    )

When ``lat_range`` or ``lon_range`` is ``None`` (the default), the mosaic uses
the full valid range for that axis. If ``dynamic=False`` and no range is
specified, the mosaic is pre-allocated to the full global grid.

Coordinate systems
^^^^^^^^^^^^^^^^^^

All angular values are in **radians**. The latitude/longitude coordinate
system is controlled by two parameters:

- ``latlon_type``: one of ``'centric'`` (default), ``'graphic'``, or
  ``'squashed'``.
- ``lon_direction``: ``'east'`` (default) or ``'west'``.

Choosing dtypes
^^^^^^^^^^^^^^^

By default, the reprojected brightness image uses ``float64`` and all geometry
arrays (resolution, phase, emission, incidence) and the ``time`` array use
``float32``::

    from nav.reproj import BodyMosaic
    import numpy as np

    # Defaults: image in float64, geometry in float32
    mosaic = BodyMosaic(body_name='MIMAS')

    # All float64 for maximum precision
    mosaic = BodyMosaic(
        body_name='MIMAS',
        image_dtype=np.float32,     # smaller image storage
        metadata_dtype=np.float64,  # full-precision geometry
    )

The ``time`` field is always stored as ``float64`` regardless of ``metadata_dtype``.
The ``image_number`` field is always ``uint16``, capping a single mosaic at
65 535 contributing images.

Photometric correction
^^^^^^^^^^^^^^^^^^^^^^

Pass a photometric model to apply a correction during reprojection::

    from nav.reproj import BodyMosaic, LambertModel

    mosaic = BodyMosaic(
        body_name='MIMAS',
        photometric_model=LambertModel(),
    )

Available models are :class:`~nav.reproj.photometric_model.LambertModel`,
:class:`~nav.reproj.photometric_model.LommelSeeligerModel`, and
:class:`~nav.reproj.photometric_model.MinnaertModel`. When ``photometric_model``
is ``None`` (the default), pixel values are reprojected without correction.

Pixel conflict resolution
^^^^^^^^^^^^^^^^^^^^^^^^^

``BodyMosaic`` uses the ``BEST_RESOLUTION`` strategy (see
:class:`~nav.reproj.bodies.BodyMosaicMergeStrategy`): empty (masked) pixels are
filled unconditionally and existing data is replaced only when the new
observation has strictly better effective resolution (lower km/pixel).

Longitude wraparound
^^^^^^^^^^^^^^^^^^^^

The internal storage uses a shifted circular buffer so that data spanning
the 0/2\ |pi| boundary (e.g., a body centered on the meridian) is handled
correctly. The retrieval methods unwrap longitude automatically.

Retrieval methods
^^^^^^^^^^^^^^^^^

All retrieval methods return a :class:`~nav.reproj.bodies.BodyMosaicData`
frozen dataclass with masked arrays for image data, resolution, phase,
emission, incidence, and observation metadata:

- :meth:`~nav.reproj.bodies.BodyMosaic.to_bounded` -- return the mosaic
  clipped to the data bounds or a user-specified range.
- :meth:`~nav.reproj.bodies.BodyMosaic.to_full` -- return the full
  -|pi|/2 to |pi|/2 x 0 to 2\ |pi| grid.
- :attr:`~nav.reproj.bodies.BodyMosaic.bounds` -- the current (lat, lon)
  extents of accumulated data, or ``None`` if the mosaic is empty.

Ring reprojection and mosaicing
---------------------------------

:class:`~nav.reproj.rings.RingMosaic` works similarly but uses **sparse**
longitude storage: only longitude columns that contain at least one valid
pixel are stored. This is memory-efficient for the common case where only a
fraction of the ring plane is observed::

    from nav.reproj import RingMosaic

    mosaic = RingMosaic('SATURN', radius_inner=70000, radius_outer=140000)
    for obs in observations:
        result = mosaic.reproject(obs)
        mosaic.add(result)

    data = mosaic.to_sparse()  # RingMosaicData with longitude_antimask

The ``longitude_antimask`` field in the result indicates which full-grid
longitude bins are present in the sparse storage.

Choosing dtypes (rings)
^^^^^^^^^^^^^^^^^^^^^^^

The same ``image_dtype`` / ``metadata_dtype`` kwargs are available on
:class:`~nav.reproj.rings.RingMosaic`::

    import numpy as np
    from nav.reproj import RingMosaic

    mosaic = RingMosaic(
        'SATURN', radius_inner=70000, radius_outer=140000,
        metadata_dtype=np.float64,  # full-precision geometry
    )

Orbit model
^^^^^^^^^^^

The ring geometry (eccentricity, ring plane) is handled by
:class:`~nav.reproj.ring_orbit_model.RingOrbitModel`. Pre-defined instances
are available::

    from nav.reproj import FRING_CORE, BRING_OUTER_EDGE

**Longitude convention.** When no orbit model is specified
(``orbit_model=None``, which is the default), the longitudes stored in
reprojection results and mosaics are **inertial J2000 ring longitudes** —
measured eastward from the ascending node of the ring plane on the J2000
reference plane.  When an orbit model is supplied, each inertial longitude is
converted to the **co-rotating frame** of the model before binning; mosaic
column *i* then corresponds to co-rotating longitude ``i ×
longitude_resolution``.  See also :ref:`orbit-model-longitude` in the
command-line reference.

Pass a custom model via the ``orbit_model`` parameter::

    from nav.reproj import RingMosaic, RingOrbitModel

    my_orbit = RingOrbitModel(
        name='MY-RING',
        a=140220.0,
        e=0.0,
        w0=0.0,
        dw=0.0,
        mean_motion=581.964 * 3.14159 / 180.0,
        epoch_utc='2007-01-01',
    )
    mosaic = RingMosaic('SATURN', radius_inner=70000, radius_outer=140000,
                        orbit_model=my_orbit)

Merge strategy
^^^^^^^^^^^^^^

The ``merge_strategy`` parameter controls how longitude columns are updated
when multiple observations overlap::

    from nav.reproj import RingMosaic, RingMosaicMergeStrategy

    mosaic = RingMosaic(
        'SATURN', radius_inner=70000, radius_outer=140000,
        merge_strategy=RingMosaicMergeStrategy.BEST_RESOLUTION,
    )

- ``MOST_COVERAGE_THEN_RESOLUTION`` (default): fill empty longitude columns
  first; for already-present columns, replace only when the new data has
  better mean radial resolution.
- ``BEST_RESOLUTION``: replace an existing longitude column only when the new
  data has strictly better mean radial resolution.

Retrieval methods
^^^^^^^^^^^^^^^^^

- :meth:`~nav.reproj.rings.RingMosaic.to_sparse` -- sparse storage (only
  present longitude columns). The ``longitude_antimask`` field marks present
  columns.
- :meth:`~nav.reproj.rings.RingMosaic.to_bounded` -- dense array clipped to
  a longitude range.
- :meth:`~nav.reproj.rings.RingMosaic.to_full` -- dense full 0 to 2\ |pi|
  longitude grid.

Saving and loading
------------------

All four result dataclasses (:class:`~nav.reproj.bodies.BodyMosaicData`,
:class:`~nav.reproj.bodies.BodyReprojResult`,
:class:`~nav.reproj.rings.RingMosaicData`,
:class:`~nav.reproj.rings.RingReprojResult`) support ``save()`` and ``load()``
methods. Two file formats are supported:

- **npz** (NumPy archive, default) — format inferred from a ``.npz``
  extension.
- **FITS** — format inferred from a ``.fits`` or ``.fit`` extension. Requires
  the ``astropy`` package (included as a runtime dependency).

Paths may be a string, a :class:`pathlib.Path`, or a :class:`filecache.FCPath`
(for example ``gs://…`` URIs handled by the project’s ``FileCache``). Remote
paths are fetched into the local cache on ``load()`` and written locally then
uploaded on ``save()``.

Body mosaic examples::

    from nav.reproj import BodyMosaic, BodyMosaicData

    data = mosaic.to_bounded()

    # Save — format inferred from extension
    data.save('mimas.npz')                    # compressed npz
    data.save('mimas.npz', compress=False)    # uncompressed npz (faster I/O)
    data.save('mimas.fits')                   # FITS
    data.save('mimas.fits', format='fits')    # explicit format

    # Load
    reloaded = BodyMosaicData.load('mimas.npz')
    reloaded = BodyMosaicData.load('mimas.fits')

Body reprojection result::

    from nav.reproj import BodyReprojResult

    result = mosaic.reproject(obs, image_name='N1234567890')
    result.save('reproj.npz')
    reloaded = BodyReprojResult.load('reproj.npz')

Ring mosaic examples::

    import math
    from nav.reproj import RingMosaicData

    data = ring_mosaic.to_bounded(longitude_range=(0.0, math.pi))
    data.save('saturn_rings.npz')
    reloaded = RingMosaicData.load('saturn_rings.npz')

Ring reprojection result::

    from nav.reproj import RingReprojResult

    result = ring_mosaic.reproject(obs, image_name='N1234567890')
    result.save('ring_reproj.fits')
    reloaded = RingReprojResult.load('ring_reproj.fits')

When loading, the dtypes of all arrays are verified against the ``image_dtype``
and ``metadata_dtype`` fields stored in the file. A ``ValueError`` is raised if
any mismatch is detected, guarding against files produced by external tools
that may have coerced dtypes.

Image labels (reprojection and mosaic)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each :class:`~nav.reproj.rings.RingReprojResult` and
:class:`~nav.reproj.bodies.BodyReprojResult` carries an ``image_name`` string
(typically the source image stem).  The ``save()`` and ``load()`` methods
preserve this value.

Each :class:`~nav.reproj.rings.RingMosaicData` and
:class:`~nav.reproj.bodies.BodyMosaicData` carries ``contributing_image_names``,
a tuple of strings in the same order as the ``image_number`` indices stored in
the mosaic (pixel value ``k`` refers to ``contributing_image_names[k]`` when
``k`` is in range).  The tuple grows by one entry each time ``mosaic.add()``
finishes incorporating a reprojection and advances the internal image counter.

In Python, pass ``image_name=...`` to :meth:`~nav.reproj.rings.RingMosaic.reproject`
and :meth:`~nav.reproj.bodies.BodyMosaic.reproject`.  The ``nav_mosaic`` CLI
stores the dataset image stem per file by default; pass ``--image-name LABEL``
to use the same label for every image in the run instead.

Cartographic navigation model
-------------------------------

Once a body mosaic is built, it can be projected back onto image coordinates
to produce a navigation model for correlation::

    from nav.reproj import create_cartographic_model

    result = create_cartographic_model(
        mosaic.to_bounded(),
        obs,
        body_name='MIMAS',
    )
    if result is not None:
        model_img = result.model_img          # [v, u] float array
        ratio    = result.resolution_ratio   # mosaic res / image res

The function returns ``None`` if the mosaic has no valid data. The
``resolution_ratio`` field gives the median mosaic effective resolution divided
by the image center resolution; a value greater than 1.0 means the model
will be blurrier than the image.

.. |pi| replace:: *π*

.. _cli-mosaic:

Command-line mosaic generation
-------------------------------

The ``nav_mosaic_rings`` and ``nav_mosaic_body`` commands (entry points into
the single ``nav_mosaic`` program) reproject a dataset of images and combine
them into a mosaic using a two-pass workflow:

1. **Reprojection pass** — for each image in the dataset, load the observation,
   optionally apply a pre-computed navigation offset, call
   ``BodyMosaic.reproject()`` / ``RingMosaic.reproject()`` (with ``image_name``
   set to that image's file stem, or to ``--image-name`` when that option is
   given), and save the result as
   ``<output-dir>/<prefix>_<body_or_planet>_<image_stem>_reproj.<fmt>`` (body
   name for ``nav_mosaic body``, planet name for ``nav_mosaic rings``). Existing
   files are skipped unless ``--overwrite`` is given, enabling interrupted runs
   to be resumed.

2. **Mosaic pass** — re-iterate the same image list, load each reprojection file
   that exists, call ``mosaic.add()`` (which extends ``contributing_image_names``
   in lockstep with ``image_number``), and save the final mosaic as
   ``<output-dir>/<prefix>_<body_or_planet>_mosaic.<fmt>``.

Either pass may be skipped with ``--skip-reproject`` / ``--skip-mosaic``.

Ring mosaics quick example::

    nav_mosaic_rings coiss_saturn \
        --volumes COISS_2001 \
        --pds3-holdings-root /data/pds3 \
        --nav-results-root /data/nav_results \
        --planet SATURN \
        --radius-inner 139500 \
        --radius-outer 140220 \
        --output-dir /data/mosaics \
        --prefix fring_2004

Body mosaics quick example::

    nav_mosaic_body coiss_saturn \
        --volumes COISS_2001 \
        --pds3-holdings-root /data/pds3 \
        --nav-results-root /data/nav_results \
        --body-name MIMAS \
        --output-dir /data/mosaics \
        --prefix mimas_2004

Offset application
^^^^^^^^^^^^^^^^^^

When ``--nav-results-root`` is provided, ``nav_mosaic`` looks up a
``_metadata.json`` file for each image (written by ``nav_offset``).  If the
file exists and has ``status == 'success'``, the stored ``(dv, du)`` offset is
applied to the observation's FOV via ``oops.fov.OffsetFOV`` before reprojection.
If the file is absent, invalid JSON, or has a non-success status, a warning is
logged and uncorrected pointing is used.

Output format
^^^^^^^^^^^^^

The default output format is FITS (``.fits``).  Pass ``--format npz`` to use
compressed NumPy archives instead.  Reprojection and mosaic files live directly
under ``<output-dir>``; the only subdirectory used is ``logs/`` for per-image
reprojection logs from pass 1:

- Per-image reprojection: ``<output-dir>/<prefix>_<body_or_planet>_<image_stem>_reproj.<fmt>``
- Per-image reprojection log: ``<output-dir>/logs/<results_path_stub>_<timestamp>.log``
- Final mosaic: ``<output-dir>/<prefix>_<body_or_planet>_mosaic.<fmt>``

If ``--prefix`` is empty (the default), the leading underscore is omitted.

Common options reference
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--output-dir DIR``
     - *(required)*
     - Directory for output files.
   * - ``--prefix STR``
     - ``''``
     - Filename prefix.
   * - ``--format {fits,npz}``
     - ``fits``
     - Output file format.
   * - ``--overwrite``
     - ``False``
     - Re-compute and overwrite existing per-image reprojection files.
   * - ``--skip-reproject``
     - ``False``
     - Skip the reprojection pass.
   * - ``--skip-mosaic``
     - ``False``
     - Skip the mosaic-building pass.
   * - ``--nav-results-root DIR``
     - ``None``
     - Root written by ``nav_offset``; enables offset application.
   * - ``--dry-run``
     - ``False``
     - Print what would be done without writing files.
   * - ``--image-name LABEL``
     - *(use each file's stem)*
     - Override the ``image_name`` stored on every reprojection and the names
       listed in ``contributing_image_names`` on the mosaic.

Ring-specific options
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--planet NAME``
     - *(required)*
     - Planet name (e.g. ``SATURN``).
   * - ``--radius-inner KM``
     - *(required)*
     - Inner mosaic radius (km).
   * - ``--radius-outer KM``
     - *(required)*
     - Outer mosaic radius (km).
   * - ``--longitude-resolution DEG``
     - ``0.02``
     - Column pitch (degrees/pixel).
   * - ``--radius-resolution KM``
     - ``5.0``
     - Row pitch (km/pixel).
   * - ``--orbit-model {none,fring_core,bring_outer_edge}``
     - ``none``
     - Ring orbit model for co-rotating longitude (see below).
   * - ``--merge-strategy {best_resolution,most_coverage_then_resolution}``
     - ``most_coverage_then_resolution``
     - Conflict-resolution strategy.
   * - ``--margin N``
     - ``3``
     - Edge pixels to exclude.
   * - ``--zoom N or R,L``
     - ``1``
     - Zoom factor for sub-pixel interpolation.
   * - ``--no-omit-shadow``
     - *(flag; default: shadow masked)*
     - Include pixels inside the planet shadow.
   * - ``--longitude-range START END``
     - ``None``
     - Restrict reprojected longitude range (degrees).
   * - ``--radius-range INNER OUTER``
     - ``None``
     - Restrict reprojected radius range (km).
   * - ``--image-dtype DTYPE``
     - ``float64``
     - NumPy dtype for the brightness array.
   * - ``--metadata-dtype DTYPE``
     - ``float32``
     - NumPy dtype for geometry metadata arrays.
   * - ``--photometric-model {none,lambert,lommel-seeliger,minnaert}``
     - ``none``
     - Photometric correction during ring ``reproject()`` (optional; same models as body).

.. _orbit-model-longitude:

Orbit model and longitude convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``--orbit-model none`` (the default), the longitude values stored in both
the per-image reprojection files and the final mosaic are **inertial J2000 ring
longitudes** — measured eastward from the ascending node of the ring plane on
the J2000 reference plane, in degrees (internally radians).  This is the
default behavior of ``oops.backplane.Backplane.ring_longitude``.

When an orbit model is supplied (``--orbit-model fring_core`` or
``--orbit-model bring_outer_edge``), each inertial longitude is transformed
to the **co-rotating frame** of that model before binning.  In the resulting
file, mosaic column *i* corresponds to co-rotating longitude
``i × longitude_resolution``; the column index no longer has a fixed
relationship to J2000 north.  All files in the same mosaic must use the same
orbit model setting.

Body-specific options
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--body-name NAME``
     - *(required)*
     - Body to reproject (e.g. ``MIMAS``).
   * - ``--lat-resolution DEG``
     - ``0.1``
     - Row pitch (degrees/pixel).
   * - ``--lon-resolution DEG``
     - ``0.1``
     - Column pitch (degrees/pixel).
   * - ``--lat-range MIN MAX``
     - ``None``
     - Latitude extent (degrees); default full range.
   * - ``--lon-range MIN MAX``
     - ``None``
     - Longitude extent (degrees); default full range.
   * - ``--max-incidence DEG``
     - ``70``
     - Maximum incidence angle for valid pixels.
   * - ``--max-emission DEG``
     - ``70``
     - Maximum emission angle for valid pixels.
   * - ``--max-resolution KM``
     - ``None``
     - Maximum resolution (km/pixel) for valid pixels.
   * - ``--edge-margin N``
     - ``3``
     - Edge pixels to discard.
   * - ``--zoom N``
     - ``1``
     - Sub-pixel zoom factor.
   * - ``--latlon-type {centric,graphic,squashed}``
     - ``centric``
     - Latitude/longitude coordinate system.
   * - ``--lon-direction {east,west}``
     - ``east``
     - Longitude direction convention.
   * - ``--photometric-model {none,lambert,lommel-seeliger,minnaert}``
     - ``none``
     - Photometric correction to apply.
   * - ``--no-dynamic``
     - *(flag; default: dynamic growth enabled)*
     - Disable dynamic mosaic growth.
   * - ``--resolution-threshold F``
     - ``1.0``
     - Improvement factor required to overwrite a pixel.
   * - ``--copy-slop N``
     - ``0``
     - Extra pixels around each copied pixel to reduce artefacts.
   * - ``--image-dtype DTYPE``
     - ``float64``
     - NumPy dtype for the brightness array.
   * - ``--metadata-dtype DTYPE``
     - ``float32``
     - NumPy dtype for geometry metadata arrays.

Command-line mosaic display
----------------------------

The ``nav_mosaic_display_rings`` and ``nav_mosaic_display_body`` commands
(entry points into the single ``nav_mosaic_display`` program) open an
interactive PyQt6 window for browsing reprojection and mosaic files.  Multiple
files can be passed; the window shows one file at a time and includes
**Prev / Next** navigation buttons.

Ring display quick example::

    nav_mosaic_display_rings /data/mosaics/fring_2004_mosaic.fits

Body display quick example::

    nav_mosaic_display_body /data/mosaics/mimas_2004_reproj_N1234567890.fits

Display options
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--stretch-black F``
     - auto
     - Initial black-point for image stretch.
   * - ``--stretch-white F``
     - auto
     - Initial white-point for image stretch.
   * - ``--stretch-gamma F``
     - ``0.5``
     - Initial gamma (``data ** gamma`` convention; < 1 brightens mid-tones).
   * - ``--show-radii``
     - ``False``
     - (Rings) Overlay green horizontal lines at user-configured radii.
   * - ``--show-parallels``
     - ``False``
     - (Bodies) Overlay latitude parallel lines.
   * - ``--show-meridians``
     - ``False``
     - (Bodies) Overlay longitude meridian lines.

Interactive controls
^^^^^^^^^^^^^^^^^^^^

- **Scroll wheel** — zoom both axes simultaneously.
- **Shift + scroll** — zoom the X axis (longitude) only.
- **Ctrl + scroll** — zoom the Y axis (radius / latitude) only.
- **Shift + left-drag** — rubber-band zoom to a selected region.
- **Left-drag** — pan.
- **Right-click** (rings only) — display a radial profile at the clicked
  longitude column.
- **Save FOV** button — save the current viewport to a PNG file.
- **Stretch sliders** (Black / White / Gamma) — adjust contrast.
- **Color by** radio buttons — tint the image by a per-column or per-pixel
  metadata field (radial resolution, angular resolution, phase, emission,
  image number, etc.).  On the ring window, options that required ephemeris
  columns not present in the file (inertial longitude, true anomaly) are omitted.
- **Cursor info** — for mosaics, the source-image line uses stored contributing
  names in the form ``imagename (#k)`` when available.
