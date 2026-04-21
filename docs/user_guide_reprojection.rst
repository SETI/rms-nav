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

Coordinate systems
^^^^^^^^^^^^^^^^^^

All angular values are in **radians**. The latitude/longitude coordinate
system is controlled by two parameters:

- ``latlon_type``: one of ``'centric'`` (default), ``'graphic'``, or
  ``'squashed'``.
- ``lon_direction``: ``'east'`` (default) or ``'west'``.

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

Merge strategy
^^^^^^^^^^^^^^

When multiple observations overlap, the ``merge_strategy`` parameter controls
which pixel wins::

    from nav.reproj import BodyMosaic, BodyMosaicMergeStrategy

    mosaic = BodyMosaic(
        body_name='MIMAS',
        merge_strategy=BodyMosaicMergeStrategy.MOST_COVERAGE_THEN_RESOLUTION,
    )

- ``BEST_RESOLUTION`` (default): the image with the highest resolution (lowest
  km/pixel) wins at each pixel.
- ``MOST_COVERAGE_THEN_RESOLUTION``: prefer the image with the most total
  coverage; break ties using resolution.

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

    mosaic = RingMosaic(planet_name='SATURN')
    for obs in observations:
        result = mosaic.reproject(obs)
        mosaic.add(result)

    data = mosaic.to_sparse()  # RingMosaicData with longitude_antimask

The ``longitude_antimask`` field in the result indicates which full-grid
longitude bins are present in the sparse storage.

Orbit model
^^^^^^^^^^^

The ring geometry (eccentricity, ring plane) is handled by
:class:`~nav.reproj.ring_orbit_model.RingOrbitModel`. Pre-defined instances
are available::

    from nav.reproj import FRING_CORE, BRING_OUTER_EDGE

Pass a custom model via the ``ring_orbit_model`` parameter::

    from nav.reproj import RingMosaic, RingOrbitModel

    my_orbit = RingOrbitModel(
        a=140220.0,
        ae=0.0,
        e=0.0,
        long_peri=0.0,
        precession_rate=0.0,
        epoch_et=0.0,
    )
    mosaic = RingMosaic(planet_name='SATURN', ring_orbit_model=my_orbit)

Merge strategy
^^^^^^^^^^^^^^

The ``merge_strategy`` parameter controls how longitude columns are updated
when multiple observations overlap::

    from nav.reproj import RingMosaic, RingMosaicMergeStrategy

    mosaic = RingMosaic(
        planet_name='SATURN',
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
