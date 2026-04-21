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

The always-sparse design means that ``reproject()`` always returns a
:class:`~nav.reproj.rings.RingReprojResult` with only the valid longitude
columns populated. There is no ``compress_longitude`` flag; sparsity is the
invariant.

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
