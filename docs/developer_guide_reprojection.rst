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

The always-sparse design means that ``reproject()`` always returns a
:class:`~nav.reproj.rings.RingReprojResult` with only the valid longitude
columns populated. There is no ``compress_longitude`` flag; sparsity is the
invariant.

dtype propagation
-----------------

Each ``BodyMosaic`` and ``RingMosaic`` instance holds two authoritative dtype
attributes set at construction:

- ``_image_dtype`` (default ``np.float64``) — dtype for reprojected brightness
  ``img`` arrays.
- ``_metadata_dtype`` (default ``np.float32``) — dtype for all geometry arrays
  (``resolution``, ``eff_resolution``, ``phase``, ``emission``, ``incidence``)
  **and** for ``time``.

These propagate through the pipeline as follows:

1. ``_allocate`` / ``_expand_lat`` / ``_expand_lon_impl`` allocate internal
   arrays using these dtypes directly.
2. ``reproject()`` casts backplane arrays with
   ``bp_xyz.mvals.astype(self._metadata_dtype)`` and constructs per-pixel
   intermediate arrays at ``_image_dtype`` (for ``img``) or
   ``_metadata_dtype`` (for all geometry).
3. Every ``BodyReprojResult`` and ``BodyMosaicData`` (and ring equivalents)
   carries explicit ``image_dtype`` and ``metadata_dtype`` fields so that the
   dtype contract is self-describing and survives a save/load round-trip.

``time`` is always ``float64`` regardless of the dtype kwargs, preserving
sub-second precision for Cassini ET values (~5×10⁸ s). ``image_number`` is
always ``uint16`` regardless of the dtype kwargs, capping a single mosaic at
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
    dtype) and ``<name>__mask`` (a ``bool_`` array). Tuples of length 2 are
    stored as 1-D length-2 arrays. Strings, dtype names, and scalar floats/ints
    are stored as 0-D unicode or numeric arrays.

fits
    ``astropy.io.fits``. Scalar metadata (strings, numbers, dtype names) go
    into the PrimaryHDU header. Each array occupies a separate ImageHDU with
    ``EXTNAME = <FIELDNAME>``; masks are stored as a companion ImageHDU with
    ``EXTNAME = <FIELDNAME>_MASK`` (uint8, 0 = valid).

Format inference
^^^^^^^^^^^^^^^^

When ``format=None`` (the default), the format is inferred from the file
extension:

- ``.npz`` → ``'npz'``
- ``.fits``, ``.fit``, ``.fits.gz``, ``.fz`` → ``'fits'``

An explicit ``format='npz'`` or ``format='fits'`` keyword overrides inference.

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
