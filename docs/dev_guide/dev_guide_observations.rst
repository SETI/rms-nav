============
Observations
============

Overview
========

The :mod:`nav.obs` subsystem wraps an ``oops`` snapshot in a
navigation-aware class that adds backplane caching, extended-FOV
accessors, image masks, and per-instrument calibration hooks.  Every
navigation pipeline takes an
:class:`~nav.obs.obs_snapshot_inst.ObsSnapshotInst` instance as its
input and reads the per-image data, geometry, and instrument-specific
calibration through that object.

The class hierarchy splits responsibility across three axes:

- :class:`~nav.obs.obs.Obs` — abstract observation root, wires
  :class:`~nav.support.nav_base.NavBase` into the ``oops`` class tree so
  every concrete observation inherits ``config`` and ``logger``.
- :class:`~nav.obs.obs_snapshot.ObsSnapshot` — extends
  :class:`~nav.obs.obs.Obs` and ``oops.observation.snapshot.Snapshot``.
  Adds the FOV / extended-FOV accessors, backplane caching, and the
  per-image mask helpers that every navigation model and technique
  consumes.
- :class:`~nav.obs.obs_inst.ObsInst` — abstract mix-in carrying
  per-instrument calibration: the ``from_file`` constructor contract,
  the optical PSF, the per-instrument visual-magnitude window, and the
  per-image public-metadata projection.
- :class:`~nav.obs.obs_snapshot_inst.ObsSnapshotInst` — concrete mix-in
  of :class:`~nav.obs.obs_snapshot.ObsSnapshot` and
  :class:`~nav.obs.obs_inst.ObsInst`.  Per-mission subclasses derive
  from this base.

ObsSnapshot
===========

:class:`~nav.obs.obs_snapshot.ObsSnapshot` is the navigation-side wrapper
around an ``oops`` snapshot.  It exposes three families of helpers:

- **FOV / extended-FOV geometry.**  ``data_shape_uv`` / ``data_shape_vu``
  report the sensor shape; ``fov_v_min`` / ``fov_v_max`` /
  ``fov_u_min`` / ``fov_u_max`` give the in-sensor pixel bounds;
  ``extfov_margin_v`` / ``extfov_margin_u`` give the per-axis margin
  appended by :class:`~nav.nav_orchestrator.instrument_config.InstrumentSettings`;
  the corresponding ``extfov_*`` accessors give the extended bounds and
  shape.  ``clip_fov`` / ``clip_extfov`` clamp ``(u, v)`` coordinates
  into either grid; ``clip_rect_fov`` / ``clip_rect_extfov`` clamp full
  rectangles.
- **Mask and template constructors.**  ``make_fov_zeros`` /
  ``make_extfov_zeros`` allocate float arrays of the right shape;
  ``make_extfov_false`` allocates the boolean equivalent;
  ``unpad_array_to_extfov`` crops a sensor-shaped array down to the
  extended-FOV grid.  ``extfov_data_sensor_mask`` returns a boolean
  mask that is ``True`` where the extended-FOV pixel corresponds to a
  real sensor pixel and ``False`` in the margin.
- **Inventory predicates.**  ``inventory_body_in_fov`` /
  ``inventory_body_in_extfov`` consume an ``oops`` inventory entry and
  return whether the predicted body bounding box overlaps the sensor /
  extended FOV.  Per-NavModel ``instances_for_obs`` hooks (e.g.
  :meth:`~nav.nav_model.nav_model_body.NavModelBody.instances_for_obs`)
  call these to decide which bodies to instantiate.

Backplanes are cached in the underlying ``oops`` snapshot, so repeated
queries of the same backplane on the same ``ObsSnapshot`` reuse the
prior computation.

ObsInst
=======

:class:`~nav.obs.obs_inst.ObsInst` is the per-instrument calibration
mix-in.  It defines the abstract contract every per-mission subclass
must implement:

- ``from_file(path, *, config=None, extfov_margin_vu=None, **extra_params)``
  — load an image file and return the matching
  :class:`~nav.obs.obs_snapshot_inst.ObsSnapshotInst`.  Subclasses
  delegate the actual decode to ``oops.hosts.<mission>.<inst>.from_file``,
  then wrap the resulting ``oops`` snapshot.
- ``star_psf()`` — returns the per-instrument optical
  :class:`~psfmodel.PSF` (typically a
  :class:`~psfmodel.GaussianPSF`).  Used by
  :class:`~nav.nav_model.stars.nav_model_stars.NavModelStars` to predict
  the per-star detection footprint.
- ``star_psf_size(star)`` — returns the per-star kernel support
  rectangle in pixels.
- ``star_min_usable_vmag()`` / ``star_max_usable_vmag()`` — the
  per-instrument photometric window.  Stars outside this window do not
  contribute predicted detections.
- ``get_public_metadata()`` — returns a JSON-friendly dict of per-image
  metadata fields (mission, instrument, exposure, filter wheel
  positions, etc.) for the per-image sidecar.

The ``inst_config`` property exposes the per-instrument YAML block
loaded from ``src/nav/config_files/config_4N0_inst_*.yaml`` so subclass
methods can read instrument-specific knobs without hard-coding them.

Per-instrument subclasses
=========================

Concrete subclasses live in ``src/nav/obs/`` and are registered (via
the :mod:`nav.obs` package's ``__init__.py``) under a per-mission /
per-instrument key consumed by :class:`~nav.dataset.dataset.DataSet`.
Today's shipping subclasses:

- :class:`~nav.obs.obs_inst_cassini_iss.ObsCassiniISS` — Cassini ISS
  NAC and WAC.  Delegates to ``oops.hosts.cassini.iss.from_file``.
- :class:`~nav.obs.obs_inst_voyager_iss.ObsVoyagerISS` — Voyager 1 / 2
  ISS NA and WA cameras.  Delegates to
  ``oops.hosts.voyager.iss.from_file``.
- :class:`~nav.obs.obs_inst_galileo_ssi.ObsGalileoSSI` — Galileo SSI
  (uses ``full_fov=True`` to read the full sensor regardless of the
  on-chip ROI).  Delegates to ``oops.hosts.galileo.ssi.from_file``.
- :class:`~nav.obs.obs_inst_newhorizons_lorri.ObsNewHorizonsLORRI` —
  New Horizons LORRI (passes ``calibration=False`` so the raw pixel
  values pass through).  Delegates to
  ``oops.hosts.newhorizons.lorri.from_file``.
- :class:`~nav.obs.obs_inst_sim.ObsSim` — simulated-image observation
  backed by a description of bodies and stars, consumed by
  ``nav_create_simulated_image`` and the simulated-image GUI driver.

Each subclass overrides ``from_file`` to pull the right ``oops`` host,
wires up the per-instrument PSF and photometric window, and forwards
per-image metadata into ``get_public_metadata``.

Adding a new instrument
=======================

The end-to-end checklist lives at
:doc:`dev_guide_extending`; the obs-side bullet is:

1. Subclass :class:`~nav.obs.obs_snapshot_inst.ObsSnapshotInst` in
   ``src/nav/obs/obs_inst_<mission>_<inst>.py``.
2. Implement ``from_file`` (delegate to the matching
   ``oops.hosts.<mission>.<inst>`` host module), ``star_psf``,
   ``star_min_usable_vmag``, ``star_max_usable_vmag``, and
   ``get_public_metadata``.
3. Register the subclass in :mod:`nav.obs` (``src/nav/obs/__init__.py``)
   under the per-mission / per-instrument key the corresponding
   :class:`~nav.dataset.dataset.DataSet` subclass passes to ``from_file``.
4. Add the per-instrument config block at
   ``src/nav/config_files/config_4N0_inst_<mission>_<inst>.yaml`` so
   ``inst_config`` carries the instrument's tuning knobs.

API reference
=============

The autodocumented API surface is at :doc:`/api_reference/api_obs`.
