===================
The Image Simulator
===================

Overview
========

The image simulator (the ``nav.sim`` package) renders synthetic spacecraft
frames -- stars, planetary bodies, and rings, with a realistic detector model --
from operator-supplied geometry rather than from SPICE. It exists to **test and
validate the navigation pipeline**, not as an end-user product: because every
simulated frame's true pointing offset is known by construction, a simulated
image is the only frame whose navigation answer can be checked exactly. The
simulator drives the algorithmic-invariant tests, the regression baselines, the
single-variable sensitivity sweeps, and the sensitivity report
(:doc:`/simulator_report/simulator_report`).

The simulator has three equally valid entry points (the "three peers"):

- the **Python API** (:func:`nav.sim.render.render_combined_model` and the
  per-feature renderers),
- the **YAML scene catalog** under ``tests/integration/sim_scenes/`` (validated by
  :mod:`nav.sim.scene`), which is the durable test artifact, and
- the **GUI** ``nav_create_simulated_image``, an interactive editor for the same
  parameters.

Every parameter is reachable from all three; adding a physical effect means
adding it to the renderer, the scene schema, and a GUI control together. The
navigator side -- how a simulated frame is turned into ``NavFeature`` objects and
navigated -- is documented in the simulated-model chapters
(:doc:`dev_guide_navigation_models_body_simulated`,
:doc:`dev_guide_navigation_models_ring_simulated`,
:doc:`dev_guide_navigation_models_star_simulated`); this chapter documents the
rendering side and the scene formats.

What can be simulated
=====================

A scene is composed from these ingredients, each independently optional:

- **Ellipsoidal bodies** -- a triaxial ellipsoid with Lambertian shading at a
  chosen illumination azimuth and phase angle, optional in-plane rotation and
  tilt, and optional procedurally generated craters.
- **Irregular (polyhedral-mesh) bodies** -- a procedurally generated lumpy mesh
  at a chosen three-axis pose, for non-ellipsoidal shapes (Hyperion-like,
  Phoebe-like). See :ref:`sim-body-params`.
- **Planetary rings** -- bright ringlets and dark gaps with elliptical
  (mode-1, eccentric) edges and per-edge anti-aliased shading.
- **Stars** -- an explicit list of stars at chosen positions and magnitudes, and
  a procedurally generated field of random background stars.
- **A camera-roll and pointing offset** -- the planted ground truth the navigator
  must recover: a sub-pixel ``(dv, du)`` translation and a boresight roll.
- **A realistic detector model** -- Poisson shot noise, Gaussian read noise, a
  bias pedestal, cosmic-ray spikes, missing-data markers, full-well saturation
  with optional column bloom, and a per-instrument PSF -- all keyed to the
  emulated instrument's configuration.
- **A stray-light gradient** -- a linear brightness ramp or a radial flare bump,
  to exercise the source-image background filter.

Per-instrument coupling means a simulated "Cassini ISS NAC raw" frame goes
through the same noise sigma, signal scale, saturation, marker value, and PSF as
a real CISS NAC raw frame; the emulated instrument is chosen by the
``instrument`` field (see :ref:`Instruments <sim-instruments>`).

Scene ingredients
=================

The panels below are rendered by ``python -m tests.integration.sim_doc_images``
(see :ref:`sim-png-export`); each isolates one ingredient.

.. figure:: _sim_images/ellipsoid_body.png
   :width: 45%
   :align: center

   Ellipsoidal body (Lambertian, moderate phase). ``axis1`` is the vertical
   extent, ``axis2`` the horizontal.

.. figure:: _sim_images/mesh_body.png
   :width: 45%
   :align: center

   Irregular polyhedral-mesh body of the same axes at a three-axis pose.

.. figure:: _sim_images/body_craters.png
   :width: 45%
   :align: center

   Ellipsoid with procedurally generated craters.

.. figure:: _sim_images/crescent_body.png
   :width: 45%
   :align: center

   High-phase (130 deg) mesh body rendered as a thin lit crescent.

.. figure:: _sim_images/rings.png
   :width: 45%
   :align: center

   Two eccentric ringlets with a gap between them.

.. figure:: _sim_images/star_field.png
   :width: 45%
   :align: center

   A random background star field.

.. figure:: _sim_images/multi_body.png
   :width: 45%
   :align: center

   Multiple bodies (ellipsoid and mesh) at different sizes, depth-ordered by
   ``range``.

.. figure:: _sim_images/body_and_stars.png
   :width: 45%
   :align: center

   A body against a background star field.

.. figure:: _sim_images/detector_noise.png
   :width: 45%
   :align: center

   Detector model: read + shot noise, sparse cosmic-ray spikes (bright) and
   missing-data dropouts (dark).

.. figure:: _sim_images/stray_light_gradient.png
   :width: 45%
   :align: center

   A linear stray-light gradient behind a body.

.. figure:: _sim_images/composite_scene.png
   :width: 45%
   :align: center

   A composite frame: a mesh moon, a ring, and a star field.

The render pipeline
===================

:func:`nav.sim.render.render_combined_model` takes a ``sim_params`` dict and
returns ``(image, metadata)``. It composes the frame in a fixed order so each
later stage sees the accumulated signal: background stars, then the explicit
star list, then bodies and rings (depth-sorted far-to-near by their ``range``),
then the noise-free stray-light gradient, then the detector model (signal
scaling, Poisson shot noise, Gaussian read noise, bias pedestal, cosmic rays,
missing-data markers, and saturation with optional bloom). The output is an array
of detector counts (DN).

**Determinism.** Every random effect draws from a per-effect sub-seed derived
from the scene's ``random_seed`` (:mod:`nav.sim.seeds`), so the same scene
renders byte-identically, and adding a new randomized effect does not perturb the
output of existing ones. Crater placement uses a stable hash of the body's
geometry when the body gives no explicit ``seed``.

.. _sim-instruments:

**Instruments.** The ``instrument`` field selects which per-instrument
configuration block drives the detector model and PSF
(:mod:`nav.sim.instruments`). The recognized names are ``coiss_nac``,
``coiss_wac``, ``coiss_calib_nac``, ``coiss_calib_wac``, ``gossi``, ``nhlorri``,
and ``vgiss``, plus ``generic`` (alias ``sim``) for the instrument-agnostic
defaults. Calibrated (``*_calib_*``) instruments are in I/F units with a NaN
missing-data marker and no full-well; raw instruments are in DN with a 0 marker.
A scene can pin or override individual instrument settings with
``instrument_config`` (see :ref:`sim-instrument-config`).

.. _sim-scene-formats:

Scene format
============

A scene is a single YAML file whose fields are the flat runtime ``sim_params``
names the renderer consumes, so a validated scene file *is* the ``sim_params``
dict with no translation layer.

The scene catalog (:mod:`nav.sim.scene`) is the durable test artifact, laid out
as ``tests/integration/sim_scenes/<scene_class>/<scene_name>.yaml`` (the
directory is the registry). :func:`nav.sim.scene.load_sim_scene` parses and
validates a file and returns the flat ``sim_params`` dict the renderer consumes;
the GUI's "Save Scene (YAML)" / "Load Scene (YAML)" buttons read and write the
same format via :func:`nav.sim.scene.save_sim_scene`. The scene classes (for
example ``algorithmic_invariants``, ``phase_sweep_regular_body``,
``phase_sweep_irregular_body``, ``range_sweep``, ``noise_sweep``,
``multi_body_geometry``, ``regression``) scope what each scene is testing and are
enforced by the structural test. The scene README at
``tests/integration/sim_scenes/README.txt`` documents the schema alongside the
code.

A complete YAML scene -- a noisy Cassini NAC frame with one irregular mesh body, a
ring, a couple of stars, and a planted offset the navigator must recover -- reads:

.. code-block:: yaml

   schema_version: 1
   scene_name: example_scene
   instrument: coiss_nac
   size_v: 220
   size_u: 220
   random_seed: 42
   exposure_sec: 1.0
   bodies:
     - name: HYPERION
       shape_model: polyhedral_mesh
       mesh_lumpiness: 0.4
       mesh_seed: 7
       pose_euler_deg: [10.0, 35.0, 0.0]
       center_v: 110.0
       center_u: 110.0
       axis1: 150.0
       axis2: 110.0
       axis3: 95.0
       illumination_angle: 25.0
       phase_angle: 40.0
   rings:
     - name: RINGLET
       feature_type: RINGLET
       center_v: 110.0
       center_u: 110.0
       inner_data: [{mode: 1, a: 90.0, ae: 6.0}]
       outer_data: [{mode: 1, a: 98.0, ae: 6.0}]
       shading_distance: 10.0
       range: 1000.0
   background_stars_num: 40
   stars:
     - {name: S1, v: 30.0, u: 60.0, vmag: 6.0}
     - {name: S2, v: 180.0, u: 150.0, vmag: 7.5}
   noise:
     poisson: true
     read_noise_dn: 4.0
   offset_v: 1.43
   offset_u: -0.61

The ``schema_version`` and ``scene_name`` keys are metadata the renderer
ignores; ``scene_name`` must equal the filename stem. Every other key is a flat
``sim_params`` field consumed directly by the renderer.

Scene parameter reference
=========================

Top-level fields
----------------

.. list-table::
   :widths: 22 13 12 53
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``size_v`` / ``size_u``
     - int
     - required
     - Image height and width in pixels.
   * - ``instrument``
     - str
     - ``generic``
     - Emulated instrument; selects the detector model and PSF (see
       :ref:`Instruments <sim-instruments>`).
   * - ``random_seed``
     - int
     - 42
     - Scene seed; per-effect sub-seeds derive from it.
   * - ``exposure_sec``
     - float
     - 1.0
     - Exposure time; scales the cosmic-ray count.
   * - ``offset_v`` / ``offset_u``
     - float
     - 0.0
     - Planted pointing offset (px) applied to all bodies, rings, and stars.
   * - ``offset_rotation_deg``
     - float
     - 0.0
     - Planted boresight roll (deg) applied about the image center.
   * - ``bodies``
     - list
     - ``[]``
     - Per-body parameter dicts (see :ref:`sim-body-params`).
   * - ``rings``
     - list
     - ``[]``
     - Per-ring parameter dicts (see :ref:`sim-ring-params`).
   * - ``stars``
     - list
     - ``[]``
     - Explicit star dicts (see :ref:`sim-star-params`).
   * - ``background_stars_num``
     - int
     - 0
     - Random background-star count (0-1000).
   * - ``noise``
     - dict
     - instrument
     - Detector-noise block (see :ref:`sim-noise`).
   * - ``stray_light``
     - dict
     - off
     - Stray-light block (see :ref:`sim-stray-light`).
   * - ``instrument_config``
     - dict
     - none
     - Per-instrument config overrides (see :ref:`sim-instrument-config`).

.. _sim-body-params:

Body parameters
---------------

Each entry of ``bodies`` is a dict. Common fields:

.. list-table::
   :widths: 22 13 12 53
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``name``
     - str
     - generated
     - Body label used in metadata and annotations.
   * - ``center_v`` / ``center_u``
     - float
     - frame center
     - Body center in pixels.
   * - ``axis1`` / ``axis2`` / ``axis3``
     - float
     - 0.0
     - Full extents of the three body axes in pixels (``axis3`` defaults to
       ``min(axis1, axis2)``).
   * - ``illumination_angle``
     - float (deg)
     - 0.0
     - Image-plane light azimuth (0 = from the top).
   * - ``phase_angle``
     - float (deg)
     - 0.0
     - Phase angle (0 = fully lit, 180 = back-lit crescent).
   * - ``range``
     - float
     - body index
     - Depth-ordering key; smaller renders in front.
   * - ``shape_model``
     - str
     - ``ellipsoid``
     - ``ellipsoid`` or ``polyhedral_mesh``.

Ellipsoid bodies add ``rotation_z`` and ``rotation_tilt`` (degrees), the crater
controls ``crater_fill``, ``crater_min_radius``, ``crater_max_radius``,
``crater_power_law_exponent``, ``crater_relief_scale``, and an optional crater
``seed``. Mesh bodies (``shape_model: polyhedral_mesh``) instead read:

.. list-table::
   :widths: 22 13 12 53
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``mesh_lumpiness``
     - float
     - 0.3
     - Surface relief amplitude as a fraction of the unit radius.
   * - ``mesh_seed``
     - int
     - 0
     - Seed selecting which irregular shape is generated.
   * - ``mesh_n_lat`` / ``mesh_n_lon``
     - int
     - 16 / 32
     - Mesh latitude bands and longitude divisions.
   * - ``pose_euler_deg``
     - [float, float, float]
     - ``[0, 0, 0]``
     - Body orientation as intrinsic X, Y, Z Euler angles (degrees).
   * - ``nav_override``
     - dict
     - none
     - Overlay applied to *the predicted body only*, diverging the navigation
       geometry from the rendered one (see below).

.. note::

   **Axis convention.** Both renderers use the same mapping: ``axis1`` is the
   vertical (``v``) extent and ``axis2`` the horizontal (``u``) extent. An
   ellipsoid and a mesh declared with identical ``axis1``/``axis2`` are oriented
   the same way, so the two are interchangeable for the same geometry.

**Render geometry vs navigation geometry.** By default the navigator predicts the
body from the same parameters the renderer drew, so it knows the truth. A body
may carry an optional ``nav_override`` mapping that is overlaid on the predicted
body only -- the renderer ignores it and always draws the true geometry. This
separates the *render geometry* (ground truth) from the *navigation geometry*
(what the navigator assumes), the channel the irregular-body scenarios use to
render a lumpy mesh but predict its smooth (ellipsoidal) limit, or to predict the
same body at a disagreeing pose. The override never moves the center, so the
planted offset is still recoverable. See
:doc:`dev_guide_navigation_models_body_simulated`.

.. _sim-ring-params:

Ring parameters
---------------

Each entry of ``rings`` is a dict with ``name``, a ``feature_type`` of
``RINGLET`` (adds brightness) or ``GAP`` (subtracts it), a ``center_v`` /
``center_u``, a ``shading_distance`` (edge-fade width in pixels), a ``range``
depth key, and ``inner_data`` / ``outer_data`` edge lists. At least one edge is
required. Each edge is a list of mode dicts; the required mode-1 dict carries the
elliptical orbit: ``a`` (semi-major axis, px), ``ae`` (eccentricity times ``a``,
px), ``long_peri`` (longitude of pericenter, deg), and ``rate_peri`` (precession
rate, deg/day, applied across ``time`` minus ``ring_epoch``).

.. _sim-star-params:

Star parameters
---------------

Each entry of ``stars`` is a dict with ``name``, a ``v`` / ``u`` position, a
``vmag`` (visual magnitude; lower is brighter), a ``psf_sigma``, and an optional
``spectral_class``. Random background stars are added by setting
``background_stars_num``; their PSF sigma and brightness power-law are configured
by ``background_stars_psf_sigma`` and
``background_stars_distribution_exponent``. Stars are rendered with a half-pixel
PSF-evaluation offset so a star's brightness centroid lands exactly at its
predicted ``(v, u)``, which keeps star navigation free of a constant half-pixel
bias.

.. _sim-noise:

Detector-noise block
--------------------

The optional ``noise`` dict overrides the emulated instrument's detector
defaults:

.. list-table::
   :widths: 30 12 12 46
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``poisson``
     - bool
     - True
     - Apply Poisson shot noise to the signal.
   * - ``read_noise_dn``
     - float
     - instrument
     - Gaussian read-noise sigma in DN.
   * - ``bias_dn``
     - float
     - instrument
     - Additive bias pedestal; lifts dark sky off zero so it is not confused with
       the missing-data marker.
   * - ``cosmic_ray_rate_per_sec``
     - float
     - 0.0
     - Cosmic-ray fluence (events / cm^2 / sec), scaled by ``exposure_sec``.
   * - ``missing_data_rate``
     - float
     - 0.0
     - Fraction of pixels (0-1) set to the missing-data marker.

Saturation clips at the instrument's full-well DN after noise; cameras with
documented column bloom can carry a ``bloom_length`` that spreads saturated
excess along the column.

.. _sim-stray-light:

Stray-light block
-----------------

The optional ``stray_light`` dict adds a background contribution before the
detector model: ``amplitude`` (peak fraction of full scale; 0 disables it),
``direction_deg`` (ramp direction for the ``linear`` model), and ``model``
(``linear`` ramp or ``radial`` bump). It exercises the navigator's source-image
background filter.

.. _sim-instrument-config:

Instrument-config overrides
---------------------------

The optional ``instrument_config`` dict is deep-merged over the resolved
per-instrument block, so a scene can pin individual settings (PSF sigma, noise,
data units, extfov margin) without tracking later camera-config edits. Omit it to
inherit everything; name ``generic`` and override everything to fully
self-specify. The top-level ``noise`` block still wins over
``instrument_config.noise`` for rendering. See :doc:`dev_guide_observations`.

The simulated-image GUI
=======================

``nav_create_simulated_image`` is a PyQt6 editor for the same parameters, with a
live preview. Launch it with:

.. code-block:: bash

   nav_create_simulated_image

The GUI exposes the full scene parameter surface, so any scene that can be
written by hand in YAML can also be built in the GUI. The **General** tab carries
the image size, the planted offset and camera roll, the exposure time, the seed,
the instrument selector, the camera-rotation-fit override and midtime, the
detector-noise panel (Poisson, read noise, bias, bloom, signal full-scale,
pixel area, cosmic-ray rate, missing-data rate), the stray-light panel
(amplitude, direction, model, radial centre), a PSF preview, a saturation overlay
toggle, the closest-planet selector and ring times, and the background-star
controls. Each per-**body** tab carries that body's geometry, a shape-model
dropdown (ellipsoid or mesh) with the mesh lumpiness / seed / resolution / pose
controls, the crater controls and crater seed, an optional physical scale, and a
navigation-override group (the predicted geometry that diverges from the rendered
one). Each per-**star** tab carries the star's position, magnitude, PSF, smear
vector, and catalog label; each per-**ring** tab carries the ring's edges and
shading. The parameters the GUI does not edit are the nested
``instrument_config`` overrides, multi-mode ring edges (the renderer reads only
mode 1), and the absolute ``signal_full_scale_dn`` alias (its fractional form is
exposed instead). Scenes round-trip through the **Load / Save Scene (YAML)**
buttons, so a scene rendered in the GUI can be saved as a catalog artifact and a
catalog scene can be loaded back to edit. The GUI is one peer, not the sole
control surface; the YAML and the Python API are equally authoritative.

Running navigation on a simulated image
=======================================

A simulated image is navigated through the same pipeline as a real frame, via the
``sim`` dataset. With a saved YAML scene file:

.. code-block:: bash

   nav_offset sim /path/to/scene.yaml

The ``sim`` dataset (``DataSetSim``) builds an :class:`~nav.obs.obs_inst_sim.ObsSim`
that loads the scene via :func:`nav.sim.scene.load_sim_scene`, renders the frame,
and carries the ``sim_params`` on the snapshot. (A runtime scene file must still
satisfy the validator, so its ``scene_name`` must equal the filename stem.) The
model-selection layer routes a simulated obs to the simulated NavModels --
``NavModelBodySimulated``, ``NavModelRingsSimulated``, ``NavModelStarsSimulated``
-- which build one feature set per body / ring / star field from the scene
parameters, while the SPICE-backed models decline a simulated obs. From there the
same techniques run and produce the same ``NavResult``. In tests the scene is
usually driven directly:
``ObsSim.from_file(path, sim_params=load_sim_scene(path))``.

.. _sim-png-export:

Exporting viewable PNGs
=======================

The renderer emits detector counts whose absolute range depends on the instrument
and on cosmic-ray spikes, so :mod:`nav.sim.png_export` stretches a DN image to a
viewable grayscale PNG with a percentile clip (a few hot pixels do not crush the
signal) and an optional gamma that lifts dim features such as a crescent or a
faint star field:

.. code-block:: python

   from nav.sim.png_export import render_scene_png
   render_scene_png(sim_params, 'frame.png', gamma=1.4, upscale=2)

Two tools build on it. The sweep runner can dump every frame behind a response
curve for inspection:

.. code-block:: bash

   python -m tests.integration.sim_sweep_runner --dump-images out/ --only phase_regular_body

writes one PNG per sweep step under ``out/<sweep_name>/``. The documentation-image
generator rebuilds the galleries in this chapter and the scene images in the
sensitivity report:

.. code-block:: bash

   python -m tests.integration.sim_doc_images

Both galleries carry a ``NOTES.md`` describing how to regenerate them after a
rendering change.

See also
========

- :doc:`dev_guide_navigation_models_body_simulated`,
  :doc:`dev_guide_navigation_models_ring_simulated`,
  :doc:`dev_guide_navigation_models_star_simulated` -- the navigator side.
- :doc:`dev_guide_testing` -- the test kinds the simulator drives.
- :doc:`/simulator_report/simulator_report` -- the sensitivity and
  algorithmic-invariant results.
- API reference: :doc:`/api_reference/api_sim`.
