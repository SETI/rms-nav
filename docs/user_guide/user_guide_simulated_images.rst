================
Simulated Images
================

RMS-NAV can synthesize images on demand for arbitrary geometry and run them
through the same navigation pipeline as real spacecraft images. A simulated
frame whose true offset (and camera roll) is known *by construction* lets you
verify that a technique recovers what it should, sweep a single parameter to see
how a diagnostic responds, and cover geometries that no real archive happens to
contain -- all without a real observation.

A simulated scene has three equally-valid entry points, each backed by the same
renderer and parameter set:

* the interactive **GUI** (``nav_create_simulated_image``), for dialling in a
  scene and watching it render live;
* a **YAML scene catalog** under ``tests/integration/sim_scenes/``, the durable,
  diffable artifact a test or a reviewer consumes;
* the **Python API** (:func:`nav.sim.render.render_combined_model`,
  :class:`~nav.obs.obs_inst_sim.ObsSim`), for programmatic use.

Creating a simulated image in the GUI
=====================================

Launch the interactive GUI with:

.. code-block:: bash

   nav_create_simulated_image

The GUI lets you set the image size and planted offset; choose the instrument
the frame emulates; configure the detector-noise model and random background
stars; add individual stars, planetary bodies (ellipsoid or polyhedral mesh),
and rings; preview the image live; and save the scene as a JSON parameter file
or a catalog YAML.

Running navigation on a simulated image
=======================================

To navigate a saved JSON parameter file, pass the ``sim`` dataset name and the
file path:

.. code-block:: bash

   nav_offset sim /path/to/simulated_image.json

You can restrict the models and techniques the same way as for a real image:

.. code-block:: bash

   nav_offset sim /path/to/simulated_image.json \
     --nav-models stars,rings \
     --nav-techniques correlate_all

Because the simulated observation is flagged as such, the autonomous registry
builds the *simulated* navigation models -- :class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated`,
:class:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`, and
:class:`~nav.nav_model.stars.nav_model_stars_simulated.NavModelStarsSimulated` --
which read the operator parameters instead of SPICE, while the SPICE-backed
models stand down. The simulated body emits the same BODY_DISC / BODY_BLOB /
LIMB_ARC features a real body does (depending on its resolution and phase), the
simulated ring emits RING_ANNULUS plus a RING_EDGE per edge, and the simulated
stars emit STAR features, so every technique that runs on a real frame can run
on a simulated one.

The YAML scene catalog
======================

The scene catalog under ``tests/integration/sim_scenes/<scene_class>/<name>.yaml``
mirrors the real-image library's directory-as-registry layout. Each file is one
synthetic frame plus the planted ground truth the navigator should recover. The
schema is :mod:`nav.sim.scene` (``load_sim_scene`` validates a file into a
:class:`~nav.sim.scene.SimScene`; ``SimScene.to_sim_params`` maps it to the dict
the renderer and :class:`~nav.obs.obs_inst_sim.ObsSim` consume).

A scene names its instrument, image size, random seed, and the geometry, then
declares the planted offset and roll under ``ground_truth``:

.. code-block:: yaml

   schema_version: 1
   scene_name: planted_offset_star_field
   instrument: coiss_nac
   image_size_vu: [128, 128]
   random_seed: 7
   exposure_sec: 1.0
   bodies: []
   rings: []
   stars:
     list:
       - {name: F1, v: 30.0, u: 40.0, vmag: 3.0}
       - {name: F2, v: 80.0, u: 35.0, vmag: 3.6}
   noise:
     poisson: true
     read_noise_dn: 4.0
   ground_truth:
     planted_offset_dv_px: 1.5
     planted_offset_du_px: -0.5
     planted_rotation_deg: 0.0

Key scene fields:

* ``instrument`` -- the camera the frame emulates (e.g. ``coiss_nac``, ``gossi``),
  selecting the PSF, noise, saturation, and units the renderer and navigator use.
* ``image_size_vu`` -- ``[height, width]`` in pixels.
* ``bodies`` / ``rings`` / ``stars`` -- the geometry. Bodies and rings are lists
  of the same parameter dicts the GUI and JSON use; ``stars.list`` holds the
  per-star dicts and ``stars.background_count`` adds a random background field.
* ``noise`` -- the detector-noise block (``poisson``, ``read_noise_dn``,
  ``cosmic_ray_rate_per_sec``, ``missing_data_rate``, ...).
* ``fit_camera_rotation`` -- optional boolean. A scene may ask the navigator to
  solve for a camera roll regardless of whether the emulated instrument fits
  rotation by default, so a clean-PSF camera can exercise the 3-DoF path.
* ``ground_truth.planted_offset_dv_px`` / ``planted_offset_du_px`` -- the
  translation the renderer applies; the navigator predicts the unshifted
  geometry and must recover it.
* ``ground_truth.planted_rotation_deg`` -- a camera roll about the boresight,
  applied to stars and bodies before the translation.

Render and navigate a catalog scene from Python:

.. code-block:: python

   from nav.nav_model import build_models_for_obs
   from nav.nav_orchestrator import NavOrchestrator
   from nav.obs.obs_inst_sim import ObsSim
   from nav.sim.scene import load_sim_scene

   scene = load_sim_scene(path_to_scene_yaml)
   obs = ObsSim.from_file('/tmp/frame.json', sim_params=scene.to_sim_params())
   result = NavOrchestrator(
       build_models_for_obs(obs), only_models='*', only_techniques='*'
   ).navigate(obs)
   # result.offset_px should recover scene.ground_truth's planted offset.

Algorithmic-invariant tests
===========================

Scenes under ``sim_scenes/algorithmic_invariants/`` carry a planted offset (or
roll) and are navigated by ``tests/integration/test_sim_algorithmic_invariants.py``,
which asserts the navigator recovers the planted value within tolerance. Because
the ground truth is correct *by construction*, these tests never need
re-blessing (unlike a regression baseline). The catalog includes scenes designed
so each technique is the load-bearing one: a disc scene, a high-phase blob
crescent, a star field, a camera-roll field, a resolved-body limb, and a ring
edge.

Regression baselines
====================

``sim_baselines/<name>.json`` records the rounded ``(offset, confidence,
status)`` the navigator produces for every catalog scene, and
``tests/integration/test_sim_baselines.py`` re-navigates each scene and asserts
an exact rounded match -- a tripwire on any unintended change. Regenerate the
baselines after an intended change with:

.. code-block:: bash

   python -m tests.integration.update_sim_baselines

Parameter sweeps
================

A sweep drives one base scene by varying a single parameter (or a group of
parameters that move together) across a list of values and navigates each step,
so you can see how a diagnostic *responds* to a controlled change. A sweep spec
lives at ``tests/integration/sim_sweeps/<name>.yaml``:

.. code-block:: yaml

   sweep_name: range_body_size
   base_scene: phase_sweep_regular_body/regular_sphere_base.yaml
   parameters: [bodies.0.axis1, bodies.0.axis2, bodies.0.axis3]
   values: [130.0, 90.0, 60.0, 40.0, 20.0, 12.0]

Run all sweeps and write the per-step response curves (status, offset error,
confidence, primary technique) to ``sim_sweeps/results/<name>.json`` with:

.. code-block:: bash

   python -m tests.integration.sim_sweep_runner

``tests/integration/test_sim_sweeps.py`` asserts the per-sweep invariants: the
read-noise sweep recovers at low noise and fails at the navigability cliff; the
phase sweep recovers at every phase; and the range sweep's primary technique
transitions ``BodyLimbNav -> BodyDiscCorrelateNav -> BodyBlobNav`` as the body
shrinks past the limb, disc, and blob resolution thresholds.

JSON parameter file reference
=============================

The full description of the GUI/JSON parameter structure -- every supported
top-level field, star parameter, body parameter, ring parameter, and ring edge
mode -- is documented under :doc:`/introduction_simulated_images`.
