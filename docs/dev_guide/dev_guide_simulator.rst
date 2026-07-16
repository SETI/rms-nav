===================
The Image Simulator
===================

Overview
========

The image simulator (the ``spindoctor.sim`` package) renders synthetic spacecraft
frames -- stars, planetary bodies, and rings, with a realistic detector model --
from operator-supplied geometry rather than from SPICE. It exists to **test and
validate the navigation pipeline**, not as an end-user product: because every
simulated frame's true pointing offset is known by construction, a simulated
image is the only frame whose navigation answer can be checked exactly. The
simulator drives the algorithmic-invariant tests, the regression baselines, the
single-variable sensitivity sweeps, and the sensitivity report
(:doc:`/simulator_report/simulator_report`).

The simulator has three equally valid entry points (the "three peers"):

- the **Python API** (:func:`spindoctor.sim.render.render_combined_model`, a thin
  cached driver over the stage pipeline in :mod:`spindoctor.sim.forward`),
- the **YAML scene catalog** under ``tests/integration/sim_scenes/`` (validated by
  :mod:`spindoctor.sim.scene`), which is the durable test artifact, and
- the **GUI** ``sd_create_simulated_image`` (the ``spindoctor.cli.sim_editor``
  package), an interactive editor for the same parameters.

Every parameter is reachable from all three; adding a physical effect means
adding it to the renderer, the scene schema, and a GUI control together. The
navigator side -- how a simulated frame is turned into ``NavFeature`` objects and
navigated -- is documented in the simulated-model chapters
(:doc:`dev_guide_navigation_models_body_simulated`,
:doc:`dev_guide_navigation_models_ring_simulated`,
:doc:`dev_guide_navigation_models_star_simulated`); this chapter documents the
architecture, the rendering side, and the scene formats.

.. _sim-two-sided:

The two-sided architecture
==========================

The simulator is split into two sides that exchange information through exactly
one channel. Understanding this split comes before everything else in this
chapter, because a change that quietly bridges it destroys the property the
simulator exists to provide: that a recovery measurement's only error is the
error the scene planted.

**The image side** renders the frame the navigator is given, reading the *full*
scene -- including every planted error, noise knob, and contaminant. It lives in
:mod:`spindoctor.sim.forward`:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Module
     - Role
   * - ``stages.py``
     - The :class:`~spindoctor.sim.forward.stages.SimFrame` dataclass and the
       :class:`~spindoctor.sim.forward.stages.Stage` protocol (see
       :ref:`sim-stage-pipeline`).
   * - ``pipeline.py``
     - The fixed-order stage registry and ``run_pipeline`` driver.
   * - ``scene_radiance.py``
     - Composes the noise-free signal image: background stars, catalog stars,
       and the body/ring stack depth-sorted far to near.
   * - ``body.py`` / ``body_mesh.py``
     - Ellipsoid and polyhedral-mesh body renderers.
   * - ``ring.py``
     - Ring-feature renderer (mode-1 elliptical edges, edge shading).
   * - ``star.py``
     - Star and background-star-field renderers.
   * - ``optics.py``
     - The optical-path stage: it runs motion smear, residual distortion, the
       whole-scene PSF, ghost reflections, and the stray-light field in a fixed
       internal order (see :ref:`sim-optics-stage`). The smear, distortion,
       PSF, and ghost sub-stages live in the sibling modules ``smear.py``,
       ``distortion.py``, ``psf.py``, and ``ghosts.py``.
   * - ``detector/``
     - The detector stage as a package: the CCD electron unit chain
       (``chain.py``), the resolved-parameter view (``params.py``), and the
       stochastic / structured noise sub-effects (``noise_stages.py`` -- dark,
       hot pixels, banding, bias structure, cosmic rays). Carries the vidicon
       DN path and the calibrated-I/F inversion (see :ref:`sim-detector-stage`).
   * - ``telemetry.py``
     - Transmission loss; carries the per-pixel missing-data markers today.
   * - ``atmosphere.py``
     - Reserved slot for haze-limb (Titan-class) rendering; currently empty.
   * - ``artifacts_catalog.py``
     - Per-instrument PSF, distortion-residual, and detector-chain defaults the
       ``instrument_defaults`` switch turns on (see
       :ref:`sim-artifacts-catalog`).

**The navigator side** predicts what the scene *should* look like, exactly the
way the SPICE-backed models predict a real frame. Its renderers live under
``nav_model/`` so the package layout states who owns them:
:mod:`spindoctor.nav_model.sim_body` and :mod:`spindoctor.nav_model.sim_ring`
draw the predicted templates that
:class:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated` and
:class:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`
emit as features, and
:class:`~spindoctor.nav_model.stars.nav_model_stars_simulated.NavModelStarsSimulated`
builds its catalog star records directly.

.. _sim-information-boundary:

The information boundary
------------------------

Every key in the scene schema is classified as either **idealized** --
information the production pipeline could know from catalogs, SPICE, labels, or
configuration: instrument identity, image size, exposure, body ellipsoid/mesh
geometry and pose, ring orbits and epochs, star catalog positions and
magnitudes -- or **truth** -- nature's values and the test's contaminants: the
planted pointing offset and roll, the RNG realization, the noise and
stray-light blocks, crater terrain, per-star PSF anomalies, and the
``nav_override`` channel. :data:`spindoctor.sim.scene.TRUTH_KEYS` is the
machine-readable truth inventory; an import-time assertion in
:mod:`spindoctor.sim.scene` fails if any schema key is left unclassified, so a
schema change cannot dodge the classification.

The boundary is enforced structurally, not by convention:

- :class:`~spindoctor.obs.obs_inst_sim.ObsSim` exposes the navigator-side
  models a **filtered view**, ``obs.nav_params``, built by
  :func:`spindoctor.sim.scene.build_nav_params`: every truth key is stripped
  (the filter is default-deny -- an unclassified or unknown key stays behind),
  a body's ``nav_override`` mapping is overlaid onto its idealized view and
  then dropped (the navigator sees what it *believes*, never the true values
  underneath), and all values are deep copies. The renderer consumes the full
  scene; the navigator side structurally cannot read what is not there.
- The navigator-side models read **only** ``obs.nav_params``. None of the
  renderer's output -- rendered star records, body masks, z-order maps, all
  accumulated on the frame's ``truth`` metadata -- ever crosses.
- Navigator-side quantities that depend on the detector are derived from the
  emulated instrument's **published** configuration, never from the scene's
  truth-side blocks. The star detection limit
  (:meth:`~spindoctor.obs.obs_inst_sim.ObsSim.star_max_usable_vmag`) is the
  worked example: it comes from the resolved per-instrument config block
  (including any idealized ``instrument_config`` overrides) and is independent
  of the scene's ``noise`` block, so a scene that plants noise different from
  the published values produces an honestly-wrong detection limit -- planted
  model error, exactly what such a scene is for.
- ``tests/spindoctor/sim/test_information_boundary.py`` constructs a scene
  exercising every ``TRUTH_KEYS`` entry and asserts none is reachable through
  the filtered view. This test is the independence guarantee: any change that
  adds a truth key must extend it in the same change (the test iterates the
  frozenset, so an unextended sample table fails loudly).

Sharing code is fine; sharing information is not
------------------------------------------------

Both sides deliberately call the same geometry helpers --
:mod:`spindoctor.sim.ellipsoid_geometry`, :mod:`spindoctor.sim.mesh_geometry`,
:mod:`spindoctor.sim.ring_geometry`, and :mod:`spindoctor.sim.star_records`.
This is not a boundary leak; it is what makes the measurement clean. With
shared conventions (pixel centers, the sign of ``dv``, edge rasterization,
record defaults) the planted error is the *only* error in a recovery
measurement. Independent implementations would each carry their own
conventions, and any delta between them would land as an unknown systematic
inside the measured error -- contaminating the truth reference every simulator
result rests on. The discipline the shared helpers follow: they never read the
scene dict and never accept a truth key. Most take explicit geometry
arguments; the two record parsers
(:func:`spindoctor.sim.star_records.star_record_from_params` and
:func:`spindoctor.sim.mesh_geometry.mesh_spec_from_params`) parse per-object
mappings restricted to idealized keys, and on the navigator side they only
ever receive entries that have already passed the boundary filter. Either
way, truth cannot be smuggled across.

The honest residual: a bug in a shared helper that makes rendered features
subtly unlike *real* features leaves both sides consistent and recovery clean
-- scenes wrong, measurement blind. Its detectors are the reviewed render-diff
contact sheet (see :ref:`sim-catalog-workflow`) and the per-instrument realism
comparison against real cohorts, not any structural test.

One working rule follows from the blind spot. When implementing or extending
the image-side renderers in ``sim/forward/``, derive behavior from instrument
references and physics, **not** by consulting the navigator-side predicted
renderers. Copying the navigator's rendering choices manufactures exactly the
shared assumption the boundary cannot detect; the shared geometry helpers are
the sanctioned channel for conventions that must agree, and anything beyond
them agreeing is evidence, not a goal.

.. _sim-stage-pipeline:

The stage pipeline
==================

The forward model renders in a fixed physical order that does not vary per
scene (:mod:`spindoctor.sim.forward.pipeline`):

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Stage name
     - Callable
     - What it does today
   * - ``scene_radiance``
     - :func:`~spindoctor.sim.forward.scene_radiance.compose_scene_radiance`
     - Composes the noise-free signal: background stars, catalog stars, then
       bodies and rings depth-sorted far to near (nearer objects overwrite),
       with the planted offset and camera roll applied. Accumulates feature
       truth (star records, masks, inventory, z-order) into ``frame.truth``.
   * - ``optics``
     - :func:`~spindoctor.sim.forward.optics.apply_optics`
     - Applies the optical-path effects in a fixed internal order -- motion
       smear, residual distortion, the whole-scene PSF, ghost reflections, then
       the stray-light field -- each contributing only when its ``optics``
       sub-block is present (see :ref:`sim-optics-stage`).
   * - ``downsample``
     - :func:`~spindoctor.sim.forward.stages.downsample_to_detector`
     - Box-downsamples the oversampled planes to the detector grid. A scene
       with an active PSF renders on a 4x oversampled grid and downsamples here;
       with no PSF the factor is 1 and this is a no-op.
   * - ``detector``
     - :func:`~spindoctor.sim.forward.detector.apply_detector`
     - Runs the electron unit chain: converts the composed [0, 1] signal to
       electrons through the exposure, applies Poisson shot noise, full-well
       bloom, read noise, banding, gain to DN, bias structure, quantization, and
       the ADC clip (see :ref:`sim-detector-stage`). The Voyager vidicon takes
       the DN-domain path; a calibrated instrument inverts the calibration
       transform afterward.
   * - ``telemetry``
     - :func:`~spindoctor.sim.forward.telemetry.apply_telemetry`
     - Applies the per-pixel missing-data markers.

Each stage is a callable matching the
:class:`~spindoctor.sim.forward.stages.Stage` protocol: it mutates a
:class:`~spindoctor.sim.forward.stages.SimFrame` in place, reads its parameters
from the full scene mapping, and draws randomness only from the
``numpy.random.Generator`` it is handed. Every effect is **off unless asked
for**: with no ``stray_light`` block no field is added, with no
``missing_data_rate`` no pixels are dropped, and with no ``noise``, ``detector``,
or ``artifacts`` block the detector adds no shot noise, read noise, dark, hot
pixels, banding, or bias structure. The detector always converts the composed
signal to detector counts -- that is what makes a DN frame -- but the stochastic
and structured noise activates only through a scene ``noise`` block (which pins
the individual parameters, e.g. ``poisson: true`` with a ``read_noise_dn``) or
the ``artifacts: {instrument_defaults: true}`` opt-in (which turns on the
emulated camera's physical signal chain at catalog values). A scene with none of
those blocks renders a clean DN frame: the self-consistency floor. A
single-variable sweep therefore relies on every other block staying absent so it
attributes error to the one effect it varies.

``SimFrame`` carries the mutable image state between stages:

- ``signal`` -- the ``(V*os, U*os)`` float64 image. It holds normalized
  [0, ~1] intensive scene units through the radiance and optics stages; the
  detector stage converts it to electrons through the exposure and digitizes it
  to DN in place (the electron unit chain).
- ``point_e`` -- a same-shaped plane reserved for point sources in electrons.
  The detector adds it into the electron image after the intensive conversion
  and before Poisson, so point sources never pass through the signal scale.
  Star deposition into this plane is a later fidelity step; today the radiance
  stage still draws stars PSF-spread in signal units, so the plane stays zeroed.
- ``oversample`` -- the oversampling factor. A scene with an active PSF (an
  ``optics.psf`` block or ``instrument_defaults``) renders the radiance on a 4x
  oversampled grid and the box downsample returns it to the detector grid;
  otherwise it is 1.
- ``truth`` -- renderer output metadata (rendered star records, body masks,
  inventory, z-order maps). None of it crosses the information boundary.

**Per-stage seeding.** Each stage receives its own generator seeded via
:func:`spindoctor.sim.seeds.derive_effect_seed` from the scene's single
``random_seed`` and the stage's registry name, so one stage's noise realization
is independent of which other stages are enabled, and adding a stage later
leaves existing stages' realizations unchanged. A stage name is therefore part
of its scenes' noise realization: renaming a stage reseeds every scene that
exercises it and regenerates the affected baselines -- allowed, but not to be
done idly. (The seed derivation is a stable digest, never Python's per-process
salted ``hash``.) Sub-effects inside a stage derive their own named streams the
same way (background stars, cosmic rays, crater placement).

**The driver.** :func:`spindoctor.sim.render.render_combined_model` takes a
``sim_params`` dict and returns ``(image, metadata)``. It normalizes the
parameters into a deterministic JSON cache key, runs the pipeline on a fresh
frame, and caches the result (``lru_cache(maxsize=1)``, plus small shape caches
inside the body/mesh/star renderers; ``clear_render_caches`` drops them all for
determinism tests). The caching contract: scene parameters are
JSON-serializable scalars/lists/maps, no wall-clock or global RNG state is
consulted, and two renders of the same scene are bit-identical on one machine.

.. _sim-instruments:

**Instruments.** The ``instrument`` field selects which per-instrument
configuration block drives the detector model and PSF
(:mod:`spindoctor.sim.instruments`). The recognized names are ``coiss_nac``,
``coiss_wac``, ``coiss_calib_nac``, ``coiss_calib_wac``, ``gossi``, ``nhlorri``,
and ``vgiss``, plus ``generic`` (alias ``sim``) for the instrument-agnostic
defaults. Calibrated (``*_calib_*``) instruments are in I/F units with a NaN
missing-data marker and no full-well; raw instruments are in DN with a 0 marker.
Calibrated scenes render through the full DN chain and then invert the
calibration transform, so they carry propagated shot/read noise and quantization
texture in I/F units. A scene can pin or override individual instrument settings
with
``instrument_config`` (see :ref:`sim-instrument-config`).

.. _sim-optics-stage:

The optics stage
================

The optics stage runs on the oversampled radiance image in a fixed internal
order that mirrors image formation
(:func:`~spindoctor.sim.forward.optics.apply_optics`). Each sub-stage
contributes only when its ``optics`` sub-block is present, so a scene names
exactly the optical effects it wants and leaves the rest at the floor.

1. **Motion smear** (:mod:`~spindoctor.sim.forward.smear`) averages the
   radiance over the exposure along the pointing drift. It runs first, while
   the per-object-class radiance layers are still separable, so the optics
   below form the image of the time-averaged radiance. A single
   ``object_class: all`` entry smears the whole scene; several entries give
   differential smear, where each class carries its own drift vector.
2. **Residual distortion** (:mod:`~spindoctor.sim.forward.distortion`) warps
   the geometric image by the low-order radial polynomial (``k1``, ``k2`` about
   the optical centre) plus an optional seeded non-radial wander -- the field
   error left after the navigator applies each camera's known distortion model.
   A limb fitted at the frame edge then disagrees with a ring fitted through the
   centre by the differential residual, which the navigator gets no model to
   remove.
3. **Whole-scene PSF** (:mod:`~spindoctor.sim.forward.psf`) convolves the image
   by a core-Gaussian-plus-Moffat-wing kernel, so the limb gradient, the
   ring-edge gradient, and every star shape inherit one profile. The core is
   elliptical (``sigma_v`` may differ from ``sigma_u``); ``w`` is exactly the
   fraction of the kernel's energy in the isotropic wing, and each term is
   normalized so the kernel conserves flux. The Cassini cameras get a wider
   truncation window for their documented long wings.
4. **Ghost reflections** (:mod:`~spindoctor.sim.forward.ghosts`) add displaced,
   defocused, low-amplitude copies of the formed focal-plane image (internal
   reflections). Each ghost copies the pre-ghost signal, so ghosts do not
   reflect one another.
5. **Stray light** adds the smooth low-frequency scattered-light field (a linear
   ramp or a radial bump) last, before the detector stage. It is additive, so it
   brightens dark sky as well as lit features; the navigator's source-image
   background filter is meant to remove it.

Only the distortion non-radial field draws randomness, and it derives its own
seeded stream from the scene seed, so the optics stage does not consume the
pipeline generator. A sub-stage whose block is a true no-op (all-zero radial
distortion, an amplitude-0 ghost) renders bit-identically to one without it.

.. _sim-detector-stage:

The detector stage
==================

The detector stage (:mod:`spindoctor.sim.forward.detector`, a package) is the
normative unit chain of the forward model. For a CCD instrument the composed
intensive signal passes through the electron unit chain in
:func:`~spindoctor.sim.forward.detector.chain.apply_detector`:

1. **Signal to electrons.** The intensive [0, 1] signal is scaled to electrons
   through ``signal_full_scale_frac * full_well_e * (exposure_sec /
   exposure_ref_sec)``.
2. **Point sources.** The point-source electron plane (``frame.point_e``) is
   added after the intensive conversion and before Poisson, so point sources
   never pass through the signal scale. (Star deposition into ``point_e`` is a
   later fidelity step; today the radiance stage still draws stars PSF-spread in
   signal units, so the plane stays zeroed.)
3. **Dark current, then Poisson.** A dark pedestal accumulates over the
   exposure before the shot term, so the shot noise grows with the dark signal.
   Poisson shot noise then acts on the electron image.
4. **Hot pixels and full-well bloom.** Hot pixels are stamped, then charge above
   ``full_well_e`` spills along the column up to ``bloom_length`` pixels each
   way and the image is capped at the well. A camera's physical saturation
   therefore emerges from ``full_well_e / gain_e_per_dn`` (below the ADC ceiling
   for Cassini), not from the ADC clip.
5. **Cosmic rays and read noise.** Cosmic rays deposit above the well after the
   bloom cap, so they reach the ADC ceiling and land on the orchestrator's
   masks. Gaussian read noise is added in electrons.
6. **Banding, gain to DN, bias structure.** Coherent banding is added in
   electrons; the image is divided by ``gain_e_per_dn`` and offset by the bias
   pedestal into DN; low-order bias structure (per-image pedestal jitter,
   row/column gradients) is added in DN.
7. **Quantize and clip.** The DN image is quantized by the selected ADC sub-mode
   and clipped at ``saturation_dn``.

The stochastic and structured sub-effects (dark, hot pixels, banding, bias
structure, cosmic rays) live in
:mod:`~spindoctor.sim.forward.detector.noise_stages`; the chain draws each one
an independent RNG stream from the scene seed, so toggling one never perturbs
another's realization. The deterministic signal-to-DN conversion always runs --
it is what makes a DN frame -- while each stochastic sub-effect is a no-op when
its gating amplitude, fraction, or rate is zero.

**Quantization sub-modes.** ``exact`` rounds to integer DN (uniform bins);
``8bit`` bins for an 8-bit ADC; ``uneven_12bit`` snaps values near the
power-of-two carry boundaries to reproduce the histogram spikes of an ADC with
unequal bit weights; ``sqrt_lut`` companding encodes to 8 bits through a
square-root LUT and back, leaving a signal-dependent quantization residual.

**The vidicon path.** The Voyager vidicon
(:func:`~spindoctor.sim.forward.detector.chain.apply_detector`, vidicon branch)
is not photon-noise dominated, so it skips the electron conversion: the signal
maps straight to the 8-bit DN full scale and the noise is applied in DN -- a
line-correlated read-noise term (a per-line offset plus a within-line white
component) and a faint vertical coherent periodic component -- then 8-bit
quantization.

**The calibrated-I/F path.** A calibrated (``coiss_calib_*``) instrument renders
through the full DN chain and then inverts the calibration transform: the bias
and dark pedestals are subtracted before the exposure divide (matching the real
pipeline), so a calibrated frame carries no spurious 1/exposure pedestal and
comes out in I/F units with propagated shot / read noise and quantization
texture. The calibration scale is derived so a noise-free signal of 1.0 at the
reference exposure round-trips to I/F 1.0.

:func:`~spindoctor.sim.forward.detector.params.resolve_detector_params`
collapses the emulated instrument's config block, the per-instrument catalog
defaults, the scene ``detector`` / ``noise`` blocks, and the
``artifacts.instrument_defaults`` switch into one flat
:class:`~spindoctor.sim.forward.detector.params.DetectorParams` view the chain
reads. Resolution precedence, highest first: an explicit scene key (``detector``
block, then ``noise`` block), then the catalog value when ``instrument_defaults``
is on, then the disabled floor -- the physical-chain artifacts default to zero,
so an unconfigured scene renders a clean DN frame. A scene that selects a gain
state the instrument does not catalogue is a validation error, not a silent
guess.

.. _sim-artifacts-catalog:

The instrument-defaults switch and the artifacts catalog
========================================================

``artifacts: {instrument_defaults: true}`` opts the whole scene into the
emulated camera's physical signal chain at catalog values: the whole-scene PSF
kernel and the residual-distortion amplitude
(:func:`~spindoctor.sim.forward.optics.instrument_defaults_on` gates the optics
side), and the detector electron-chain read noise, dark current, hot pixels,
banding, and bias structure (the detector side reads the same switch). It turns
on only the *physical* chain: the per-mode loss incidences (whole-line
telemetry loss, compression-block dropouts, and the like) stay at zero, so a
defaults scene is a clean-but-realistic frame, not a damaged one.

:mod:`spindoctor.sim.forward.artifacts_catalog` is the single home for those
per-instrument values: ``PSF_KERNELS`` (core sigma and wing parameters),
``DISTORTION_RESIDUAL_RMS_PX`` (the residual field error), and
``DETECTOR_DEFAULTS`` (the full electron-chain, gain-table, read-noise, and
vidicon numbers). Every value is provenance-tagged in a comment beside it, and
every value is interim -- sized from published FWHMs, gain tables, and
documented residual-error bounds, pending the per-instrument measurement passes
-- so the wing parameters and noise amplitudes are the first quantities the
realism-match pass revisits. The calibrated Cassini instrument names alias their
raw entries, and ``generic`` (alias ``sim``) is an instrument-agnostic ideal
12-bit detector whose electron well equals its DN depth at unit gain.

.. _sim-floor:

The self-consistency floor
==========================

With no ``optics``, ``noise``, ``detector``, or ``artifacts`` block a scene
renders the *self-consistency floor*: the detector converts the composed signal
to DN and nothing else acts -- no PSF, no shot or read noise, no dark, hot
pixels, banding, bias structure, distortion, smear, ghosts, or stray light. Each
effect is off unless asked for, and a single-variable sweep relies on every
other block staying absent so it attributes error to the one effect it varies.
The floor's matching PSF configuration is ``optics: {psf: {match_navigator:
true}}``, which the validator resolves into the navigator's own model -- a pure
Gaussian at the emulated instrument's ``star_psf_sigma``, no Moffat wing, no
field variation -- so the image-side PSF equals the navigator-side PSF exactly
and the only residual is the one the scene plants elsewhere.

.. _sim-perf-budget:

The render performance budget
=============================

A 512x512 scene with a whole-scene PSF plus the full detector stack at
oversample 4 must render in under 2 s single-core, and a 1024x1024
Cassini-class scene in under 8 s (``tests/integration/test_sim_perf.py``). The
budget is a *cold-render* budget: the render caches are cleared so the timed
render pays the kernel-build and compile costs a first render pays. The harness
pins itself: it sets the process CPU affinity to one core and caps the
BLAS/OpenMP thread-count environment variables for the duration, so an unpinned
numpy FFT cannot silently multithread and fake the budget. Under heavy machine
load the timed render can exceed the budget purely from contention; a failure is
reported and investigated, not blessed by raising the budget.

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
   ``range_km``.

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

.. _sim-scene-formats:

Scene format (schema version 2)
===============================

A scene is a single YAML file whose fields are the flat runtime ``sim_params``
names the renderer consumes, so a validated scene file *is* the ``sim_params``
dict with no translation layer. The current ``schema_version`` is **2**; the
validator rejects any other version, and any file that turns up with an older
version is converted, not accommodated.

The scene catalog (:mod:`spindoctor.sim.scene`) is the durable test artifact, laid out
as ``tests/integration/sim_scenes/<scene_class>/<scene_name>.yaml`` (the
directory is the registry). :func:`spindoctor.sim.scene.load_sim_scene` parses and
validates a file and returns the flat ``sim_params`` dict the renderer consumes;
the GUI's "Save Scene (YAML)" / "Load Scene (YAML)" buttons read and write the
same format via :func:`spindoctor.sim.scene.save_sim_scene`; and programmatic
scene authors -- the calibration-campaign generator
``util/calibration/scene_gen.py`` and the doc-gallery definitions in
``tests/integration/sim_doc_images.py`` -- validate their in-memory dicts
through the same core, :func:`spindoctor.sim.scene.validate_sim_params`.
Validation is strict: an unknown key at the top level or inside a ``bodies`` /
``rings`` / ``stars`` entry fails loudly, so a typo cannot silently render the
default scene.

The scene classes (for example ``algorithmic_invariants``,
``phase_sweep_regular_body``, ``phase_sweep_irregular_body``, ``range_sweep``,
``noise_sweep``, ``multi_body_geometry``, ``regression``) scope what each scene
is testing and are enforced by the structural test. The scene README at
``tests/integration/sim_scenes/README.txt`` documents the schema alongside the
code.

A complete YAML scene -- a noisy Cassini NAC frame with one irregular mesh body, a
ring, a couple of stars, and a planted offset the navigator must recover -- reads:

.. code-block:: yaml

   schema_version: 2
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

**Boundary classification.** Every key carries an information-boundary side
(see :ref:`sim-information-boundary`): *idealized* keys reach the navigator
through ``obs.nav_params``; *truth* keys are readable only by the image-side
renderer. The tables below state each key's side. A key added to the schema
without a classification fails the import-time completeness assertion in
:mod:`spindoctor.sim.scene`, so every schema change must extend the boundary in
the same change.

Scene parameter reference
=========================

Top-level fields
----------------

.. list-table::
   :widths: 22 11 11 10 46
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Side
     - Meaning
   * - ``size_v`` / ``size_u``
     - int
     - required
     - idealized
     - Image height and width in pixels.
   * - ``instrument``
     - str
     - required
     - idealized
     - Emulated instrument; selects the detector model and PSF (see
       :ref:`Instruments <sim-instruments>`).
   * - ``random_seed``
     - int
     - required
     - truth
     - Scene seed; per-stage and per-effect sub-seeds derive from it.
   * - ``exposure_sec``
     - float
     - 1.0
     - idealized
     - Exposure time; scales the cosmic-ray count.
   * - ``offset_v`` / ``offset_u``
     - float
     - 0.0
     - truth
     - Planted pointing offset (px) applied to all bodies, rings, and stars.
   * - ``offset_rotation_deg``
     - float
     - 0.0
     - truth
     - Planted boresight roll (deg) applied about the image center.
   * - ``midtime_utc``
     - str
     - none
     - idealized
     - Observation midtime the snapshot reports (label knowledge).
   * - ``closest_planet``
     - str
     - none
     - idealized
     - Planet context the snapshot reports.
   * - ``time`` / ``ring_epoch``
     - float
     - 0.0
     - idealized
     - Observation time and orbit epoch for ring-edge precession
       (``rate_peri`` applies across ``time - ring_epoch``); catalog knowledge
       both sides read.
   * - ``fit_camera_rotation``
     - bool
     - none
     - idealized
     - Overrides the instrument's camera-rotation-fit setting for this scene.
   * - ``bodies``
     - list
     - ``[]``
     - idealized
     - Per-body parameter dicts (see :ref:`sim-body-params`); some per-body
       keys are truth-side.
   * - ``rings``
     - list
     - ``[]``
     - idealized
     - Per-ring parameter dicts (see :ref:`sim-ring-params`).
   * - ``stars``
     - list
     - ``[]``
     - idealized
     - Explicit star dicts (see :ref:`sim-star-params`); ``psf_sigma`` is
       truth-side.
   * - ``background_stars_num``
     - int
     - 0
     - truth
     - Random background-star count (0-1000). Background stars are
       contaminants: the navigator receives no catalog for them.
   * - ``background_stars_psf_sigma``
     - float
     - none
     - truth
     - PSF sigma of the background-star field.
   * - ``background_stars_distribution_exponent``
     - float
     - none
     - truth
     - Brightness power-law exponent of the background-star field.
   * - ``noise``
     - dict
     - off
     - truth
     - Detector-noise block (see :ref:`sim-noise`).
   * - ``oversample``
     - int
     - auto
     - truth
     - Pins the render oversampling factor; omit to let the renderer choose (4
       with an active PSF, else 1).
   * - ``optics``
     - dict
     - off
     - truth
     - Optical-path block: PSF, smear, distortion, ghosts, stray light (see
       :ref:`sim-optics`).
   * - ``spk_error``
     - dict
     - off
     - truth
     - Planted spacecraft-ephemeris parallax error (see :ref:`sim-spk-error`).
   * - ``detector``
     - dict
     - off
     - truth
     - Detector-chain override: gain state, detector model, exposure reference
       (see :ref:`sim-detector-block`).
   * - ``artifacts``
     - dict
     - off
     - truth
     - Physical-chain opt-in switch (see :ref:`sim-artifacts-block`).
   * - ``shade_solid_rings``
     - bool
     - none
     - truth
     - Image-side ring-appearance knob (the navigator's ring template is
       always solid-shaded by its own convention).
   * - ``instrument_config``
     - dict
     - none
     - idealized
     - Per-instrument config overrides (see :ref:`sim-instrument-config`).

.. _sim-body-params:

Body parameters
---------------

Each entry of ``bodies`` is a dict. The geometry, pose, lighting, and physical
scale are idealized (the published shape model of a body is catalog knowledge;
a scene plants shape error through ``nav_override``, not by hiding the mesh).
Common fields:

.. list-table::
   :widths: 22 11 11 10 46
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Side
     - Meaning
   * - ``name``
     - str
     - generated
     - idealized
     - Body label used in metadata and annotations.
   * - ``center_v`` / ``center_u``
     - float
     - frame center
     - idealized
     - Body center in pixels.
   * - ``axis1`` / ``axis2`` / ``axis3``
     - float
     - 0.0
     - idealized
     - Full extents of the three body axes in pixels (``axis3`` defaults to
       ``min(axis1, axis2)``).
   * - ``illumination_angle``
     - float (deg)
     - 0.0
     - idealized
     - Image-plane light azimuth (0 = from the top).
   * - ``phase_angle``
     - float (deg)
     - 0.0
     - idealized
     - Phase angle (0 = fully lit, 180 = back-lit crescent).
   * - ``range_km``
     - float
     - body index
     - idealized
     - Subject distance in km; also the depth-ordering key (smaller renders
       in front).
   * - ``km_per_pixel``
     - float
     - none
     - idealized
     - Optional physical scale at the limb.
   * - ``shape_model``
     - str
     - ``ellipsoid``
     - idealized
     - ``ellipsoid`` or ``polyhedral_mesh``.

Ellipsoid bodies add ``rotation_z`` and ``rotation_tilt`` (degrees, idealized).
Mesh bodies (``shape_model: polyhedral_mesh``) instead read the idealized mesh
keys ``mesh_lumpiness`` (relief amplitude as a fraction of the unit radius,
default 0.3), ``mesh_seed`` (which irregular shape is generated, default 0),
``mesh_n_lat`` / ``mesh_n_lon`` (mesh resolution, defaults 16 / 32), and
``pose_euler_deg`` (intrinsic X, Y, Z Euler angles, default ``[0, 0, 0]``).

The truth-side body keys never reach the navigator:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Field (truth)
     - Meaning
   * - ``crater_fill``, ``crater_min_radius``, ``crater_max_radius``,
       ``crater_power_law_exponent``, ``crater_relief_scale``
     - Procedural crater terrain -- nature's surface, which the navigator's
       smooth predicted template does not know.
   * - ``seed``
     - The crater realization (a stable hash of the body geometry when
       absent).
   * - ``anti_aliasing``
     - Image-side rendering-fidelity knob; the predicted template always
       renders at full anti-aliasing.
   * - ``nav_override``
     - Mapping overlaid on the body's *idealized view* by the boundary
       filter, then dropped (see below). Only idealized body keys may appear
       in it.

.. note::

   **Axis convention.** Both body renderers use the same mapping: ``axis1`` is
   the vertical (``v``) extent and ``axis2`` the horizontal (``u``) extent. An
   ellipsoid and a mesh declared with identical ``axis1``/``axis2`` are oriented
   the same way, so the two are interchangeable for the same geometry.

**Render geometry vs navigation geometry.** By default the navigator predicts
the body from the same idealized parameters the renderer drew, so it knows the
geometry exactly. A body may carry an optional ``nav_override`` mapping that
the boundary filter (:func:`spindoctor.sim.scene.build_nav_params`) overlays
onto the body's idealized view before the navigator sees it -- the renderer
ignores it and always draws the true geometry, and the overridden true values
never cross. This separates the *render geometry* (ground truth) from the
*navigation geometry* (what the navigator assumes): the channel the
irregular-body scenarios use to render a lumpy mesh but predict its smooth
(ellipsoidal) limit, or to predict the same body at a disagreeing pose. The
override never moves the center, so the planted offset is still recoverable.
See :doc:`dev_guide_navigation_models_body_simulated`.

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
rate, deg/day, applied across the scene-level ``time`` minus ``ring_epoch``).

All ring keys are idealized at the current fidelity: the mode-1 orbits *are*
the catalog orbits, with no planted per-feature error.

.. _sim-star-params:

Star parameters
---------------

Each entry of ``stars`` is a dict with ``name`` and an optional
``catalog_name``, a ``v`` / ``u`` position, a ``vmag`` (visual magnitude; lower
is brighter), an optional ``spectral_class``, an optional per-star smear vector
``move_v`` / ``move_u``, and an optional PSF fitting-window size ``psf_size``
(a two-integer list). All of those are idealized -- they are the catalog and
instrument knowledge a real pipeline has. The one truth-side star key is
``psf_sigma``: a per-star PSF width override is an anomaly of the rendered
image, and the navigator knows only the instrument's published PSF.

Random background stars are added by the truth-side top-level keys
``background_stars_num``, ``background_stars_psf_sigma``, and
``background_stars_distribution_exponent``.

Stars are rendered with a half-pixel PSF-evaluation offset so a star's
brightness centroid lands exactly at its predicted ``(v, u)``, which keeps star
navigation free of a constant half-pixel bias.

.. _sim-noise:

Detector-noise block
--------------------

The optional ``noise`` dict (truth-side) pins the truth-side detector noise the
scene plants. The detector stage consumes most of it; the missing-data markers
are applied by the telemetry stage. Each field is off at the floor unless the
block sets it (or ``artifacts.instrument_defaults`` turns on the catalog chain).

.. list-table::
   :widths: 30 12 12 46
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``poisson``
     - bool
     - False
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

.. _sim-optics:

Optics block
------------

The optional ``optics`` dict (truth-side) carries the optical-path sub-blocks
the optics stage applies in its fixed internal order (see
:ref:`sim-optics-stage`). Every sub-block is optional and absent by default; a
sub-block that is present but a true no-op renders bit-identically to one that
is absent.

``optics.psf`` -- the whole-scene point-spread function. Either the explicit
core-plus-wing form or the exclusive navigator-matched form:

.. list-table::
   :widths: 24 12 12 52
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``sigma_v`` / ``sigma_u``
     - float
     - required
     - Gaussian core sigma along v / u, in detector pixels.
   * - ``w``
     - float
     - 0.0
     - Moffat wing energy fraction, in [0, 1].
   * - ``r0``
     - float
     - 2.0
     - Moffat core radius in detector pixels.
   * - ``n``
     - float
     - 3.0
     - Moffat index.
   * - ``match_navigator``
     - bool
     - --
     - Exclusive alternative: resolve the PSF to the navigator's own Gaussian
       at the instrument ``star_psf_sigma`` (no other PSF key may appear).

``optics.smear`` -- a list of per-object-class motion-smear entries. Each entry
has ``dv_px`` / ``du_px`` (drift in pixels) and ``object_class`` (one of
``all``, ``stars``, ``bodies``, ``rings``; default ``all``). One ``all`` entry
smears the whole scene; several give differential smear.

``optics.distortion`` -- the residual geometric distortion:

.. list-table::
   :widths: 24 12 12 52
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``k1`` / ``k2``
     - float
     - 0.0
     - Radial polynomial coefficients about the optical centre.
   * - ``center_v`` / ``center_u``
     - float
     - frame centre
     - Optical-centre position in pixels.
   * - ``nonradial_rms_px``
     - float
     - 0.0
     - RMS amplitude of the seeded smooth non-radial wander, in pixels.

``optics.ghosts`` -- a list of ghost reflections. Each entry has ``dv_px`` /
``du_px`` (offset in pixels), ``amplitude`` (fraction; 0 disables the entry),
and ``defocus_sigma`` (blur sigma in pixels).

.. _sim-stray-light:

``optics.stray_light`` -- the smooth scattered-light field applied last, before
the detector stage: ``amplitude`` (peak fraction of full scale; 0 disables it),
``direction_deg`` (ramp direction for the ``linear`` model), ``model``
(``linear`` ramp or ``radial`` bump), and ``center_v`` / ``center_u`` (the
radial-model bump centre; omit for the frame centre). It exercises the
navigator's source-image background filter.

.. _sim-spk-error:

Spacecraft-ephemeris error block
--------------------------------

The optional ``spk_error`` dict (truth-side) plants a spacecraft-ephemeris
parallax error: ``dv_px`` / ``du_px`` (the displacement at the reference range)
and ``reference_range_km`` (the range the displacement is quoted at). The
parallax displacement scales as ``reference_range_km / range_km`` per object, so
a scene that sets ``spk_error`` must give every body and ring a physical
``range_km`` -- the validator enforces this.

.. _sim-detector-block:

Detector block
--------------

The optional ``detector`` dict (truth-side) overrides the resolved detector
chain: ``gain_state`` (the electron-chain gain state, which must be catalogued
for the instrument), ``detector_model`` (``ccd`` electron chain or ``vidicon``
DN chain), ``exposure_ref_sec`` (the exposure the signal full-scale fraction
references), and ``quantization`` (the ADC sub-mode: ``exact``, ``8bit``,
``uneven_12bit``, or ``sqrt_lut``). Omitted keys fall back to the instrument's
catalog defaults.

.. _sim-artifacts-block:

Artifacts block
---------------

The optional ``artifacts`` dict (truth-side) carries the physical-chain opt-in
switch ``instrument_defaults`` (see :ref:`sim-artifacts-catalog`). With it on,
the emulated camera's catalog PSF, distortion residual, and detector noise chain
render at their per-instrument values; the per-mode loss incidences stay at
zero. Leaving it off keeps those keys absent (the self-consistency floor).

.. _sim-instrument-config:

Instrument-config overrides
---------------------------

The optional ``instrument_config`` dict is deep-merged over the resolved
per-instrument block, so a scene can pin individual settings (PSF sigma, noise,
data units, extfov margin) without tracking later camera-config edits. Omit it to
inherit everything; name ``generic`` and override everything to fully
self-specify. The top-level ``noise`` block still wins over
``instrument_config.noise`` for rendering. See :doc:`dev_guide_observations`.

.. _sim-catalog-workflow:

Catalog, baselines, and the render-diff sheet
=============================================

The scene catalog is guarded at three levels, all regenerated inside whichever
change alters rendered output:

- **Structural tests** (``tests/integration/test_sim_scenes.py``): every scene
  validates, sits in a declared class, has a unique name, and renders.
- **Regression baselines** (``tests/integration/sim_baselines/``): every
  catalog scene re-navigates to its recorded rounded outcome
  (``test_sim_baselines.py``, integration tier); regenerate with
  ``python -m tests.integration.update_sim_baselines`` and review the diff.
- **The render-diff contact sheet**
  (``python -m tests.integration.render_contact_sheet``): one
  before / after / amplified-absolute-difference panel per scene, composed into one
  sheet PNG per scene class under ``tests/integration/render_diffs/``, with
  the per-scene current renders committed under ``render_diffs/current/`` (the
  *before* images of the next regeneration). Any change that alters rendered
  output regenerates the sheets and the ``current/`` PNGs in the same change,
  and the sheet is reviewed panel by panel. The review criterion: *the scene
  still renders what it asks for -- same ingredients, same geometry, same
  planted truth; differences confined to discretization and reseeding.* This
  review exists because a converted scene can recover its planted offset and
  still be rendering the wrong thing; the recovered offset cannot catch that,
  only eyes on the render can.

The two documentation galleries (``docs/dev_guide/_sim_images/`` and
``docs/simulator_report/_scene_images/``, both written by
``python -m tests.integration.sim_doc_images``) are re-rendered under the same
rule.

The simulated-image GUI
=======================

``sd_create_simulated_image`` is a PyQt6 editor for the same parameters, with a
live preview. Launch it with:

.. code-block:: bash

   sd_create_simulated_image

The editor is the :mod:`spindoctor.cli.sim_editor` package: a single
``QMainWindow`` (``CreateSimulatedImageModel``) composed from one mixin per
schema block, so each block's controls -- and any control group a later
realism addition needs -- live in their own module:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Module
     - Controls / role
   * - ``main_window.py``
     - Assembles the mixins; owns the cross-cutting scaffolding (data model,
       render scheduling, window layout).
   * - ``global_fields.py``
     - Image size, planted offset and camera roll, exposure, seed, instrument
       selector, camera-rotation-fit override, midtime, closest planet.
   * - ``noise.py``
     - Poisson, read noise, bias, cosmic-ray rate, missing-data rate, bloom,
       signal full-scale fraction, pixel area (General tab).
   * - ``optics_tab.py``
     - The Optics tab: PSF, motion smear, distortion, ghosts, the relocated
       stray-light panel, oversample, and the spacecraft-ephemeris error group.
       Each optical sub-block is a checkable group under absent-key discipline.
   * - ``artifacts_tab.py``
     - The Artifacts tab: the instrument-defaults switch and the detector
       override (gain state, detector model, exposure reference).
   * - ``stray_light.py``
     - Stray-light amplitude, direction, model, radial center (its panel is
       hosted in the Optics tab's stray-light group).
   * - ``background_stars.py``
     - Background-star count, PSF sigma, distribution exponent.
   * - ``body_tab.py`` / ``ring_tab.py`` / ``star_tab.py``
     - The per-object tabs (geometry, shape model and mesh controls, crater
       controls, navigation-override group; ring edges and shading; star
       position, magnitude, PSF, smear, catalog label).
   * - ``tabs.py``
     - Object-tab lifecycle (the ``+`` tab, ordering, rebuild).
   * - ``scene_io.py``
     - Load / Save Scene (YAML) via :mod:`spindoctor.sim.scene`.
   * - ``render_display.py``
     - The live preview (render, stretch, zoom, saturation overlay).
   * - ``widgets.py`` / ``base.py``
     - Shared widget helpers and the mixin-facing protocol.

The GUI exposes the full scene parameter surface, so any scene that can be
written by hand in YAML can also be built in the GUI. The parameters the GUI
does not edit are the nested ``instrument_config`` overrides, multi-mode ring
edges (the renderer reads only mode 1), the absolute ``signal_full_scale_dn``
alias (its fractional form is exposed instead), and the detector
``quantization`` sub-mode (the catalog default applies). Scenes
round-trip through the **Load / Save Scene (YAML)** buttons, so a scene rendered
in the GUI can be saved as a catalog artifact and a catalog scene can be loaded
back to edit; ``tests/main/test_sim_editor_round_trip.py`` asserts the
round-trip is loss-free over the full key inventory. The GUI is one peer, not
the sole control surface; the YAML and the Python API are equally
authoritative.

Running navigation on a simulated image
=======================================

A simulated image is navigated through the same pipeline as a real frame, via the
``sim`` dataset. With a saved YAML scene file:

.. code-block:: bash

   sd_offset sim /path/to/scene.yaml

The ``sim`` dataset (``DataSetSim``) builds an :class:`~spindoctor.obs.obs_inst_sim.ObsSim`
that loads the scene via :func:`spindoctor.sim.scene.load_sim_scene`, renders the
frame through the forward model, and exposes the navigator the filtered
idealized scene view ``obs.nav_params`` (see
:ref:`sim-information-boundary`). The model-selection layer routes a simulated
obs to the simulated NavModels -- ``NavModelBodySimulated``,
``NavModelRingsSimulated``, ``NavModelStarsSimulated`` -- which build one
feature set per body / ring / star field from that filtered view, while the
SPICE-backed models decline a simulated obs. From there the same techniques run
and produce the same ``NavResult``. In tests the scene is usually driven
directly: ``ObsSim.from_file(path, sim_params=load_sim_scene(path))``.

.. _sim-png-export:

Exporting viewable PNGs
=======================

The renderer emits detector counts whose absolute range depends on the instrument
and on cosmic-ray spikes, so :mod:`spindoctor.sim.png_export` stretches a DN image to a
viewable grayscale PNG with a percentile clip (a few hot pixels do not crush the
signal) and an optional gamma that lifts dim features such as a crescent or a
faint star field:

.. code-block:: python

   from spindoctor.sim.png_export import render_scene_png
   render_scene_png(sim_params, 'frame.png', gamma=1.4, upscale=2)

Three tools build on it. The sweep runner can dump every frame behind a response
curve for inspection:

.. code-block:: bash

   python -m tests.integration.sim_sweep_runner --dump-images out/ --only phase_regular_body

writes one PNG per sweep step under ``out/<sweep_name>/``. The documentation-image
generator rebuilds the galleries in this chapter and the scene images in the
sensitivity report:

.. code-block:: bash

   python -m tests.integration.sim_doc_images

Both galleries carry a ``NOTES.md`` describing how to regenerate them after a
rendering change. And the render-diff contact sheet
(:ref:`sim-catalog-workflow`) stretches every catalog scene into its committed
review panels:

.. code-block:: bash

   python -m tests.integration.render_contact_sheet

See also
========

- :doc:`dev_guide_navigation_models_body_simulated`,
  :doc:`dev_guide_navigation_models_ring_simulated`,
  :doc:`dev_guide_navigation_models_star_simulated` -- the navigator side.
- :doc:`dev_guide_testing` -- the test kinds the simulator drives, including
  the information-boundary test.
- :doc:`/simulator_report/simulator_report` -- the sensitivity and
  algorithmic-invariant results.
- API reference: :doc:`/api_reference/api_sim`.
