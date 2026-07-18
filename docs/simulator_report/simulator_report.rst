============================================
Simulator Performance and Sensitivity Report
============================================

This chapter reports how the SpinDoctor navigation pipeline performs on simulated
images: how accurately each technique recovers a known transform, and how its
navigation responds as a single scene parameter is swept. It is a standalone
chapter, separate from the user and developer guides, and is regenerable on
demand from the simulator scene catalog and sweep harness.

The numbers below are representative measurements from the committed scene
catalog and sweeps. The technique solvers carry sub-millipixel floating-point
jitter across processes and machines, so treat the recovery errors as
"comfortably within the stated bound" rather than exact constants; the
qualitative results (which technique is load-bearing, where navigation fails, how
error trends with phase) are stable.

.. contents::
   :local:
   :depth: 2

Purpose and scope
=================

This report summarises two measurements taken on simulated frames with ground
truth that is correct by construction:

* **Algorithmic-invariant recovery** -- a planted offset (or camera roll) the
  navigator must recover, for each technique in the ladder.
* **Single-variable sensitivity** -- a base scene driven across noise, phase, and
  body size, showing the navigability cliff, the phase response, and the
  technique-selection transitions.

The report keys on the *recovered geometry* (offset error, roll error, primary
technique, success/fail). The per-technique confidence coefficients are
sim-calibrated (see ``config_510_techniques.yaml``), so the confidence
column reflects the shipped formulas; tier interpretation against real frames
still rests on the operator-curated image library, not on these clean scenes.

Methodology
===========

A **sweep** drives one base scene by overriding a single parameter across a list
of values and navigating each step. For an offset or camera-roll sweep the
planted ground truth is read from the (overridden) parameter itself, so the error
is ``recovered - planted``. A sweep optionally **pins** one technique
(``only_techniques=<name>``) and reads that technique's own recovered offset, so
each technique is characterised independently. The harness, spec schema, runner,
and plotting live in ``tests/integration/sim_sweep.py``, ``sim_sweep_runner.py``,
and ``sim_sweep_plots.py``.

**Offset value sets.** Two offset sweeps per technique probe the offset axis at
two scales:

- A **dense sub-pixel** sweep plants every offset in

  ::

     0.0, 0.05, 0.1, 0.137, 0.2, 0.25, 0.31, 0.382, 0.45, 0.5, 0.55, 0.611,
     0.667, 0.7, 0.75, 0.823, 0.9, 0.95, 1.0, 1.25, 1.5, 1.618, 1.75   (px)

  -- the quarter- and half-pixel anchors, the thirds, golden-ratio fractions, and
  a spread of other non-power-of-2 fractions, sampling the pixel densely enough to
  expose any fraction-dependent residual.
- A **wide-range** sweep plants offsets across the full navigable range with
  varied fractional parts, up to each technique's ceiling.

**Navigable range.** The recoverable offset is bounded by the extended-FOV search
margin. It is size-keyed per instrument: Cassini NAC is ``[13, 25]`` px at 256,
``[25, 50]`` at 512, and ``[50, 140]`` at the full 1024. The sweeps run at 220 px
(margin ~50 px, the generic fallback) for tractable runtime -- a 1024 px
navigation costs ~35 s -- and sweep to the measured per-technique ceiling:

.. list-table:: Per-technique offset sweep (base scene + navigable ceiling)
   :header-rows: 1
   :widths: 22 34 14 14

   * - Technique
     - Base scene (pinned technique)
     - Wide ceiling
     - Limit set by
   * - BodyDiscCorrelateNav
     - ``regular_sphere_base`` (90 px sphere)
     - ~48 px
     - extfov margin
   * - RingEdgeNav
     - ``planted_offset_ring`` (ringlet)
     - ~48 px
     - extfov margin
   * - BodyLimbNav
     - ``planted_offset_limb`` (130 px sphere)
     - ~40 px
     - extfov margin
   * - StarFieldFromCatalogNav
     - ``planted_offset_star_field`` (6-star field)
     - ~20 px
     - frame size / per-star search window
   * - BodyBlobNav
     - ``small_sphere_base`` (20 px sphere)
     - ~extfov margin
     - lit-shape (disc / crescent) coarse acquisition

Beyond a technique's ceiling the navigator correctly reports failure (the feature
is outside the searchable region); the wide sweeps run to the ceiling and, for the
blob, a little past it to show the degradation.

``BodyBlobNav`` previously stopped at ~6 px (the predicted bounding box plus its
per-body slop), past which the brightness-weighted centroid clipped and silently
biased. A coarse lit-shape correlation now re-centres each blob's box on the body
across the full search window before the centroid is taken, so the capture range
matches the other techniques (recovery holds to a few hundredths of a pixel out to the
extfov margin on the low-phase ``small_sphere_base`` sweep). The template tracks phase:
a filled disc at or below half phase, and a synthesised crescent above it, oriented
along the sub-solar direction the blob feature carries. A high-phase crescent displaced
~20 px beyond its bounding box recovers to a few hundredths of a pixel on the
``planted_offset_blob_crescent_displaced`` invariant scene. See
:doc:`../dev_guide/dev_guide_techniques_body_blob`.

Running the sweeps
==================

The sweeps are **not** part of the normal pytest run. Generate every sweep's
response curve (JSON under ``tests/integration/sim_sweeps/results/``) and the
figures in this chapter with:

.. code-block:: bash

   python -m tests.integration.sim_sweep_runner

A single sweep can be inspected from Python via
``tests.integration.sim_sweep.load_sweep`` / ``run_sweep``. The planted-recovery
table below comes from the algorithmic-invariant scenes, which *do* run in the
deliberate (integration-marked) tier:

.. code-block:: bash

   pytest tests/integration/test_sim_algorithmic_invariants.py -m "" -n auto --dist=loadfile

Targeted regression scenes under ``sim_scenes/regression/`` guard specific
behaviours in the normal suite without running the full sweep.

See :doc:`/dev_guide/dev_guide_simulator` for the scene catalog, scene formats,
and the sweep / image-dump tooling, :doc:`/dev_guide/dev_guide_testing` for the
test tiers, and :doc:`/dev_guide/dev_guide_navigation_models` for the simulated
models that emit the features each technique consumes.

Example scenes
==============

The frames below are the actual catalog scenes behind the measurements in this
chapter, rendered from their YAML by
``python -m tests.integration.sim_doc_images``.

.. figure:: _scene_images/disc.png
   :width: 45%
   :align: center

   Resolved disc

.. figure:: _scene_images/mesh_disc.png
   :width: 45%
   :align: center

   Irregular mesh body

.. figure:: _scene_images/limb_mesh.png
   :width: 45%
   :align: center

   Mesh limb

.. figure:: _scene_images/blob_crescent.png
   :width: 45%
   :align: center

   High-phase crescent

.. figure:: _scene_images/mesh_crescent.png
   :width: 45%
   :align: center

   Mesh crescent

.. figure:: _scene_images/ring.png
   :width: 45%
   :align: center

   Ring edge

.. figure:: _scene_images/star_field.png
   :width: 45%
   :align: center

   Star field

.. figure:: _scene_images/regular_sphere_base.png
   :width: 45%
   :align: center

   Sweep base sphere

Algorithmic-invariant recovery
==============================

Each scene below plants a known transform and the navigator predicts the
unshifted geometry, so the recovered offset (or roll) should equal the planted
value. The planted offsets are deliberately off-grid (no integer, half-, or
quarter-pixel values), so a technique cannot land on a sub-pixel-bias null and
report a flatteringly small error; these are single-sample correctness checks at
one arbitrary phase, while per-technique sub-pixel precision across many offsets is
characterized in the offset-accuracy section below. The technique column names the
load-bearing technique -- pinned for the single-technique scenes (blob, limb,
ring, roll) so each is characterised in isolation, and the full ensemble for the
disc and star scenes.

.. list-table:: Planted-transform recovery by technique
   :header-rows: 1
   :widths: 26 30 16 16 12

   * - Scene
     - Technique
     - Planted
     - Recovered
     - Error
   * - ``planted_offset_disc``
     - BodyDiscCorrelateNav
     - (1.43, -0.61) px
     - (1.50, -0.60) px
     - 0.07 px
   * - ``planted_offset_irregular``
     - BodyDiscCorrelateNav (mesh)
     - (1.43, -0.61) px
     - (1.43, -0.62) px
     - 0.01 px
   * - ``planted_offset_blob``
     - BodyBlobNav
     - (1.43, -0.61) px
     - (1.42, -0.62) px
     - 0.02 px
   * - ``planted_offset_blob_crescent``
     - BodyBlobNav (120 deg)
     - (1.43, -0.61) px
     - (1.24, -0.59) px
     - 0.19 px
   * - ``planted_offset_star_field``
     - StarField + UniqueMatch + Refine
     - (1.43, -0.61) px
     - (1.32, -0.65) px
     - 0.12 px
   * - ``planted_offset_limb``
     - BodyLimbNav
     - (1.43, -0.61) px
     - (1.40, -0.66) px
     - 0.06 px
   * - ``planted_offset_ring``
     - RingEdgeNav
     - (1.43, -0.61) px
     - (1.42, -0.62) px
     - 0.01 px
   * - ``planted_rotation_star_field``
     - StarFieldFromCatalogNav (roll)
     - (1.43, -0.61) px, 1.37 deg
     - 1.356 deg
     - 0.01 deg

Observations:

* All scenes plant the **same** off-grid offset ``(1.43, -0.61)`` px, so the
  per-technique errors are at one common sub-pixel phase and directly comparable.
  Every technique recovers well within the 1.0 px (and third-of-a-degree) bound.
* At this phase the mesh disc and ring edge recover to ~0.01 px, the blob to
  ~0.02 px, the limb to ~0.06 px, and the disc to ~0.07 px; the fused star field
  to ~0.12 px.
* The high-phase blob crescent (~0.19 px at 120 deg) is the hardest case: only a
  thin lit crescent constrains the centroid. It still recovers sub-pixel.
* These are single-phase samples, so a given technique can sit above or below its
  multi-offset median in the offset-accuracy section (e.g. the disc's ~0.07 px here
  versus its ~0.006 px sweep median); the table is a same-phase comparison and a
  correctness check, not the precision benchmark.

Single-variable sensitivity
===========================

The noise, phase, and range sweeps drive one base scene -- a well-resolved sphere
with a planted ``(1.5, -0.5)`` offset -- so the offset error column is the
recovery error at each step. (The offset and roll sweeps that follow use their
own base scenes.)

Read-noise sweep
----------------

Varying ``noise.read_noise_dn`` from a clean frame to past the navigability
cliff:

.. list-table:: Read-noise response
   :header-rows: 1
   :widths: 16 16 16 16 28

   * - ``read_noise_dn``
     - Status
     - Offset error
     - Confidence
     - Primary technique
   * - 1
     - success
     - 0.00 px
     - 0.75
     - BodyDiscCorrelateNav
   * - 4
     - success
     - 0.00 px
     - 0.75
     - BodyDiscCorrelateNav
   * - 8
     - success
     - 0.00 px
     - 0.75
     - BodyDiscCorrelateNav
   * - 16
     - success
     - 0.00 px
     - 0.75
     - BodyDiscCorrelateNav
   * - 32
     - success
     - 0.00 px
     - 0.75
     - BodyDiscCorrelateNav
   * - 64
     - **failed**
     - --
     - 0.00
     - --

The disc correlation is robust: the offset is recovered exactly until the read
noise overwhelms the body signal, at which point the frame is classified
unnavigable and navigation fails cleanly. The calibrated confidence is flat
across the navigable range -- the correlation-quality diagnostics the formula
consumes do not degrade until the cliff itself.

Phase-angle sweep
-----------------

Varying ``bodies.0.phase_angle`` across the full range on the resolved sphere:

.. list-table:: Phase-angle response
   :header-rows: 1
   :widths: 14 16 16 16 28

   * - Phase (deg)
     - Status
     - Offset error
     - Confidence
     - Primary technique
   * - 0
     - success
     - 0.00 px
     - 0.40
     - BodyBlobNav
   * - 30
     - success
     - 0.00 px
     - 0.75
     - BodyDiscCorrelateNav
   * - 60
     - success
     - 0.01 px
     - 0.75
     - BodyDiscCorrelateNav
   * - 90
     - success
     - 0.00 px
     - 0.74
     - BodyDiscCorrelateNav
   * - 120
     - success
     - 0.00 px
     - 0.73
     - BodyDiscCorrelateNav
   * - 150
     - success
     - 0.00 px
     - 0.65
     - BodyDiscCorrelateNav

The resolved body navigates to success at every phase and recovers the planted
offset to within ~0.01 px throughout. The disc correlation holds its accuracy as
the terminator sweeps across the lit disc -- the gradient-domain matched filter
keys on the sunward limb, which stays a sharp, well-defined feature at every
phase -- so there is no mid-phase accuracy penalty. At zero phase the blob's
lit-weighted centroid wins the technique selection outright: a fully-lit disc has
no correlation-gradient advantage for the disc to exploit. From 30 deg onward the
disc carries the frame.

Body-size (range) sweep
-----------------------

Shrinking ``bodies.0.axis{1,2,3}`` together from well-resolved to unnavigable.
This is the technique ladder the range regime is meant to exercise:

.. list-table:: Body-size response
   :header-rows: 1
   :widths: 16 16 16 16 28

   * - Diameter (px)
     - Status
     - Offset error
     - Confidence
     - Primary technique
   * - 130
     - success
     - 0.01 px
     - 0.99
     - BodyLimbNav
   * - 90
     - success
     - 0.00 px
     - 0.75
     - BodyDiscCorrelateNav
   * - 60
     - success
     - 0.01 px
     - 0.74
     - BodyDiscCorrelateNav
   * - 40
     - success
     - 0.00 px
     - 0.74
     - BodyDiscCorrelateNav
   * - 20
     - success
     - 0.00 px
     - 0.40
     - BodyBlobNav
   * - 12
     - success
     - 0.02 px
     - 0.40
     - BodyBlobNav
   * - 6
     - success
     - 0.01 px
     - 0.40
     - BodyBlobNav
   * - 4
     - **failed**
     - --
     - 0.00
     - --

The primary technique transitions cleanly as resolution falls: a well-resolved
body (130 px) is navigated by the limb fit; a mid-size body by the disc
correlation; a small body (6-20 px) falls to the orientation-free blob
centroid; and the smallest body (4 px, below the 5 px ``BODY_BLOB`` emission
floor) is unnavigable. Every navigable step recovers the planted offset
exactly. This transition is the sim's most direct verification that the
orchestrator selects the right technique for the available resolution.

Offset accuracy by technique
============================

The invariant scenes above plant a single offset near the middle of a pixel. To
look for pixel-boundary, quantization, and range corner cases, each technique is
swept across the dense sub-pixel set and the wide range described under
*Methodology*. The dense sweep traces accuracy across the pixel:

.. figure:: _figures/offset_accuracy_fine.png
   :width: 100%
   :alt: Sub-pixel offset recovery error by technique.

   Recovered-offset error (log scale) vs planted sub-pixel offset, each technique
   pinned. The disc, blob, and star field sit at or below a few hundredths of a
   pixel; the ring edge (~0.03 px) and the limb (~0.09 px) hold distance-transform
   residuals.

.. list-table:: Sub-pixel recovery error (px) over the dense fractional sweep
   :header-rows: 1
   :widths: 34 14 14 14

   * - Technique
     - min
     - median
     - max
   * - BodyDiscCorrelateNav
     - 0.000
     - 0.006
     - 0.050
   * - BodyLimbNav
     - 0.022
     - 0.092
     - 0.165
   * - RingEdgeNav
     - 0.021
     - 0.028
     - 0.034
   * - BodyBlobNav
     - 0.002
     - 0.006
     - 0.011
   * - StarFieldFromCatalogNav (dim field)
     - 0.002
     - 0.025
     - 0.065
   * - StarFieldFromCatalogNav (bright field)
     - 0.001
     - 0.005
     - 0.012

The correlation and centroid techniques sit at or below a few hundredths of a
pixel; the distance-transform techniques carry a larger sub-pixel residual:

- **Disc and blob recover to <0.01 px** (median 0.006), and the bright star field
  to ~0.005 px -- the most accurate. A zero shift between identical frames recovers
  the disc exactly at ``(0, 0)``.
- **The ring edge holds a ~0.03 px residual** and **the limb a ~0.09 px residual**
  across the dense sweep, both distance-transform effects that vary with the
  sub-pixel phase; they stay well inside the invariant bound. See
  :doc:`/dev_guide/dev_guide_techniques_body_limb` for the mechanism.
- **The star field is pinned here**, so its sub-pixel rows below ~1 px are absent:
  the field matcher alone does not recover a sub-pixel translation of the whole
  field (the two-star path in the full ensemble does -- see the planted-offset star
  invariant). Above ~1 px the dim field recovers to a median ~0.025 px and the
  bright field to ~0.005 px. The dim/bright split is detailed below; the per-star
  centroiding mechanism is in :doc:`/dev_guide/dev_guide_techniques_star_field`.

The wide-range sweep confirms each technique recovers across the navigable range:

.. figure:: _figures/offset_accuracy_wide.png
   :width: 100%
   :alt: Wide-range offset recovery error by technique.

   Recovered-offset error vs planted offset across the navigable range. Disc,
   ring, limb, blob, and star each recover to their ceilings; the blob now tracks
   the body across the full window via its coarse acquisition.

The disc, ring, limb, and star recover with the same accuracy out to their
ceilings (~48, ~48, ~40, ~20 px). The **blob now holds across the navigable
range too**: for the 20 px body it stays under ~0.01 px out to ~45 px, where it
was previously a ~6 px small-offset technique (degrading to ~5.8 px at 20 px) --
the blob-shaped-disc coarse acquisition re-centres the integration window on the
body before the centroid, so the body no longer clips out of the predicted bbox.
The disc template models only a near-full disc, so this holds for bodies at least
half-lit; a high-phase crescent beyond its bbox still needs a prior. See
:doc:`/dev_guide/dev_guide_techniques_body_blob`.

Star-field centroiding: dim vs bright
=====================================

Two sweeps measure the star field on the same six-star geometry and planted
offset, varying only the stars' brightness: ``star_offset_fine`` plants a dim
field (vmag 3-4, ~100-150 DN net peak) and ``star_offset_fine_bright`` a bright
field (vmag 0-0.8, ~1000-2000 DN net peak, below the 4095 DN full well).

.. figure:: _figures/star_regime_accuracy.png
   :width: 100%
   :alt: Star-field sub-pixel accuracy for dim vs bright fields.

   Recovered-offset error (log scale) vs planted offset for the dim and bright
   star fields. The dim field sits near ~0.02 px; the bright field reaches
   ~0.005 px -- below every other technique.

The dim field recovers to a median ~0.025 px and the bright field to ~0.005 px,
the most accurate of any technique. The centroiding mechanism behind this split
is documented in :doc:`/dev_guide/dev_guide_techniques_star_field`.

A finer characterization sweeps a uniform-brightness field across a 20x
integrated-SNR range and overlays three centroiding modes -- moment-only,
PSF-everywhere, and the shipped SNR-adaptive choice -- under three backgrounds:
clean, elevated read noise, and a stray-light gradient. (Run by
``python -m tests.integration.star_snr_characterization``; not part of pytest.)

.. figure:: _figures/star_snr_clean.png
   :width: 100%
   :alt: Star-field centroiding error vs SNR, clean background.

   Clean background. The moment (blue) improves as the field brightens; the PSF
   fit (orange) wins at low SNR but plateaus at ~0.05-0.07 px. The shipped
   adaptive choice (green) rides the lower envelope of the two, crossing near an
   integrated SNR of ~30.

.. figure:: _figures/star_snr_highnoise.png
   :width: 100%
   :alt: Star-field centroiding error vs SNR, elevated read noise.

   Elevated read noise (read_noise_dn 20). The crossover sits near an integrated
   SNR of ~16.

.. figure:: _figures/star_snr_gradient.png
   :width: 100%
   :alt: Star-field centroiding error vs SNR, stray-light gradient.

   Stray-light linear gradient. The crossover sits near an integrated SNR of ~21;
   the gradient barely shifts the curves.

The moment/PSF crossover sits near an integrated SNR of ~30 on a clean
background, ~16 under heavy read noise, and ~21 under a stray-light gradient. The
estimator selection and its tuning are documented in
:doc:`/dev_guide/dev_guide_techniques_star_field`.

Per-technique accuracy across SNR and injected offset
=====================================================

Each technique's accuracy versus signal-to-noise and versus the injected offset
is characterized by ``python -m tests.integration.technique_snr_characterization``
(runner-only). Two figure families come out of it; offsets and noise levels are
stated on each panel.

**Accuracy versus SNR (fixed injected offset).** Each technique's base scene is
rendered at a fixed off-grid ``(dv, du) = (+0.317, -0.211)`` px offset. The
per-image read noise is swept from a clean frame down toward the navigability
cliff; the x-axis is a uniform per-image SNR proxy
``(peak - background) / robust_noise``. Each technique's feature has a different
intrinsic brightness, so the curves occupy different SNR bands.

.. figure:: _figures/technique_snr_nominal.png
   :width: 100%
   :alt: Per-technique recovered-offset error vs SNR, nominal background.

   Nominal background. The disc (~0.01 px) and blob (~0.008 px) are the most
   accurate and both flat with SNR. The star improves steadily with SNR. The ring
   is flat at ~0.06 px and the limb at ~0.15 px, both distance-transform residuals
   independent of SNR.

.. figure:: _figures/technique_snr_gradient.png
   :width: 100%
   :alt: Per-technique recovered-offset error vs SNR, stray-light gradient.

   A gentle stray-light gradient (linear ramp ~3% of full scale). The blob stays
   most accurate (~0.016 px), the disc rises to ~0.04 px, the ring holds ~0.06 px;
   the limb and star field return no result (their curves are absent). Stray light
   is a larger threat to the faint-feature techniques than read noise alone.

The disc and ring sub-pixel residuals are off-grid effects (zero at integer and
half-pixel offsets), measured here against an off-grid planted offset. Their
mechanisms are documented in
:doc:`/dev_guide/dev_guide_techniques_body_disc` and
:doc:`/dev_guide/dev_guide_techniques_ring_edge`.

**Accuracy versus injected offset (fixed SNR).** Holding the read noise at three levels, a
pure-vertical offset is swept from 0 to 1.75 px (``u`` held at 0) for every technique. The
panels share a y-range so the degradation as SNR drops reads directly.

.. figure:: _figures/technique_offset_high_snr.png
   :width: 100%
   :alt: Per-technique error vs injected offset, high SNR.

   High SNR (read noise 1 DN). Every technique recovers to a few hundredths of a pixel with
   no dependence on the fractional part (the disc drops below 0.001 px at several offsets);
   only the limb's bias floor stands out. This panel matches the dense fractional sweep.

.. figure:: _figures/technique_offset_medium_snr.png
   :width: 100%
   :alt: Per-technique error vs injected offset, medium SNR.

   Medium SNR (read noise 8 DN). The star field degrades to ~0.05-0.07 px and the others
   hold; the limb stays at its bias floor.

.. figure:: _figures/technique_offset_low_snr.png
   :width: 100%
   :alt: Per-technique error vs injected offset, low SNR.

   Low SNR (read noise 32 DN). The disc, blob, and ring still recover (their features are
   bright), but the limb and star field have crossed their navigability cliff and return no
   result at any offset -- the missing curves are the failure, not an omission.

Irregular-body navigation
=========================

Non-ellipsoidal bodies (Hyperion-like, Phoebe-like) are rendered from a
procedurally generated polyhedral mesh at a chosen three-axis pose. By default the
navigator predicts the body from the same mesh and pose the renderer drew, so the
recovery is exact by construction; the cases below also drive the navigator with a
deliberately *wrong* shape or pose through the ``nav_override`` channel
(:doc:`/dev_guide/dev_guide_simulator`), which is what makes a chaotic rotator's
genuinely unknown orientation testable.

.. list-table:: Mesh-body planted-offset recovery
   :header-rows: 1
   :widths: 30 26 16 16 12

   * - Scene
     - Technique (geometry)
     - Planted
     - Recovered
     - Error
   * - ``planted_offset_irregular``
     - BodyDiscCorrelateNav (mesh = mesh)
     - (1.43, -0.61) px
     - (1.43, -0.61) px
     - 0.00 px
   * - ``planted_offset_limb_mesh``
     - BodyLimbNav (mesh = mesh)
     - (1.43, -0.61) px
     - (1.26, -0.53) px
     - 0.19 px
   * - ``planted_offset_blob_mesh_crescent``
     - BodyBlobNav (mesh, 120 deg)
     - (1.43, -0.61) px
     - (1.24, -0.52) px
     - 0.21 px
   * - ``planted_offset_shapemismatch``
     - full ensemble (predict ellipsoid)
     - (1.43, -0.61) px
     - (1.99, -0.90) px
     - 0.63 px

When the predicted geometry matches the rendered mesh, the mesh disc, mesh limb,
and mesh crescent recover the planted offset as tightly as their ellipsoid
counterparts (0.00-0.21 px). The fourth row is the shape-mismatch case: the frame
renders a mildly irregular mesh but the navigator predicts its smooth
(ellipsoidal) limit at the same pose, and the body still navigates -- the disc
correlation aligns the two filled silhouettes and recovers to within
two-thirds of a pixel.

Shape mismatch vs irregularity
------------------------------

Holding the navigator's prediction at the smooth (zero-relief) limit and walking
the rendered mesh's surface relief up isolates the centroid bias an ellipsoidal
model cannot remove -- the regime the navigator's ``phase_irregularity_factor``
term is meant to capture.

.. figure:: _figures/mesh_irregularity.png
   :width: 100%
   :alt: Shape-mismatch centroid bias and confidence vs mesh relief.

   Recovered-offset error (red) and fused confidence (blue) vs rendered mesh
   lumpiness, with the prediction pinned to the zero-relief limit. The bias grows
   from ~0.3 px (no mismatch) to ~4 px at heavy relief while the confidence falls
   from ~0.99 to ~0.71 -- the navigator both mis-locates the body and reports
   lower confidence as the shape diverges.

The recovered error grows monotonically with relief (0.33, 0.47, 1.33, 2.17,
4.02, 4.03 px at lumpiness 0.0 through 0.5) and the fused confidence falls in
step (0.99 down to 0.71). At low relief the limb fit carries the frame at high
confidence; from lumpiness 0.2 the primary hands off to the disc correlation,
which still locks onto the silhouette but drifts with the unmodelled relief --
exactly the failure the irregular-body confidence penalty must learn to
distrust. The body keeps navigating to a ``success`` status throughout.

Pose disagreement
-----------------

For a body whose orientation we cannot trust, the useful question is what happens
when the assumed pose is wrong. Rendering the mesh at its true pose and walking
the navigator's *predicted* pose away from it degrades the orientation-dependent
limb fit:

.. figure:: _figures/mesh_pose_disagreement.png
   :width: 100%
   :alt: Mesh-limb recovery error vs predicted-pose disagreement.

   Pinned-limb recovered-offset error vs the predicted pose's disagreement with
   the true (rendered) pose (a tumble about the body's long axis). The limb keeps
   returning a fix across the swept range, but its error climbs from 0.27 px at
   the true pose to 4.2 px at a 45 deg disagreement -- a confidently-wrong limb
   that does not self-flag here.

The limb error climbs monotonically from 0.27 px at the true pose to 4.2 px at a
45 deg tumble, all the while still reporting ``success`` at ~0.76 confidence: it
does not self-flag in this range. A wrong in-plane roll degrades it far more
sharply -- tens of pixels, and there it does self-flag spurious. The pose-free
blob centroid, by contrast, stays accurate on the same wrong-pose body, because a
centrally-symmetric (low-relief triaxial) body's lit-weighted centroid barely
moves under rotation. This is the behaviour the ``test_sim_irregular_pose``
per-technique test pins: on a wrong-pose body the
system should demote from the confidently-wrong limb to the orientation-free
blob.

Camera-roll sensitivity and roll / translation separability
============================================================

Sweeping the planted camera roll on a star field shows both the working window
and the separability floor:

.. list-table:: Camera-roll recovery
   :header-rows: 1
   :widths: 14 22 30

   * - Planted roll
     - Full-ensemble error
     - StarFieldFromCatalogNav alone
   * - 0.25 deg
     - --
     - collapses to 0 (spurious)
   * - 0.5 deg
     - 0.05 deg
     - collapses to 0 (spurious)
   * - 0.75 deg
     - --
     - 0.69 deg (partial)
   * - 1.0 deg
     - 0.01 deg
     - 1.04 deg
   * - 1.5 deg
     - 0.01 deg
     - 1.51 deg
   * - 2.0 deg
     - 0.12 deg
     - 1.89 deg

A small roll is not separable from a translation. ``StarFieldFromCatalogNav``'s
RANSAC pattern matcher collapses a roll below ~0.75 deg toward zero (often with a
spurious flag); a fitted roll below the configurable
``rotation_separability_floor_deg`` (default 0.75 deg) is therefore reported with
the rotation-unobservable sentinel variance and the
``rotation_below_separability_floor`` diagnostics flag, never as a confident
near-zero value.
The two-star ``StarUniqueMatchNav`` path recovers down to ~0.5 deg, so the *full
ensemble* recovers the 0.5 deg roll even where the field matcher alone does not.
At exactly zero roll no technique reports a rotation. Above ~2-3 deg the inlier
count falls below quorum. The usable window for the field matcher is therefore
roughly 0.75-2 deg, widening to ~0.5 deg with the two-star path. See
:doc:`/dev_guide/dev_guide_techniques_star_field` and
:doc:`/dev_guide/dev_guide_techniques_star_unique_match`.

Small-body navigation floor
===========================

The range sweep above navigates 12 px and 6 px bodies by ``BodyBlobNav`` and
fails at 4 px, below the 5 px ``BODY_BLOB`` emission floor where the model emits
no body feature at all. The floor is the emission gate (a sub-5-px silhouette
covers so few pixels that the centroid is PSF- and noise-dominated), not the
reliability gate: the ``BODY_BLOB`` reliability is driven by a measured
detection SNR, so a bright body is admitted at any emitted size while a
predicted body with no image signal anywhere in its search window is culled
regardless of size. See :doc:`/dev_guide/dev_guide_techniques_body_blob`.

I/F-calibrated vs raw-DN navigation
===================================

Every sweep and scene above renders in raw DN. The ``planted_offset_disc_if``
invariant scene confirms navigation is unit-agnostic: the same body on the
I/F-calibrated ``coiss_calib_nac`` instrument recovers the planted offset exactly.
That scene carries no ``artifacts`` block, so it renders through the stage floor
(noise-free); a ``calibrated_if`` scene with ``instrument_defaults`` runs the
full DN detector chain and then the calibration inverse, so calibrated frames
carry propagated shot and read noise and quantization texture in I/F units.
The realism section below compares exactly that path against the real
calibrated cohorts.

Realism match against real cohorts
==================================

Everything above measures the navigator against frames the simulator
rendered; this section measures the simulator against reality.  The realism
runner (``python -m tests.integration.sim_realism``; see
:ref:`sim-realism-match` for the machinery) compares the curated real-image
library against matched simulated frames -- one per real frame, same
instrument signal chain, same exposure, same content class -- on seven
figures of merit, per instrument.  Each compared statistic is reported as a
distribution overlay plus one scalar: the Wasserstein-1 distance on
quantile-clipped samples, normalized by the real distribution's IQR (a
value of 1.0 means the two distributions are displaced by about one real
IQR).  **No pass/fail threshold is attached**; the numbers below are read
against each cohort's support.  Figure of merit 7 (technique diagnostics)
is read-only and was not consulted for any tuned value -- it is built from
the navigator's own outputs, and tuning the image side until the
navigator's diagnostics agree would re-admit circularity through parameter
fitting.

Cohort support (measured 2026-07-18, 75 frames):

.. list-table::
   :header-rows: 1
   :widths: 22 10 13 13 13 13 13 13

   * - Instrument (cohort)
     - Frames
     - 1 noise
     - 2 PSF
     - 3 limbs
     - 4 rings
     - 5 range
     - 6 artifacts
   * - Cassini NAC (CALIB)
     - 58
     - supported
     - supported
     - supported
     - supported
     - supported
     - supported
   * - Cassini WAC (CALIB)
     - 4
     - limited
     - limited
     - unsupported
     - unsupported
     - limited
     - limited
   * - Galileo SSI (REDR)
     - 8
     - limited
     - unsupported
     - unsupported
     - unsupported
     - supported
     - supported
   * - Voyager ISS (GEOMED)
     - 3
     - limited
     - unsupported
     - unsupported
     - unsupported
     - limited
     - limited
   * - New Horizons LORRI (sci)
     - 2
     - unsupported
     - limited
     - unsupported
     - unsupported
     - limited
     - limited

Where a figure of merit is *unsupported* and no independent evidence
exists, that instrument's sim accuracy is **bounded by unverified
forward-model fidelity**: the Galileo cohort holds only negative-case
frames (no star, limb, or ring content), so nothing currently verifies the
gossi PSF or shape rendering; the Voyager and LORRI cohorts support only
fragments.  This is a Cassini match first, and the labels say exactly how
far the others reach.

Cassini match quality
---------------------

The tuning pass (2026-07-17) adjusted the forward-model defaults in
``sim/forward/artifacts_catalog.py`` against FOMs 1-6 -- the tuned values
and their cohort statistics are recorded there -- and this table records
the headline divergences before and after (normalized W1; smaller is
better, ~1 means displaced by one real IQR):

.. list-table::
   :header-rows: 1
   :widths: 34 22 11 11 11 11

   * - Statistic (FOM)
     - Cohort median (real)
     - W1/IQR before
     - W1/IQR after
     - n real
     - n sim
   * - sky noise sigma (FOM 1)
     - 2.1e-4 I/F
     - 10.0
     - 5.0
     - 5263
     - 2184
   * - sky level above floor (FOM 1)
     - 3.8e-4 I/F
     - 13.9
     - 173 [#tail]_
     - 5263
     - 2184
   * - noise at signal (FOM 1)
     - 2.9e-4 I/F
     - 13.8
     - 5.1
     - 15700
     - 6440
   * - star EE50 (FOM 2)
     - 0.91 px
     - 1.09
     - 0.57
     - 23
     - 79
   * - star EE80 (FOM 2)
     - 1.79 px
     - 1.28
     - 0.23
     - 23
     - 79
   * - limb rise width, co-populated strata (FOM 3)
     - 2.54 px
     - -- [#copop]_
     - 0.16
     - 3755
     - 4338
   * - ring-edge rise width (FOM 4)
     - 2.87 px
     - 0.47
     - 0.43
     - 8480
     - 21858
   * - near-floor fraction, 0.05-0.5 s (FOM 5)
     - 0.107
     - 0.76
     - 1.84 [#floor]_
     - 27
     - 27
   * - p99 - p50 stretch, 0.05-0.5 s (FOM 5)
     - 0.104 I/F
     - 0.89
     - 0.28
     - 27
     - 27
   * - spike fraction (FOM 6)
     - 1.1e-4
     - 11.0
     - 1.2
     - 58
     - 58

.. [#tail] Medians agree to 25% (3.0e-4 sim vs 3.8e-4 real); the large W1
   is carried by a tail of simulated sky patches near bright content in
   body- and ring-bearing frames -- residual halo and shading structure
   the darkest-quartile patch selection admits on the sim side.

.. [#copop] The pooled FOM 3 statistic is computed over strata populated
   on *both* sides.  Earlier records pooled every stratum, but the
   simulated body model's limb emission gates (diameter >= 100 px, phase
   <= 60 deg) left six of the eight real-populated strata -- every
   high-phase and small-body stratum -- with no sim counterpart, so the
   previously reported pooled values (0.36 before / 0.46 after) compared
   different stratum mixtures and were not like-for-like.  The
   measurement path now bypasses those navigation-policy gates and the
   matched ring-scene moon tracks the real body's diameter and phase, so
   every real-populated stratum has a sim counterpart in the current
   record (the summary's ``limb_bins_real_only`` is empty); a
   like-for-like "before" number cannot be reconstructed for the
   pre-tuning model, so the cell is blank.

.. [#floor] Simulated sky is flatter than real sky relative to its own
   noise (real frames carry background gradients and glow), so more sim
   pixels sit within one sigma of the frame floor.  This statistic folds
   in scene content beyond the detector chain; see the known gaps.

The FOM 3 strata behind the pooled number (2026-07-18; p = phase bin
< 60 / 60-120 / > 120 deg, r = diameter bin < 100 / 100-400 / > 400 px):

.. list-table::
   :header-rows: 1
   :widths: 18 14 17 17 17 17

   * - Stratum
     - W1/IQR
     - real median (px)
     - sim median (px)
     - n real
     - n sim
   * - p0 r0
     - 0.21
     - 2.87
     - 2.64
     - 522
     - 1408
   * - p0 r1
     - 0.33
     - 2.75
     - 2.75
     - 400
     - 400
   * - p0 r2
     - 0.80
     - 3.09
     - 2.62
     - 733
     - 780
   * - p1 r0
     - 0.92
     - 2.18
     - 2.45
     - 435
     - 398
   * - p1 r1
     - 0.72
     - 2.24
     - 2.41
     - 922
     - 796
   * - p1 r2
     - 0.45
     - 2.58
     - 2.39
     - 525
     - 200
   * - p2 r0
     - 1.00
     - 1.75
     - 2.29
     - 70
     - 156
   * - p2 r1
     - 2.04
     - 1.71
     - 2.18
     - 148
     - 200

The headline reading: the star PSF now matches the cohort through the
same estimator on both sides (EE50 0.90 sim vs 0.91 real; EE80 1.72 vs
1.79); the stationary artifact incidence matches after the
calibrated-chain hot-pixel retune, while the transient (cosmic-ray)
share stays unmodeled on every chain -- each catalog entry retains an
explicit zero beside its measured rate and the chain-model reason (see
known gap 7); limb rise widths agree in the like-for-like pool
(co-populated-strata medians 2.539 sim vs 2.539 px real, W1/IQR 0.16)
with per-stratum divergences from 0.21 to 2.04 -- worst at high phase,
where the simulated crescent limb measures *wider* than the real one
(2.2-2.3 px vs 1.7 px medians); ring-edge widths agree at the
few-tenths level (2.45 vs 2.87 px); and the sky-noise *level* matches
at the median (2.2e-4 vs 2.1e-4 I/F) while the distributional
divergence remains floored by the quantization-scar gap below.  The
low-phase residual is not a PSF error -- FOM 2 pins the PSF
independently -- but limb content (real topography, the Lommel-Seeliger
approximation); the high-phase excess runs the other way and points at
the crescent shading model.  The curve *shapes* agree far tighter than
the scalar samples: the density-W1 of the frame-averaged curves is
0.06 for the sky power spectrum, 0.13 for the star radial profile,
0.02 for the limb profile, and 0.02 for the ring radial profile (axis
units over the real IQR; ``curve_divergences`` in the summary).

Two caveats apply to every FOM 2/3 width above.  First, a registration
asymmetry: real-side star cutouts and limb profiles are centred through
the sidecar offset and catalog geometry, so operator-verified
registration residuals inflate the real widths one-sidedly -- the
matched sim frames are registered exactly by construction -- and part of
the tuned PSF wing may absorb registration error rather than optics.
Second, the absolute rise-width medians are estimator-specific (the
10-90% estimator biases roughly +17% at the widest profiles and -6% at
the narrowest); what makes the comparison fair is that both sides run
the identical estimator, so the *difference* is meaningful even where
the absolute value is not.

.. figure:: _figures/realism_coiss_calib_nac_noise.png
   :width: 100%
   :alt: Cassini NAC sky-noise distributions, real vs sim

   FOM 1, Cassini NAC: sky-patch noise sigma, sky level above floor,
   noise at signal, and the sky power spectrum.

.. figure:: _figures/realism_coiss_calib_nac_psf.png
   :width: 100%
   :alt: Cassini NAC star radial profile and encircled energy, real vs sim

   FOM 2, Cassini NAC: star radial profiles and encircled-energy radii.
   The tuned kernel reproduces the cohort's EE50/EE80.

.. figure:: _figures/realism_coiss_calib_nac_limb.png
   :width: 100%
   :alt: Cassini NAC limb profile and rise widths, real vs sim

   FOM 3, Cassini NAC: normalized limb profiles and 10-90% rise widths by
   phase (p) and resolution (r) bin.

.. figure:: _figures/realism_coiss_calib_nac_ring.png
   :width: 100%
   :alt: Cassini NAC ring-edge profiles, real vs sim

   FOM 4, Cassini NAC: ring-edge radial profiles and rise widths.

.. figure:: _figures/realism_coiss_calib_nac_dynrange.png
   :width: 100%
   :alt: Cassini NAC exposure-stratified dynamic range, real vs sim

   FOM 5, Cassini NAC: near-floor fractions and signal stretch per
   exposure stratum.

.. figure:: _figures/realism_coiss_calib_nac_artifacts.png
   :width: 100%
   :alt: Cassini NAC artifact incidence, real vs sim

   FOM 6, Cassini NAC: measured line-loss and single-pixel-spike rates
   against the catalog defaults.

The other instruments
---------------------

**Cassini WAC** (4 frames, limited).  The cohort's two star frames give
9 usable star cutouts that pin the tuned WAC kernel: cohort EE50/EE80 =
1.33/2.16 px against simulated 1.13/1.90 px, normalized W1 0.42 and 0.80
(from 1.32 and 3.16 before tuning).  The WAC *noise* comparison, though,
diverges by two orders of magnitude one-sidedly: simulated sky sigma
sits at 2.4e-5 I/F against 1.9e-4 real (W1/IQR 101), noise at signal
likewise (101), and the sky level above floor at 3.7e-5 vs 3.2e-4
(W1/IQR 2481).  A one-sided deficit of that size on the calibrated
chain is a chain symptom, not a small-n artifact -- most plausibly the
nominal DN-to-I/F calibration scale of the simulated chain (the
per-filter CISSCAL scale is not modeled; the same family as known gap
3), though the 4-frame cohort cannot close the attribution.  The
remaining WAC statistics rest on 1-2 frames per stratum and are
reported in the summary JSON without distributional claims.

**Galileo SSI** (8 frames).  Every Galileo frame in the library is a
negative case, so FOMs 2-4 are unsupported and *gossi sim accuracy is
bounded by unverified forward-model fidelity* for PSF and shape rendering
until the star-calibration frames land.  What the cohort does support:
FOM 6 -- the measured spike rate matched the catalog after tuning
(median measured 1.2e-4 of pixels vs simulated 1.7e-4; normalized W1 0.63, from 27.8 before tuning) -- and the observation that the real REDR sky floor
is quantization-locked at exactly 1 LSB while the frames carry an extended
background glow the matched sky frames do not model (see the known gaps).

**Voyager ISS** (3 frames, limited).  The GEOMED star frame gives 8 real
star cutouts (EE50 median 1.22 px) against two simulated ones (EE50
median 3.46 px, after the plateau guard rejects the brightest simulated
core, whose noise-free 8-bit quantization ties its central pixels at 1-2
LSB -- a quantization plateau with no sub-pixel shape information, not a
clipped core) -- one frame and two noise-dominated cutouts
constrain nothing, so the vgiss PSF comparison is *unconstrained*: the
retained interim sigma stands on its published-range provenance alone,
not on cohort evidence.  The vidicon read noise was tuned down from its
high-gain interim value (see the catalog); the sky-noise floor itself is
quantization-limited on the sim side (known gaps).

**New Horizons LORRI** (2 frames, limited).  Both frames are 4x4-binned
star fields; their binned-pixel EE50 (median 0.59 px) cannot constrain the
1x1-mode kernel the catalog carries, so the LORRI kernel is retained and
LORRI sim accuracy for the PSF is bounded by unverified per-mode
fidelity.  The sky statistics are pedestal-invariant by construction (the
sci products are bias-subtracted; the sim raw chain is not).

Technique diagnostics (read-only)
---------------------------------

For a handful of matched pairs per instrument (at most three: one star
field, one limb frame, one ring frame) the full navigator ran on both the
real frame and its matched simulated frame; the per-technique diagnostics
are recorded in the summary JSON (``fom7_rows``).  Headline reading of the
2026-07-18 pairs:

- Star-field pairs behave comparably where both sides lock:
  ``StarFieldFromCatalogNav`` reports 0.95 confidence on both sides of the
  WAC and LORRI star pairs, with the simulated inlier residuals tighter
  than the real ones (0.02-0.05 px vs 0.09-0.33 px) -- the sim errs
  optimistic on star sharpness.
- The Cassini ring pair is indistinguishable: ``RingEdgeNav`` reports
  0.952 confidence on both the real frame and its matched render.
- The limb pairs split: on the NAC pair both sides succeed (0.77 real vs
  0.81 sim) with the simulated DT residual three times the real one;
  on the WAC pair the real limb fit succeeds at 0.83 while the matched
  simulated result is discarded by its own ensemble.
- The NAC star pair inverts: the real frame's field lock fails (3
  inliers; a known caveat of that cohort frame) while its matched render
  locks at 0.95 -- the painted-silhouette star occlusion gap and the
  clean simulated field both push in the optimistic direction.

These rows are the evidence a human reads to judge whether the techniques
*behave* comparably on matched frames.  They were not consulted during
tuning and never will be: they are built from the navigator's own outputs,
and fitting the image model to them would make navigator errors invisible
by construction.

Known gaps
----------

Where a figure of merit reveals a forward-model gap that parameter tuning
cannot close, it is recorded here rather than force-fitted:

1. **Stars shine through dark limbs.**  Star occlusion by bodies uses the
   painted *lit* silhouette (the body mask is where the rendered body is
   brighter than zero), so a star behind the un-lit part of a limb still
   shines.  Simulated star-technique success on body-crossing fields is
   therefore optimistic: real frames lose stars behind dark limbs that
   simulated frames keep.
2. **Occluded limbs claim full reliability.**  In mutual-event scenes the
   simulated body model emits its limb arc with full
   ``visible_arc_fraction`` and reliability even where another body
   occludes it; the occlusion is recorded in the scene truth but does not
   yet reduce the emitted feature's claim.  Pending the recalibration
   pass, mutual-event confidences remain provisional.
3. **Calibrated-product quantization scars.**  The sim's calibrated path
   retains full LSB quantization texture: its measured sky-noise floor is
   ~1 DN-equivalent at every exposure (Cassini CALIB and Voyager GEOMED
   paths alike).  Real calibrated products sit *below* the LSB (NAC CALIB
   ~0.1-0.4 DN-equivalent, GEOMED ~0.22 DN) because their float-valued
   corrections -- flat fields, dark/bias frames, 2-Hz filtering, geometric
   resampling -- dither the quantization away.  This floors the FOM 1
   sky-noise divergence for every calibrated cohort; closing it needs a
   calibration-scar model (sub-LSB dithering / resample texture), not
   parameter tuning.  Tuning read noise below its published value to
   compensate would misattribute the mismatch and was not done.  The
   per-filter absolute DN-to-I/F scale is likewise a nominal identity
   (signal 1.0 = I/F 1.0 at the reference exposure) rather than the real
   per-filter CISSCAL scale, so I/F noise amplitudes carry that
   uncertainty too.  The Cassini WAC comparison shows that scale
   uncertainty at full size: its simulated sky sigma undershoots the
   real cohort by roughly 8x one-sidedly (2.4e-5 vs 1.9e-4 I/F; see the
   WAC subsection).
4. **Hot pixels are per-scene, not per-detector.**  The simulated
   hot-pixel population is drawn from each scene's seeded stream, so it
   never recurs at fixed detector positions across frames the way real
   hot pixels do.  The FOM 6 stationary/transient split therefore cannot
   match by construction; only the total incidence is comparable (and now
   tuned).
5. **No per-readout-mode PSF.**  The catalog carries one kernel per
   instrument; the LORRI cohort's 4x4-binned frames (and any summed
   Cassini modes that enter the library later) see a different effective
   kernel that the catalog cannot express.
6. **Matched-frame content limits.**  Simulated ring edges are sharp
   optical-depth steps and simulated matched bodies are relief-free
   ellipsoids, so residual FOM 3/4 width differences fold in real edge
   structure (ring radial profiles, limb topography) beyond the PSF that
   FOM 2 pins independently.  The Galileo negative frames carry an
   extended background glow (scattered light) that matched sky frames do
   not model, which dominates that cohort's FOM 5 comparison.
7. **Measured transients exceed what the cosmic-ray stage can express.**
   Every cohort measures a nonzero transient spike fraction (NAC 2.75e-4,
   WAC 4.89e-4, Galileo 1.17e-4, LORRI 3.36e-4 per frame), yet no
   catalog entry adopts a ``cosmic_ray_rate_per_sec``: the Cassini
   incidence is exposure-independent (readout-dominated) and
   modest-amplitude while the chain's stage scales event counts with
   exposure and deposits near full well -- an adoption attempt
   (2026-07-17) inflated the matched frames' sky-noise statistics ~5x
   and was reverted; the Galileo population is already carried by the
   tuned per-scene hot pixels, so a separate term double-counts it; the
   LORRI measurement is star-contaminated (both cohort frames are binned
   star fields); and the Voyager vidicon path has no transient stage.
   Each catalog entry records its measured rate and the unblocking
   condition beside the retained zero.

Summary
=======

* All seven feature techniques (disc, mesh disc, blob, high-phase blob, star
  field, limb, ring edge) plus the camera-roll fit recover their planted
  transform to sub-pixel / sub-half-degree accuracy on clean simulated frames.
* The disc, blob, ring edge, and star field are quantization-free: they recover
  any offset -- whole, near-boundary, or arbitrary fraction -- to a few hundredths
  of a pixel. The disc and blob are the most accurate (~0.01 px). The limb holds a
  ~0.09 px distance-transform residual and the ring a ~0.03 px residual.
* The star field improves with SNR: a dim field recovers to ~0.02 px and a bright
  field to ~0.005 px, the most accurate of any technique.
* Irregular (mesh) bodies navigate as accurately as ellipsoids when the predicted
  geometry matches the rendered one (mesh disc, limb, and crescent all recover to
  0.00-0.21 px). When the navigator predicts the wrong shape the centroid bias
  grows with the rendered relief (to ~4 px) and the confidence falls; when it
  predicts the wrong pose the limb degrades to a confidently-wrong fix (or, for a
  wrong in-plane roll, far enough that it self-flags) while the pose-free blob
  holds -- the demote-to-pose-free behaviour a chaotic rotator needs.
* The sweeps show the expected qualitative behaviour: navigation degrades to a
  clean failure past the noise cliff, the resolved body recovers across the full
  phase range with no mid-phase accuracy penalty, and the primary technique walks
  the limb -> disc -> blob ladder as a body shrinks.
* A small camera roll is not separable from a translation: the field matcher
  recovers rolls only above ~0.75 deg (the two-star fit extends this to ~0.5 deg),
  and the small-body navigation floor (the 5 px ``BODY_BLOB`` emission floor) is
  set by the feature-emission gate, not the centroid algorithm. Navigation is
  unit-agnostic: I/F frames navigate identically to raw DN.
* The confidence column reflects the sim-calibrated per-technique formulas
  (limb ~0.99, disc ~0.65-0.77, blob capped at 0.40 on clean frames); the report
  verifies the recovered geometry and the technique selection, while tier
  behaviour on real frames is validated against the operator-curated library.
