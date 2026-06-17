============================================
Simulator Performance and Sensitivity Report
============================================

This chapter reports how the RMS-NAV navigation pipeline performs on simulated
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

Per the simulator's cardinal principle, **the sim verifies, real data
calibrates**. The sim does not set the per-technique confidence coefficients --
those are tuned against the operator-curated real-image library. What the sim
does is exercise the technique *algorithms* against ground truth that is correct
by construction, and characterise how a navigation diagnostic *responds* to a
controlled single-parameter change. This report is the summary of those two
measurements:

* **Algorithmic-invariant recovery** -- a planted offset (or camera roll) the
  navigator must recover, for each technique in the ladder.
* **Single-variable sensitivity** -- a base scene driven across noise, phase, and
  body size, showing the navigability cliff, the phase response, and the
  technique-selection transitions.

Because the confidence coefficients are uncalibrated placeholders, the report
keys on the *recovered geometry* (offset error, roll error, primary technique,
success/fail), not on the absolute confidence value, which barely moves on these
clean frames and is a Phase 10 calibration concern.

How to regenerate
=================

The recovery numbers come from navigating the algorithmic-invariant scenes:

.. code-block:: bash

   pytest tests/integration/test_sim_algorithmic_invariants.py -m "" -n auto --dist=loadfile

The sweep response curves come from the sweep runner, which writes one JSON per
sweep under ``tests/integration/sim_sweeps/results/``:

.. code-block:: bash

   python -m tests.integration.sim_sweep_runner

See :doc:`/user_guide/user_guide_simulated_images` for the scene-catalog and
sweep workflow, and :doc:`/dev_guide/dev_guide_navigation_models` for the
simulated models that emit the features each technique consumes.

Algorithmic-invariant recovery
==============================

Each scene below plants a known transform and the navigator predicts the
unshifted geometry, so the recovered offset (or roll) should equal the planted
value. The technique column names the load-bearing technique -- pinned for the
scenes whose fused confidence sits below the success threshold on a clean frame
(blob, limb, ring, roll), and the full ensemble for the disc and star scenes.

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
     - (3.5, -2.0) px
     - (3.50, -1.80) px
     - 0.20 px
   * - ``planted_offset_irregular``
     - BodyDiscCorrelateNav (mesh)
     - (2.0, 1.5) px
     - (1.75, 1.50) px
     - 0.25 px
   * - ``planted_offset_blob``
     - BodyBlobNav
     - (1.5, -0.5) px
     - (1.49, -0.51) px
     - 0.02 px
   * - ``planted_offset_blob_crescent``
     - BodyBlobNav (120 deg)
     - (1.5, -0.5) px
     - (1.32, -0.45) px
     - 0.18 px
   * - ``planted_offset_star_field``
     - StarField + UniqueMatch + Refine
     - (1.5, -0.5) px
     - (1.52, -0.51) px
     - 0.02 px
   * - ``planted_offset_limb``
     - BodyLimbNav
     - (1.5, -0.5) px
     - (1.51, -0.51) px
     - 0.02 px
   * - ``planted_offset_ring``
     - RingEdgeNav
     - (1.5, -0.5) px
     - (1.50, -0.50) px
     - 0.004 px
   * - ``planted_rotation_star_field``
     - StarFieldFromCatalogNav (roll)
     - 1.50 deg
     - 1.507 deg
     - 0.007 deg

Observations:

* Every technique recovers its planted transform to well under a pixel (or a
  third of a degree for the roll). The point-feature techniques -- ring edge,
  blob, limb, and the star field -- are the most precise (a few hundredths of a
  pixel), because their predicted geometry aligns exactly with a sharp image
  feature.
* The disc correlation carries the largest error (~0.2 px) -- not an inherent
  limit but a correlator bug the offset sweep below pins down: its gradient-mode
  sub-pixel refinement is biased near the peak, while raw-intensity NCC and the
  feature techniques reach ~1/128 px. The mesh-body disc adds a shape error on
  top, since the body is irregular but the correlation template is the same
  rendered shape. Both stay well inside the 1.0 px invariant bound.
* The high-phase blob crescent (~0.18 px at 120 deg) is the hardest case: only a
  thin lit crescent constrains the centroid. It still recovers sub-pixel because
  the blob subtracts the bias pedestal and thresholds against sky noise rather
  than the body-inflated global noise.

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
     - 0.30
     - BodyDiscCorrelateNav
   * - 4
     - success
     - 0.00 px
     - 0.30
     - BodyDiscCorrelateNav
   * - 8
     - success
     - 0.00 px
     - 0.30
     - BodyDiscCorrelateNav
   * - 16
     - success
     - 0.00 px
     - 0.30
     - BodyDiscCorrelateNav
   * - 32
     - success
     - 0.00 px
     - 0.30
     - BodyDiscCorrelateNav
   * - 64
     - **failed**
     - --
     - 0.00
     - --

The disc correlation is robust: the offset is recovered exactly until the read
noise overwhelms the body signal, at which point the frame is classified
unnavigable and navigation fails cleanly. The flat confidence reflects the
uncalibrated placeholder coefficients -- a calibrated formula would taper the
confidence down toward the cliff rather than stepping off it (a Phase 10 / T7
concern).

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
     - 0.30
     - BodyDiscCorrelateNav
   * - 60
     - success
     - 0.27 px
     - 0.30
     - BodyDiscCorrelateNav
   * - 90
     - success
     - 0.29 px
     - 0.29
     - BodyDiscCorrelateNav
   * - 120
     - success
     - 0.34 px
     - 0.28
     - BodyDiscCorrelateNav
   * - 150
     - success
     - 0.01 px
     - 0.30
     - BodyDiscCorrelateNav

The resolved body navigates to success at every phase and recovers within a third
of a pixel. The disc-correlation error rises through the mid-phase range (60-120
deg) as the terminator eats into the lit disc and the correlation template
matches less of the image, then falls back near full phase (150 deg) where the
crescent is again a sharp, well-defined feature. At zero phase the blob's
lit-weighted centroid wins outright -- a fully-lit disc has no correlation
gradient advantage.

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
     - 0.00 px
     - 0.28
     - BodyLimbNav
   * - 90
     - success
     - 0.00 px
     - 0.30
     - BodyDiscCorrelateNav
   * - 60
     - success
     - 0.00 px
     - 0.28
     - BodyDiscCorrelateNav
   * - 40
     - success
     - 0.00 px
     - 0.31
     - BodyDiscCorrelateNav
   * - 20
     - success
     - 0.00 px
     - 0.40
     - BodyBlobNav
   * - 12
     - **failed**
     - --
     - 0.00
     - --

The primary technique transitions cleanly as resolution falls: a well-resolved
body (130 px) is navigated by the limb fit; a mid-size body by the disc
correlation; a small body (20 px) falls to the orientation-free blob centroid;
and the smallest body (12 px) is unnavigable. Every navigable step recovers the
planted offset exactly. This transition is the sim's most direct verification
that the orchestrator selects the right technique for the available resolution.

Sub-pixel offset accuracy across the pixel
==========================================

The invariant scenes above all plant a single offset near the middle of a pixel.
To check for pixel-boundary and quantization artifacts, two sweeps plant the same
body offset across a range that includes whole pixels, the near-boundary 0.99 px,
the exact half- and quarter-pixel, and arbitrary non-repeating fractions
(0.12783, 0.73912) -- one on a small body navigated by the blob centroid, one on
a resolved body navigated by the disc correlation.

.. list-table:: Recovery error vs planted offset (px)
   :header-rows: 1
   :widths: 18 20 20

   * - Planted offset
     - Blob centroid
     - Disc correlation
   * - 0.0
     - 0.007
     - 0.33
   * - 0.12783
     - 0.010
     - 0.36
   * - 0.25
     - 0.009
     - 0.38
   * - 0.5
     - 0.003
     - 0.00
   * - 0.73912
     - 0.003
     - 0.24
   * - 0.99
     - 0.006
     - 0.33
   * - 1.0
     - 0.007
     - 0.33
   * - 1.5
     - 0.003
     - 0.00
   * - 2.99
     - 0.006
     - 0.33

The **blob centroid is quantization-free**: it recovers every offset --
whole-pixel, near-boundary, perfect-fraction, or arbitrary -- to a few
hundredths of a pixel, with no dependence on the fractional part. The feature
techniques (limb, ring edge, star field) behave the same way, because they fit a
sharp predicted geometry to a sharp image feature.

The **disc correlation shows a striking periodic error**: ~0.33 px at whole-pixel
offsets, falling to ~0 at the exact half-pixel, with a period of one pixel. This
is *not* a fundamental NCC limit. The correlator upsamples its correlation
spectrum to 1/128 px and reaches that accuracy on raw intensity -- a zero shift
between two identical frames recovers exactly ``(0, 0)``. The bias appears only on
the **gradient-magnitude** pass, which the disc's ``auto`` mode selects for a
smooth body because the edge gives a higher-contrast correlation peak. The Sobel
*magnitude* rectifies the signal, so the gradient correlation peak is non-smooth
at its apex, and the sub-pixel estimator is biased whenever the residual from the
nearest whole pixel is small -- which recurs at every integer offset, vanishing at
the half-pixel where the residual is farthest from the cusp.

This is a core-correlator behaviour (it lives in
:func:`nav.support.correlate.navigate_with_pyramid_kpeaks`, not in the simulator),
so it applies to real-image disc navigation too. It is flagged for a fix --
refining the final sub-pixel offset on raw intensity rather than gradient
magnitude -- which should be validated against the real-image library before
landing. The offset sweep is the regression that will confirm the fix and guard
against recurrence; it is the clearest example of the simulated layer surfacing a
real navigation defect.

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

The key limitation: **a small roll is not separable from a translation**.
``StarFieldFromCatalogNav``'s RANSAC pattern matcher cannot distinguish a sub-0.75
deg roll from a pure shift of the field, so it returns a zero roll (and a spurious
flag) below that floor. The two-star ``StarUniqueMatchNav`` path -- a rigid
two-point fit -- extends recovery down to ~0.5 deg, which is why the *full
ensemble* recovers the 0.5 deg roll even though the field matcher alone does not.
At exactly zero roll there is no rotation signal to fit, so no technique reports
one. Above ~2-3 deg the outer stars rotate past the matcher's inlier tolerance and
the inlier count falls below quorum. The usable window for the field matcher is
therefore roughly 0.75-2 deg, widening to ~0.5 deg with the two-star path.

Small-body navigation floor
===========================

The range sweep above fails at a 12 px body, which is small but not
fundamentally unnavigable. The cause is not the algorithm: at a 16 px body the
blob centroid is still exact (the correlation log reports the right peak), but the
``BODY_BLOB`` feature's reliability falls just below the gate
(``reliability 0.18 < threshold 0.20``) because its ``blob_extent_px`` term scores
a small body low. At 24 px the reliability clears the gate (0.22) and the body
navigates. The floor is thus set by the **reliability calibration**, not the
centroid: the gate deliberately distrusts tiny bodies because on a *real* frame a
handful of pixels is noise- and PSF-dominated. The centroid itself works well
below the gate, so navigating down to a few pixels is a calibration decision (a
Phase 10 concern), not an algorithmic barrier -- and one that must be tuned
against real data rather than the noise-light sim.

I/F-calibrated vs raw-DN navigation
===================================

Every sweep and scene above renders in raw DN. The
``planted_offset_disc_if`` invariant scene confirms navigation is **unit-agnostic**:
the same body on the I/F-calibrated ``coiss_calib_nac`` instrument recovers the
planted offset exactly. The navigation techniques key on scale-invariant
quantities -- normalised cross-correlation for the disc, a MAD-relative noise
threshold for detection and the blob -- so they do not care whether a pixel is in
DN or I/F. The differences are in the detector model, not the navigation: the
``calibrated_if`` render path leaves the composed signal in [0, 1] I/F units and
applies **no** DN detector model (no Poisson shot noise, no full-well saturation
gate, no bias pedestal or missing-data markers), because those map onto DN, not
I/F. A consequence worth noting is that simulated I/F frames are currently
noise-light -- realistic I/F noise is a deferred sim feature -- so an I/F scene
exercises the navigation algorithms but not yet a realistic I/F noise regime.

Summary
=======

* All seven feature techniques (disc, mesh disc, blob, high-phase blob, star
  field, limb, ring edge) plus the camera-roll fit recover their planted
  transform to sub-pixel / sub-half-degree accuracy on clean simulated frames.
* The point-feature techniques (blob, limb, ring edge, star field) are
  quantization-free: they recover any offset -- whole, near-boundary, or arbitrary
  fraction -- to a few hundredths of a pixel.
* The offset sweep surfaced a real core-correlator defect: the disc's
  gradient-magnitude sub-pixel refinement is biased (~0.3 px, periodic in the
  fractional offset) where raw NCC reaches ~1/128 px. It affects real disc
  navigation and is flagged for a fix to be validated against the real library.
* The sweeps confirm the expected qualitative behaviour: navigation degrades to a
  clean failure past the noise cliff, the resolved body handles the full phase
  range with a mid-phase accuracy dip, and the primary technique walks the limb
  -> disc -> blob ladder as a body shrinks.
* A small camera roll is not separable from a translation: the field matcher
  recovers rolls only above ~0.75 deg (the two-star fit extends this to ~0.5 deg),
  and the small-body navigation floor (~16-24 px) is set by the blob reliability
  calibration, not the centroid algorithm. Navigation is unit-agnostic: I/F frames
  navigate identically to raw DN.
* The confidence column is flat by design -- the coefficients are uncalibrated
  placeholders, so the absolute confidence is not yet a calibrated tier. Turning
  these sweeps into confidence-monotonicity tripwires is the Phase 10 calibration
  / T7 work; today they verify the recovered geometry and the technique
  selection, which is what the simulated layer is for.
