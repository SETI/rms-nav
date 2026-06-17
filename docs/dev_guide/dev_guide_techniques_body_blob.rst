==========================================================
Body Blob Centroid (BodyBlobNav)
==========================================================

Overview
========

:class:`~nav.nav_technique.nav_technique_body_blob.BodyBlobNav` recovers a single translation
from one or more body brightness centroids. For each offered ``BODY_BLOB`` feature the technique
computes a brightness-weighted moment inside the predicted bounding box, compares the
observed centroid to the predicted lit-weighted centroid carried on the feature, and runs an
inverse-variance-weighted joint fit across every consumed body to recover the per-image
translation. With ``N >= 2`` blobs the fit is over-determined and the joint solution is
robust to centroid errors on any single body.

Feasibility passes when at least one offered ``BODY_BLOB`` carries a non-zero predicted
diameter; feasibility fails when every blob has degenerate geometry (a sub-pixel body that
collapses the brightness-weighted moment).

The technique reports a confidence intrinsically capped at 0.4 via the spec's
:attr:`~nav.nav_technique.confidence.ConfidenceSpec.hard_cap` — a brightness-weighted
centroid is much weaker evidence than a limb fit, so even an ideal blob match cannot dominate
the ensemble combine.

Theory
======

The technique fits a per-image translation by minimising the inverse-variance-weighted
squared residual between the per-blob observed and predicted centroids.

Per-blob centroid
-----------------

For each consumed body, the technique computes the brightness-weighted moment over the
predicted bounding box:

.. math::

    \bar{x}_{\mathrm{obs}} =
        \frac{\sum_{(v, u) \in \mathrm{bbox}} I(v, u) \,(v, u)}
             {\sum_{(v, u) \in \mathrm{bbox}} I(v, u)}.

The predicted centroid carried on the feature is the lit-weighted moment of the rendered
model (see :doc:`dev_guide_navigation_models_body`); the per-blob residual is

.. math::

    r_{i} = \bar{x}_{\mathrm{obs},\,i} - \bar{x}_{\mathrm{pred},\,i}.

Per-blob covariance
-------------------

The per-blob centroid uncertainty follows the standard CRLB scaling for a uniform-brightness
disc:

.. math::

    \sigma_{\mathrm{centroid}} \approx
        \frac{D_{\mathrm{px}}}{2 \, \sqrt{N_{\mathrm{lit}}} \, \mathrm{SNR}}

where :math:`D_{\mathrm{px}}` is the predicted disc diameter in pixels,
:math:`N_{\mathrm{lit}}` is the number of lit pixels inside the predicted bounding box, and
SNR is the per-pixel signal-to-noise ratio. Two additional contributions are added in
quadrature: a shape-irregularity sigma scaling with the body's
phase-and-irregularity coupling :math:`\kappa`
(see :doc:`dev_guide_navigation_models_body`) times half the predicted disc diameter, and the
photon-noise floor. The total per-blob sigma populates a 2x2 isotropic covariance.

Joint translation fit
---------------------

The joint translation minimises

.. math::

    C(\Delta v, \Delta u) = \sum_{i} w_{i} \,\bigl\| r_{i} - (\Delta v, \Delta u) \bigr\|^{2},
    \qquad w_{i} = \frac{1}{\sigma_{i}^{2}}.

The closed-form minimum is the inverse-variance-weighted mean of the per-blob residuals:

.. math::

    (\Delta v, \Delta u)^{*} = \frac{\sum_{i} w_{i} \, r_{i}}{\sum_{i} w_{i}}.

The reported translation covariance is :math:`\sigma^{2} I` with :math:`\sigma^{2} =
1 / \sum_{i} w_{i}` — the standard precision-weighted-mean variance.

Restrictions and assumptions
----------------------------

- Per-blob centroids assume the bounding box truly contains the body's flux. When a
  cosmic-ray hit, an in-band stellar source, or a neighbouring body's halo lands inside the
  predicted bounding box, the moment skews and the technique reports a wrong centroid. The
  upstream ``BODY_BLOB`` emission gates filter pathological cases (see
  :doc:`dev_guide_navigation_models_body`).
- A vanishing total flux (an entirely-in-shadow body whose predicted bounding box happens to
  cover the right part of the FOV) collapses the moment; the technique drops such blobs
  before the joint fit and reports a no-signal failure when every blob is dropped.
- **Very small bodies are deliberately gated out.** The ``BODY_BLOB`` feature's reliability
  carries a ``blob_extent_px`` term that drives reliability below the keep threshold for a
  body only a handful of pixels across (on the simulated catalog a 20 px body passes but a
  ~24 px-or-smaller body sits just under the gate). This is intentional: on a *real* frame a
  body a few pixels wide is dominated by the point-spread function, cosmic rays, and
  background structure, so its brightness-weighted centroid is not trustworthy even though the
  arithmetic still produces a number. Navigating bodies below that floor is therefore held
  back pending calibration against the operator-curated real-image library; the floor is a
  config-tunable gate, not a hard algorithmic limit.
- The technique carries no rotation evidence — a brightness-weighted centroid is rotation-
  invariant about itself. When the per-instrument
  :attr:`~nav.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true, the
  technique returns the rank-deficient 3x3 covariance from
  :func:`~nav.nav_technique.nav_technique.embed_rotation_unobservable` and reports
  :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.rotation_rad` as zero with
  the rotation-unobservable sentinel sigma.

Sources of uncertainty
----------------------

The reported covariance is the precision-weighted-mean variance described above. It does
not capture systematic biases from a body whose true rotational orientation differs from the
rendered ellipsoid (the
:attr:`~nav.feature.flags.BodyBlobFlags.phase_irregularity_factor` term tracks this so the
confidence formula can down-weight the technique on irregular high-phase scenes, but the
centroid itself remains biased). When the converged offset sits within
:math:`\mathtt{at\_edge\_tolerance\_px}` of any axis bound, the result is flagged
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` and the hard-zero
gate forces confidence to zero.

Configuration
=============

All numeric tunables for this technique live in ``techniques.BodyBlobNav.tuning`` in
``src/nav/config_files/config_510_techniques.yaml``.

- ``at_edge_tolerance_px`` — float, default ``1.0`` px. A converged offset whose absolute
  distance from any search-window axis bound falls within this tolerance is flagged
  :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge`.

The remaining numeric thresholds (the per-blob CRLB scaling constants, the noise-floor
detection threshold) are derived from the per-image
:attr:`~nav.nav_orchestrator.nav_context.NavContext.image_noise_sigma` and the per-blob
geometry; no YAML knob is exposed.

Per-instrument overrides
------------------------

The ``at_edge_tolerance_px`` knob is global; per-instrument YAML files in
``src/nav/config_files/config_4N0_inst_*.yaml`` do not override it. The
search-window margin used by the at-edge test comes from the per-instrument
:class:`~nav.nav_orchestrator.instrument_config.InstrumentSettings`.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared
sigmoid combination; see :doc:`dev_guide_techniques_confidence` for the per-term arithmetic.
The formula spec is ``techniques.BodyBlobNav`` in the same YAML file and consumes attributes
off :class:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics` plus
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge`.

- :attr:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics.body_snr_inside_predicted_bbox`
  — alpha = 0.5, offset = 0.0, divisor = 4.0, cap at 1.0. Per-image SNR inside the
  predicted bounding box. Brightness-weighted centroid uncertainty shrinks with SNR.
- :attr:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics.body_extent_px` — alpha = 1.0,
  offset = 8.0, divisor = 8.0, cap at 1.0. Predicted body's longer-axis extent in pixels.
  Larger blobs carry more centroid signal up to a 16-pixel saturation point.
- :attr:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics.blob_count` — alpha = 0.4,
  offset = 0.0, divisor = 3.0, cap at 1.0. Number of ``BODY_BLOB`` features fused.
  Multi-body geometry over-determines the joint translation up to a 3-blob saturation.
- :attr:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics.max_phase_irregularity_factor`
  — alpha = 0.0, offset = 0.0, divisor = 0.15, cap at 1.0. Maximum phase-and-irregularity
  factor across the consumed blobs (see :doc:`dev_guide_navigation_models_body` for the
  formula). The term carries no weight in the current confidence formula; the wiring is in place so a downstream
  recalibration can tune the alpha without code changes.

Hard-zero gate: :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge`
firing forces confidence to zero before the sigmoid evaluates. The constant baseline is
:math:`\alpha_{0} = -1.0`. A post-sigmoid ``hard_cap`` of ``0.4`` clamps the result: a
brightness-weighted centroid cannot drive the ensemble past 0.4 confidence even when every
term saturates.

Implementation
==============

Source files:

- ``src/nav/nav_technique/nav_technique_body_blob.py`` —
  :class:`~nav.nav_technique.nav_technique_body_blob.BodyBlobNav`, the per-blob residual
  collector, and the joint-translation helper.
- ``src/nav/nav_technique/confidence.py`` — shared sigmoid-combination evaluator;
  documented at :doc:`dev_guide_techniques_confidence`.
- ``src/nav/nav_technique/diagnostics.py`` —
  :class:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics`; documented at
  :doc:`dev_guide_techniques_diagnostics`.

Public class :class:`~nav.nav_technique.nav_technique_body_blob.BodyBlobNav`, base
:class:`~nav.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__`` so ``NavTechnique._registry`` discovers it.

Class attributes:

- :attr:`~nav.nav_technique.nav_technique_body_blob.BodyBlobNav.name` — ``'BodyBlobNav'``.
- :attr:`~nav.nav_technique.nav_technique_body_blob.BodyBlobNav.accepts_feature_types` —
  ``frozenset({BODY_BLOB})``.
- :attr:`~nav.nav_technique.nav_technique_body_blob.BodyBlobNav.requires_prior` — ``False``.
  Runs in pass 1 of the orchestrator's two-pass pipeline.
- :attr:`~nav.nav_technique.nav_technique_body_blob.BodyBlobNav.confidence_attributes` —
  ``{'at_edge', 'body_snr_inside_predicted_bbox', 'body_extent_px', 'blob_count',
  'residual_px', 'max_phase_angle_deg', 'max_phase_irregularity_factor'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~nav.nav_technique.nav_technique_body_blob.BodyBlobNav.is_feasible` and
:meth:`~nav.nav_technique.nav_technique_body_blob.BodyBlobNav.navigate`.

Diagnostics
-----------

:class:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics`:

- :attr:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics.body_snr_inside_predicted_bbox`
  — per-image SNR inside the predicted bounding box. Consumed by the confidence formula.
- :attr:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics.body_extent_px` — predicted
  body's longer-axis extent in pixels. Consumed by the confidence formula.
- :attr:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics.blob_count` — number of
  ``BODY_BLOB`` features fused. Consumed by the confidence formula.
- :attr:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics.residual_px` — joint-fit RMS
  residual after solving the precision-weighted mean. Diagnostic only; not in the formula.
- :attr:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics.max_phase_angle_deg` — maximum
  raw phase angle across consumed blobs. Diagnostic only; the formula consumes
  ``max_phase_irregularity_factor`` instead because raw phase understates the centroid
  uncertainty for an irregular body.
- :attr:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics.max_phase_irregularity_factor` —
  maximum :math:`\sin(\phi/2) \cdot \sigma_{\mathrm{ellipsoid}} / R_{\mathrm{body}}` across
  consumed blobs. Consumed by the confidence formula (alpha=0.0 — the wiring is
  in place so the formula picks up a recalibrated alpha without code changes).

Call path
---------

Call path traced through
:meth:`~nav.nav_technique.nav_technique_body_blob.BodyBlobNav.navigate`:

1. Open a logged section. Filter the offered features down to ``BODY_BLOB`` entries with a
   non-zero predicted diameter via the private eligibility helper.
2. Read the search-window margin off the observation via
   :func:`~nav.nav_technique.nav_technique.search_window_for_obs`, the extfov image off
   :attr:`~nav.nav_orchestrator.nav_context.NavContext.image_ext`, and the per-image noise
   sigma off :attr:`~nav.nav_orchestrator.nav_context.NavContext.image_noise_sigma`
   (clamped at a tiny floor so the noise-floor test stays well-defined on near-blank
   inputs).
3. For each eligible blob, the private residual collector slices the predicted bounding
   box from the extfov image, evaluates the brightness-weighted-moment centroid, drops the
   blob when total flux falls below the noise floor, computes the per-blob residual against
   the predicted lit-weighted centroid, and accumulates the per-blob inverse-variance
   weight.

   - **No surviving blobs.**  The technique returns a spurious zero-confidence result via
     the private fail helper, with the corresponding
     :class:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics`.

4. The private joint-fit helper computes the inverse-variance-weighted-mean translation and
   the precision-weighted-mean covariance.
5. Apply the at-edge test against the search-window axis bounds.
6. Result-shape branches on
   :attr:`~nav.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`:

   - **No rotation fit.**
     :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.covariance_px2` is the
     (2, 2) translation block.
     :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.rotation_rad` and
     :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad` are
     ``None``.
   - **Rotation fit.**  The technique embeds the (2, 2) translation block in a (3, 3)
     covariance via
     :func:`~nav.nav_technique.nav_technique.embed_rotation_unobservable`, sets
     :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.rotation_rad` to ``0.0``,
     and reports
     :func:`~nav.nav_technique.nav_technique.rotation_unobservable_sigma_rad` as the
     :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad`.
     A brightness-weighted centroid is rotation-invariant about itself, so the technique
     carries no rotation evidence.

7. Build a :class:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics` from the per-blob
   residuals (max-SNR, max-extent, blob-count, RMS residual, max raw phase, max
   phase-irregularity factor), evaluate the confidence spec via
   :func:`~nav.nav_technique.confidence.evaluate_sigmoid_combination`, log the per-term
   breakdown via :func:`~nav.nav_technique.nav_technique.log_confidence_breakdown`, and
   assemble the :class:`~nav.nav_technique.technique_result.NavTechniqueResult`.

The :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.feature_ids` field
preserves every consumed
:attr:`~nav.feature.feature.NavFeature.feature_id` so the orchestrator's curator can
attribute each contribution at audit time.

Examples
========

``below_resolution_body`` (Cassini ISS NAC, image ``N1777325846_1``)
    Mimas approximately 20 px in diameter in the lower left, at phase angle 72 degrees. The
    body model emits a single ``BODY_BLOB`` feature (the per-pixel ellipsoid uncertainty exceeds
    :data:`~nav.nav_model.nav_model_body.LIMB_ARC_MAX_UNCERTAINTY_PX` so ``LIMB_ARC`` is
    suppressed in favour of the centroid path).
    :class:`~nav.nav_technique.nav_technique_body_blob.BodyBlobNav` consumes the blob and
    converges within ~1 px of the operator-verified offset
    :math:`(\Delta v, \Delta u) = (6.08, -1.53)` px. The post-sigmoid hard cap of 0.4
    keeps the technique from outranking a hypothetical limb fit on a similar but
    well-resolved scene.

``multi_body`` (Cassini ISS NAC, image ``N1487595731_1``)
    Dione and Rhea both visible at phase angle approximately 90 degrees. When the body
    model emits ``BODY_BLOB`` features for both bodies (or one body's limb fails the uncertainty
    gate), :class:`~nav.nav_technique.nav_technique_body_blob.BodyBlobNav` fuses the two
    centroids into a joint translation. The 3-blob saturation in the confidence formula is
    not reached, but the multi-body
    :attr:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics.blob_count` term still
    contributes a positive offset. Operator-verified offset is
    :math:`(\Delta v, \Delta u) = (7.03, -18.42)` px.
