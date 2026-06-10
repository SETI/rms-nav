====================
Body Blob Navigation
====================

Overview
========

The body-blob technique exploits the brightness centroid of a body too small to resolve as a
shape: when a body spans only a handful of pixels its limb, terminator, and disc carry no
usable geometric signal, but the brightness-weighted centroid of its lit pixels still localises
it.  For each body the technique computes the moment centroid of the above-noise pixels inside
the body's predicted bounding box, takes the offset between that observed centroid and the
predicted centre as a residual, and solves one precision-weighted translation that maps the
predicted centroids onto the observed ones across all bodies.  Because a centroid is a far
weaker observation than a limb fit, the technique's confidence is hard-capped at 0.4 so an
ideal blob match still cannot dominate the ensemble.  Feasibility passes when at least one
body-blob feature has a non-zero predicted diameter.  Feasibility fails when no body-blob
feature carries a predicted diameter, since the centroid moment is then degenerate.

Theory
======

A point-like or barely-resolved body contributes a compact brightness peak whose
intensity-weighted centroid is the natural position estimate.  The centroid is taken only over
pixels exceeding a noise threshold inside the predicted bounding box, so the background never
biases the moment.  For body :math:`b` with above-noise pixel set :math:`S_b` and intensities
:math:`w_k`, the observed centroid is

.. math::

   \hat{\mathbf{c}}_b = \frac{\sum_{k \in S_b} w_k\,\mathbf{x}_k}{\sum_{k \in S_b} w_k},

and the per-body residual is the difference between this observed centroid and the predicted
centre.  Under a rigid pointing error every body shares one translation
:math:`\boldsymbol{\delta}`, recovered as the precision-weighted mean of the per-body
residuals,

.. math::

   \boldsymbol{\delta} = \frac{\sum_b W_b\,\left(\hat{\mathbf{c}}_b -
   \mathbf{c}_b^{\mathrm{pred}}\right)}{\sum_b W_b}.

The per-body weight follows the Cramer-Rao bound for the centroid of a uniform-brightness
disc, in which the centroid uncertainty scales as the diameter divided by the lit pixel count
and the signal-to-noise ratio:

.. math::

   W_b = \frac{N_b\,\mathrm{SNR}_b^{2}}{R_b^{2}},

with :math:`N_b` the lit pixel count, :math:`\mathrm{SNR}_b` the mean above-noise signal
divided by the noise sigma, and :math:`R_b` the predicted radius.  With two or more bodies the
fit is over-determined, so it tolerates a centroid error on any one body; with a single body it
is exactly determined and there is no residual scatter to estimate.

The covariance is diagonal.  With :math:`N` bodies and two fitted translation parameters the
per-axis reduced chi-square is the weighted residual sum divided by the degrees of freedom, and
the per-axis variance is that reduced chi-square divided by the total weight, floored at the
pure inverse-precision value.  A single body cannot constrain two parameters, so the result
collapses to the inverse-precision floor -- the large per-blob centroid variance -- correctly
reflecting that one point is near-unobservable for two parameters rather than over-confident.
The cross-covariance term is intentionally zero because the two axis errors come from
independent moment integrals.  An optional uncalibrated model-error variance can be added in
quadrature to the diagonal.

The dominant uncertainty source the reported covariance does not fully capture is the
lit-hemisphere bias: at non-zero phase the brightness centroid sits toward the sunlit limb
rather than the body centre, and for an irregular body at high phase the shadowing on an
unknown rotational orientation is not predictable from an ellipsoidal model.  A rotation about
a body's own centroid leaves the centroid unchanged, so the technique carries no rotation
information and abstains on rotation entirely.

Configuration
=============

Runtime tunables live under ``techniques.BodyBlobNav.tuning`` in
``src/nav/config_files/config_510_techniques.yaml``.

- ``at_edge_tolerance_px`` -- float, default ``1.0`` px.  Slack around the search-window axis
  bounds for the at-edge check; a converged offset within this distance of a bound is flagged
  at-edge.
- ``model_error_floor_px`` -- float, default ``0.0`` px.  Uncalibrated model-error sigma added
  in quadrature to the reported covariance diagonal; the default is a no-op.

Confidence formula
-------------------

The confidence coefficients live alongside ``tuning`` in the same
``techniques.BodyBlobNav`` stanza.  The sigmoid argument starts from ``alpha0`` of ``-1.0``
and adds the linear terms below; the sigmoid mathematics is documented in
:doc:`dev_guide_techniques_confidence`.  The gate ``hard_zero_if`` forces confidence to zero
when ``at_edge`` is true, and a post-sigmoid ``hard_cap`` of ``0.4`` ceilings the confidence
because a brightness-weighted centroid is a weaker observation than a limb fit.

- ``body_snr_inside_predicted_bbox`` -- alpha = 0.5, offset = 0, divisor = 4.0, cap at 1.0.
  Mean signal-to-noise inside the predicted bounding boxes; higher SNR tightens the centroid.
- ``body_extent_px`` -- alpha = 1.0, offset = 8.0, divisor = 8.0, cap at 1.0.  Mean predicted
  body extent; larger blobs carry more centroid signal, with the offset placing the eight-pixel
  emission floor at zero contribution.
- ``blob_count`` -- alpha = 0.4, offset = 0, divisor = 3.0, cap at 1.0.  Number of fused
  blobs; more bodies over-determine the joint translation.
- ``max_phase_irregularity_factor`` -- alpha = 0.0, offset = 0, divisor = 0.15, cap at 1.0.
  Maximum combined shape-irregularity and phase-shadowing factor across the consumed blobs;
  wired into the formula but carrying no weight at the current coefficient.

Implementation
==============

The technique lives in ``src/nav/nav_technique/nav_technique_body_blob.py``;
:py:class:`~nav.nav_technique.nav_technique_body_blob.BodyBlobNav` subclasses
:py:class:`~nav.nav_technique.nav_technique.NavTechnique`.  It declares
``accepts_feature_types`` of ``frozenset({NavFeatureType.BODY_BLOB})``, ``requires_prior`` of
``False`` (it runs in pass 1), a ``tier`` of ``'fallback'``, and a ``confidence_attributes``
set of ``at_edge``, ``body_snr_inside_predicted_bbox``, ``body_extent_px``, ``blob_count``,
``residual_px``, ``max_phase_angle_deg``, and ``max_phase_irregularity_factor``.

:py:meth:`~nav.nav_technique.nav_technique_body_blob.BodyBlobNav.is_feasible` reads only
feature metadata, via the module-level ``_eligible_blobs`` filter, and returns a feasibility
report (see :doc:`dev_guide_techniques_feasibility`) that is feasible when at least one
body-blob feature has a non-zero predicted diameter.

:py:meth:`~nav.nav_technique.nav_technique_body_blob.BodyBlobNav.navigate` opens a logged
technique section, filters to eligible blobs, reads the search half-window from
``search_window_for_obs``, and collects the per-blob residuals through the module-level
``_collect_per_blob_residuals``.  That helper computes each blob's brightness-weighted centroid
with ``_brightness_weighted_centroid`` (which clamps the predicted bounding box to the extended
field with ``_clamp_bbox`` and keeps only above-noise pixels), forms the centroid CRLB weight,
and drops any blob with no above-noise signal in its box.  The joint translation and its
covariance come from ``_joint_offset_from_residuals``, which delegates the diagonal covariance
to ``_joint_covariance``.

The result shape branches on two conditions.  First, when no blob carries above-noise signal
the technique returns through the private ``_fail_no_signal``, a zero-confidence
``spurious`` result with empty diagnostics.  Second, the covariance and rotation fields branch
on ``context.fit_camera_rotation``: when the flag is false the covariance is two by two and
``rotation_rad`` / ``sigma_rotation_rad`` are ``None``; when it is true the two-by-two
covariance is widened to the rank-deficient three-by-three form by
:py:func:`~nav.nav_technique.nav_technique.embed_rotation_unobservable`, ``rotation_rad`` is
zero, and ``sigma_rotation_rad`` is the unobservable sentinel from
:py:func:`~nav.nav_technique.nav_technique.rotation_unobservable_sigma_rad`.  The at-edge flag
fires when either translation axis reaches the search-window bound within
``at_edge_tolerance_px``.

The diagnostics object is a
:py:class:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics` with fields
``body_snr_inside_predicted_bbox`` (the mean per-blob SNR), ``body_extent_px`` (the mean
predicted extent), ``blob_count`` (the number of consumed blobs), ``residual_px`` (the joint
fit residual RMS), ``max_phase_angle_deg`` (the maximum raw phase angle, recorded for
inspection only), and ``max_phase_irregularity_factor`` (the maximum combined
irregularity-and-phase factor that the confidence formula consumes).  Confidence is evaluated
by ``evaluate_sigmoid_combination`` against an internal adapter exposing those diagnostics
alongside the ``at_edge`` flag; the calibration is documented in
:doc:`dev_guide_techniques_confidence`, the per-term breakdown is logged through
``log_confidence_breakdown``, and the shared diagnostics dataclass is described in
:doc:`dev_guide_techniques_diagnostics`.

Examples
========

In the ``below_resolution_body`` scene (Cassini NAC ``N1777325846_1``), Mimas appears about
twenty pixels across in the lower-left corner at seventy-two degrees phase, slightly
overexposed.  Feasibility passes and the blob technique is the primary technique.  It recovers
a translation near the operator ground truth ``(6.08, -1.53)`` px and the overall verdict is a
medium-confidence success.  Confidence is driven by ``body_snr_inside_predicted_bbox`` from the
bright lit pixels and by ``body_extent_px`` near the twenty-pixel mark, with a single-blob
``blob_count`` and no ``at_edge`` firing; the hard cap of ``0.4`` ceilings the score in line
with the centroid's weaker constraint.
