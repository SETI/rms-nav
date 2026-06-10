====================
Body Disc Navigation
====================

Overview
========

The body-disc technique exploits the whole sunlit silhouette of a resolved body rather than
its boundary: it takes the per-body brightness template the model renders for each body in the
field, fuses the templates into one composite by Z-buffer paint so a nearer body's pixels
overwrite a farther body's, and correlates the composite against the image to recover one
translation (and, on rotation-fitted cameras, one rotation).  The correlation runs a pyramid
normalised-cross-correlation that self-selects between the raw image and its gradient, so a
smooth Lambert-shaded disc that fills the field is matched on its interior brightness while a
sparse crescent is matched on its limb signal.  Feasibility passes when at least one body-disc
feature carries a rendered template payload.  Feasibility fails when no body-disc feature
carries a template.

Theory
======

A resolved body projects a filled brightness pattern -- the lit fraction of its disc shaded by
the local photometry -- whose position in the image is known up to the pointing error.  If a
synthetic template of that pattern is rendered at the predicted position, the pointing error
is exactly the translation that maximises the agreement between template and image.
Normalised cross-correlation measures that agreement while removing additive and multiplicative
brightness offsets, so the match does not depend on absolute calibration.  For a candidate
offset :math:`\boldsymbol{\delta}` over the masked template support, the correlation score is

.. math::

   \mathrm{NCC}(\boldsymbol{\delta}) = \frac{\sum_{k} \left(I_k(\boldsymbol{\delta}) -
   \bar{I}\right)\left(T_k - \bar{T}\right)}
   {\sqrt{\sum_{k}\left(I_k(\boldsymbol{\delta}) - \bar{I}\right)^2}
   \sqrt{\sum_{k}\left(T_k - \bar{T}\right)^2}},

where :math:`T_k` is the template, :math:`I_k(\boldsymbol{\delta})` is the image sampled under
the shifted template, and the bars denote means over the mask.  The peak of this surface is
the offset; its sharpness and its separation from the next-best peak measure how unambiguous
the match is.

The search is a coarse-to-fine image pyramid: the correlation is computed at a downsampled
scale to localise the basin, then refined at successively finer scales with sub-pixel peak
interpolation.  Two parallel correlation surfaces are available -- one on raw brightness and
one on gradient magnitude -- and the fitter selects whichever yields the stronger peak per
image, because raw brightness carries the unique-alignment signal on a full smooth disc while
the gradient carries it when only the limb is informative.

When a rotation is fitted, the template is rotated about the centroid of the bodies'
predicted centres across a multi-level angular schedule: a coarse sweep across the rotation
cap, then two refinement passes that halve the angular step around the running winner.  Each
sample is a full pyramid correlation, and the rotation with the highest peak quality wins.
The disc rotation uncertainty is reported as unobservable: the correlation peak quality is a
peak-separation ratio rather than a log-likelihood, so its curvature carries no calibrated
Fisher information about the rotation angle, and the technique therefore contributes a
translation estimate while abstaining on rotation by routing a sentinel variance into the
rotation slot of the covariance.

A multi-body composite sharpens the peak roughly as the square root of the body count when
the backgrounds are independent, and the joint geometric constraint removes the "swap moon
assignments" mode-failure that solo per-body correlation suffers.  The technique fails when
the silhouette carries no unique alignment signal -- a featureless, partially-clipped, or
below-resolution disc -- and a spurious flag fires when the correlation peak migrates between
pyramid levels by more than a diameter-scaled tolerance, signalling that the coarse and fine
scales disagree about where the body is.  The reported covariance captures only the
correlation-peak curvature; it does not capture SPICE bias or template-shape error.

Configuration
=============

Runtime tunables live under ``techniques.BodyDiscCorrelateNav.tuning`` in
``src/nav/config_files/config_510_techniques.yaml``.

- ``rotation_at_edge_fraction`` -- float, default ``0.95`` (dimensionless).  Fraction of the
  maximum rotation at which the fitted rotation trips the at-edge flag; lower values surface a
  rotation pegged against its cap earlier.
- ``consistency_max_fraction_of_diameter`` -- float, default ``0.025`` (dimensionless).
  Fraction of the largest body diameter contributing to the inter-pyramid consistency cap, so
  a large textured body is allowed a proportionally larger peak walk between levels.
- ``consistency_max_px`` -- float, default ``4.0`` px.  Floor on the inter-pyramid
  consistency cap that covers pyramid quantisation regardless of body size; the applied cap is
  the larger of this floor and the diameter-scaled value.

The FFT upsample factor the correlation uses is read from ``config.offset`` (key
``correlation_fft_upsample_factor``), not from this stanza.

Confidence formula
-------------------

The confidence coefficients live alongside ``tuning`` in the same
``techniques.BodyDiscCorrelateNav`` stanza.  The sigmoid argument starts from ``alpha0`` of
``-2.0`` and adds the linear terms below; the sigmoid mathematics is documented in
:doc:`dev_guide_techniques_confidence`.  The gate ``hard_zero_if`` forces confidence to zero
when ``at_edge`` or ``spurious`` is true.

- ``ncc_peak`` -- alpha = 1.5, offset = 0, divisor = 6.0, cap at 1.0.  Peak-separation quality
  of the chosen correlation peak; the dominant positive term, saturating over the healthy
  quality range.
- ``consistency_ratio`` -- alpha = -1.0, offset = 0, divisor = 1.0, no cap.  Inter-pyramid
  peak migration normalised by the per-image diameter-scaled spurious threshold; a ratio at
  the spurious edge contributes minus one to the sigmoid argument.
- ``body_count`` -- alpha = 0.4, offset = 0, divisor = 3.0, cap at 1.0.  Number of bodies
  fused into the composite; more bodies sharpen the joint constraint, capped so a three-body
  scene saturates.
- ``peak_to_runner_up_ratio`` -- alpha = 0.0, offset = 0, divisor = 2.0, cap at 1.0.  Ratio of
  the winning peak to the runner-up; wired into the formula but carrying no weight at the
  current coefficient.

Implementation
==============

The technique lives in ``src/nav/nav_technique/nav_technique_body_disc.py``;
:py:class:`~nav.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav` subclasses
:py:class:`~nav.nav_technique.nav_technique.NavTechnique`.  It declares
``accepts_feature_types`` of ``frozenset({NavFeatureType.BODY_DISC})``, ``requires_prior`` of
``False`` (it runs in pass 1), and a ``confidence_attributes`` set of ``at_edge``,
``spurious``, ``ncc_peak``, ``peak_to_runner_up_ratio``, ``consistency_px``,
``consistency_ratio``, ``used_gradient``, and ``body_count``.

:py:meth:`~nav.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav.is_feasible` reads
only feature metadata, via the private ``_filter_disc_features`` filter, and returns a
feasibility report (see :doc:`dev_guide_techniques_feasibility`) that is feasible when at
least one body-disc feature carries a template payload.

:py:meth:`~nav.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav.navigate` opens a
logged technique section, filters to template-bearing features, and fuses their templates into
a single composite with :py:func:`~nav.feature.composition.compose_template_features`.  The
search half-window comes from ``search_window_for_obs``; the FFT upsample factor from the
private ``_upsample_factor``; the diameter-scaled consistency cap from the private
``_consistency_tol_for``.

The result shape branches on ``context.fit_camera_rotation``.  When the flag is false the
technique runs a single pyramid correlation,
:py:func:`~nav.support.correlate.navigate_with_pyramid_kpeaks`, in ``'auto'`` gradient mode
against the unrotated composite, and reports a two-by-two covariance with ``rotation_rad`` /
``sigma_rotation_rad`` of ``None``.  When the flag is true the technique runs the rotation
schedule in the private ``_run_3dof_pyramid``, which calls
``_evaluate_rotation_samples`` over the coarse and first refinement levels and
``_ncc_at_rotation`` per sample (each rotating the composite about the centroid pivot
returned by the private ``_composite_pivot_vu`` and running one pyramid correlation), and
derives the rotation uncertainty from ``_rotation_sigma_from_quality``; the result then
carries a three-by-three covariance whose rotation slot holds the unobservable sentinel
variance, ``rotation_rad`` is the winning angle, and ``sigma_rotation_rad`` is the square root
of that slot.  A correlation covariance whose shape is not two by two raises
:py:exc:`RuntimeError`.  The at-edge flag is the disjunction of the correlation's own at-edge
flag and, with rotation, the rotation exceeding ``rotation_at_edge_fraction`` of the maximum;
the spurious flag is taken directly from the correlation result.

The diagnostics object is a
:py:class:`~nav.nav_technique.diagnostics.BodyDiscDiagnostics` with fields ``ncc_peak`` (the
peak quality), ``peak_to_runner_up_ratio`` (computed by the module-level
``_peak_to_runner_up_ratio`` over the returned top-K peaks), ``consistency_px`` (the raw
inter-pyramid migration), ``consistency_ratio`` (that migration divided by the diameter-scaled
cap), ``used_gradient`` (whether the auto picker chose gradient mode), and ``body_count`` (the
number of fused bodies).  Confidence is evaluated by ``evaluate_sigmoid_combination`` against
an internal adapter exposing those diagnostics alongside the ``at_edge`` and ``spurious``
flags; the calibration is documented in :doc:`dev_guide_techniques_confidence`, the per-term
breakdown is logged through ``log_confidence_breakdown``, and the shared diagnostics dataclass
is described in :doc:`dev_guide_techniques_diagnostics`.  The unobservable-rotation sentinel is
:py:data:`~nav.nav_technique.nav_technique.ROTATION_UNOBSERVABLE_VARIANCE`.

Examples
========

In the ``body_full_fov`` scene (Cassini NAC ``N1572105349_1``), Dione fills the centre of the
field, mostly lit with a sliver of terminator and a predicted diameter near one hundred
fifty-five pixels.  Feasibility passes and the disc technique runs; the gradient-mode
correlation finds a strong peak (quality near twenty) and converges to ``(9.17, -17.01)`` px,
within half a pixel of the operator ground truth ``(8.68, -17.37)`` px.  The fit nonetheless
flags itself ``spurious`` because the inter-pyramid ``consistency_px`` of about ``2.78`` px
exceeds the consistency tolerance, so the ``hard_zero_if`` gate drives confidence to zero even
though the headline offset is correct -- the dominant ``ncc_peak`` term is overridden by the
spurious gate.

In the ``multi_body`` scene (Cassini NAC ``N1487595731_1``), Dione and Rhea overlap at about
ninety degrees phase.  Feasibility passes and the disc technique converges to ``(6.76,
-17.71)`` px, within about one pixel of the operator ground truth ``(7.03, -18.42)`` px, with
a confidence of about ``0.246``; ``ncc_peak`` and the two-body ``body_count`` drive the score,
and the consistency stays within budget.
