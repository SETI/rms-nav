=======================
Ensemble Reconciliation
=======================

Overview
========

The ensemble is the single point in the pipeline where several per-technique offset estimates
become one :py:class:`~nav.nav_orchestrator.nav_result.NavResult`.  It takes the list of
per-technique results from one or both passes, discards the ones that cannot be trusted, groups the
survivors by statistical agreement, picks the most-supported group, fuses its members into one
offset with a combined covariance and confidence, and decides whether the surviving disagreement is
benign or a genuine conflict.  It is exercised in isolation against synthetic per-technique results,
because its correctness is what makes the rest of the pipeline trustworthy.

Theory
======

The reconciliation runs as seven ordered steps over the per-technique estimates.  Each estimate is
a translation offset (and, when the camera-rotation fit is enabled, a small rotation angle) with an
associated covariance, a scalar confidence, and two boolean flags: one marking the result as
spurious, one marking it as resting against the image edge.

**1. Drop spurious results.**  Any estimate the producing technique flagged as untrustworthy is
removed.  If every estimate was spurious, the reconciliation fails with no offset.

**2. Drop at-edge results unless that would empty the set.**  Estimates whose fitted feature sat
against the image boundary are removed, because a curve cut off by the frame edge is poorly
constrained along the cut.  This drop is conditional: if removing the edge-resting estimates would
leave nothing, they are kept, so a scene whose only signal lies at the frame edge can still produce
an answer.

**3. Group by agreement via single-link clustering.**  Two estimates are linked when *either* their
Mahalanobis distance is at most a threshold *or* their plain Euclidean translation distance is at
most a pixel floor.  For estimates :math:`a` and :math:`b` with means :math:`\mu_a, \mu_b` and
covariances :math:`\Sigma_a, \Sigma_b`, the Mahalanobis distance is

.. math::
   d(a, b) = \sqrt{(\mu_a - \mu_b)^{\mathsf{T}}\, (\Sigma_a + \Sigma_b)^{+}\, (\mu_a - \mu_b)},

where :math:`(\cdot)^{+}` is the symmetric pseudoinverse, so a rank-deficient summed covariance is
handled cleanly.  Any component of :math:`\mu_a - \mu_b` lying in the null space of the summed
covariance is treated as infinite distance: two estimates cannot agree along an axis that neither
of them observes.  The pixel floor exists because the per-technique covariances report only their
estimator-tight precision, far below the true position uncertainty driven by model error and
pointing residuals; without it, estimates that agree visually to a few pixels would register as
hundreds of standard deviations apart and never link.  Transitive closure over the pairwise links
yields the final groups.

**4. Select the highest summed-confidence group.**  The groups are ranked by the sum of their
members' confidences, and the top group is chosen as the answer set.

**5. Combine the group in information form.**  Within the chosen group, the translation components
are fused by a precision-weighted (Kalman information-form) merge:

.. math::
   \Sigma_{\mathrm{comb}} = \left( \sum_i \Sigma_i^{+} \right)^{+}, \qquad
   \mu_{\mathrm{comb}} = \Sigma_{\mathrm{comb}} \sum_i \Sigma_i^{+}\, \mu_i.

Tighter covariances pull the combined estimate harder.  When every member carries a rotation
component, the rotation is not averaged as a plain coordinate (which would wrap incorrectly near
:math:`\pm\pi`); it is combined on the circle as the precision-weighted circular mean

.. math::
   \theta_{\mathrm{comb}} =
   \operatorname{atan2}\!\left( \sum_i w_i \sin\theta_i,\ \sum_i w_i \cos\theta_i \right),

with :math:`w_i` the rotation-component information of each member.  Every contributing rotation is
required to lie strictly inside the small-angle bound that each technique already clamps to, so the
linear differencing used in the agreement test and this circular mean are both valid.  The combined
confidence is itself a precision-weighted average of the members' confidences, boosted by a factor
that grows with the count of significantly-weighted contributors and capped at a fixed ceiling.

**6. Apply the disagreement penalty.**  When more than one group existed before the combine, the
combined confidence is multiplied by a penalty below one, because the presence of a competing group
means the answer is less certain than its internal agreement alone would suggest.

**7. Detect conflict.**  When a runner-up group exists, the gap between the best and runner-up
summed confidences is compared against a threshold.  A gap below the threshold means the second
group is nearly as well-supported as the first: the result is reported as conflicted, with its
confidence further multiplied by a conflict penalty, so a downstream consumer must opt in
explicitly before trusting it.  A sufficiently large gap clears the conflict check, and the
combined estimate is emitted as a success -- subject to a final minimum-confidence floor and a
confidence-tier assignment, below which the reconciliation fails instead.

The reported covariance captures only the precision the contributing techniques claimed, propagated
through the information-form merge; it does not model correlated systematic error shared across
techniques.  When the summed information matrix has a near-zero eigenvalue relative to its largest,
the combine is flagged rank-deficient: one axis is effectively unobservable, and the result records
an infinite uncertainty along that direction.  When every contributing covariance shares a single
null direction, the total weight is zero and the offset is unobservable, which fails the
reconciliation outright.

Configuration
=============

The ensemble's tunables live on :py:class:`~nav.nav_orchestrator.ensemble.EnsembleConfig`, a frozen
dataclass whose defaults match ``config_540_orchestrator.yaml``.  Each field:

- ``agreement_sigma`` -- float, default ``2.0`` (dimensionless).  Mahalanobis-distance threshold
  for linking two estimates; larger groups more aggressively.
- ``agreement_pixel_floor`` -- float, default ``5.0`` px.  Euclidean translation-distance fallback
  for linking; two estimates link when they agree to within this many pixels even if their
  Mahalanobis distance is large.  Set to ``0.0`` to disable the floor.
- ``agreement_gap`` -- float, default ``0.5`` (dimensionless).  Minimum best-vs-runner-up
  summed-confidence gap before the result is declared conflicted; larger values declare conflict
  more readily.
- ``disagreement_penalty`` -- float, default ``0.7`` (dimensionless).  Multiplier on combined
  confidence when more than one group existed; smaller values penalise disagreement harder.
- ``conflicted_confidence_multiplier`` -- float, default ``0.3`` (dimensionless).  Additional
  multiplier applied to the confidence of a conflicted result.
- ``min_confidence`` -- float, default ``0.2`` (dimensionless).  Floor below which the
  reconciliation returns a failure instead of a success.
- ``pinvh_rcond`` -- float, default ``1.0e-9`` (dimensionless).  Relative cutoff passed to the
  symmetric pseudoinverse; smaller values keep more near-singular directions.
- ``max_allowed_rotation_deg`` -- float, default ``5.0`` deg.  Small-angle bound a 3-DoF result's
  rotation may take; the linear rotation differencing and circular-mean combine assume every input
  stays strictly inside this magnitude, and a result arriving at or beyond it is an upstream
  programming error that trips an assertion.
- ``tier_thresholds`` -- mapping, default ``{high: {min_confidence: 0.8, max_sigma_px: 0.5},
  medium: {min_confidence: 0.5, max_sigma_px: 2.0}, low: {min_confidence: 0.2, max_sigma_px:
  null}}``.  Maps each confidence rank to the minimum confidence and maximum per-axis sigma it
  requires; a ``max_sigma_px`` of ``null`` imposes no sigma bound on that tier.

Implementation
==============

Source file: ``src/nav/nav_orchestrator/ensemble.py``.

The public entry point is :py:func:`~nav.nav_orchestrator.ensemble.ensemble`; its signature is
deferred to autodoc.  It is driven by the
:py:class:`~nav.nav_orchestrator.ensemble.EnsembleConfig` passed from the orchestrator and returns a
:py:class:`~nav.nav_orchestrator.nav_result.NavResult` constructed through that class's
:py:meth:`~nav.nav_orchestrator.nav_result.NavResult.success`,
:py:meth:`~nav.nav_orchestrator.nav_result.NavResult.conflicted`, or
:py:meth:`~nav.nav_orchestrator.nav_result.NavResult.failed` classmethod depending on the outcome.

The seven steps map onto private helpers.  Spurious filtering and the conditional at-edge filtering
run inline.  ``_drop_superseded_fallbacks`` removes fallback-tier results for any body already
covered by a non-spurious primary-tier result (a redundant safety net behind the orchestrator's own
tier filtering), using ``_source_bodies`` and ``_technique_tier`` to read the body names and tier
from the technique registry.  ``_agreement_groups`` builds the single-link clusters: it computes
pairwise distances with ``_mahalanobis_distance`` and the parameter vectors with
``_result_param_vector``, builds a sparse adjacency matrix, and resolves the transitive closure with
:py:func:`scipy.sparse.csgraph.connected_components`.  The chosen group is fused by
``_combine_precision_weighted`` (translation by the information-form merge, rotation by the
circular mean) and its confidence by ``_combine_confidence``; both call
:py:func:`scipy.linalg.pinvh` for the pseudoinverse, which is also where the rank-deficiency check
and the zero-total-weight unobservable check live.  The public
:py:func:`~nav.nav_orchestrator.ensemble.derive_confidence_rank` helper maps the combined confidence
and per-axis sigma onto a confidence rank using the configured ``tier_thresholds``.

Examples
========

The ``multi_body`` scene (Cassini NAC ``N1487595731_1_CALIB``) is the canonical conflict case.
After dropping no spurious or at-edge results, three estimates survive:
:py:class:`~nav.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav` at ``(6.76, -17.71)``
confidence 0.246, :py:class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav` at
``(7.00, -18.00)`` confidence 0.239, and
:py:class:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav` at ``(11.58, 12.64)``
confidence 0.744.  The disc and limb estimates agree to within a pixel and fuse into one group with
summed confidence ``0.246 + 0.239 = 0.485``; the terminator estimate is roughly 31 px away in the U
axis and forms its own group at 0.744.  The best group is the lone terminator, but the gap to the
runner-up is ``0.744 - 0.485 = 0.259``, below the default ``agreement_gap`` of 0.5, so step seven
declares the result conflicted and returns a
:py:class:`~nav.nav_orchestrator.nav_result.NavResult` through
:py:meth:`~nav.nav_orchestrator.nav_result.NavResult.conflicted` with the confidence further scaled
by ``conflicted_confidence_multiplier``.

The ``body_partial_overflow`` scene (Cassini NAC ``N1484593951_2_CALIB``) shows the opposite
outcome.  The disc and terminator techniques both flag themselves spurious, so step one removes
them, leaving :py:class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav` at
``(12.06, 30.53)`` as the sole survivor.  With one estimate there is exactly one group and no
runner-up, the conflict check is skipped, and the reconciliation emits a success through
:py:meth:`~nav.nav_orchestrator.nav_result.NavResult.success` at confidence rank ``low``.
