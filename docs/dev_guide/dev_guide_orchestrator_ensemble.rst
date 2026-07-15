==========================================================
Ensemble Combine (ensemble + EnsembleConfig)
==========================================================

Overview
========

:func:`~spindoctor.nav_orchestrator.ensemble.ensemble` is the function that reconciles every
per-technique :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult` into a
single :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`. The orchestrator invokes the
ensemble twice per image: once after pass 1 (to derive the pass-2 prior) and once on the
union of pass-1 and pass-2 results (to produce the final answer). The reconciliation
discipline is honest: spurious results are dropped, at-edge results are dropped unless
removing them empties the set, the surviving results are grouped by Mahalanobis-distance
agreement, the highest summed-confidence group wins, and the within-group results are
fused via precision-weighted (Kalman-style) merging.

Theory
======

The ensemble's reconciliation is a seven-step pipeline.

Step 1 — drop spurious
----------------------

Every result with
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` ``True`` is
dropped unconditionally. Spurious is the technique's self-assessed structural failure
flag; the ensemble does not second-guess it. A supersession filter then drops every
fallback-tier result whose source body is already covered by a non-spurious
primary-tier result on the same body (for example a
:class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav` fallback when
:class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` succeeded on that
body); a fallback with no primary coverage for its body stays in the set.

Step 2 — drop at-edge
---------------------

Every result with
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` ``True`` is dropped
*unless* dropping the at-edge cohort would empty the surviving set. The exception
preserves an at-edge result when it is the only signal the orchestrator has — better a
hint at a search-window edge than no answer at all.

Step 3 — pairwise agreement
---------------------------

Two results :math:`(\mu_{a}, \Sigma_{a})` and :math:`(\mu_{b}, \Sigma_{b})` agree when

.. math::

    d_{M}(a, b) = \sqrt{(\mu_{a} - \mu_{b})^{\top} (\Sigma_{a} + \Sigma_{b})^{+}
                        (\mu_{a} - \mu_{b})}

is at most ``agreement_sigma``, *or* when their Euclidean translation distance is at
most ``agreement_pixel_floor`` (default 5 px). The pixel floor exists because
per-technique covariances are CRLB-tight — well below the actual position uncertainty
driven by model error — so results agreeing visually to a few pixels can register as
hundreds of sigmas apart.

Both distances are measured **only in the intersection of the two results' observable
subspaces**, not in the full parameter space. A rank-deficient result (a flat-ring
rank-1 fit, or a rotation-unobservable star result) carries meaningless values along
the axis it cannot observe; differencing those against a result that *does* observe
that axis would manufacture disagreement out of nothing. Restricting the comparison to
the directions both results genuinely constrain lets a rank-1 ring edge and a full-rank
blob that agree radially group — and Step 5 then fuses the ring's radial precision with
the blob's along-edge constraint. Two results with no shared observable direction have
nothing to disagree on and are treated as agreeing, so their complementary constraints
combine. The pseudo-inverse used here and in Step 5 quarantines the huge
unobservable-axis sentinel variances (the ``1e15`` rotation sentinel and the
``1e12``-scale translation sentinels) and inverts them separately from the genuine
block, so a sentinel axis cannot raise the eigenvalue cutoff and silently zero the
well-constrained axes.

Step 4 — consensus-subset selection and outlier rejection
---------------------------------------------------------

Every surviving result sponsors a candidate subset: the results that agree with it
pairwise under Step 3. The subset with the highest summed confidence wins and is the
set the ensemble fuses; the winning technique names are reported on
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.consensus_techniques` and
results outside it are *excluded from the consensus* and reported on
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.excluded_from_consensus`.

Only *corroborating* members count toward summed confidence, quorum, and the Step 6
agreement boost. A pass-2 result — for example a
:class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` refine that
searches a small window around the pass-1 prior — is conditionally dependent on the
techniques that set that prior. The orchestrator records those techniques on the
result's :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.prior_source_techniques`;
when such a technique sits in the same subset, the descendant re-observed its own seed
and casts **no independent vote** (it may still refine the fused offset in Step 5, just
not corroborate its prior). This stops a marginal refine seeded by a wrong pass-1
answer from "confirming" that answer and inflating consensus.

Exclusion forces the conflicted branch only when the excluded results constitute a
genuine alternative answer: the strongest consensus formable among the excluded
results alone either has at least two corroborating members (an alternative with
quorum) or the winning subset is itself a singleton (no quorum anywhere). The
summed-confidence gap test that then decides a conflict is applied in two regimes:

- **Runner-up has quorum:** an *absolute* gap — conflict when
  ``best - runner_up < agreement_gap``.
- **Lone-vs-lone standoff** (singleton winner against a singleton excluded dissenter):
  a *relative* gap — conflict only when ``best - runner_up < agreement_gap * best``.
  Because the consensus selection has already excluded the dissenter as an outlier, it
  keeps veto power only while its confidence is comparable to the winner's; a dissenter
  at well under ``(1 - agreement_gap)`` of the winner's confidence is outlier-rejected,
  not a conflict.

A lone dissenter against a multi-technique consensus is outlier-rejected: the consensus
is fused normally (with the Step 6 disagreement penalty) and the dissenter appears only
in ``excluded_from_consensus`` and ``per_technique``.

Step 5 — precision-weighted merge
---------------------------------

Inside the winning group, fuse the per-technique offsets into one estimate via
Kalman-style information addition. The fused information matrix is the sum of the
per-technique information matrices :math:`I_{i} = \Sigma_{i}^{+}`; the fused offset is
:math:`\mu = \Sigma \, \sum_{i} I_{i} \mu_{i}`, where :math:`\Sigma` is the
pseudo-inverse of the summed information matrix. The pseudoinverse handles rank-deficient
inputs (e.g. a flat-ring-only result) gracefully — the unobservable axis carries an
unbounded marginal sigma.

Step 6 — disagreement and conflict penalties
--------------------------------------------

When any result was excluded from the consensus, the fused confidence is multiplied
by ``disagreement_penalty`` (default 0.7). When the conflict branch fired in Step 4 the
``status='conflicted'`` :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` is returned with a further
``conflicted_confidence_multiplier`` (default 0.3) applied to the winning group's
combined confidence so the JSON sidecar reflects the conflict's severity.

Step 7 — confidence-rank assignment
-----------------------------------

The fused confidence and the per-axis sigma are mapped to a five-bucket rank
(``'high'`` / ``'medium'`` / ``'low'`` / ``'conflicted'`` / ``'failed'``) by
:func:`~spindoctor.nav_orchestrator.ensemble.derive_confidence_rank` against the per-rank
``min_confidence`` / ``max_sigma_px`` thresholds. Below the ``min_confidence`` floor the
ensemble returns ``status='failed'``.

Two honesty guards then cap the tier at ``'medium'`` regardless of how well the
confidence and sigma score:

- **Rank-deficient fused covariance.** When one translation axis is unobservable, the
  observable axis may be pinned precisely but the other is an assumption, not a
  measurement, so the result is not a ``'high'``-tier absolute fix. The
  :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_along_unobservable_px`
  field carries the same verdict to the metadata.
- **Single-star consensus.** When every combined member is a single-star solution — a
  one-star :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`
  match or a one-inlier
  :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` refine —
  there is no independent cross-check, so it tops out at ``'medium'`` even when the
  single-inlier confidence cap sits exactly on the ``'high'`` boundary and the star
  centroid's CRLB sigma is tight. A single-star result cross-checked by a non-star
  technique can still earn ``'high'``.

Restrictions and assumptions
----------------------------

- Per-technique covariances must be 2x2 (translation-only) or 3x3 (translation +
  rotation). The ensemble does not handle scale-disagreement or arbitrary-shape
  parameter spaces.
- The Mahalanobis grouping assumes the per-technique covariances are calibrated. An
  over-confident covariance shrinks the apparent agreement region and may cause a
  legitimate match to land in its own cluster.
- The pseudoinverse cutoff (``pinvh_rcond``) is global; rank-deficient detection uses the
  same threshold for grouping and merging so behaviour is consistent across the two
  passes.

Sources of uncertainty
----------------------

The fused covariance is the pseudo-inverse of the summed information matrix; it is the
standard precision-weighted-merge form. When the input set has no full-rank result, the
fused covariance is rank-deficient along the unconstrained axis; the
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_along_unobservable_px` field
captures the unbounded eigenvalue's direction. When the disagreement-penalty fires the
fused confidence is reduced multiplicatively.

Configuration
=============

Tunables live on :class:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig`. The defaults are
module-level constants in :mod:`spindoctor.nav_orchestrator.ensemble`; the orchestrator's
constructor accepts an :class:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig` override.

- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.agreement_sigma` — float, default
  ``2.0``. Mahalanobis-distance threshold for grouping.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.agreement_pixel_floor` — float,
  default ``5.0``. Euclidean translation-distance fallback for grouping: two results
  agree when either distance test passes (see Step 3). ``0.0`` disables the floor.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.agreement_gap` — float, default
  ``0.5``. Minimum summed-confidence gap between best and runner-up groups before
  declaring a conflict. Applied as an absolute gap when the runner-up has a quorum; in a
  lone-vs-lone standoff it is measured relative to the winner (``best - runner_up <
  agreement_gap * best``).
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.disagreement_penalty` — float,
  default ``0.7``. Multiplier on combined confidence when more than one group existed.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.conflicted_confidence_multiplier` —
  float, default ``0.3``. Additional multiplier when the conflicted branch fires.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.min_confidence` — float, default
  ``0.2``. Final-result threshold below which the ensemble returns
  :meth:`~spindoctor.nav_orchestrator.nav_result.NavResult.failed` instead of
  :meth:`~spindoctor.nav_orchestrator.nav_result.NavResult.success`.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.pinvh_rcond` — float, default
  ``1.0e-9``. Cutoff for :func:`scipy.linalg.pinvh`.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.max_allowed_rotation_deg` — float,
  default ``5.0``. Small-angle bound on a 3-DoF result's rotation magnitude. Every
  contributing technique clamps its rotation fit to this bound, so a result arriving
  outside it is an upstream programming error and raises
  :class:`~spindoctor.support.exceptions.NavContractError`.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.tier_thresholds` — mapping
  ``rank -> {min_confidence, max_sigma_px}``; default thresholds give ``'high'`` for
  confidence at or above 0.5 with sigma at most 0.5 px, ``'medium'`` for confidence at
  or above 0.2 with sigma at most 2.0 px, and ``'low'`` for confidence at or above 0.2
  with no sigma cap. The tiers are sigma-differentiated: with calibrated covariances the
  ``max_sigma_px`` gate carries most of the discrimination, so the ``'medium'`` and
  ``'low'`` confidence floors rest at the same value. The Step 7 tier caps apply on top
  of these thresholds.

Implementation
==============

The ensemble is split across three modules under ``src/spindoctor/nav_orchestrator/``:

- ``ensemble.py`` — the reconciler itself:
  :func:`~spindoctor.nav_orchestrator.ensemble.ensemble`,
  :func:`~spindoctor.nav_orchestrator.ensemble.derive_confidence_rank`, and
  :class:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig`.
- ``ensemble_consensus.py`` — consensus-subset selection:
  :func:`~spindoctor.nav_orchestrator.ensemble_consensus.consensus_selection`,
  :func:`~spindoctor.nav_orchestrator.ensemble_consensus.corroborating_members`,
  :func:`~spindoctor.nav_orchestrator.ensemble_consensus.corroborating_confidence`, and
  :func:`~spindoctor.nav_orchestrator.ensemble_consensus.result_param_vector`.
- ``ensemble_observability.py`` — observable-subspace linear algebra:
  :func:`~spindoctor.nav_orchestrator.ensemble_observability.mixed_scale_pinvh`,
  :func:`~spindoctor.nav_orchestrator.ensemble_observability.observable_basis`,
  :func:`~spindoctor.nav_orchestrator.ensemble_observability.observable_intersection_basis`,
  and :func:`~spindoctor.nav_orchestrator.ensemble_observability.mahalanobis_distance`.

Public surface (autodocumented at :doc:`/api_reference/api_nav_orchestrator`):

- :func:`~spindoctor.nav_orchestrator.ensemble.ensemble` — the reconciler. Returns one
  :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`.
- :func:`~spindoctor.nav_orchestrator.ensemble.derive_confidence_rank` — assign the
  five-bucket rank from a confidence / sigma pair.
- :class:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig` — frozen dataclass carrying
  the nine tunables documented above.

Grouping is sponsored-neighborhood consensus selection
(:func:`~spindoctor.nav_orchestrator.ensemble_consensus.consensus_selection`): every
result sponsors the subset of results that agree with it pairwise, and the subset with
the highest corroborating summed confidence wins. Unlike single-link transitive-closure
grouping, a result that agrees with only one member of the winning subset does not drag
the whole subset toward it — membership requires pairwise agreement with the sponsoring
result. :func:`scipy.linalg.pinvh` (via
:func:`~spindoctor.nav_orchestrator.ensemble_observability.mixed_scale_pinvh`) serves
both the per-pair distance test and the precision-weighted merge.

Examples
========

**Two agreeing techniques.**  Pass 1 produces
:class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav`
(:math:`(6.76, -17.71)` ± 0.5 px) and
:class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`
(:math:`(7.00, -18.00)` ± 0.3 px). The Mahalanobis distance is well below
``agreement_sigma=2.0``; both end up in the same group. The fused offset is
:math:`(6.93, -17.92)` px with combined per-axis sigma ~0.26 px. No disagreement
penalty fires (only one group existed). The fused confidence is the precision-weighted
average of the two per-technique confidences, boosted by the agreement factor
:math:`1 + 0.5 \log_{2} n` over the :math:`n` significant corroborating members — here
:math:`n = 2`, so a factor of 1.5 — and capped by the project-wide combined-confidence
ceiling.

**Consensus selection with three techniques.**  Three techniques converge:
:math:`(7.0, -18.0)` ± 0.3, :math:`(8.0, -17.5)` ± 0.5, :math:`(11.6, 12.6)` ± 0.4. The
first two agree pairwise; the third is several sigma off in both axes. The candidate
subsets are the agreeing pair (summed confidence 0.49) and the singleton (0.74); the
singleton wins on summed confidence, and the excluded pair is an alternative *with
quorum*. The gap :math:`0.74 - 0.49 = 0.25` falls below ``agreement_gap=0.5``, so the
ensemble flags the conflict and returns ``status='conflicted'`` rather than picking the
higher-confidence isolated wrong answer (this is the documented ``multi_body`` test
scene's behaviour). Had a *third* technique joined the pair, the pair-plus-one subset
would have outweighed the singleton and the ensemble would have fused it, excluding the
dissenter as an outlier instead of conflicting.

**Rank-deficient ring-edge fit.**  A flat-ring-only scene produces a
:class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav` result whose covariance is
rank-1 along radial only. The ensemble's pseudoinverse handles the rank deficiency: the
fused covariance has unbounded variance along the along-edge tangent and the
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_along_unobservable_px` field
captures it. When a star or body limb supplies an orthogonal-axis constraint the fused
result becomes full-rank.
