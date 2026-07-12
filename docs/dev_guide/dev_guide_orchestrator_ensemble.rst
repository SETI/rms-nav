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
flag; the ensemble does not second-guess it.

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

is at most ``agreement_sigma``, where the pseudoinverse uses
:func:`scipy.linalg.pinvh` so rank-deficient covariances are handled, *or* when their
Euclidean translation distance is at most ``agreement_pixel_floor`` (default 5 px). The
pixel floor exists because per-technique covariances are CRLB-tight — well below the
actual position uncertainty driven by model error — so results agreeing visually to a
few pixels can register as hundreds of sigmas apart. A result whose
:math:`(\mu_{a} - \mu_{b})` projects into the null space of the summed covariance is
treated as infinite Mahalanobis distance — estimates cannot agree along an
unobservable axis.

Step 4 — consensus-subset selection and outlier rejection
---------------------------------------------------------

Every surviving result sponsors a candidate subset: the results that agree with it
pairwise under Step 3. The subset with the highest summed confidence wins and is the
set the ensemble fuses; results outside it are *excluded from the consensus* and
reported on
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.excluded_from_consensus`.

Exclusion forces the conflicted branch only when the excluded results constitute a
genuine alternative answer: the strongest consensus formable among the excluded
results alone either has at least two members (an alternative with quorum) or the
winning subset is itself a singleton (no quorum anywhere). In that case, when the
summed-confidence gap between winner and runner-up falls below ``agreement_gap``, the
ensemble returns a ``status='conflicted'``
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult` instead of fusing. A lone
dissenter against a multi-technique consensus is outlier-rejected: the consensus is
fused normally (with the Step 6 disagreement penalty) and the dissenter appears only
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
``conflicted_confidence_multiplier`` (default 0.3) applied to the runner-up's summed
confidence so the JSON sidecar reflects the conflict's severity.

Step 7 — confidence-rank assignment
-----------------------------------

The fused confidence and the per-axis sigma are mapped to a five-bucket rank
(``'high'`` / ``'medium'`` / ``'low'`` / ``'conflicted'`` / ``'failed'``) by
:func:`~spindoctor.nav_orchestrator.ensemble.derive_confidence_rank` against the per-rank
``min_confidence`` / ``max_sigma_px`` thresholds. Below the ``min_confidence`` floor the
ensemble returns ``status='failed'``.

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
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.agreement_gap` — float, default
  ``0.5``. Minimum summed-confidence gap between best and runner-up groups before
  declaring a conflict.
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
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.tier_thresholds` — mapping
  ``rank -> {min_confidence, max_sigma_px}``; default thresholds give ``'high'`` for
  confidence at or above 0.8 with sigma at most 0.5 px, ``'medium'`` for 0.5 confidence
  with sigma at most 2.0 px, ``'low'`` for 0.2 confidence with no sigma cap.

Implementation
==============

Source file: ``src/spindoctor/nav_orchestrator/ensemble.py`` —
:func:`~spindoctor.nav_orchestrator.ensemble.ensemble`,
:func:`~spindoctor.nav_orchestrator.ensemble.derive_confidence_rank`, and
:class:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig`.

Public surface (autodocumented at :doc:`/api_reference/api_nav_orchestrator`):

- :func:`~spindoctor.nav_orchestrator.ensemble.ensemble` — the reconciler. Returns one
  :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`.
- :func:`~spindoctor.nav_orchestrator.ensemble.derive_confidence_rank` — assign the
  five-bucket rank from a confidence / sigma pair.
- :class:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig` — frozen dataclass carrying
  the seven tunables documented above.

The function uses :func:`scipy.sparse.csgraph.connected_components` to find the
Mahalanobis-distance clusters and :func:`scipy.linalg.pinvh` for both the per-pair
distance test and the precision-weighted merge.

Examples
========

**Two agreeing techniques.**  Pass 1 produces
:class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav`
(:math:`(6.76, -17.71)` ± 0.5 px) and
:class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`
(:math:`(7.00, -18.00)` ± 0.3 px). The Mahalanobis distance is well below
``agreement_sigma=2.0``; both end up in the same group. The fused offset is
:math:`(6.93, -17.92)` px with combined per-axis sigma ~0.26 px. No disagreement
penalty fires (only one group existed) so the fused confidence is the summed per-technique
confidence (capped by the project-wide ceiling).

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
