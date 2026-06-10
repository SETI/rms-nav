=================
Navigation Result
=================

Overview
========

:py:class:`~nav.nav_orchestrator.nav_result.NavResult` is the frozen, full in-memory output of a
single navigation.  It carries the headline answer (the offset and its per-axis uncertainty plus a
coarse confidence rank) alongside the complete diagnostic record: every technique that ran, every
feature that was extracted and whether it survived the reliability gate, the image-quality
classifier verdict, per-model metadata, optional annotations for the summary preview, and the
reproducibility envelope.  It is not serialized to JSON directly; the curator distils a
JSON-friendly subset from it.

The canonical constructors are the three classmethods
:py:meth:`~nav.nav_orchestrator.nav_result.NavResult.success`,
:py:meth:`~nav.nav_orchestrator.nav_result.NavResult.failed`, and
:py:meth:`~nav.nav_orchestrator.nav_result.NavResult.conflicted`, each invoked by the orchestrator
at the corresponding terminal branch of a navigation.  The primary consumer is the curator (which
reads the headline and diagnostic fields to build the per-image JSON metadata) and the
summary-PNG renderer (which reads the annotations); downstream science consumers read the offset,
uncertainty, and confidence rank.

Theory
======

The result enforces the consistency rules that tie its top-level status to the rest of the record.
The status is one of three values: success, failed, or conflicted.  Two invariants bind status to
the offset.  A failed result must carry no offset, because a failed navigation has, by definition,
produced no usable answer.  A successful result must carry an offset, because success means an
answer exists.  A third rule binds the coarse confidence rank to the status: the failed rank is
reserved for failed results and may not appear on any other status.

The confidence score is constrained to the unit interval.  The covariance, when present, must be a
square two-dimensional matrix; it is frozen read-only on construction so a consumer cannot mutate
the uncertainty after the fact.  Beyond these structural guarantees the result is an inert record;
it computes no geometry and runs no algorithm of its own.

Configuration
=============

:py:class:`~nav.nav_orchestrator.nav_result.NavResult` has no configuration.  It is constructed at
the end of a navigation from values the techniques and the ensemble have already produced; no YAML
knobs apply.

Implementation
==============

Source file: ``src/nav/nav_orchestrator/nav_result.py``.  Public class
:py:class:`~nav.nav_orchestrator.nav_result.NavResult`, a frozen
:py:func:`dataclasses.dataclass` with ``eq=False``.  The module also defines the type aliases
``Status`` (the three-value status literal) and ``ConfidenceRank`` (the five-bucket rank literal).

The headline fields are :py:attr:`~nav.nav_orchestrator.nav_result.NavResult.status`,
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.offset_px` (the ``(dv, du)`` offset, ``None``
on failure), :py:attr:`~nav.nav_orchestrator.nav_result.NavResult.sigma_px` (per-axis 1-sigma
marginal uncertainty, ``None`` on failure),
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.sigma_along_unobservable_px` (set only when
the covariance is rank-1, as on flat-ring-only scenes),
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.confidence_rank`,
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.confidence` (the calibrated score in
``[0, 1]``), :py:attr:`~nav.nav_orchestrator.nav_result.NavResult.status_reason` (a
:py:class:`~nav.support.status_reason.NavStatusReason` value), and
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.covariance_px2` (the full 2x2 or 3x3
covariance, ``None`` on failure).

The diagnostic fields are :py:attr:`~nav.nav_orchestrator.nav_result.NavResult.per_technique` (a
list of :py:class:`~nav.nav_technique.technique_result.NavTechniqueResult`, including dropped
techniques), :py:attr:`~nav.nav_orchestrator.nav_result.NavResult.feature_inventory` (a list of
:py:class:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary`),
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.image_classifier` (the
:py:class:`~nav.nav_orchestrator.image_classifier_result.NavImageClassifierResult`),
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.provenance` (the
:py:class:`~nav.nav_orchestrator.provenance.Provenance` envelope),
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.model_metadata` (per-model diagnostic dicts),
and :py:attr:`~nav.nav_orchestrator.nav_result.NavResult.annotations` (an
:py:class:`~nav.annotation.annotations.Annotations` collection, empty by default).  The optional
rotation fields are :py:attr:`~nav.nav_orchestrator.nav_result.NavResult.rotation_rad` and
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.sigma_rotation_rad`, both ``None`` when camera
rotation is not fitted.

The :py:meth:`~nav.nav_orchestrator.nav_result.NavResult.__post_init__` invariant validates the
status-to-offset rules (failed implies no offset, success implies an offset), the failed-rank rule
(the failed rank requires the failed status), the confidence-range rule (confidence in ``[0, 1]``),
and the covariance shape; it freezes any supplied covariance into a read-only array.  Each
violation raises :py:exc:`ValueError`.

The three classmethod constructors set the status-dependent fields automatically.
:py:meth:`~nav.nav_orchestrator.nav_result.NavResult.failed` produces a result with no offset, zero
confidence, and the failed rank.  :py:meth:`~nav.nav_orchestrator.nav_result.NavResult.success`
derives :py:attr:`~nav.nav_orchestrator.nav_result.NavResult.sigma_px` from the diagonal of the
supplied covariance and carries the optional rotation outputs.
:py:meth:`~nav.nav_orchestrator.nav_result.NavResult.conflicted` hard-sets the conflicted rank and
the conflicted-techniques status reason so downstream consumers refuse the result without explicit
opt-in.

Examples
========

For ``body_partial_overflow`` (Cassini ISS frame ``N1484593951_2_CALIB``, a body overflowing one
edge), a successful navigation yields a result with
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.status` equal to ``'success'``,
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.offset_px` holding the fitted ``(dv, du)``,
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.sigma_px` holding the two per-axis sigmas
derived from the covariance diagonal, and
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.feature_inventory` listing the limb and
terminator features the body model emitted, each tagged kept or gated.

A failed navigation on a frame with no usable signal yields
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.status` equal to ``'failed'``,
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.offset_px` ``None``,
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.confidence` ``0.0``,
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.confidence_rank` equal to ``'failed'``, and a
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.status_reason` recording why.

A conflicted navigation, in which more than one agreement group survives, reports the best group:
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.status` equal to ``'conflicted'``,
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.confidence_rank` equal to ``'conflicted'``,
and :py:attr:`~nav.nav_orchestrator.nav_result.NavResult.status_reason` fixed to the
conflicted-techniques value, while
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.offset_px` still carries the reported group's
offset.
