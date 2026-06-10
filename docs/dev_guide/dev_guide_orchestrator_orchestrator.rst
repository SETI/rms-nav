=======================
Navigation Orchestrator
=======================

Overview
========

The orchestrator is the top-level driver that turns one observation into one
:py:class:`~nav.nav_orchestrator.nav_result.NavResult`.  It builds the per-image
:py:class:`~nav.nav_orchestrator.nav_context.NavContext`, instantiates every requested navigation
model, extracts and reliability-gates their features, and then runs the registered navigation
techniques in two passes.  The first pass runs prior-free techniques and reconciles their results
into a coarse offset prior; the second pass runs prior-required techniques against that prior.  A
final ensemble reconciliation over the union of both passes' results produces the single
``NavResult`` returned to the caller.

Four short-circuit gates can terminate the run before the final ensemble, each mapping to a
distinct :py:class:`~nav.support.status_reason.NavStatusReason`.  The first is the hard-failure
image-class gate, driven by the ``_HARD_FAILURE_TO_REASON`` table: a blank, fully overexposed,
mostly-missing, or corrupt image fails before any model runs.  The remaining three fire after
feature extraction: a no-features gate when no model emitted anything
(:py:attr:`~nav.support.status_reason.NavStatusReason.NO_FEATURES_EXTRACTED`), an all-gated gate
when every emitted feature fell below its reliability threshold
(:py:attr:`~nav.support.status_reason.NavStatusReason.ALL_FEATURES_GATED`), and a
no-feasible-techniques gate when pass one produced no technique results at all
(:py:attr:`~nav.support.status_reason.NavStatusReason.NO_FEASIBLE_TECHNIQUES`).

Theory
======

The orchestrator implements a fixed pipeline; the stages run in order, and any stage may
short-circuit the run.

1. Build a per-image context from the observation: the extended-field-of-view image (the sensor
   rectangle zero-padded by a per-instrument margin), the sensor, saturation, and cosmic-ray
   masks, the image-quality classification, the noise estimate, and the gradient-magnitude, signed
   gradient-vector, and edge distance-transform images that the matching techniques reuse.

2. Classify the image.  If the verdict is one of the hard-failure classes (a blank frame, a frame
   whose pixels are almost all at full well, a frame dominated by the missing-data sentinel, or a
   frame whose file failed to parse), the pipeline stops immediately and reports the matching
   refusal reason without running any model or technique.

3. Build every requested model and gather the synthetic features each predicts.  Apply the
   reliability gate to drop features whose photometric or geometric reliability falls below a
   per-feature-type threshold.  If no feature was emitted at all, or if every emitted feature was
   gated away, the pipeline stops with the corresponding refusal reason.

4. Run pass one: every feasible technique that does not require a prior.  Primary-tier techniques
   run first; fallback-tier techniques run only for bodies that no primary-tier technique already
   covered with a non-spurious result.  If pass one produced no result, the pipeline stops with the
   no-feasible-techniques reason.

5. Reconcile the pass-one results into a single estimate.  When that estimate carries an offset and
   a covariance, it becomes the prior for the second pass.

6. Run pass two: every feasible technique that requires a prior, against the pass-one prior.

7. Reconcile the union of both passes' results into the final estimate.

The hard-failure short-circuit policy is deliberate: a corrupted or empty frame should fail in
milliseconds with an unambiguous reason rather than waste time rendering models that have nothing
to match.  Each short-circuit returns a failure carrying the image classification and provenance so
a downstream reader can see exactly why navigation refused.

Models and techniques are treated as sandboxed plugins.  Each model render, each feature
extraction, each annotation pass, and each technique run is wrapped so that any exception it raises
is logged with a full traceback and treated as zero output rather than propagated.  A misbehaving
model contributes no features; a misbehaving technique contributes no result.  The pipeline never
raises through to its caller -- every failure mode surfaces on the returned result instead.  This
isolation is what lets an operator add an experimental model or technique without risking the rest
of the pipeline.

Configuration
=============

The orchestrator consumes no configuration of its own.  Every per-image tunable arrives by way of
:py:func:`~nav.nav_orchestrator.instrument_config.instrument_settings_from_obs`, which reads the
observation's resolved per-camera block from ``config_4N0_inst_*.yaml`` (for example
``src/nav/config_files/config_400_inst_coiss.yaml``).  The keys that flow in are:

- ``data_units`` -- string, default ``raw_dn``.  Selects the DN-keyed or I/F-keyed threshold set
  and whether a saturation mask is computed at all.
- ``noise`` -- the per-instrument noise block.  Its ``saturation_dn`` (float, default ``4095`` DN
  on raw cameras) drives the per-pixel saturation mask, and its ``marker_value`` (numeric or
  ``NaN``, default ``0``) is the missing-data sentinel; for calibrated-I/F cameras the marker is
  literally ``NaN`` and is sanitised to a finite fill before the derivative kernels run.
- ``image_quality_thresholds`` -- the per-instrument block feeding the image classifier (blank
  floor, saturation fraction cap, missing fraction cap, noisy threshold); see
  :doc:`dev_guide_orchestrator_image_classifier` for each field.
- ``camera_rotation`` keys ``fit_camera_rotation`` (bool, default ``false``) and
  ``max_rotation_deg`` (float, default ``5.0`` deg).  These enable 3-degree-of-freedom technique
  fits and bound the fitted rotation magnitude.
- ``signal_dn_to_image_unit_scale`` -- the per-instrument scale relating raw signal DN to image
  units, carried on the instrument block for the noise and saturation reasoning.

Constructor-level overrides on
:py:class:`~nav.nav_orchestrator.orchestrator.NavOrchestrator` adjust which components run and
which defaults apply: ``only_models`` and ``only_techniques`` are glob-pattern filters,
``ensemble_config`` is an :py:class:`~nav.nav_orchestrator.ensemble.EnsembleConfig` override (see
:doc:`dev_guide_orchestrator_ensemble`), ``image_derivatives_config`` is an
:py:class:`~nav.nav_orchestrator.image_derivatives.ImageDerivativesConfig` override,
``image_quality_thresholds`` is an
:py:class:`~nav.nav_orchestrator.image_classifier.ImageQualityThresholds` override that supersedes
the per-instrument block (see :doc:`dev_guide_orchestrator_image_classifier`), and
``rms_nav_version`` is the version string written into provenance.

Implementation
==============

Source files: ``src/nav/nav_orchestrator/orchestrator.py``, with collaborators ``ensemble.py``,
``nav_context.py``, ``nav_result.py``, ``image_classifier.py``, ``instrument_config.py``,
``provenance.py``, ``feature_summary.py``, ``image_derivatives.py``, and ``status_reason_info.py``.

The public class is :py:class:`~nav.nav_orchestrator.orchestrator.NavOrchestrator`, which derives
from :py:class:`~nav.support.nav_base.NavBase` for the shared config and logger.  It exposes two
public methods, :py:meth:`~nav.nav_orchestrator.orchestrator.NavOrchestrator.prepare` and
:py:meth:`~nav.nav_orchestrator.orchestrator.NavOrchestrator.navigate`.  The
:py:meth:`~nav.nav_orchestrator.orchestrator.NavOrchestrator.prepare` method runs the same
pre-technique pipeline as navigate -- provenance, context, model builds, feature extraction,
inventory, per-model metadata, merged annotations -- and bundles them in an
:py:class:`~nav.nav_orchestrator.orchestrator.OrchestratorPrep`; it does not short-circuit on a
hard-failure class, so the manual-navigation driver can override gate decisions visually.

The :py:meth:`~nav.nav_orchestrator.orchestrator.NavOrchestrator.navigate` call path runs through
private helpers in this order.  It builds the provenance envelope (``_make_provenance``, which reads
runtime git, SPICE, and static-data state through
:py:func:`~nav.nav_orchestrator.provenance.collect_provenance_metadata`), builds the context and the
classifier verdict (``_make_context``, which calls
:py:func:`~nav.nav_orchestrator.instrument_config.instrument_settings_from_obs`, sanitises the
missing-data marker to a finite fill, runs
:py:class:`~nav.nav_orchestrator.image_classifier.NavImageClassifier`, builds the saturation and
cosmic-ray masks, and calls
:py:func:`~nav.nav_orchestrator.image_derivatives.compute_all_image_derivatives`), and logs the
verdict.  If the class is a hard-failure class, it returns through ``_fail`` with the matching
reason.  Otherwise it builds models (``_build_models``), extracts features (``_extract_features``),
gates them (``_extract_and_gate``), builds the inventory (``_build_inventory``), snapshots model
metadata (``_collect_model_metadata``), and merges annotations (``_collect_annotations``).

The two failure gates after extraction route through ``_fail``: no features at all, and every
feature gated.  Pass one then runs ``_run_pass`` twice -- once for primary-tier prior-free
techniques and once for fallback-tier prior-free techniques with the already-covered bodies
excluded -- and a third ``_fail`` gate fires when pass one produced no results.  The pass-one
results are reconciled by :py:func:`~nav.nav_orchestrator.ensemble.ensemble`; when that result
carries an offset and covariance it is threaded into a prior-bearing context (via the context's
``with_prior``) for pass two, which is a third ``_run_pass`` call with ``requires_prior=True``.  The
final :py:func:`~nav.nav_orchestrator.ensemble.ensemble` call reconciles the union of both passes
and is logged and returned.

The four ``_fail`` gate paths return the reasons
:py:attr:`~nav.support.status_reason.NavStatusReason.NO_SIGNAL_IN_IMAGE`,
:py:attr:`~nav.support.status_reason.NavStatusReason.IMAGE_OVEREXPOSED`,
:py:attr:`~nav.support.status_reason.NavStatusReason.MISSING_DATA_DOMINANT`, or
:py:attr:`~nav.support.status_reason.NavStatusReason.IMAGE_CORRUPT` (hard-failure class, looked up
from the ``_HARD_FAILURE_TO_REASON`` table);
:py:attr:`~nav.support.status_reason.NavStatusReason.NO_FEATURES_EXTRACTED` (no feature emitted);
:py:attr:`~nav.support.status_reason.NavStatusReason.ALL_FEATURES_GATED` (every feature gated); and
:py:attr:`~nav.support.status_reason.NavStatusReason.NO_FEASIBLE_TECHNIQUES` (pass one produced no
results).  Every ``_fail`` and every successful return also emits the operator-readable INFO lines
from ``STATUS_REASON_INFO_TEMPLATE`` via ``_log_status_reason``, so the per-image log carries the
human-readable narrative for the final reason.

Model selection uses the glob filter ``filter_by_glob`` on the per-image registry: model names
follow the ``prefix:VALUE`` convention (``rings:SATURN``, ``body:DIONE``, plain ``stars``), and a
colon-free token such as ``rings`` is expanded to ``rings:*`` while the value part is upper-cased.
Technique selection applies the constructor's ``only_techniques`` glob through
:py:func:`~nav.nav_technique.nav_technique.filter_technique_names`, restricting each pass to the
techniques whose ``requires_prior`` and tier match.

Examples
========

These examples use named scenes from ``tests/integration/image_library/images/``.

The ``body_partial_overflow`` scene (Cassini NAC ``N1484593951_2_CALIB``) shows a successful
navigation.  Large Rhea is partially off-frame to the upper right with a good limb.
:py:class:`~nav.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav` runs but flags itself
spurious because the disc-template correlation peak collapses on the heavily cropped silhouette, and
:py:class:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav` runs but flags itself
spurious as well, so :py:class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav` is the only
non-spurious technique.  Its offset of ``(12.06, 30.53)`` px lies within the 1.0 px ground-truth
uncertainty of the operator's ``(11.0, 29.5)``.  The final
:py:class:`~nav.nav_orchestrator.nav_result.NavResult` carries ``status == 'success'``,
``confidence_rank == 'low'``, the fused offset, a 2x2 covariance, and the full per-technique list
including the spurious entries for diagnostics.

The ``multi_body`` scene (Cassini NAC ``N1487595731_1_CALIB``) shows a conflicted navigation.  Dione
and Rhea overlap at roughly 90 deg phase.
:py:class:`~nav.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav` and
:py:class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav` both converge within about a pixel
of the operator's ``(7.03, -18.42)`` ground truth (disc ``(6.76, -17.71)`` confidence 0.246; limb
``(7.00, -18.00)`` confidence 0.239), but
:py:class:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav` latches onto a wrong
local minimum at ``(11.58, 12.64)`` with confidence 0.744.  The ensemble forms two disagreeing
groups: disc-plus-limb sum to 0.485 against the lone terminator's 0.744, a gap of 0.259 below the
agreement-gap threshold of 0.5.  The final
:py:class:`~nav.nav_orchestrator.nav_result.NavResult` therefore carries ``status == 'conflicted'``
and ``confidence_rank == 'conflicted'`` (see :doc:`dev_guide_orchestrator_ensemble` for the
conflict-detection step).

The ``star_dominated`` scene (Cassini WAC ``W1580760393_1_CALIB``) shows a failed navigation: the
dense CLEAR-filter star field produces a result whose combined confidence does not earn any
confidence tier, so the orchestrator returns a
:py:class:`~nav.nav_orchestrator.nav_result.NavResult` with ``status == 'failed'``, ``offset_px`` of
``None``, and ``confidence_rank == 'failed'``.

The ``one_bright_star_no_body`` scene (Cassini WAC ``W1449079117_1_CALIB``) walks the pass-one to
pass-two hand-off.  A single star (Vega) under the RED filter is matched in pass one by
:py:class:`~nav.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`, whose pass-one
ensemble estimate becomes the prior threaded into the pass-two context.
:py:class:`~nav.nav_technique.nav_technique_star_refine.StarRefineNav` consumes that prior in pass
two to refine the alignment, and the final
:py:class:`~nav.nav_orchestrator.nav_result.NavResult` carries ``status == 'success'`` with
``confidence_rank == 'low'`` and an offset near the operator's ``(3.06, -0.02)``.
