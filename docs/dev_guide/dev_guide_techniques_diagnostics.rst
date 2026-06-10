=====================
Technique Diagnostics
=====================

Overview
========

Every navigation technique returns, alongside its offset, a typed diagnostics object summarising the
internal quantities that explain how the fit went: peak heights, residual RMS values, inlier counts,
arc fractions, mode flags. These objects feed two consumers. The confidence formula reads named
diagnostic attributes to compute a calibrated score (see :doc:`dev_guide_techniques_confidence`), and
the orchestrator's JSON curator walks a per-class field map to decide which diagnostics land in the
output metadata. This page documents the module of diagnostics dataclasses as a family; each
technique's own page covers how its specific fields drive its confidence formula.

Theory
======

A diagnostics object is a pure container, not an algorithm: it records the measurements a technique
already computed so that downstream stages can score and report them without re-deriving anything.
The design has two recurring conventions. First, each diagnostic is a flat scalar — a count, a pixel
length, a ratio, a flag, or a mode string — so it can be normalised by the confidence formula's
affine transform and serialised to JSON without nesting. Second, every technique's diagnostics class
carries a static field map naming, for each attribute, the JSON key it serialises to; a field that
is present on the dataclass but absent from that map is a build-time error, which guarantees the
published metadata stays in lock-step with the dataclass as fields are added or removed.

The diagnostics also encode the difference between a raw measurement and the quantity the confidence
formula actually consumes. Where a raw value would mislead the formula — a raw pixel disagreement
that should be scaled by body diameter, a raw phase angle that understates centroid uncertainty on
an irregular body — the dataclass carries both the raw value (for inspection) and a normalised
companion (for the formula). This separation is a documented property of the containers, not a
computation they perform.

Configuration
=============

The diagnostics dataclasses have no configuration of their own: they are runtime containers
populated by each technique, with no YAML knobs and no module-level tunables. The coefficients that
consume their fields live in each technique's confidence-formula block in
``config_510_techniques.yaml`` (see :doc:`dev_guide_techniques_confidence`), and the field-to-JSON
mapping is a class constant on each dataclass rather than a configurable value.

Implementation
==============

Source file: ``src/nav/nav_technique/diagnostics.py``. The module defines one frozen dataclass per
concrete technique plus a union alias spanning all of them. Each dataclass carries a ``CURATOR_FIELDS``
class variable mapping every public attribute to its JSON key, which the orchestrator's curator walks
when assembling metadata.

The body techniques' diagnostics are
:py:class:`~nav.nav_technique.diagnostics.BodyDiscDiagnostics` (``ncc_peak``,
``peak_to_runner_up_ratio``, ``consistency_px``, ``consistency_ratio``, ``used_gradient``,
``body_count``), :py:class:`~nav.nav_technique.diagnostics.BodyLimbDiagnostics`
(``visible_limb_arc_fraction``, ``visible_arc_px``, ``dt_fit_rms_px``, ``lm_iterations``,
``tukey_inlier_count``), :py:class:`~nav.nav_technique.diagnostics.BodyTerminatorDiagnostics` (the
same shape with ``visible_terminator_arc_fraction`` substituted), and
:py:class:`~nav.nav_technique.diagnostics.BodyBlobDiagnostics`
(``body_snr_inside_predicted_bbox``, ``body_extent_px``, ``blob_count``, ``residual_px``,
``max_phase_angle_deg``, ``max_phase_irregularity_factor``).

The ring techniques' diagnostics are :py:class:`~nav.nav_technique.diagnostics.RingEdgeDiagnostics`
(``total_edge_length_px``, ``per_edge_dt_rms_summed``, ``edge_count``, ``is_rank_1``) and
:py:class:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics` (``ncc_peak``,
``peak_to_runner_up_ratio``, ``annulus_count``, ``used_gradient``).

The star techniques' diagnostics are
:py:class:`~nav.nav_technique.diagnostics.StarFieldDiagnostics` (``n_inliers``,
``median_residual_px``, ``n_detected_sources``, ``n_catalog_predicted``, ``n_triplets_evaluated``),
:py:class:`~nav.nav_technique.diagnostics.StarUniqueMatchDiagnostics` (``mode``, ``predicted_snr``,
``brightness_margin_mag``, ``residual_px``), and
:py:class:`~nav.nav_technique.diagnostics.StarRefineDiagnostics` (``n_stars_used``,
``median_pos_err_px``, ``residual_scatter_px``). The interactive technique reports
:py:class:`~nav.nav_technique.diagnostics.ManualNavDiagnostics` (``operator_accepted``).

The module-level alias :py:obj:`~nav.nav_technique.diagnostics.NavTechniqueDiagnostics` is the union
of all ten dataclasses; the curator and the technique-result type both consume it, so adding a
technique means adding both its dataclass and a new union member.

Examples
========

Body-limb diagnostics on the ``body_partial_overflow`` scene (Cassini NAC ``N1484593951_2_CALIB``,
a large partially-cropped Rhea with a good limb). The body-limb technique converges to about
``(12.06, 30.53)`` px and populates a
:py:class:`~nav.nav_technique.diagnostics.BodyLimbDiagnostics` whose ``visible_arc_px`` and
``visible_limb_arc_fraction`` reflect the surviving limb, ``dt_fit_rms_px`` records the final DT
residual, ``lm_iterations`` the Levenberg-Marquardt iteration count, and ``tukey_inlier_count`` the
number of vertices that kept positive Tukey weight; the curator serialises all five into the JSON
metadata through ``CURATOR_FIELDS``.

Degenerate terminator diagnostics on the same image. The body-terminator technique on
``N1484593951_2_CALIB`` rejects every one of its 895 vertices and does not iterate, so its
:py:class:`~nav.nav_technique.diagnostics.BodyTerminatorDiagnostics` carries
``tukey_inlier_count`` of 0 and ``lm_iterations`` of 0 — the values the technique's hard-zero
confidence gate reads to force its confidence to zero.
