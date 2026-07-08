==========================================================
Feasibility Reporting (Shared NavFeasibilityReport)
==========================================================

Overview
========

Feasibility reporting is the small dataclass every navigation technique returns from its
``is_feasible`` method. The orchestrator consults the report before invoking ``navigate``;
infeasible techniques are skipped silently and the human-readable reason is recorded so the
per-image log surfaces "skipped — no ``LIMB_ARC`` features with sufficient visible arc" without
the technique's per-image work running. Feasibility checks read feature *metadata* only —
never image pixels — so they are cheap to obtain on every image.

Theory
======

Feasibility is a binary outcome with a stable human-readable reason. Every feasibility check
reads the offered feature set's metadata (feature types, surviving polyline vertex counts,
predictable-star cohort sizes, etc.) and returns either feasible-with-consumed-count or
infeasible-with-reason.

Stable reasons
--------------

The reason string carries an English description that is *stable across images* — the
orchestrator's diagnostics use it as a key to correlate similar refusals across an entire
imaging campaign. A change to the wording is therefore a visible change to downstream
consumers; reasons are written in a fixed lower-case-with-underscore style ("ok",
"no_limb_arc_features_with_sufficient_visible_arc", "no_prior_offset_on_context",
"too_few_inliers (N < min M)").

Consumed-feature count
----------------------

When a feasibility report is positive it carries the count of features the technique would
consume if invoked. This number is the size of the post-filter inlier set and is recorded on
the per-image diagnostics. A technique whose feasibility check counts fewer features than the
orchestrator offered is usually applying a within-type filter (e.g. drop ``LIMB_ARC`` polylines
whose surviving vertex count is below a threshold) before the actual fit runs.

Restrictions and assumptions
----------------------------

- ``is_feasible`` reads metadata only. Anything that requires a pixel read or a backplane
  query belongs in ``navigate``, not in feasibility — the orchestrator runs feasibility on
  every offered technique on every image, so a per-pixel read in feasibility would multiply
  the runtime.
- The reason field must be non-empty when the report is infeasible. The dataclass constructor
  rejects an empty reason string in that case.
- The consumed-feature count must be non-negative. Feasibility reports for infeasible
  techniques typically set the count to zero.

Sources of uncertainty
----------------------

There is none — feasibility is a deterministic predicate over feature metadata.

Configuration
=============

Feasibility reporting carries no YAML configuration of its own. The per-technique thresholds
that drive feasibility decisions (minimum surviving polyline length, minimum predictable-star
count, etc.) live on each technique's ``tuning`` block; the feasibility check just reads them.

Implementation
==============

Source file: ``src/spindoctor/nav_technique/feasibility.py`` —
:class:`~spindoctor.nav_technique.feasibility.NavFeasibilityReport`.

Public surface (autodocumented at :doc:`/api_reference/api_nav_technique`):

- :class:`~spindoctor.nav_technique.feasibility.NavFeasibilityReport` — the dataclass. Frozen,
  three-field. Fields:

  - :attr:`~spindoctor.nav_technique.feasibility.NavFeasibilityReport.feasible` — bool. True when
    the technique can run on the supplied feature set; False when not.
  - :attr:`~spindoctor.nav_technique.feasibility.NavFeasibilityReport.reason` — str. Human-readable
    reason; required non-empty when ``feasible`` is False; ignored (but commonly set to
    ``"ok"``) when True.
  - :attr:`~spindoctor.nav_technique.feasibility.NavFeasibilityReport.consumed_feature_count` — int.
    Number of features the technique *would* consume after its own type filter. Defaults to
    zero; safe to leave at zero when ``feasible`` is False.

The dataclass enforces its own invariants in ``__post_init__``:

- :attr:`~spindoctor.nav_technique.feasibility.NavFeasibilityReport.feasible` must be a real
  :class:`bool` (not an int or numpy bool).
- :attr:`~spindoctor.nav_technique.feasibility.NavFeasibilityReport.reason` must be a real
  :class:`str`.
- :attr:`~spindoctor.nav_technique.feasibility.NavFeasibilityReport.consumed_feature_count` must be
  a real non-bool :class:`int` and at least zero.
- An infeasible report with an empty ``reason`` raises :exc:`ValueError`.

The dataclass is consumed by every concrete
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique` subclass's
:meth:`~spindoctor.nav_technique.nav_technique.NavTechnique.is_feasible` method and the
orchestrator's two-pass driver. See :doc:`dev_guide_techniques` for the family-level
overview of how feasibility plugs into the pipeline.

Examples
========

**Feasible report, body limb fit.**  When :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`
sees three offered ``LIMB_ARC`` features, two of which carry surviving vertex counts at or above
``min_arc_px``, the report is::

    NavFeasibilityReport(
        feasible=True,
        reason='ok',
        consumed_feature_count=2,
    )

The orchestrator invokes
:meth:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav.navigate` with the full feature
set; the technique itself drops the third polyline before fitting.

**Infeasible report, body limb fit.**  When every offered ``LIMB_ARC`` has fewer surviving
vertices than ``min_arc_px``, the report is::

    NavFeasibilityReport(
        feasible=False,
        reason='no_limb_arc_features_with_sufficient_visible_arc',
        consumed_feature_count=0,
    )

The orchestrator records the reason in the per-image log and skips
:meth:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav.navigate`.

**Reason-keyed correlation.**  An imaging campaign that produced 1,000 images and reports
``no_limb_arc_features_with_sufficient_visible_arc`` on 380 of them tells the operator that
38 % of the campaign's body coverage was either off-frame or below the resolution threshold —
a campaign-level metric they can read off the per-image JSON sidecars without re-running the
pipeline.
