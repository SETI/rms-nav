=====================
Feasibility Reporting
=====================

Overview
========

Before the orchestrator invokes a technique it asks the technique whether it can run at all on the
feature set extracted from the current image. The technique answers with a feasibility report: a
small frozen value stating yes-or-no, a stable human-readable reason when the answer is no, and the
number of features the technique would consume if invoked. The report is what lets the orchestrator
skip an inapplicable technique silently while still recording, for diagnostics, why it was skipped.

Theory
======

A feasibility check is a cheap pre-flight test, distinct from the navigation itself. A technique is
feasible only when the image actually contains the kind of evidence it needs: a limb technique needs
visible limb arcs, a star technique needs detectable stars, a ring technique needs ring edges. The
report carries three pieces of information. The first is a single boolean: can the technique run?
The second is a textual reason, meaningful only when the answer is no; the wording is required to be
stable so that the same refusal on different images can be correlated across a batch (for example,
"too few surviving limb vertices" should read identically every time it occurs). The third is a
count of how many features the technique would consume after applying its own type filter, used for
diagnostic bookkeeping and free to be zero when the technique is infeasible.

The value enforces one invariant: an infeasible report must carry a non-empty reason, because a
silent refusal with no explanation is useless to the operator reading the diagnostics. A feasible
report's reason is ignored. The consumed-feature count must be non-negative. The report encodes no
algorithm of its own; it is the typed contract between a technique's feasibility check and the
orchestrator's scheduling logic.

Configuration
=============

Feasibility reporting has no configuration: the report is a plain value object constructed at
runtime by each technique's feasibility check, and there are no YAML knobs or module-level tunables
that apply to it. The thresholds a technique uses to decide feasibility live in that technique's own
``tuning`` block (documented on the technique's page); the report merely carries the decision.

Implementation
==============

Source file: ``src/nav/nav_technique/feasibility.py``. The module's entire public surface is the
single frozen dataclass :py:class:`~nav.nav_technique.feasibility.NavFeasibilityReport`, with fields
:py:attr:`~nav.nav_technique.feasibility.NavFeasibilityReport.feasible`,
:py:attr:`~nav.nav_technique.feasibility.NavFeasibilityReport.reason`, and
:py:attr:`~nav.nav_technique.feasibility.NavFeasibilityReport.consumed_feature_count`. Its
``__post_init__`` raises :py:exc:`TypeError` if any field has the wrong type and :py:exc:`ValueError`
if the consumed-feature count is negative or if an infeasible report is constructed with an empty
reason.

Each concrete technique's feasibility check returns one of these reports; the orchestrator consults
the :py:attr:`~nav.nav_technique.feasibility.NavFeasibilityReport.feasible` flag to decide whether to
call the technique's navigate step and records the
:py:attr:`~nav.nav_technique.feasibility.NavFeasibilityReport.reason` for any technique it skips.

Examples
========

Infeasible report on the ``one_bright_star_no_body`` scene (Cassini WAC ``W1449079117_1_CALIB``, a
single star in an empty FOV). The body and ring techniques have no body or ring features to work
with on this image, so their feasibility checks each return a report with
:py:attr:`~nav.nav_technique.feasibility.NavFeasibilityReport.feasible` set to False, a stable
reason naming the missing feature kind, and a
:py:attr:`~nav.nav_technique.feasibility.NavFeasibilityReport.consumed_feature_count` of zero. The
orchestrator skips them silently and proceeds to the star techniques, whose feasibility checks
return feasible reports.

Feasible report on the ``body_partial_overflow`` scene (Cassini NAC ``N1484593951_2_CALIB``, a
large partially-cropped Rhea with a good limb). The body-limb technique's check sees the surviving
LIMB_ARC feature, returns a report with
:py:attr:`~nav.nav_technique.feasibility.NavFeasibilityReport.feasible` True and a positive
:py:attr:`~nav.nav_technique.feasibility.NavFeasibilityReport.consumed_feature_count`, and the
orchestrator runs the technique — which converges to roughly ``(12.06, 30.53)`` px and becomes the
primary.
