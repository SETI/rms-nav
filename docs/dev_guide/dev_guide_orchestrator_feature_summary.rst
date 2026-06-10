===============
Feature Summary
===============

Overview
========

:py:class:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary` is the frozen per-feature
post-mortem entry that populates
:py:attr:`~nav.nav_orchestrator.nav_result.NavResult.feature_inventory`.  One entry is recorded for
every feature a model emits, capturing the feature's identity, type, producing model, self-assessed
reliability, the reliability-gate decision (kept or dropped, with a reason), and the feature's
bounding box on the extended-FOV canvas.

The orchestrator constructs one entry per emitted feature as it walks the models' output and
applies the reliability gate, and stores the list on the result.  The sole consumer is the curator,
which writes each entry into the per-image JSON metadata so an operator can see what was extracted
and why each feature was kept or dropped.  The summary deliberately omits the heavy parts of a full
feature (templates, polylines, covariance) so the metadata stays compact.

Theory
======

The summary encodes a small set of field invariants that keep the post-mortem record meaningful:
the feature identifier and producing-model name must be non-empty strings, the reliability must lie
in the unit interval, the gate decision must be boolean, and a dropped feature must carry a
non-empty human-readable reason while a kept feature carries none.  The bounding box is a four-tuple
of integers in extended-FOV coordinates, expressing a half-open row/column range.  Beyond enforcing
those invariants the dataclass runs no algorithm; it is an inert record.

Configuration
=============

:py:class:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary` has no configuration.  Its
fields are populated at navigate time from each emitted feature and the gate decision; no YAML knobs
apply.

Implementation
==============

Source file: ``src/nav/nav_orchestrator/feature_summary.py``.  Public class
:py:class:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary`, a frozen
:py:func:`dataclasses.dataclass`.

The public fields are :py:attr:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.feature_id`
(matching the producing feature's identifier),
:py:attr:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.feature_type` (a
:py:class:`~nav.feature.feature_type.NavFeatureType` value),
:py:attr:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.source_model` (the producing model
name, such as ``'stars'``, ``'body:MIMAS'``, or ``'rings:SATURN'``),
:py:attr:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.reliability` (the self-assessed
score in ``[0, 1]``), :py:attr:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.gated`
(``True`` when the reliability gate dropped the feature),
:py:attr:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.gate_reason` (the drop reason, or
``None``), and
:py:attr:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.bbox_extfov_vu` (the
``(v_min, u_min, v_max, u_max)`` integer bounding box).

The :py:meth:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.__post_init__` invariant
validates every field: it raises :py:exc:`ValueError` for an empty feature identifier or source
model, a reliability outside the unit interval, or a gated feature with no reason, and
:py:exc:`TypeError` for a feature type that is not a
:py:class:`~nav.feature.feature_type.NavFeatureType`, a non-numeric reliability, a non-boolean gate
flag, or a bounding box that is not a four-tuple of integers.  The class defines no classmethods.

Examples
========

For ``body_partial_overflow`` (Cassini ISS frame ``N1484593951_2_CALIB``), the body model emits a
limb arc and a terminator arc, each of which yields one
:py:class:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary`.  A kept limb arc has
:py:attr:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.feature_type` equal to the
``LIMB_ARC`` :py:class:`~nav.feature.feature_type.NavFeatureType` value,
:py:attr:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.source_model` equal to a
``'body:'``-prefixed name, :py:attr:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.gated`
``False``, and :py:attr:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.gate_reason`
``None``, with a :py:attr:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.bbox_extfov_vu`
spanning the visible limb on the extended-FOV canvas.  A feature dropped for low reliability has
:py:attr:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.gated` ``True`` and a non-empty
:py:attr:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.gate_reason` such as a below-floor
reliability message, with its
:py:attr:`~nav.nav_orchestrator.feature_summary.NavFeatureSummary.reliability` recorded so the
operator can see how close it came to surviving the gate.
