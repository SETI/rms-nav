================
Metadata Curator
================

Overview
========

The curator projects an in-memory navigation result into the JSON-friendly metadata block written
to disk for every image.  It selects the serializable fields, rounds every float to a documented
precision so the output is byte-stable across runs, replaces infinite values with a finite
sentinel, and assembles the ``navigation_result`` dictionary the orchestrator merges into the
per-image metadata file.  It also enforces an invariant: every per-technique diagnostic field that
ships in the JSON must be listed in that technique's curator allow-list, so a newly added diagnostic
cannot silently vanish from the output.

Theory
======

The transformation is mechanical, not algorithmic.  A navigation result holds rich in-memory state:
the headline offset and its covariance, per-technique estimates with their own diagnostic records,
a per-feature inventory, the image classification, and the provenance envelope.  The curator walks
that structure and emits a plain nested dictionary of strings, numbers, lists, and nested
dictionaries -- nothing that a standard JSON encoder cannot serialize.

Two policies govern the numeric output.  First, every float is rounded to a fixed number of decimal
places chosen per quantity kind: pixel-valued quantities (offsets, sigmas, covariance entries) to a
finer precision, confidence-like scores to a coarser one, and ephemeris-time timestamps to the
finest, because a coarse timestamp would lose sub-second alignment.  These precisions are tighter
than the per-image tolerance budget; their purpose is reproducibility, so two runs on the same
inputs produce identical bytes.  Second, an infinite value -- which arises legitimately when one
axis of the offset is unobservable -- cannot be encoded as standard JSON, so it is replaced by a
large finite sentinel of the matching sign, and a not-a-number value is replaced by zero.

The allow-list invariant is a guardrail rather than a computation.  Each technique declares which of
its diagnostic fields are exported and under what JSON key; a verification pass compares the
declared set against the actual fields on the diagnostic record and refuses to proceed if any field
is undeclared, turning an easy-to-miss omission into a loud failure.

Configuration
=============

The curator is configured entirely by module-level constants in
``src/nav/nav_orchestrator/curator.py``; there is no YAML override path.  The rounding policy
constants:

- ``PIXEL_DECIMALS`` -- int, default ``4`` (decimal places).  Precision for pixel-valued quantities:
  offsets, per-axis sigmas, and covariance matrix entries.
- ``CONFIDENCE_DECIMALS`` -- int, default ``3`` (decimal places).  Precision for confidence scores,
  reliabilities, and the per-technique diagnostic floats.
- ``ET_DECIMALS`` -- int, default ``6`` (decimal places).  Precision for the ephemeris-time image
  timestamp.

The infinite-value sentinel is the shared constant
:py:data:`nav.feature.constants.JSON_INF_SENTINEL`: a positive infinity is written as this finite
value and a negative infinity as its negation, so the unobservable-axis sigma survives JSON
serialization as a large finite number rather than an encoder error.

Implementation
==============

Source file: ``src/nav/nav_orchestrator/curator.py``.

The module exposes two public functions, both with signatures deferred to autodoc.
:py:func:`~nav.nav_orchestrator.curator.build_metadata_dict` is the entry point: given a
:py:class:`~nav.nav_orchestrator.nav_result.NavResult` it returns the JSON-ready
``navigation_result`` dictionary.  It first calls
:py:func:`~nav.nav_orchestrator.curator.assert_diagnostic_fields_present` to enforce the allow-list
invariant, then assembles the headline fields (status, status reason, rounded offset and sigma,
unobservable-axis sigma, confidence and rank, covariance, techniques used, and a per-type feature
count over the kept inventory entries), and finally curates the per-technique results, the feature
inventory, the image classifier verdict, and the provenance envelope.  Optional rotation and
rotation-sigma fields are appended in degrees when the result carries a fitted rotation.

The rounding and serialization run through private helpers: ``_round_float`` applies the
per-quantity decimal policy and the infinity / not-a-number substitution, ``_round_pair`` and
``_round_2x2`` extend it to offset pairs and covariance matrices, and ``_curate_technique_result``,
``_curate_diagnostics``, ``_curate_feature_summary``, ``_curate_image_classifier``, and
``_curate_provenance`` project each nested record.  ``_curate_diagnostics`` reads each technique's
``CURATOR_FIELDS`` allow-list to decide which diagnostic fields to export and under which JSON key.
:py:func:`~nav.nav_orchestrator.curator.assert_diagnostic_fields_present` walks the per-technique
diagnostic records and raises :py:exc:`AssertionError` when a record lacks a ``CURATOR_FIELDS``
declaration or carries a field absent from it, so continuous integration fails the build before an
undocumented diagnostic reaches the JSON.

Examples
========

For the ``body_partial_overflow`` scene (Cassini NAC ``N1484593951_2_CALIB``), navigation succeeds
on :py:class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav` with offset ``(12.06, 30.53)``
px.  :py:func:`~nav.nav_orchestrator.curator.build_metadata_dict` emits a ``navigation_result`` block
whose ``status`` is ``success``, whose ``offset_px`` is the offset rounded to ``PIXEL_DECIMALS``
(``[12.06, 30.53]``), whose ``confidence`` is rounded to ``CONFIDENCE_DECIMALS``, and whose
``confidence_rank`` is ``low``.  The ``per_technique`` list carries an entry for every technique
that ran, including the disc and terminator entries that flagged themselves spurious, each with its
diagnostics filtered through the technique's ``CURATOR_FIELDS`` allow-list; if any diagnostic field
were missing from that allow-list,
:py:func:`~nav.nav_orchestrator.curator.assert_diagnostic_fields_present` would have raised
:py:exc:`AssertionError` before the dictionary was built.  A result whose offset axis was
unobservable would carry ``sigma_along_unobservable_px`` written as the finite
:py:data:`nav.feature.constants.JSON_INF_SENTINEL` rather than a raw infinity.
