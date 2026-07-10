==========================================================
JSON Curation (build_metadata_dict)
==========================================================

Overview
========

The curator turns a :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` into a JSON-friendly
metadata dict consumed by downstream readers. Two functions form the public surface:
:func:`~spindoctor.nav_orchestrator.curator.build_metadata_dict` does the conversion, and
:func:`~spindoctor.nav_orchestrator.curator.assert_diagnostic_fields_present` runs at startup to
enforce the per-technique ``CURATOR_FIELDS`` allow-list discipline so a new diagnostic
field cannot silently disappear from the JSON output.

Theory
======

The curator picks JSON-friendly fields from a
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult`, rounds floats to documented
precision, substitutes the ``JSON_INF_SENTINEL`` finite sentinel for non-finite floats (or
zero for NaN), and emits the ``navigation_result`` block consumed by downstream readers.

Float rounding policy
---------------------

Three precision constants govern the rounding:

- ``PIXEL_DECIMALS = 4`` — pixel-domain quantities (offsets, sigmas, covariance
  entries).
- ``CONFIDENCE_DECIMALS = 3`` — confidence scores in :math:`[0, 1]`.
- ``ET_DECIMALS = 6`` — ET timestamps (seconds past J2000 TDB).

The constants are chosen tighter than the per-image tolerance budget so the JSON output
is byte-identical across runs of the same input — a regression-baseline comparator can
diff the JSON directly.

Allow-list discipline
---------------------

Every per-technique diagnostic field that ships in the JSON appears in the technique's
``CURATOR_FIELDS`` class attribute (a mapping of dataclass-field name to JSON-key name, or
``None`` to skip). The curator walks ``CURATOR_FIELDS`` rather than the dataclass's
``fields()`` directly, so a new field added to a diagnostics dataclass without an entry
in the mapping does not silently leak into the JSON.
:func:`~spindoctor.nav_orchestrator.curator.assert_diagnostic_fields_present` runs at startup
(or in CI) and fails the build with :exc:`AssertionError` when any dataclass field is
missing from its ``CURATOR_FIELDS``.

Restrictions and assumptions
----------------------------

- The curator does not handle nested dataclasses generically. Per-technique diagnostic
  classes are flat (every public field is a Python primitive or numpy scalar); when a
  future diagnostic dataclass needs nested structure the curator will need a recursive
  variant.
- Non-finite floats (``+inf``, ``-inf``, ``nan``) are mapped: ``+inf`` becomes
  ``JSON_INF_SENTINEL``, ``-inf`` becomes ``-JSON_INF_SENTINEL``, ``nan`` becomes ``0.0``.
  The sentinel is a documented finite value the JSON schema reserves for "unbounded".
- The curator does not include the per-image image array in the JSON (it would balloon
  the sidecar to multi-megabyte sizes). An external image-export step writes the image
  alongside the JSON when needed.

Sources of uncertainty
----------------------

The curator reports no uncertainty. Every output is a deterministic projection of the
input :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`.

Configuration
=============

The curator carries no YAML configuration of its own. The three rounding constants
(``PIXEL_DECIMALS``, ``CONFIDENCE_DECIMALS``, ``ET_DECIMALS``) and the
``JSON_INF_SENTINEL`` live as module-level constants; ``JSON_INF_SENTINEL`` lives in
:mod:`spindoctor.feature.constants` and is shared with other JSON producers.

Implementation
==============

Source file: ``src/spindoctor/nav_orchestrator/curator.py`` —
:func:`~spindoctor.nav_orchestrator.curator.build_metadata_dict`,
:func:`~spindoctor.nav_orchestrator.curator.assert_diagnostic_fields_present`, plus the private
``_round_float`` / ``_round_pair`` / ``_round_2x2`` rounding helpers.

Public surface (autodocumented at :doc:`/api_reference/api_nav_orchestrator`):

- :func:`~spindoctor.nav_orchestrator.curator.build_metadata_dict` — turns a
  :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` into a JSON-friendly nested dict.
  Public entry point for the per-image-sidecar writer.
- :func:`~spindoctor.nav_orchestrator.curator.assert_diagnostic_fields_present` — verifies every
  per-technique diagnostic field has a ``CURATOR_FIELDS`` entry. Run at config-load
  time; raises :exc:`AssertionError` on the first unmapped field.

Per-technique diagnostics dataclasses (documented at
:doc:`dev_guide_techniques_diagnostics`) declare their own ``CURATOR_FIELDS`` class
attributes; the curator picks fields from each via the dataclass's
``CURATOR_FIELDS`` rather than from
:func:`dataclasses.fields` directly.

Examples
========

**Per-image JSON sidecar shape.**  After a successful
:class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` fit on
``body_partial_overflow``, the curator emits::

    {
      "navigation_result": {
        "status": "ok",
        "offset_px": [11.06, 30.53],
        "sigma_px": [0.125, 0.122],
        "confidence_rank": "high",
        "confidence": 0.794,
        "confidence_provisional": true,
        "status_reason": "OK",
        "covariance_px2": [[0.0156, 0.0017], [0.0017, 0.0148]],
        "per_technique": [
          {
            "technique_name": "BodyLimbNav",
            "feature_ids": ["limb_arc:RHEA"],
            "offset_px": [12.06, 30.53],
            "covariance_px2": [[0.0156, 0.0017], [0.0017, 0.0148]],
            "confidence": 0.794,
            "spurious": false,
            "at_edge": false,
            "diagnostics": {
              "visible_limb_arc_fraction": 0.85,
              "visible_arc_px": 120.0,
              "dt_fit_rms_px": 0.4,
              "lm_iterations": 5,
              "tukey_inlier_count": 118
            }
          }
        ],
        ...
      }
    }

Every per-technique diagnostic key under ``"diagnostics"`` corresponds to a non-``None``
entry in the diagnostics dataclass's ``CURATOR_FIELDS``.

The literal ``"confidence_provisional": true`` marker flags that the confidence values
and ``confidence_rank`` tiers are calibrated against simulated planted-truth recovery
only (sim-anchored) and must not be read as probabilities of real-image accuracy; the
curator emits it unconditionally until a calibration anchored to real-image error
measurements lands.

**Allow-list catches a missed field.**  An operator adds a new field
``mean_polarity_score`` to
:class:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics` without updating
``CURATOR_FIELDS``. At startup
:func:`~spindoctor.nav_orchestrator.curator.assert_diagnostic_fields_present` runs over the
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult` returned by the smoke test, walks the
diagnostic's :func:`dataclasses.fields`, and raises::

    AssertionError: BodyLimbDiagnostics has unmapped fields ['mean_polarity_score'];
    add them to CURATOR_FIELDS or set value to None to skip

The build fails before the new field can silently disappear from the JSON sidecar.

**Non-finite handling.**  A pathological technique reports ``rotation_rad = +inf`` (a
genuine cost-collapse case the LM refiner mapped to the rotation-unobservable sentinel).
The curator emits ``JSON_INF_SENTINEL`` (``1.0e30``) in the JSON instead of ``+inf``,
keeping the file JSON-spec-compliant; downstream readers consult the sentinel to
distinguish "intentionally unbounded" from a numerical NaN.
