============================
Developer Guide: Static Data
============================

The autonomous-navigation pipeline relies on a small set of static-data
YAML catalogues that capture per-body shape parameters, per-ring radial
uncertainties, and per-instrument noise / image-quality / photometry
constants. These tables substitute for cross-image statistical learning:
no run depends on the result of any other run, but every run benefits
from values that astronomers and instrument teams have already calibrated.

The static-data files live in ``src/nav/config_files/`` alongside the
runtime configuration YAMLs and use the same numeric-prefix loading
contract (lower number loads first; later files override earlier ones).

File layout
===========

* ``config_220_body_shape.yaml`` populates ``config.body_shape`` —
  per-body radii / ellipsoid residuals / albedo / crater scale; consumed
  by the body-feature extractors to derive per-image limb uncertainty
  and reliability scores.
* ``config_3N0_*_rings.yaml`` populates ``config.rings`` — per-ring-
  feature radii, eccentricities, RMS radial precision; consumed by the
  ring-edge extractor to derive per-edge ``sigma_radial``.
* ``config_4N0_inst_*.yaml`` populates ``config.<camera>`` — per-camera
  ``noise:``, ``mag_offset:``, ``image_quality_thresholds:``, and
  ``source_image_filter:`` blocks; consumed by the orchestrator
  preflight and by the star photometry helper.

Citation requirement
====================

Every numeric value in ``config_220_body_shape.yaml`` and any new value
added to a ``config_4N0_inst_*.yaml`` ``noise:`` / ``mag_offset:`` block
**requires an accurate, non-fabricated citation**. The reasoning:

* Navigation trust is downstream-safety-critical. An invented
  ``ellipsoid_rms_residual_km`` propagates silently into every
  per-feature uncertainty estimate for that body for every image
  forever.
* The runtime has no cross-image cross-check that would catch a wrong
  value; the orchestrator trusts the static data and feeds it directly
  into reliability scores and technique covariances.

Schema
------

Each body block in ``config_220_body_shape.yaml`` is wrapped in a
top-level ``body_shape:`` mapping; each entry is keyed by SPICE body
name and carries an optional sibling ``_sources`` mapping. Keys
beginning with ``_`` are stripped at config-load time so the
documentation does not bloat the parsed ``Config`` — the citation lives
in the file for human review only.

.. code-block:: yaml

    body_shape:
      MIMAS:
        radii_km: [207.4, 196.8, 190.6]
        ellipsoid_rms_residual_km: 1.4
        crater_scale_km: 3.0
        albedo_mean: 0.96
        albedo_variation: 0.05
        shape_class_hint: regular
        _sources:
          radii_km: 'Thomas (2010), Icarus 208(1):395-401, Table 3, Mimas row.
                     doi:10.1016/j.icarus.2010.01.025'
          ellipsoid_rms_residual_km: '...'
          # ... and so on for every numeric field.

Anti-hallucination procedure
----------------------------

AI agents drafting body-shape entries:

1. **Cite only documents fetched in-session.** Every citation must be
   traceable to a ``WebFetch`` / ``WebSearch`` lookup performed in the
   same session, or to an ``oops``-package data file read directly. No
   citing from training-data memory.
2. If a value cannot be sourced from a fetched document, leave it as
   ``null`` and write
   ``'PLACEHOLDER — no source found, calibrate in Phase 10'`` as the
   ``_sources`` entry. The runtime fallback (10 % radius default plus a
   reliability cap of 0.3) handles ``null`` values.
3. DOIs and paper titles must verify against a real
   ``https://doi.org/<DOI>`` lookup; agents do not invent identifiers.
4. Any draft PR that lists a citation an AI agent invented (caught in
   human review) is reverted in full and re-drafted by a different
   process.

Human review
------------

Every PR touching ``config_220_body_shape.yaml`` requires a reviewer to
spot-check **at least 5 randomly-selected citations** by opening the
cited document and verifying the value appears at the cited location.
PRs are merged only after the reviewer marks the PR with the
``cited-values-spot-checked`` label. To keep review tractable, an
initial-population PR is broken into ≤ 10 bodies per PR.

Validation tests
================

``tests/nav/config_files/test_body_shape_citations.py`` enforces:

* Every body declares a ``_sources`` mapping.
* Every required numeric / list field on a body has a corresponding
  ``_sources`` entry that is a non-empty string.
* No ``_sources`` value contains the substrings ``TODO`` / ``FIXME`` /
  ``XXX`` (case-insensitive).
* ``PLACEHOLDER`` is allowed only when the value itself is ``null``.

The same validation pattern extends to per-camera ``noise:`` /
``mag_offset:`` blocks in ``config_4N0_inst_*.yaml`` and to any new
entries added to ``config_3N0_*_rings.yaml``. Existing ring-catalogue
values are grandfathered (they were curated by orbit-fitting astronomers
and the catalogues document their pedigree in the file header) — only
*new* additions need explicit ``_sources`` entries.

Loader behaviour
================

``Config._load_yaml`` strips every mapping key whose name starts with
``_`` before merging, so ``_sources`` blocks never appear in the parsed
``Config`` object. The runtime accessors (``config.body_shape``,
``config.<camera>.mag_offset``, etc.) see only the value-bearing
fields. Tests assert this behaviour explicitly so the strip rule cannot
regress silently.
