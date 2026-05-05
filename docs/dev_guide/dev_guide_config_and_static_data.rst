==========================
Config and Static Data
==========================

Every RMS-NAV subsystem reads its tunables from a single
:class:`~nav.config.config.Config` object loaded from a stack of YAML files. The
files split cleanly into two kinds: **runtime configuration** (knobs an operator
might want to tune per run — search ranges, emission thresholds, label fonts) and
**static data** (per-body shape tables, per-ring catalogues, per-instrument
calibration constants curated from external publications). Both kinds load through
the same loader; they differ in how they are reviewed and updated.

This chapter documents the loader, the section structure, the file layout, and the
static-data citation discipline.

The Config object
=================

:class:`~nav.config.config.Config` lazily loads its YAML stack on first attribute
access. The stack is, in order (later files override earlier ones for the same key):

1. The bundled ``src/nav/config_files/*.yaml`` files, sorted by filename. The
   3-digit numeric prefix is the merge order; the file groups are documented
   under :ref:`config-file-layout` below.
2. ``nav_default_config.yaml`` in the current working directory (if present),
   for per-checkout personal defaults.
3. Any files passed via ``--config-file PATH`` on the CLI (repeatable; each
   merges in order).
4. The handful of CLI flags that map to config keys (``--pds3-holdings-root``,
   ``--nav-results-root``, etc.) override the corresponding ``environment``
   block entry.

Direct programmatic use:

.. code-block:: python

   from nav.config import Config, DEFAULT_CONFIG

   cfg = Config()                            # lazy; reads the stack on first access
   cfg.update_config('custom.yaml')          # merge in an override file
   print(cfg.offset.correlation_fft_upsample_factor)
   print(cfg.environment.pds3_holdings_root)

The module-level :data:`~nav.config.config.DEFAULT_CONFIG` singleton is what
most subsystems read when no explicit ``config=`` keyword is supplied.
Per-class ``config`` properties on
:class:`~nav.support.nav_base.NavBase` subclasses surface the same singleton
through dependency injection.

Sections
--------

Top-level YAML keys are exposed as
:class:`~nav.support.attrdict.AttrDict` properties so code can write
``cfg.bodies.use_lambert`` instead of ``cfg['bodies']['use_lambert']``. The
shipping sections:

- ``general`` — logging levels and other global settings.
- ``offset`` — correlation and star-refinement parameters.
- ``stars`` — star-model and ring-occlusion parameters; see
  :doc:`dev_guide_navigation_models_star`.
- ``bodies`` — body rendering parameters; see
  :doc:`dev_guide_navigation_models_body`.
- ``rings`` — ring-model parameters (planet shadow removal, fade widths,
  per-planet ``ring_features``); see
  :doc:`dev_guide_navigation_models_ring`.
- ``titan`` — Titan-specific parameters (placeholder).
- ``bootstrap`` — bootstrap navigation parameters.
- ``backplanes`` — the list of body and ring backplanes to generate; see
  :doc:`dev_guide_backplanes`.
- ``pds4`` — per-dataset PDS4 template directories and bundle names; see
  :doc:`dev_guide_pds4`.
- ``environment`` — runtime paths (``pds3_holdings_root``,
  ``nav_results_root``, ``backplane_results_root``, ``bundle_results_root``).
- ``body_shape`` — static per-body shape catalogue (see
  :ref:`static-data-citations` below).
- ``coiss`` / ``vgiss`` / ``gossi`` / ``nhlorri`` — per-camera blocks
  (``noise``, ``mag_offset``, ``image_quality_thresholds``,
  ``source_image_filter``, etc.).
- ``techniques`` — per-:class:`~nav.nav_technique.nav_technique.NavTechnique`
  tunables and confidence-formula
  coefficients; see :doc:`dev_guide_techniques`.
- ``satellites`` — per-planet satellite lists used by the body
  :class:`~nav.nav_model.nav_model.NavModel`'s inventory query.
- ``feature_emission`` — per-planet
  :attr:`~nav.feature.feature_type.NavFeatureType.RING_EDGE` vs
  :attr:`~nav.feature.feature_type.NavFeatureType.RING_ANNULUS` gates; see
  :doc:`dev_guide_techniques_ring_annulus`.

The user-facing tour at :doc:`/introduction_configuration` lists every shipping
file with one-sentence descriptions.

.. _config-file-layout:

File layout
===========

The numeric prefix encodes the load order and hints at the section's role. The
ranges are conventional, not enforced by the loader:

.. list-table::
   :header-rows: 1
   :widths: 18 35 47

   * - Prefix
     - Group
     - Files
   * - ``0xx``
     - Global / model-shared
     - ``config_010_general``, ``config_020_offset``, ``config_030_stars``,
       ``config_040_bodies``, ``config_050_rings``, ``config_060_titan``,
       ``config_070_bootstrap``
   * - ``1xx``
     - Catalogues
     - ``config_100_satellites``
   * - ``2xx``
     - Per-target tables (static data)
     - ``config_220_body_shape``
   * - ``3xx``
     - Per-planet ring catalogues (static data)
     - ``config_300_jupiter_rings``, ``config_310_saturn_rings``,
       ``config_320_uranus_rings``, ``config_330_neptune_rings``
   * - ``4xx``
     - Per-instrument camera blocks (mixed runtime + static)
     - ``config_400_inst_coiss``, ``config_410_inst_gossi``,
       ``config_420_inst_nhlorri``, ``config_430_inst_vgiss``,
       ``config_440_sim``
   * - ``5xx``
     - Per-technique tunables
     - ``config_510_techniques``
   * - ``9xx``
     - Downstream-product settings
     - ``config_900_backplanes``, ``config_950_pds4``

Loader rules
------------

- YAML mappings are deep-merged, not overwritten. A user override that sets
  ``bodies.use_lambert: false`` does not unset every other key under
  ``bodies``.
- Lists are overwritten wholesale. A user override of ``stars.catalogs``
  overwrites the bundled list rather than appending to it.
- Mapping keys whose name starts with ``_`` are stripped at load time. This
  is the strip-rule that lets static-data files carry ``_sources`` blocks
  alongside their numeric values without bloating the parsed
  :class:`~nav.config.config.Config` object;
  see :ref:`static-data-citations`.

Numeric values are typed by YAML; downstream consumers convert with explicit
casts where the section schema is mixed (e.g. the per-instrument
``image_quality_thresholds`` block constructs a frozen
:class:`~nav.nav_orchestrator.image_classifier.ImageQualityThresholds`
dataclass).

Path resolution
---------------

The ``environment`` block carries the four downstream output roots:

- ``pds3_holdings_root`` — read root for PDS3 holdings (default
  ``$PDS3_HOLDINGS_DIR``, falling back to
  ``https://pds-rings.seti.org/holdings``).
- ``nav_results_root`` — write root for ``_metadata.json`` and ``_summary.png``
  files produced by ``nav_offset``.
- ``backplane_results_root`` — write root for backplane FITS / NumPy products
  produced by ``nav_backplanes``.
- ``bundle_results_root`` — write root for PDS4 bundles produced by
  ``nav_create_bundle``.

Each value may be a local path or a URL; ``filecache``-aware consumers handle
both. Environment-variable overrides
(``PDS3_HOLDINGS_DIR``, ``NAV_RESULTS_ROOT``, ``BACKPLANE_RESULTS_ROOT``,
``BUNDLE_RESULTS_ROOT``) take precedence over the YAML defaults; CLI flags
take precedence over the env vars.

.. _static-data-citations:

Static data: catalogues and citation discipline
================================================

The pipeline treats a small set of YAML files as **static data**: per-body
shape parameters, per-ring radial uncertainties, and per-instrument
photometric / noise constants. These tables substitute for cross-image
statistical learning — no run depends on the result of any other run, but
every run benefits from values that astronomers and instrument teams have
already calibrated.

Static-data files
-----------------

- ``config_220_body_shape.yaml`` populates ``config.body_shape`` — per-body
  radii, ellipsoid residuals, albedo, crater scale; consumed by
  :func:`~nav.nav_model.body_shape.shape_for_body` and from there by every
  body :class:`~nav.nav_technique.nav_technique.NavTechnique`'s covariance
  and reliability formula. See
  :doc:`dev_guide_navigation_models_body`.
- ``config_3N0_*_rings.yaml`` populate ``config.rings.<planet>.ring_features``
  — per-ring-edge radii, eccentricities, RMS radial precision; consumed by
  the ring-edge extractor to derive per-edge ``sigma_radial``. See
  :doc:`dev_guide_navigation_models_ring`.
- ``config_4N0_inst_*.yaml`` populate ``config.<camera>`` — per-camera
  ``noise``, ``mag_offset``, ``image_quality_thresholds``, and
  ``source_image_filter`` blocks; consumed by the orchestrator preflight, the
  star photometry helper, and the per-instrument PSF model.

Citation requirement
--------------------

Every numeric value in ``config_220_body_shape.yaml`` and any new value added
to a ``config_4N0_inst_*.yaml`` ``noise:`` / ``mag_offset:`` block **requires
an accurate, non-fabricated citation**. The reasoning:

- Navigation trust is downstream-safety-critical. An invented
  ``ellipsoid_rms_residual_km`` propagates silently into every per-feature
  uncertainty estimate for that body for every image forever.
- The runtime has no cross-image cross-check that would catch a wrong value;
  the orchestrator trusts the static data and feeds it directly into
  reliability scores and technique covariances.

Schema
------

Each body block in ``config_220_body_shape.yaml`` is wrapped in a top-level
``body_shape:`` mapping; each entry is keyed by upper-case SPICE body name and
carries an optional sibling ``_sources`` mapping. Keys beginning with ``_``
are stripped at config-load time so the documentation does not bloat the
parsed ``Config`` — the citation lives in the file for human review only.

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

1. **Cite only documents fetched in-session.**  Every citation must be
   traceable to a ``WebFetch`` / ``WebSearch`` lookup performed in the same
   session, or to an ``oops``-package data file read directly. No citing
   from training-data memory.
2. If a value cannot be sourced from a fetched document, leave it as ``null``
   and write
   ``'PLACEHOLDER — no source found, calibration pending'`` as the
   ``_sources`` entry. The runtime fallback (10 % radius default plus a
   reliability cap of 0.3) handles ``null`` values.
3. DOIs and paper titles must verify against a real
   ``https://doi.org/<DOI>`` lookup; agents do not invent identifiers.
4. Any draft PR that lists a citation an AI agent invented (caught in human
   review) is reverted in full and re-drafted by a different process.

Human review
------------

Every PR touching ``config_220_body_shape.yaml`` requires a reviewer to
spot-check **at least 5 randomly-selected citations** by opening the cited
document and verifying the value appears at the cited location. PRs are
merged only after the reviewer marks the PR with the
``cited-values-spot-checked`` label. To keep review tractable, an
initial-population PR is broken into ≤ 10 bodies per PR.

Validation tests
----------------

``tests/nav/config_files/test_body_shape_citations.py`` enforces:

- Every body declares a ``_sources`` mapping.
- Every required numeric / list field on a body has a corresponding
  ``_sources`` entry that is a non-empty string.
- No ``_sources`` value contains the substrings ``TODO`` / ``FIXME`` /
  ``XXX`` (case-insensitive).
- ``PLACEHOLDER`` is allowed only when the value itself is ``null``.

The same validation pattern extends to per-camera ``noise`` /
``mag_offset`` blocks in ``config_4N0_inst_*.yaml`` and to any new entries
added to ``config_3N0_*_rings.yaml``. Existing ring-catalogue values are
grandfathered (they were curated by orbit-fitting astronomers and the
catalogues document their pedigree in the file header) — only *new*
additions need explicit ``_sources`` entries.

Strip-rule guarantee
--------------------

:meth:`~nav.config.config.Config._load_yaml` strips every mapping key whose
name starts with ``_`` before merging, so ``_sources`` blocks never appear in
the parsed :class:`~nav.config.config.Config` object. The runtime accessors
(``config.body_shape``, ``config.<camera>.mag_offset``, etc.) see only the
value-bearing fields. Tests assert this behaviour explicitly so the strip
rule cannot regress silently.

Adding a new tunable
====================

When a new YAML knob is added:

1. Pick the file whose section it belongs to (e.g. a new ``bodies`` key goes
   in ``config_040_bodies.yaml``). If no existing file fits, allocate a new
   numeric prefix per the layout table above.
2. Add the key with a sensible default and a one-line YAML comment naming
   the consumer.
3. Document the key on the consumer's dev-guide page (for example,
   :doc:`dev_guide_navigation_models_body` lists every key under
   ``bodies``). The page lists name, type, default, units, and consumer.
4. If the key represents static data (a measured constant rather than a
   knob), add the ``_sources`` entry per the citation discipline above.
5. Add or extend a unit test under ``tests/nav/config_files/`` that asserts
   the loader exposes the key with the expected default.
