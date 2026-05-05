==========================
Code Familiarization Plan
==========================

This is a study plan for a developer who has just cloned the repository and
wants to come up to speed on RMS-NAV in the most efficient order. Each stage
lists documentation pages to read and source modules to inspect alongside
them. The earlier stages are prerequisites for the later ones; the later
stages can be read more selectively once the core pipeline makes sense.

The plan does not explain the code — it sequences it. Read each
documentation page first, then look at the corresponding source. Every class,
function, and module name below is a hyperlink into the API reference; click
through to jump straight to the autodoc page (and from there to the
``[source]`` link to the file itself).

Stage 1 — Orientation
=====================

Goal: understand what the package does, the repository layout, and the
top-level architecture. Skim only — do not try to absorb every detail.

- ``README.md`` — PyPI-facing summary of what ``rms-nav`` is and the
  installation / quick-start commands.
- :doc:`dev_guide_introduction` — environment variables, CLI entry points,
  test / lint / docs commands, CI / release process.
- :doc:`dev_guide_navigation_overview` — the six-subsystem architectural
  sketch and the per-image data flow.
- :doc:`dev_guide_class_hierarchy` — the inheritance graph and the shared
  base classes the rest of the manual assumes.

Stage 2 — Cross-cutting foundations
===================================

Goal: understand the shared infrastructure every subsystem depends on
(configuration, logging, the universal base class, common helpers).

- :doc:`dev_guide_config_and_static_data` — how YAML defaults are layered,
  the :class:`~nav.config.config.Config` singleton, and the static-data
  citation policy.
- :doc:`dev_guide_logging` — ``pdslogger`` usage, section conventions, and
  the rule against bare ``print``.
- :mod:`nav.config` — config loader and module-level logger:

  - :mod:`nav.config.config`
  - :mod:`nav.config.logger`

- ``src/nav/config_files/`` — the bundled YAML defaults (numeric-prefix load
  order); not autodoc'd since it is data, not code.
- :mod:`nav.support` — cross-cutting helpers. Start with these two; the
  rest can be browsed on demand:

  - :mod:`nav.support.nav_base` —
    :class:`~nav.support.nav_base.NavBase`, the shared base providing
    :attr:`~nav.support.nav_base.NavBase.config` and
    :attr:`~nav.support.nav_base.NavBase.logger`.
  - :mod:`nav.support.types`

Stage 3 — Inputs: datasets, observations, features
==================================================

Goal: understand how an image goes from a file path on disk to an
in-memory ``Observation`` (from ``oops``) carrying the per-pixel features
that the rest of the pipeline consumes.

- :doc:`dev_guide_observations` — :class:`~nav.dataset.dataset.DataSet`
  enumeration, the :class:`~nav.obs.obs_snapshot_inst.ObsSnapshotInst`
  per-instrument hierarchy, and the
  :class:`~nav.feature.feature.NavFeature` dataclass / feature-type
  taxonomy.
- :mod:`nav.dataset` — file enumeration. The per-mission subclasses
  share the :class:`~nav.dataset.dataset.DataSet` /
  :class:`~nav.dataset.dataset_pds3.DataSetPDS3` bases:

  - :mod:`nav.dataset.dataset_pds3_cassini_iss`
  - :mod:`nav.dataset.dataset_pds3_voyager_iss`
  - :mod:`nav.dataset.dataset_pds3_galileo_ssi`
  - :mod:`nav.dataset.dataset_pds3_newhorizons_lorri`
  - :mod:`nav.dataset.dataset_pds4`

- :mod:`nav.obs` — observation wrappers. The per-instrument
  :class:`~nav.obs.obs_inst.ObsInst` subclasses share the
  :class:`~nav.obs.obs_snapshot_inst.ObsSnapshotInst` base:

  - :mod:`nav.obs.obs_inst_cassini_iss`
  - :mod:`nav.obs.obs_inst_voyager_iss`
  - :mod:`nav.obs.obs_inst_galileo_ssi`
  - :mod:`nav.obs.obs_inst_newhorizons_lorri`

- :mod:`nav.feature`:

  - :mod:`nav.feature.feature` — the :class:`~nav.feature.feature.NavFeature`
    dataclass itself.
  - :mod:`nav.feature.feature_type` — the feature-type taxonomy.
  - :mod:`nav.feature.geometry`
  - :mod:`nav.feature.flags`
  - :mod:`nav.feature.composition`
  - :mod:`nav.feature.reliability`

Stage 4 — First end-to-end pipeline: ``NavModelBody`` + ``BodyLimbNav``
=======================================================================

Goal: walk one image end-to-end through the simplest possible
model / technique pair, plus the minimum slice of orchestrator,
annotations, and driver code needed to produce the JSON metadata and
summary PNG. Body-limb navigation is the canonical reference
distance-transform pipeline and the smallest of the autonomous
techniques.

Read in this order:

Model
-----

- :doc:`dev_guide_navigation_models` — what a
  :class:`~nav.nav_model.nav_model.NavModel` is and the three methods
  every subclass implements:

  - :meth:`~nav.nav_model.nav_model.NavModel.create_model`
  - :meth:`~nav.nav_model.nav_model.NavModel.to_features`
  - :meth:`~nav.nav_model.nav_model.NavModel.to_annotations`

- :doc:`dev_guide_navigation_models_bodies` — the body-model family.
- :doc:`dev_guide_navigation_models_body` —
  :class:`~nav.nav_model.nav_model_body.NavModelBody` in detail.
- :mod:`nav.nav_model.nav_model` — abstract base, registry, and
  :func:`~nav.nav_model.nav_model.build_models_for_obs`.
- :mod:`nav.nav_model.body_shape` — body shape support.
- :mod:`nav.nav_model.nav_model_body_base` — shared body-annotation
  helpers (:class:`~nav.nav_model.nav_model_body_base.NavModelBodyBase`).
- :mod:`nav.nav_model.nav_model_body` — the catalog-driven body model
  (:class:`~nav.nav_model.nav_model_body.NavModelBody`).

Technique
---------

- :doc:`dev_guide_techniques` — what a
  :class:`~nav.nav_technique.nav_technique.NavTechnique` is, the
  registry, and the prior-free / prior-required two-pass split.
- :doc:`dev_guide_techniques_image_derivatives` — gradient and
  edge-distance-transform images shared by every DT technique.
- :doc:`dev_guide_techniques_dt_fitting` — the coarse-NCC + Levenberg-
  Marquardt + Tukey biweight machinery the limb / terminator / ring-edge
  techniques all share.
- :doc:`dev_guide_techniques_body_limb` — the body-limb technique itself.
- :mod:`nav.nav_orchestrator.image_derivatives` — the per-image
  derivative builder.
- :mod:`nav.nav_technique.nav_technique` — abstract base + registry
  (:class:`~nav.nav_technique.nav_technique.NavTechnique`).
- :mod:`nav.nav_technique.dt_fitting` — shared DT-fitting helpers.
- :mod:`nav.nav_technique.nav_technique_body_limb` — the technique
  implementation
  (:class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav`).

Orchestrator (minimum slice)
----------------------------

- :doc:`dev_guide_orchestrator` — the orchestrator package map.
- :doc:`dev_guide_orchestrator_nav_context` — the per-image dataclass
  every model and technique reads / writes
  (:class:`~nav.nav_orchestrator.nav_context.NavContext`).
- :doc:`dev_guide_orchestrator_orchestrator` —
  :class:`~nav.nav_orchestrator.orchestrator.NavOrchestrator`: the
  two-pass driver loop. (Skim — feasibility / confidence / ensemble are
  covered in Stage 5.)
- :doc:`dev_guide_orchestrator_nav_result` — the final per-image result
  envelope (:class:`~nav.nav_orchestrator.nav_result.NavResult`).
- :mod:`nav.nav_orchestrator` — read these three first; the rest are
  picked up in Stage 5:

  - :mod:`nav.nav_orchestrator.nav_context`
  - :mod:`nav.nav_orchestrator.orchestrator`
  - :mod:`nav.nav_orchestrator.nav_result`

Output: annotations, metadata, summary PNG, top-level driver
------------------------------------------------------------

- :doc:`dev_guide_annotations` — the summary-PNG overlay system.
- :doc:`dev_guide_orchestrator_curator` — projecting
  :class:`~nav.nav_orchestrator.nav_result.NavResult` into the JSON
  metadata block.
- :mod:`nav.annotation` — annotation primitives, plus combine logic
  (:meth:`~nav.annotation.annotations.Annotations.combine`):

  - :mod:`nav.annotation.annotation`
  - :mod:`nav.annotation.annotations`
  - :mod:`nav.annotation.annotation_text_info`

- :mod:`nav.nav_orchestrator.curator` — the metadata-dict builder
  (:func:`~nav.nav_orchestrator.curator.build_metadata_dict`).
- :mod:`nav.navigate_image_files` — the top-level per-image driver
  called by ``nav_offset``
  (:func:`~nav.navigate_image_files.navigate_image_files`).
- ``src/main/nav_offset.py`` — the CLI dispatch module that wires it up
  (the ``src/main/`` tree is not part of the autodoc API surface).

Stage 5 — Orchestrator round-trip: feasibility, confidence, diagnostics, ensemble
=================================================================================

Goal: understand how individual technique results are gated, calibrated,
classified, and combined into the single per-image answer the
orchestrator returns.

- :doc:`dev_guide_techniques_feasibility` — when a technique declines to
  run and how the reason is reported.
- :doc:`dev_guide_techniques_confidence` — calibrated [0, 1] confidence
  per technique.
- :doc:`dev_guide_techniques_diagnostics` — the typed diagnostics
  dataclass attached to each technique result.
- :doc:`dev_guide_orchestrator_ensemble` — the
  :func:`~nav.nav_orchestrator.ensemble.ensemble` combine that reconciles
  per-technique results into the final offset.
- :doc:`dev_guide_orchestrator_image_classifier` — per-image
  classification feeding feasibility / weighting.
- :doc:`dev_guide_orchestrator_instrument_config` — per-instrument
  tunables the orchestrator threads through.
- :doc:`dev_guide_orchestrator_provenance` — the provenance record
  carried through the pipeline.
- :doc:`dev_guide_orchestrator_feature_summary` — the per-image
  feature-summary dataclass.
- Technique-side helpers:

  - :mod:`nav.nav_technique.feasibility`
  - :mod:`nav.nav_technique.confidence`
  - :mod:`nav.nav_technique.confidence_config`
  - :mod:`nav.nav_technique.diagnostics`
  - :mod:`nav.nav_technique.technique_result`

- Orchestrator-side helpers:

  - :mod:`nav.nav_orchestrator.ensemble`
  - :mod:`nav.nav_orchestrator.image_classifier`
  - :mod:`nav.nav_orchestrator.image_classifier_result`
  - :mod:`nav.nav_orchestrator.instrument_config`
  - :mod:`nav.nav_orchestrator.provenance`
  - :mod:`nav.nav_orchestrator.feature_summary`
  - :mod:`nav.nav_orchestrator.status_reason_info`

- :mod:`nav.support.status_reason` — the shared status-reason enum.

Stage 6 — Remaining models and their techniques
================================================

Goal: extend the picture from Stage 4 to every other registered model
and every other technique. Each subsection pairs a model with the
techniques that consume its features, in roughly increasing order of
complexity.

Stars (model + star techniques)
-------------------------------

- :doc:`dev_guide_navigation_models_stars` — the star-model family.
- :doc:`dev_guide_navigation_models_star` —
  :class:`~nav.nav_model.stars.nav_model_stars.NavModelStars` in detail.
- :doc:`dev_guide_techniques_star_unique_match` — direct catalog-to-image
  unique matching.
- :doc:`dev_guide_techniques_star_refine` — refinement against an
  existing prior.
- :doc:`dev_guide_techniques_star_field` — full star-field correlation.
- :mod:`nav.nav_model.stars` subpackage:

  - :mod:`nav.nav_model.stars.catalog`
  - :mod:`nav.nav_model.stars.detection`
  - :mod:`nav.nav_model.stars.predicted_snr`
  - :mod:`nav.nav_model.stars.smeared_psf`
  - :mod:`nav.nav_model.stars.conflicts`
  - :mod:`nav.nav_model.stars.nav_model_stars`

- :mod:`nav.nav_technique.nav_technique_star_unique_match`
- :mod:`nav.nav_technique.nav_technique_star_refine`
- :mod:`nav.nav_technique.nav_technique_star_field`
- The shared ``_star_helpers`` module is internal and not part of the
  autodoc API.

Body (additional techniques)
----------------------------

- :doc:`dev_guide_techniques_body_terminator` — illuminated-body
  terminator fitting (DT pipeline).
- :doc:`dev_guide_techniques_body_blob` — small / unresolved body
  navigation.
- :doc:`dev_guide_techniques_body_disc` — full-disc correlation.
- :mod:`nav.nav_technique.nav_technique_body_terminator`
- :mod:`nav.nav_technique.nav_technique_body_blob`
- :mod:`nav.nav_technique.nav_technique_body_disc`

Rings (model + ring techniques)
-------------------------------

- :doc:`dev_guide_navigation_models_rings` — the ring-model family.
- :doc:`dev_guide_navigation_models_ring` —
  :class:`~nav.nav_model.nav_model_rings.NavModelRings` in detail.
- :doc:`dev_guide_techniques_ring_edge` — the ring-edge DT technique.
- :doc:`dev_guide_techniques_ring_annulus` — the ring-annulus technique.
- :mod:`nav.nav_model.nav_model_rings_base`
- :mod:`nav.nav_model.nav_model_rings`
- :mod:`nav.nav_model.rings` subpackage:

  - :mod:`nav.nav_model.rings.ring_math`
  - :mod:`nav.nav_model.rings.ring_filter`
  - :mod:`nav.nav_model.rings.ring_render_context`
  - :mod:`nav.nav_model.rings.ring_render_result`
  - :mod:`nav.nav_model.rings.ring_types`
  - :mod:`nav.nav_model.rings.ring_feature`

- :mod:`nav.nav_technique.nav_technique_ring_edge`
- :mod:`nav.nav_technique.nav_technique_ring_annulus`

Titan (model only)
------------------

- :doc:`dev_guide_navigation_models_titans` — the Titan family.
- :doc:`dev_guide_navigation_models_titan` —
  :class:`~nav.nav_model.nav_model_titan.NavModelTitan` (registered
  placeholder; emits no features pending a haze-aware extractor).
- :mod:`nav.nav_model.nav_model_titan`

Manual
------

- :doc:`dev_guide_techniques_manual` — the interactive PyQt6 driver.
- :mod:`nav.nav_technique.nav_technique_manual`
- :mod:`nav.ui.manual_nav_dialog`
- :mod:`nav.ui.library_entry`
- The ``nav.ui.common`` module is internal and not autodoc'd.

Stage 7 — Simulated images
==========================

Goal: understand the synthetic-image renderer and the simulated-image
sibling of each model family. These are used both by
``nav_create_simulated_image`` and by tests.

- :doc:`dev_guide_navigation_models_star_simulated`
- :doc:`dev_guide_navigation_models_body_simulated`
- :doc:`dev_guide_navigation_models_ring_simulated`
- :doc:`dev_guide_navigation_models_titan_simulated`
- :mod:`nav.nav_model.nav_model_body_simulated`
- :mod:`nav.nav_model.nav_model_rings_simulated`
- :mod:`nav.sim` — the synthetic-image renderer:

  - :mod:`nav.sim.render`
  - :mod:`nav.sim.sim_body`
  - :mod:`nav.sim.sim_ring`

- ``src/nav/dataset/dataset_sim.py`` and ``src/nav/obs/obs_inst_sim.py``
  — the simulated-image dataset and observation wrappers (not autodoc'd).

Stage 8 — Calibration and regression
====================================

Goal: understand how confidence calibration is anchored to a curated
cohort of real images, and how per-instrument camera-rotation correction
is calibrated.

- :doc:`dev_guide_image_library` — the operator-curated regression
  cohort (``tests/integration/image_library/``), sidecar schema, and
  baseline workflow.
- :doc:`dev_guide_rotation` — per-instrument camera-rotation correction.
- ``tests/integration/`` — the integration suite that consumes the image
  library.

Stage 9 — Downstream products
=============================

Goal: understand the products built on top of a navigated image —
mosaics / reprojections, per-pixel backplanes, and PDS4 bundles.

- :doc:`dev_guide_reprojection` — body and ring mosaicing plus the
  thread-safety constraints.
- :mod:`nav.reproj` — reprojection core:

  - :mod:`nav.reproj.bodies` — ``BodyMosaic``.
  - :mod:`nav.reproj.rings` — ``RingMosaic``.
  - :mod:`nav.reproj.cartographic_model` —
    :func:`~nav.reproj.cartographic_model.create_cartographic_model`.
  - :mod:`nav.reproj.photometric_model`
  - :mod:`nav.reproj.ring_orbit_model`

- ``src/reproj_cli/`` — CLI-only helpers shared by the mosaic drivers
  (not part of the importable ``nav`` API and not autodoc'd).
- :doc:`dev_guide_backplanes` — per-pixel geometry product generation.
- ``src/backplanes/`` — driven by ``nav_backplanes`` (not autodoc'd in
  the ``nav`` API surface).
- :doc:`dev_guide_pds4` — PDS4 bundle assembly.
- ``src/pds4/`` — driven by ``nav_create_bundle``; per-dataset hooks
  live on the :class:`~nav.dataset.dataset.DataSet` subclasses
  introduced in Stage 3 (this tree is not autodoc'd either).

Stage 10 — Extending and conventions
====================================

Goal: be ready to make changes. Read these once at the end so the
conventions land on top of a working mental model of the code.

- :doc:`dev_guide_extending` — how to add a new instrument, dataset,
  model, or technique (the registry hooks each subsystem exposes).
- :doc:`dev_guide_best_practices` — line length, docstring style, ruff /
  mypy expectations, commit conventions.
- ``src/main/`` — CLI dispatch modules for every entry point listed in
  ``[project.scripts]`` (not autodoc'd).
- The PyQt6 mosaic viewer; read after :mod:`nav.ui.manual_nav_dialog`
  from Stage 6:

  - :mod:`nav.ui.mosaic_viewer.projections`
  - :mod:`nav.ui.mosaic_viewer.sphere_render`
  - :mod:`nav.ui.mosaic_viewer.graticule`

- :doc:`/contributing` — the contributor checklist that gates every PR.
