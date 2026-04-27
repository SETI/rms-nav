============
Introduction
============

This guide is intended for developers who want to understand, modify, or
extend the RMS-NAV system.  It gives an overview of the system
architecture, the class hierarchy, and instructions for extending the
pipeline with new components.

System architecture
===================

RMS-NAV is organised around six cooperating subsystems:

1. :class:`~nav.dataset.dataset.DataSet` — handles image-file access
   and organisation across mission archives.
2. :class:`~nav.obs.obs_snapshot.ObsSnapshot` and its per-instrument
   subclasses — wrap an ``oops`` observation and supply backplanes,
   extended-FOV accessors, and per-instrument metadata.
3. :class:`~nav.nav_model.nav_model.NavModel` — generates the predicted
   appearance of one part of the scene (stars, a body, or rings) and
   emits :class:`~nav.feature.feature.NavFeature` instances ready for
   technique consumption plus
   :class:`~nav.annotation.annotations.Annotations` for the summary
   image.
4. :class:`~nav.nav_technique.nav_technique.NavTechnique` — consumes
   feature subsets and produces a
   :class:`~nav.nav_technique.technique_result.NavTechniqueResult`
   (offset, covariance, calibrated confidence, diagnostics).
5. :class:`~nav.nav_orchestrator.orchestrator.NavOrchestrator` — the
   top-level driver that runs the two-pass pipeline and reconciles
   per-technique results via
   :func:`~nav.nav_orchestrator.ensemble.ensemble`, returning a single
   :class:`~nav.nav_orchestrator.nav_result.NavResult`.
6. :mod:`nav.annotation` — composes per-NavModel annotations into the
   summary-PNG overlay.

Data flow
---------

1. ``nav_offset`` (or another CLI driver) constructs a
   :class:`~nav.dataset.dataset.DataSet` and yields one
   :class:`~nav.dataset.dataset.ImageFile` at a time.
2. The matching ``ObsSnapshotInst`` subclass reads the file via
   ``from_file(...)``.
3. :func:`~nav.nav_model.nav_model.build_models_for_obs` walks the
   :class:`~nav.nav_model.nav_model.NavModel` registry and constructs
   per-image instances applicable to the observation.
4. The :class:`~nav.nav_orchestrator.orchestrator.NavOrchestrator`
   builds a :class:`~nav.nav_orchestrator.nav_context.NavContext`,
   gathers features, gates them by reliability, runs every feasible
   technique, and ensembles the per-technique results.
5. :func:`~nav.nav_orchestrator.curator.build_metadata_dict` projects
   the resulting :class:`~nav.nav_orchestrator.nav_result.NavResult`
   into a JSON-friendly metadata block.
6. :func:`~nav.navigate_image_files.navigate_image_files` writes the
   metadata JSON and (when implemented) the summary PNG.

The orchestrator's API is described in detail in
:doc:`developer_guide_class_hierarchy`.
