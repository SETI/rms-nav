===================
Navigation Overview
===================

This page sketches the navigation-pipeline architecture and the per-image data flow.
The chapters that follow drill into each subsystem.

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
6. :mod:`nav.annotation` — composes per-:class:`~nav.nav_model.nav_model.NavModel`
   annotations into the summary-PNG overlay.

Data flow
=========

1. ``nav_offset`` (or another CLI driver) constructs a
   :class:`~nav.dataset.dataset.DataSet` and yields one
   :class:`~nav.dataset.dataset.ImageFile` at a time.
2. The matching :class:`~nav.obs.obs_snapshot_inst.ObsSnapshotInst` subclass
   reads the file via
   :meth:`~nav.obs.obs_snapshot_inst.ObsSnapshotInst.from_file`.
3. :func:`~nav.nav_model.nav_model.build_models_for_obs` walks the
   :class:`~nav.nav_model.nav_model.NavModel` registry and constructs
   per-image instances applicable to the observation.
4. The :class:`~nav.nav_orchestrator.orchestrator.NavOrchestrator`
   builds a :class:`~nav.nav_orchestrator.nav_context.NavContext`,
   gathers features and per-:class:`~nav.nav_model.nav_model.NavModel`
   annotations, gates features by reliability, runs every feasible
   prior-free technique (pass 1), ensembles the pass-1 results to derive a
   prior, runs prior-required techniques against that prior (pass 2), and
   ensembles the union.
5. :func:`~nav.nav_orchestrator.curator.build_metadata_dict` projects
   the resulting :class:`~nav.nav_orchestrator.nav_result.NavResult`
   into a JSON-friendly metadata block.
6. :func:`~nav.navigate_image_files.navigate_image_files` writes the
   metadata JSON and the summary PNG. The PNG's annotation overlay is
   composed upstream by
   :meth:`~nav.annotation.annotations.Annotations.combine` over the
   per-:class:`~nav.nav_model.nav_model.NavModel` annotations the
   orchestrator collected; the driver composites that overlay onto the
   contrast-stretched source image and writes the bytes.

The detailed class hierarchy and inheritance graph is at
:doc:`dev_guide_class_hierarchy`.
