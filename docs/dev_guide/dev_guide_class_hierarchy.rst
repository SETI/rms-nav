===============
Class Hierarchy
===============

The autonomous-navigation pipeline is built around four cooperating
groups of classes — observation snapshots, predicted-scene models,
per-feature techniques, and the orchestrator that runs them.  The
following Mermaid diagram captures the principal relationships; the
narrative below the diagram describes each group in turn.

.. mermaid::

   classDiagram
      direction RL

      class NavBase {
          +__init__(*, config=None, **kwargs)
          +logger
          +config
      }

      class DataSet {
          <<abstract>>
          +_img_name_valid(name)*
          +yield_image_files_from_arguments(args)*
          +yield_image_files_index(**kwargs)*
      }

      class Obs {
          <<abstract>>
      }

      class ObsSnapshot {
          +backplane(...)
          +ra_dec_limits_ext()
          +extfov_data_sensor_mask()
      }

      class ObsSnapshotInst {
          <<abstract>>
          +from_file(path, *, config=None, extfov_margin_vu=None)*
      }

      class NavModel {
          <<abstract>>
          +__init__(name, obs, *, config=None)
          +name
          +obs
          +metadata
          +create_model()*
          +to_features(context)* list[NavFeature]
          +to_annotations(context)* Annotations
          +instances_for_obs(obs)$ list[NavModel]
      }

      class NavModelBodyBase {
          <<abstract>>
      }

      class NavModelBodySimulated {
          +to_features(context)
          +to_annotations(context)
      }

      class NavModelRingsBase {
          <<abstract>>
      }

      class NavModelRingsSimulated {
          +to_features(context)
          +to_annotations(context)
      }

      class NavModelTitan {
          +create_model()
          +to_features(context)
          +to_annotations(context)
      }

      class NavTechnique {
          <<abstract>>
          +name
          +accepts_feature_types
          +requires_prior
          +is_feasible(features)* NavFeasibilityReport
          +navigate(features, context)* NavTechniqueResult
      }

      class NavTechniqueManual {
          +is_feasible(features)
          +navigate(features, context)
      }

      class NavOrchestrator {
          +__init__(models, *, config=None, only_models='*', only_techniques='*')
          +navigate(obs) NavResult
      }

      class NavFeature {
          <<frozen dataclass>>
          +feature_id: str
          +feature_type: NavFeatureType
          +geometry: NavFeatureGeometry
          +position_cov_px
          +preferred_filter: NavFilterSpec
          +reliability: float
          +flags: NavFeatureFlags
      }

      class NavFeatureExtractor {
          <<future>>
      }

      class NavTechniqueResult {
          <<frozen dataclass>>
          +technique_name
          +feature_ids
          +offset_px
          +covariance_px2
          +confidence
          +spurious / at_edge
          +diagnostics: NavTechniqueDiagnostics
      }

      class NavResult {
          <<frozen dataclass>>
          +status
          +offset_px / sigma_px
          +confidence_rank
          +per_technique
          +feature_inventory
          +image_classifier
          +annotations
          +provenance
      }

      class NavContext {
          <<frozen dataclass>>
          +obs
          +image_ext
          +image_noise_sigma
          +saturation_mask_ext
          +cosmic_ray_mask_ext
          +prior_offset_px
      }

      NavBase <|-- DataSet
      NavBase <|-- Obs
      NavBase <|-- NavModel
      NavBase <|-- NavTechnique
      NavBase <|-- NavOrchestrator

      Obs <|-- ObsSnapshot
      ObsSnapshot <|-- ObsSnapshotInst

      NavModel <|-- NavModelBodyBase
      NavModel <|-- NavModelRingsBase
      NavModel <|-- NavModelTitan

      NavModelBodyBase <|-- NavModelBodySimulated
      NavModelRingsBase <|-- NavModelRingsSimulated

      NavTechnique <|-- NavTechniqueManual

      NavOrchestrator --> NavModel : iterates
      NavOrchestrator --> NavTechnique : iterates registry
      NavOrchestrator --> NavContext : builds
      NavOrchestrator --> NavResult : produces
      NavModel ..> NavFeature : emits
      NavTechnique ..> NavFeature : consumes
      NavTechnique ..> NavTechniqueResult : produces
      NavResult --> NavTechniqueResult : per_technique


Top-level driver
================

:class:`~nav.nav_orchestrator.orchestrator.NavOrchestrator` is the
top-level driver.  Given a list of pre-built
:class:`~nav.nav_model.nav_model.NavModel` instances and an
:class:`~nav.obs.obs_snapshot.ObsSnapshot`, the orchestrator runs the
full two-pass navigation pipeline: it builds a
:class:`~nav.nav_orchestrator.nav_context.NavContext`, calls each
model's ``create_model``, gathers
:class:`~nav.feature.feature.NavFeature` instances via ``to_features``,
gates them by reliability, runs every feasible
:class:`~nav.nav_technique.nav_technique.NavTechnique`, and reconciles
the per-technique results through
:func:`~nav.nav_orchestrator.ensemble.ensemble`.  The output is a single
:class:`~nav.nav_orchestrator.nav_result.NavResult`.

Glob-pattern filters at construction time
(``only_models='body:MIMAS'``,
``only_techniques='!StarFieldFromCatalogNav'``) restrict which models or
techniques run, supporting debugging and per-image study without
modifying registry contents.

NavBase
=======

:class:`~nav.support.nav_base.NavBase` is the shared base class for the
orchestrator, every model, every technique, and the dataset / obs
hierarchies.  It provides ``config`` and ``logger`` properties; every
subclass calls ``super().__init__(config=...)`` to inherit them.

NavModel
========

:class:`~nav.nav_model.nav_model.NavModel` is the abstract base for
predicted-scene generators.  Subclasses implement three methods:

- ``create_model()`` populates the model's internal state and
  ``metadata`` dict.
- ``to_features(context)`` returns a list of
  :class:`~nav.feature.feature.NavFeature` instances ready for
  technique consumption.
- ``to_annotations(context)`` returns an
  :class:`~nav.annotation.annotations.Annotations` collection that the
  orchestrator merges into ``NavResult.annotations``.

Concrete subclasses self-register via ``__init_subclass__`` unless they
opt out with ``_abstract = True``.  The class method
``instances_for_obs(obs)`` is the per-class hook that
``build_models_for_obs`` iterates.  Today's concrete subclasses include
:class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated`,
:class:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`,
and :class:`~nav.nav_model.nav_model_titan.NavModelTitan` (a registered
stub).  Real-scene body / ring / star models replace the simulated ones
when available.

The :mod:`nav.nav_model.rings` subpackage carries the catalog-driven
ring-feature data model (``RingFeature``, ``RingFeatureFilter``,
``RingRenderResult``, ``RingsRenderContext``, ``ring_math``,
``ring_types``); see
:doc:`dev_guide_navigation_models_rings` for details.

NavTechnique
============

:class:`~nav.nav_technique.nav_technique.NavTechnique` is the abstract
base for navigation algorithms.  Techniques consume a subset of
:class:`~nav.feature.feature.NavFeature` instances filtered by
``accepts_feature_types`` and produce a
:class:`~nav.nav_technique.technique_result.NavTechniqueResult` with an
offset, covariance, calibrated confidence, and per-technique
diagnostics.  ``is_feasible(features)`` is consulted before invocation
and reads feature metadata only — never pixels.

Concrete subclasses self-register.
:class:`~nav.nav_technique.nav_technique_manual.NavTechniqueManual`
opts out of the auto-discovery registry (it spawns a PyQt6 dialog) and
is invoked by interactive drivers only.  Real-scene techniques
(``BodyDiscCorrelateNav``, ``BodyLimbNav``,
``StarFieldFromCatalogNav``, ...) plug in to the same registry as they
arrive.

NavFeature and NavFeatureGeometry
=================================

A :class:`~nav.feature.feature.NavFeature` is the smallest
independently-navigable scene element: a star, one body's limb arc,
one ring edge, a body disc rendered as a pixel template, and so on.
The :class:`~nav.feature.feature_type.NavFeatureType` enum names every
shipping feature category.

The ``geometry`` field carries one of the
:data:`~nav.feature.geometry.NavFeatureGeometry` payload variants
(``StarGeometry``, ``LimbPolyline``, ``TerminatorPolyline``,
``RingEdgePolyline``, ``BodyDiscGeometry``, ``BodyBlobGeometry``,
``RingAnnulusGeometry``, ``CartographicModelGeometry``); each variant
records the in-image position the technique needs.  The ``flags`` field
carries one of the
:data:`~nav.feature.flags.NavFeatureFlags` typed dataclasses, capturing
feature-type-specific booleans (for example,
``RingEdgeFlags.is_straight_line``).

Per-feature uncertainty in image-plane pixels lives on
``position_cov_px`` (or per-vertex on the polyline payloads);
``preferred_filter`` is the
:class:`~nav.support.filters.NavFilterSpec` the feature requests for
both its template and the surrounding image patch.

NavResult, ensemble, curator
============================

The orchestrator's final answer is a
:class:`~nav.nav_orchestrator.nav_result.NavResult` carrying the headline
``offset_px`` ± ``sigma_px``, a five-bucket ``confidence_rank``, the
discrete ``status_reason``, every per-technique
:class:`~nav.nav_technique.technique_result.NavTechniqueResult`, the
per-feature inventory (kept and gated entries), the image-quality
classifier verdict, the merged annotation collection, and the
reproducibility :class:`~nav.nav_orchestrator.provenance.Provenance`.

:func:`~nav.nav_orchestrator.ensemble.ensemble` is a free function (not
a class) that performs the precision-weighted Kalman-style merge.
:func:`~nav.nav_orchestrator.curator.build_metadata_dict` projects
``NavResult`` into a JSON-friendly metadata block written by
``navigate_image_files``.

Dataset, Obs, and ObsSnapshot
=============================

:class:`~nav.dataset.dataset.DataSet` handles access to image files and
metadata; per-mission subclasses
(``DataSetPDS3CassiniISS``, ``DataSetPDS3VoyagerISS``,
``DataSetPDS3GalileoSSI``, ``DataSetPDS3NewHorizonsLORRI``,
``DataSetSim``) implement archive-specific iteration and PDS4 bundle
hooks.

:class:`~nav.obs.obs.Obs` is the abstract observation base.
:class:`~nav.obs.obs_snapshot.ObsSnapshot` adds backplane handling and
extended-FOV accessors; per-instrument subclasses derive from
:class:`~nav.obs.obs_snapshot_inst.ObsSnapshotInst` and implement the
``from_file(path, ...)`` constructor.

Annotation
==========

The :mod:`nav.annotation` subsystem composes labels and graphical
elements into an overlay used by the summary PNG.
:class:`~nav.annotation.annotations.Annotations` aggregates
model-provided annotations and renders them with appropriate coloring
and contrast stretching.  Each ``NavModel.to_annotations`` returns a
fresh ``Annotations`` collection; the orchestrator merges them into
``NavResult.annotations`` via ``add_annotations``.
