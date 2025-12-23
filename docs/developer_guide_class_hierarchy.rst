==========================================
Developer Guide: Class Hierarchy
==========================================

Class Hierarchy
===============

The following Mermaid diagram shows the complete class hierarchy of the RMS-NAV system:

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
          +__init__(*, config=None)
          +_img_name_valid(name)*
          +add_selection_arguments(parser, group)*
          +yield_image_files_from_arguments(args)*
          +yield_image_files_index(**kwargs)*
          +supported_grouping()
          +pds4_bundle_template_dir()*
          +pds4_bundle_name()*
          +pds4_bundle_path_for_image(name)*
          +pds4_path_stub(image_file)*
          +pds4_template_variables(...)*
          +pds4_image_name_to_data_lidvid(name)*
          +pds4_image_name_to_browse_lidvid(name)*
      }

      class DataSetPDS3 {
          +__init__(*, config=None)
          +_img_name_valid(name)
          +read_pds3_file(path)
      }

      class DataSetPDS3CassiniISS {
          +__init__(*, config=None)
          +_img_name_valid(name)
          +pds4_bundle_template_dir()
          +pds4_bundle_name()
          +pds4_bundle_path_for_image(name)
          +pds4_path_stub(image_file)
          +pds4_template_variables(...)
          +pds4_image_name_to_data_lidvid(name)
          +pds4_image_name_to_browse_lidvid(name)
      }

      class DataSetPDS3CassiniISSCruise {
          +__init__(*, config=None)
          +_img_name_valid(name)
      }

      class DataSetPDS3CassiniISSSaturn {
          +__init__(*, config=None)
          +_img_name_valid(name)
      }

      class DataSetPDS3VoyagerISS {
          +__init__(*, config=None)
          +_img_name_valid(name)
      }

      class DataSetPDS3GalileoSSI {
          +__init__(*, config=None)
          +_img_name_valid(name)
      }

      class DataSetPDS3NewHorizonsLORRI {
          +__init__(*, config=None)
          +_img_name_valid(name)
      }

      class DataSetSim {
          +__init__(*, config=None)
          +_img_name_valid(name)
      }

      class DataSetPDS4 {
          +__init__(*, config=None)
          +_img_name_valid(name)
      }

      class Obs {
          <<abstract>>
          +__init__(*, config=None, **kwargs)
      }

      class ObsSnapshot {
          +__init__(snapshot, *, extfov_margin_vu=None, config=None, **kwargs)
          +inventory_body_in_fov(inv)
          +inventory_body_in_extfov(inv)
          +clip_rect_fov(u_min, u_max, v_min, v_max)
          +clip_rect_extfov(u_min, u_max, v_min, v_max)
          +backplanes*
          +get_backplane(name)
      }

      class ObsInst {
          <<abstract>>
          +from_file(path, *, config=None, extfov_margin_vu=None, **kwargs)*
          +star_min_usable_vmag()*
          +star_max_usable_vmag()*
          +get_public_metadata()*
      }

      class ObsSnapshotInst {
          <<abstract>>
          +from_file(path, *, config=None, extfov_margin_vu=None, **kwargs)*
      }

      class ObsCassiniISS {
          +from_file(path, *, config=None, extfov_margin_vu=None)
      }

      class ObsVoyagerISS {
          +from_file(path, *, config=None, extfov_margin_vu=None)
      }

      class ObsGalileoSSI {
          +from_file(path, *, config=None, extfov_margin_vu=None)
      }

      class ObsNewHorizonsLORRI {
          +from_file(path, *, config=None, extfov_margin_vu=None)
      }

      class ObsSim {
          +from_file(path, *, config=None, extfov_margin_vu=None)
      }

      class NavMaster {
          +__init__(obs, *, nav_models=None, nav_techniques=None, config=None)
          +obs
          +models
          +compute_all_models()
          +navigate()
          +metadata_serializable()
      }

      class NavModel {
          <<abstract>>
          +__init__(name, obs, *, config=None)
          +obs
          +model_img
          +model_mask
          +create_model(always_create_model=False, never_create_model=False, create_annotations=True)*
      }

      class NavModelStars {
          +__init__(name, obs, *, config=None)
          +create_model(always_create_model=False, never_create_model=False, create_annotations=True)
          +star_list
      }

      class NavModelBodyBase {
          <<abstract>>
          +__init__(name, obs, *, config=None)
          +create_model(always_create_model=False, never_create_model=False, create_annotations=True)*
      }

      class NavModelBody {
          +__init__(name, obs, *, config=None)
          +create_model(always_create_model=False, never_create_model=False, create_annotations=True)
      }

      class NavModelBodySimulated {
          +__init__(name, obs, *, config=None)
          +create_model(always_create_model=False, never_create_model=False, create_annotations=True)
      }

      class NavModelRings {
          +__init__(name, obs, *, config=None)
          +create_model(always_create_model=False, never_create_model=False, create_annotations=True)
      }

      class NavModelTitan {
          +__init__(name, obs, *, config=None)
          +create_model(always_create_model=False, never_create_model=False, create_annotations=True)
      }

      class NavModelCombined {
          +__init__(name, obs, models, *, config=None)
          +create_model(always_create_model=False, never_create_model=False, create_annotations=True)
      }

      class NavTechnique {
          <<abstract>>
          +__init__(nav_master, *, config=None)
          +nav_master
          +navigate()*
          +offset
          +uncertainty
          +confidence
      }

      class NavTechniqueCorrelateAll {
          +__init__(nav_master, *, config=None)
          +navigate()
          +combined_model()
      }

      class NavTechniqueManual {
          +__init__(nav_master, *, config=None)
          +navigate()
          +combined_model()
      }

      class NavTechniqueTitan {
          +__init__(nav_master, *, config=None)
          +navigate()
      }

      class Annotation {
          <<abstract>>
          +__init__(*, config=None)
          +draw(image)*
      }

      class Annotations {
          +__init__(*, config=None)
          +annotations: List[Annotation]
          +add(annotation)
          +draw_all(image)
      }

      class AnnotationTextInfo {
          +__init__(text, position, *, config=None)
          +draw(image)
      }

      class Config {
          +__init__()
          +read_config(config_path=None, reread=False)
          +update_config(config_path, read_default=True)
          +category(name)
      }

      NavBase <|-- DataSet
      NavBase <|-- Obs
      NavBase <|-- NavMaster
      NavBase <|-- NavModel
      NavBase <|-- NavTechnique
      NavBase <|-- Annotation

      DataSet <|-- DataSetPDS3
      DataSet <|-- DataSetSim
      DataSet <|-- DataSetPDS4
      DataSetPDS3 <|-- DataSetPDS3CassiniISS
      DataSetPDS3 <|-- DataSetPDS3VoyagerISS
      DataSetPDS3 <|-- DataSetPDS3GalileoSSI
      DataSetPDS3 <|-- DataSetPDS3NewHorizonsLORRI
      DataSetPDS3CassiniISS <|-- DataSetPDS3CassiniISSCruise
      DataSetPDS3CassiniISS <|-- DataSetPDS3CassiniISSSaturn

      Obs <|-- ObsSnapshot
      ObsInst <|-- ObsSnapshotInst
      ObsSnapshot <|-- ObsSnapshotInst
      ObsSnapshotInst <|-- ObsCassiniISS
      ObsSnapshotInst <|-- ObsVoyagerISS
      ObsSnapshotInst <|-- ObsGalileoSSI
      ObsSnapshotInst <|-- ObsNewHorizonsLORRI
      ObsSnapshotInst <|-- ObsSim

      NavModel <|-- NavModelStars
      NavModel <|-- NavModelBodyBase
      NavModel <|-- NavModelRings
      NavModel <|-- NavModelTitan
      NavModel <|-- NavModelCombined

      NavModelBodyBase <|-- NavModelBody
      NavModelBodyBase <|-- NavModelBodySimulated

      NavTechnique <|-- NavTechniqueCorrelateAll
      NavTechnique <|-- NavTechniqueManual
      NavTechnique <|-- NavTechniqueTitan

      Annotation <|-- AnnotationTextInfo
      Annotations --> Annotation

Key Components
==============

NavBase
-------

:class:`nav.support.nav_base.NavBase` is the base class for most components in the
system. It provides access to configuration settings and a logger via the ``config``
and ``logger`` properties. All subclasses call ``super().__init__(config=...)`` to
ensure the shared state is initialized.

NavMaster
---------

:class:`nav.nav_master.nav_master.NavMaster` coordinates the navigation process. It
initializes with an :class:`oops.observation.snapshot.Snapshot`-backed observation
and optional lists of navigation models and techniques (``nav_models`` and
``nav_techniques``). It computes models, applies techniques (for example,
``correlate_all`` and ``manual``), determines the prevailing offset based on
confidence, and produces both a summary PNG image and JSON-serializable metadata via
``metadata_serializable()``.

NavModel
--------

:class:`nav.nav_model.nav_model.NavModel` is the abstract base for synthetic model
generators. Subclasses implement ``create_model(...)`` to populate arrays and
annotations. Public properties include the model name and snapshot (``name``,
``obs``), arrays (``model_img``, ``model_mask``, ``range``), optional quality
measures (``uncertainty``, ``blur_amount``, ``confidence``), optional packed
``stretch_regions`` for per-region contrast, and ``annotations``. Implementations
include complete classes for stars, bodies (including a simulated variant), rings,
and Titan, plus a combined model used to merge the nearest visible model at each
pixel.

NavTechnique
------------

:class:`nav.nav_technique.nav_technique.NavTechnique` is the abstract base for
navigation algorithms that estimate offsets from models and the observation.
Techniques are selected by name and record technique-specific metadata. Current
implementations include ``correlate_all`` (automated correlation), ``manual``
(interactive GUI), and ``titan`` (not yet implemented).

Dataset
-------

:class:`nav.dataset.dataset.DataSet` handles access to image files and metadata. It
defines ``_img_name_valid(...)``, ``add_selection_arguments(...)``,
``yield_image_files_from_arguments(...)``, and ``yield_image_files_index(...)`` for
dataset-specific selection and iteration. For PDS4 bundle generation, it also defines
methods ``pds4_bundle_template_dir()``, ``pds4_bundle_name()``,
``pds4_bundle_path_for_image()``, ``pds4_path_stub()``, ``pds4_template_variables()``,
``pds4_image_name_to_data_lidvid()``, and ``pds4_image_name_to_browse_lidvid()``.
:class:`nav.dataset.dataset_pds3.DataSetPDS3` provides volume and index-based iteration
for archives, while instrument-specific subclasses tailor parsing and volume sets.
Instrument-specific dataset classes include
:class:`nav.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISS` (base class for all
Cassini ISS volumes), :class:`nav.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISSCruise`
(volumes 1001-1009), :class:`nav.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISSSaturn`
(volumes 2001-2116), :class:`nav.dataset.dataset_pds3_voyager_iss.DataSetPDS3VoyagerISS`,
:class:`nav.dataset.dataset_pds3_galileo_ssi.DataSetPDS3GalileoSSI`,
:class:`nav.dataset.dataset_pds3_newhorizons_lorri.DataSetPDS3NewHorizonsLORRI`, and
:class:`nav.dataset.dataset_sim.DataSetSim` (for simulated images). Dataset name mapping
is defined in ``nav.dataset.__init__`` (``coiss``, ``coiss_cruise``, ``coiss_saturn``,
``gossi``, ``nhlorri``, ``vgiss``, their ``*_pds3`` aliases, and ``sim``). Cassini ISS
adds ``--camera`` (NAC or WAC) and supports a ``botsim`` grouping that pairs NAC/WAC
images when available.

Obs and ObsSnapshot
-------------------

:class:`nav.obs.obs.Obs` is the abstract base class for observations.
:class:`nav.obs.obs_snapshot.ObsSnapshot` extends it with backplane handling and
accessors, while :class:`nav.obs.obs_inst.ObsInst` defines the instrument-specific
contract with a ``from_file(...)`` constructor and metadata helpers. Instrument
snapshots extend :class:`nav.obs.obs_snapshot_inst.ObsSnapshotInst` and must accept
``config`` and ``extfov_margin_vu`` keyword arguments. Instrument classes include
:class:`nav.obs.obs_inst_cassini_iss.ObsCassiniISS`,
:class:`nav.obs.obs_inst_voyager_iss.ObsVoyagerISS`,
:class:`nav.obs.obs_inst_galileo_ssi.ObsGalileoSSI`,
:class:`nav.obs.obs_inst_newhorizons_lorri.ObsNewHorizonsLORRI`, and
:class:`nav.obs.obs_inst_sim.ObsSim` (for simulated images).

Annotation
----------

The annotation subsystem composes labels and graphical elements into an overlay used by
the final PNG. :class:`nav.annotation.annotations.Annotations` aggregates
model-provided annotations and renders them with appropriate coloring and contrast
stretching, optionally using per-region stretching via ``stretch_regions``.
