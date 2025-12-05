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
          +__init__(config, logger_name)
          +logger
          +config
      }

      class DataSet {
          <<abstract>>
          +__init__(config, logger_name)
          +image_name_valid(name)*
          +yield_image_filenames_from_arguments(args)*
          +add_selection_arguments(parser)
      }

      class DataSetPDS3 {
          +__init__(config, logger_name)
          +image_name_valid(name)
          +yield_image_filenames_from_arguments(args)
          +add_selection_arguments(parser)
          +read_pds3_file(path)
      }

      class DataSetCassiniISS {
          +__init__(config, logger_name)
          +image_name_valid(name)
      }

      class DataSetVoyagerISS {
          +__init__(config, logger_name)
          +image_name_valid(name)
      }

      class DataSetGalileoSSI {
          +__init__(config, logger_name)
          +image_name_valid(name)
      }

      class DataSetNewHorizonsLORRI {
          +__init__(config, logger_name)
          +image_name_valid(name)
      }

      class DataSetSim {
          +__init__(config, logger_name)
          +image_name_valid(name)
      }

      class Obs {
          +__init__(config, logger_name)
          +dict
          +img
          +img_size_vu
      }

      class ObsSnapshot {
          +__init__(obs, config, logger_name)
          +dict
          +img
          +img_size_vu
          +backplanes
          +get_backplane(name)
      }

      class ObsSnapshotInst {
          <<abstract>>
          +from_file(path, extfov_margin_vu)*
          +get_public_metadata()*
      }

      class ObsCassiniISS {
          +from_file(path, extfov_margin_vu)
      }

      class ObsVoyagerISS {
          +from_file(path, extfov_margin_vu)
      }

      class ObsGalileoSSI {
          +from_file(path, extfov_margin_vu)
      }

      class ObsNewHorizonsLORRI {
          +from_file(path, extfov_margin_vu)
      }

      class ObsSim {
          +from_file(path, extfov_margin_vu)
      }

      class NavMaster {
          +__init__(obs, config, logger_name)
          +obs
          +models
          +compute_all_models()
          +navigate()
          +metadata_serializable()
      }

      class NavModel {
          <<abstract>>
          +__init__(obs, config, logger_name)
          +obs
          +model_img
          +model_mask
          +create_model()*
      }

      class NavModelStars {
          +__init__(obs, config, logger_name)
          +create_model()
          +star_list
      }

      class NavModelBodyBase {
          <<abstract>>
          +__init__(obs, config, logger_name)
          +create_model()*
      }

      class NavModelBody {
          +__init__(obs, config, logger_name)
          +create_model()
      }

      class NavModelBodySimulated {
          +__init__(obs, config, logger_name)
          +create_model()
      }

      class NavModelRings {
          +__init__(obs, config, logger_name)
          +create_model()
      }

      class NavModelTitan {
          +__init__(obs, config, logger_name)
          +create_model()
      }

      class NavModelCombined {
          +__init__(obs, config, logger_name)
          +create_model()
      }

      class NavTechnique {
          <<abstract>>
          +__init__(nav_master, config, logger_name)
          +nav_master
          +navigate()*
          +offset
          +uncertainty
          +confidence
      }

      class NavTechniqueCorrelateAll {
          +__init__(nav_master, config, logger_name)
          +navigate()
          +combined_model()
      }

      class NavTechniqueManual {
          +__init__(nav_master, config, logger_name)
          +navigate()
          +combined_model()
      }

      class NavTechniqueTitan {
          +__init__(nav_master, config, logger_name)
          +navigate()
      }

      class Annotation {
          <<abstract>>
          +__init__(config, logger_name)
          +draw(image)*
      }

      class Annotations {
          +__init__(config, logger_name)
          +annotations: List[Annotation]
          +add(annotation)
          +draw_all(image)
      }

      class AnnotationTextInfo {
          +__init__(text, position, config, logger_name)
          +draw(image)
      }

      class Config {
          +__init__(path)
          +load()
          +save()
          +get(key, default)
      }

      NavBase <|-- DataSet
      NavBase <|-- Obs
      NavBase <|-- NavMaster
      NavBase <|-- NavModel
      NavBase <|-- NavTechnique
      NavBase <|-- Annotation

      DataSet <|-- DataSetPDS3
      DataSet <|-- DataSetSim
      DataSetPDS3 <|-- DataSetCassiniISS
      DataSetPDS3 <|-- DataSetVoyagerISS
      DataSetPDS3 <|-- DataSetGalileoSSI
      DataSetPDS3 <|-- DataSetNewHorizonsLORRI

      Obs <|-- ObsSnapshot
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

:class:`nav.support.nav_base.NavBase` is the base class for most components in the system. It provides logging, access to configuration settings, and common utility methods shared across components.

NavMaster
---------

:class:`nav.nav_master.nav_master.NavMaster` coordinates the navigation process. It initializes with an observation snapshot and optional model and technique selections, computes models, applies techniques (for example, ``correlate_all`` and ``manual``), determines the prevailing offset based on confidence, and produces both a summary PNG image and JSON-serializable metadata via ``metadata_serializable()``.

NavModel
--------

:class:`nav.nav_model.nav_model.NavModel` is the abstract base for synthetic model generators. Subclasses implement ``create_model(...)`` to populate arrays and annotations. Public properties include the model name and snapshot (``name``, ``obs``), arrays (``model_img``, ``model_mask``, ``range``), optional quality measures (``uncertainty``, ``blur_amount``, ``confidence``), optional packed ``stretch_regions`` for per-region contrast, and ``annotations``. Implementations include complete classes for stars, bodies (including a simulated variant), rings, and Titan, plus a combined model used to merge the nearest visible model at each pixel.

NavTechnique
------------

:class:`nav.nav_technique.nav_technique.NavTechnique` is the abstract base for navigation algorithms that estimate offsets from models and the observation. Techniques are selected by name and record technique-specific metadata. Current implementations include ``correlate_all`` (automated correlation), ``manual`` (interactive GUI), and ``titan`` (not yet implemented).

Dataset
-------

:class:`nav.dataset.dataset.DataSet` handles access to image files and metadata. :class:`nav.dataset.dataset_pds3.DataSetPDS3` provides volume and index-based iteration for archives, while instrument-specific subclasses tailor parsing and volume sets. Instrument-specific dataset classes include :class:`nav.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISS`, :class:`nav.dataset.dataset_pds3_voyager_iss.DataSetPDS3VoyagerISS`, :class:`nav.dataset.dataset_pds3_galileo_ssi.DataSetPDS3GalileoSSI`, :class:`nav.dataset.dataset_pds3_newhorizons_lorri.DataSetPDS3NewHorizonsLORRI`, and :class:`nav.dataset.dataset_sim.DataSetSim` (for simulated images). Some datasets add flags; for example, Cassini ISS adds ``--camera`` (NAC or WAC) and supports a ``botsim`` grouping that pairs NAC/WAC images when available.

Obs and ObsSnapshot
-------------------

:class:`nav.obs.obs.Obs` is the base class for observations. :class:`nav.obs.obs_snapshot.ObsSnapshot` extends it with backplane handling and accessors, while :class:`nav.obs.obs_snapshot_inst.ObsSnapshotInst` defines the instrument-specific snapshot contract with a ``from_file(...)`` constructor and ``get_public_metadata()``. Instrument classes include :class:`nav.obs.obs_inst_cassini_iss.ObsCassiniISS`, :class:`nav.obs.obs_inst_voyager_iss.ObsVoyagerISS`, :class:`nav.obs.obs_inst_galileo_ssi.ObsGalileoSSI`, :class:`nav.obs.obs_inst_newhorizons_lorri.ObsNewHorizonsLORRI`, and :class:`nav.obs.obs_inst_sim.ObsSim` (for simulated images).

Annotation
----------

The annotation subsystem composes labels and graphical elements into an overlay used by the final PNG. :class:`nav.annotation.annotations.Annotations` aggregates model-provided annotations and renders them with appropriate coloring and contrast stretching, optionally using per-region stretching via ``stretch_regions``.
