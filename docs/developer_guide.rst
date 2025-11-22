===============
Developer Guide
===============

Introduction
============

This guide is intended for developers who want to understand, modify, or extend the RMS-NAV system. It provides an overview of the system architecture, details on the class hierarchy, and instructions for extending the system with new functionality.

System Architecture
===================

RMS-NAV follows a modular architecture organized around several key components:

1. **NavMaster**: The central controller that coordinates the navigation process
2. **NavModel**: Generates theoretical models of what should appear in images
3. **NavTechnique**: Implements algorithms to match models with actual images
4. **Dataset**: Handles image file access and organization
5. **ObsSnapshot**: Manages observation data and coordinate transformations
6. **Annotation**: Creates visual overlays and text annotations

Data Flow
---------

1. The system loads an image through a Dataset implementation
2. An ObsSnapshot is created to represent the observation
3. NavMaster coordinates the creation of models (stars, bodies, rings)
4. Navigation techniques are applied to find the best offset
5. Results are processed to create annotations and overlays
6. Output files are generated (offset data, annotated images)

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

      class Inst {
          <<abstract>>
          +__init__(config, logger_name)
          +from_file(path, extfov_margin_vu)*
      }

      class InstCassiniISS {
          +__init__(config, logger_name)
          +from_file(path, extfov_margin_vu)
      }

      class InstVoyagerISS {
          +__init__(config, logger_name)
          +from_file(path, extfov_margin_vu)
      }

      class InstGalileoSSI {
          +__init__(config, logger_name)
          +from_file(path, extfov_margin_vu)
      }

      class InstNewHorizonsLORRI {
          +__init__(config, logger_name)
          +from_file(path, extfov_margin_vu)
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

      class NavMaster {
          +__init__(obs, config, logger_name)
          +obs
          +models
          +compute_all_models()
          +navigate()
          +create_overlay()
      }

      class NavModel {
          <<abstract>>
          +__init__(obs, config, logger_name)
          +obs
          +model_img
          +model_mask
          +compute_model()*
      }

      class NavModelStars {
          +__init__(obs, config, logger_name)
          +compute_model()
          +find_stars()
      }

      class NavModelBody {
          +__init__(obs, config, logger_name)
          +compute_model()
          +find_bodies()
      }

      class NavModelRings {
          +__init__(obs, config, logger_name)
          +compute_model()
      }

      class NavModelTitan {
          +__init__(obs, config, logger_name)
          +compute_model()
      }

      class NavTechnique {
          <<abstract>>
          +__init__(nav_master, config, logger_name)
          +nav_master
          +navigate()*
      }

      class NavTechniqueStars {
          +__init__(nav_master, config, logger_name)
          +navigate()
      }

      class NavTechniqueAllModels {
          +__init__(nav_master, config, logger_name)
          +navigate()
      }

      class NavTechniqueTitan {
          +__init__(nav_master, config, logger_name)
          +navigate()
      }

      class Annotation {
          +__init__(config, logger_name)
          +draw(image)
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
      NavBase <|-- Inst
      NavBase <|-- Obs
      NavBase <|-- NavMaster
      NavBase <|-- NavModel
      NavBase <|-- NavTechnique
      NavBase <|-- Annotation

      DataSet <|-- DataSetPDS3
      DataSetPDS3 <|-- DataSetCassiniISS
      DataSetPDS3 <|-- DataSetVoyagerISS
      DataSetPDS3 <|-- DataSetGalileoSSI
      DataSetPDS3 <|-- DataSetNewHorizonsLORRI

      Inst <|-- InstCassiniISS
      Inst <|-- InstVoyagerISS
      Inst <|-- InstGalileoSSI
      Inst <|-- InstNewHorizonsLORRI

      Obs <|-- ObsSnapshot

      NavModel <|-- NavModelStars
      NavModel <|-- NavModelBody
      NavModel <|-- NavModelRings
      NavModel <|-- NavModelTitan

      NavTechnique <|-- NavTechniqueStars
      NavTechnique <|-- NavTechniqueAllModels
      NavTechnique <|-- NavTechniqueTitan

      Annotation <|-- AnnotationTextInfo
      Annotations --> Annotation

Key Components
==============

NavBase
-------

``NavBase`` is the base class for most components in the system. It provides:

logging, access to configuration settings, and common utility methods shared across components.

NavMaster
---------

``NavMaster`` coordinates the navigation process. It initializes with an observation snapshot and optional model and technique selections, computes models, applies techniques (for example, ``correlate_all`` and ``manual``), determines the prevailing offset based on confidence, and produces both an overlay image and JSON-serializable metadata via ``metadata_serializable()``.

NavModel
--------

``NavModel`` is the abstract base for synthetic model generators. Subclasses implement ``create_model(...)`` to populate arrays and annotations. Public properties include the model name and snapshot (``name``, ``obs``), arrays (``model_img``, ``model_mask``, ``range``), optional quality measures (``uncertainty``, ``blur_amount``, ``confidence``), optional packed ``stretch_regions`` for per-region contrast, and ``annotations``. Implementations include complete classes for stars, bodies (including a simulated variant), rings, and Titan, plus a combined model used to merge the nearest visible model at each pixel.

NavTechnique
------------

``NavTechnique`` is the abstract base for navigation algorithms that estimate offsets from models and the observation. Techniques are selected by name and record technique-specific metadata. Current implementations include a joint correlation across models and a manual override technique.

Dataset
-------

``DataSet`` handles access to image files and metadata. ``DataSetPDS3`` provides volume and index-based iteration for archives, while instrument-specific subclasses tailor parsing and volume sets. ``DataSetSim`` supplies images described by JSON files. Some datasets add flags; for example, Cassini ISS adds ``--camera`` (NAC or WAC) and supports a ``botsim`` grouping that pairs NAC/WAC images when available.

Obs and ObsSnapshot
-------------------

``Obs`` is the base class for observations. ``ObsSnapshot`` extends it with backplane handling and accessors, while ``ObsSnapshotInst`` defines the instrument-specific snapshot contract with a ``from_file(...)`` constructor and ``get_public_metadata()``. Instrument classes include ``ObsCassiniISS``, ``ObsVoyagerISS``, ``ObsGalileoSSI``, ``ObsNewHorizonsLORRI``, and ``ObsSim``.

Annotation
----------

The annotation subsystem composes labels and graphical elements into an overlay used by the final PNG. ``Annotations`` aggregates model-provided annotations and renders them with appropriate coloring and contrast stretching, optionally using per-region stretching via ``stretch_regions``.

Extending the System
====================

Adding a New Dataset
--------------------

To add a dataset, create a class in ``nav/dataset/`` that inherits from ``DataSet`` (or from ``DataSetPDS3`` for archives). Implement ``_img_name_valid(...)``, the file-yielding methods, and ``add_selection_arguments(...)`` to expose CLI selection flags. Register the dataset in ``nav/dataset/__init__.py`` so it becomes available to the CLI.

Example:

.. code-block:: python

   from nav.dataset.dataset_pds3 import DataSetPDS3

   class DataSetNewInstrument(DataSetPDS3):
       def __init__(self, *, config=None, logger_name=None):
           super().__init__(config=config, logger_name=logger_name)

       @staticmethod
       def image_name_valid(name):
           # Implement logic to determine if a filename is valid for this instrument
           return name.startswith("NEW") and name.endswith(".IMG")

3. Update the dataset registry in ``nav/dataset/__init__.py``
4. Add the instrument to the command-line parser in ``main/nav_main_offset.py``

Adding a New Instrument
-----------------------

To add an instrument, implement a subclass of ``ObsSnapshotInst`` in ``nav/obs/`` that provides ``from_file(...)`` and any instrument-specific helpers. Update the instrument registry in ``nav/obs/__init__.py`` so datasets can resolve the instrument class.

Example:

.. code-block:: python

   from nav.obs.obs_snapshot_inst import ObsSnapshotInst

   class ObsNewInstrument(ObsSnapshotInst):
       def __init__(self, obs, *, config=None, logger_name=None):
           super().__init__(obs, config=config, logger_name=logger_name)

       @classmethod
       def from_file(cls, path, extfov_margin_vu=None):
           # Implement logic to load an image file and return an ObsSnapshotInst
           ...

Adding a New Navigation Model
------------------------------

To implement a new model type:

1. Create a new class in ``nav/nav_model/`` inheriting from ``NavModel``.
2. Implement ``create_model(...)`` to generate arrays and annotations.
3. Update ``NavMaster.compute_all_models()`` to construct your model.

Example:

.. code-block:: python

   from nav.nav_model.nav_model import NavModel

   class NavModelNewFeature(NavModel):
       def __init__(self, name, obs, *, config=None, logger_name=None):
           super().__init__(name, obs, config=config, logger_name=logger_name)

       def create_model(self, *, always_create_model=False, never_create_model=False, create_annotations=True):
           # Implement logic to compute arrays and metadata
           ...

Adding a New Navigation Technique
---------------------------------

To implement a new navigation algorithm:

1. Create a new class in ``nav/nav_technique/`` inheriting from ``NavTechnique``.
2. Implement the ``navigate`` method with your algorithm.
3. Update ``NavMaster.navigate()`` to construct and record your technique.

Example:

.. code-block:: python

   from nav.nav_technique.nav_technique import NavTechnique

   class NavTechniqueNewMethod(NavTechnique):
       def __init__(self, nav_master, *, config=None, logger_name=None):
           super().__init__(nav_master, config=config, logger_name=logger_name)

       def navigate(self):
           # Implement your navigation algorithm
           # and update nav_master.offset_uv with the result
           ...

Configuration System
====================

RMS-NAV uses a YAML-based configuration system. The default configuration files are located
in the ``nav/config_files/`` directory.

To override configuration settings:

1. Create a custom YAML file with your settings
2. Load it using the ``Config`` class:

   .. code-block:: python

      from nav.config.config import Config

      custom_config = Config('/path/to/custom_config.yaml')

The configuration system uses a hierarchical structure with sections for:

* General settings
* Model-specific settings
* Technique-specific settings
* Instrument-specific settings

Best Practices
==============

Code Style
----------

* Follow PEP 8 for Python code style
* Use type hints for all function parameters and return values
* Document all classes and methods with docstrings
* Use abstract base classes for interface definitions
* Follow the existing logging and error handling patterns

Testing
-------

* Write unit tests for new functionality
* Put tests in the ``tests/`` directory
* Use pytest for running tests
* Ensure backward compatibility with existing functionality

Documentation
-------------

* Update docstrings for all new code
* Keep the class diagram up to date
* Document configuration options
* Add examples for new features


Building the Documentation
==========================

Prerequisites
-------------

1. Install the required Python packages:

   .. code-block:: bash

      pip install -r requirements.txt

Building HTML Documentation
---------------------------

1. Navigate to the docs directory:

   .. code-block:: bash

      cd docs

2. Build the HTML documentation:

   .. code-block:: bash

      make html

3. The built documentation will be available in ``docs/_build/html``. Open ``index.html`` in your browser to view it.

Building Other Formats
----------------------

PDF (requires LaTeX):

.. code-block:: bash

   make latexpdf

Single HTML page:

.. code-block:: bash

   make singlehtml

EPUB:

.. code-block:: bash

   make epub

Working with Mermaid Diagrams
-----------------------------

Mermaid diagrams are rendered using the sphinxcontrib-mermaid extension. To create or modify diagrams:

1. Edit the Mermaid diagram code in the RST files
2. Run ``make html`` to build the documentation
3. Check the rendered diagram in the HTML output

Example Mermaid diagram syntax:

.. code-block:: rst

   .. mermaid::

      classDiagram
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
         }
         NavBase <|-- DataSet

Updating API Documentation
--------------------------

The API documentation is automatically generated from docstrings in the code. To update it:

1. Ensure your code has proper docstrings.
2. Run ``make html`` to rebuild the documentation.

If you add new modules, you may need to update ``api_reference.rst`` to include them.

Troubleshooting
---------------

If you encounter issues with the documentation build:

1. Ensure all required packages are installed
2. Check for syntax errors in RST files
3. Look for error messages in the build output
4. Clear the build directory (``rm -rf _build``) and try again

For Mermaid diagram issues:

1. Validate your Mermaid syntax using the online Mermaid Live Editor: https://mermaid.live/
2. Ensure the sphinxcontrib-mermaid extension is properly installed and configured

Backplanes Architecture
-----------------------

Modules
~~~~~~~

- ``nav/backplanes/backplanes.py``: Orchestrates per-image flow

  - Reads prior nav metadata from ``--nav-results-root``
  - Builds ``ObsSnapshot`` with ``extfov_margin_vu=(0, 0)`` and applies ``OffsetFOV``
  - Computes bodies and rings backplanes
  - Merges sources per-pixel by distance
  - Writes FITS + PDS4 via writer

- ``nav/backplanes/backplanes_bodies.py``: Body backplanes

  - For each body in FOV, builds a clipped meshgrid (no oversampling) and evaluates OOPS backplanes
  - Embeds arrays/masks into full-size frames
  - Simulation: synthesizes fake arrays but only within the simulated body mask derived from:
    - ``snapshot.sim_body_mask_map[body_name]`` if present, else
    - ``snapshot.sim_body_index_map`` matched against ``snapshot.sim_body_order_near_to_far``

- ``nav/backplanes/backplanes_rings.py``: Ring backplanes

  - Uses full-frame ``snapshot.bp``, evaluates configured ring backplanes
  - Produces per-pixel ``distance`` used for merge ordering

- ``nav/backplanes/merge.py``: Per-pixel distance-ordered merge

  - Bodies: body-level scalar distances, broadcast within each body’s mask
  - Rings: per-pixel distances
  - BODY_ID_MAP is filled with NAIF IDs; simulation uses deterministic fake IDs when unknown

- ``nav/backplanes/writer.py``: Output writer

  - Writes BODY_ID_MAP as the first image HDU
  - Excludes any backplane that is entirely zeros from FITS and label
  - Uses ``nav/backplanes/templates/backplanes.lblx`` via ``PdsTemplate``

Snapshot Helpers
~~~~~~~~~~~~~~~~

Added methods to ``ObsSnapshot``:

- ``inventory_body_in_fov(inv: dict) -> bool``
- ``inventory_body_in_extfov(inv: dict) -> bool``
- ``clip_rect_fov(u_min, u_max, v_min, v_max) -> tuple[int, int, int, int]``
- ``clip_rect_extfov(u_min, u_max, v_min, v_max) -> tuple[int, int, int, int]``

These unify bounding-box intersection and clipping logic and are used by backplanes and navigation code.

CLI and Roots
~~~~~~~~~~~~~

Backplanes drivers accept two roots:

- ``--nav-results-root``: prior nav results (metadata); used to read ``*_metadata.json``
- ``--backplane-results-root``: destination for new backplane outputs

Offset drivers use:

- ``--nav-results-root`` for navigation outputs (offsets, overlays, etc.)

Configuration
~~~~~~~~~~~~~

``nav/config_files/config_90_backplanes.yaml`` defines:

- ``backplanes.bodies`` and ``backplanes.rings`` (name, method, units)
- ``backplanes.target_lids`` (optional) to populate PDS4 target references

Testing
~~~~~~~

There is a smoke test under ``experiments/backplanes/`` using a simulated JSON to ensure backplane generation runs end-to-end. In simulation, per-body masks are respected for fake backplanes to avoid rectangular artifacts.
