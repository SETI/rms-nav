===========================================
Developer Guide: Extending the System
===========================================

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

The dataset will automatically be available to the CLI once registered.

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
           # Set self._offset, self._uncertainty, and self._confidence
           # Record metadata in self._metadata
           ...
