=====================================
Developer Guide: Extending the System
=====================================

Extending the System
====================

Adding a New Dataset
--------------------

To add a dataset, create a class in ``src/nav/dataset/`` that inherits from ``DataSet`` (or from ``DataSetPDS3`` for archives). Implement ``_img_name_valid(...)``, the file-yielding methods, and ``add_selection_arguments(...)`` to expose CLI selection flags. Register the dataset in ``src/nav/dataset/__init__.py`` so it becomes available to the CLI.

Example:

.. code-block:: python

   from nav.dataset.dataset_pds3 import DataSetPDS3

   class DataSetNewInstrument(DataSetPDS3):
       def __init__(self, *, config=None):
           super().__init__(config=config)

       @staticmethod
       def _img_name_valid(name: str) -> bool:
           # Implement logic to determine if a filename is valid for this instrument
           return name.startswith("NEW") and name.endswith(".IMG")

3. Update the dataset registry in ``src/nav/dataset/__init__.py``

The dataset will automatically be available to the CLI once registered.

Implementing PDS4 Bundle Generation Methods
---------------------------------------------

To support PDS4 bundle generation, datasets must implement the following abstract methods
from :class:`nav.dataset.dataset.DataSet`:

* ``pds4_bundle_template_dir()``: Returns the absolute path to the template directory
  for PDS4 label generation. If a relative name is provided in config, it should be
  resolved relative to ``src/pds4/templates/``.

* ``pds4_bundle_name()``: Returns the bundle name (e.g., ``"instrument_backplanes_rsfrench2027"``).

* ``pds4_bundle_path_for_image()``: Maps an image name to a bundle directory path
  (e.g., ``"1234xxxxxx/123456xxxx"``). This is a static method.

* ``pds4_path_stub()``: Returns the full path stub including directory and filename prefix
  (e.g., ``"1234xxxxxx/123456xxxx/1234567890w"``).

* ``pds4_template_variables()``: Returns a dictionary mapping template variable names to
  values for PDS4 label generation. This should extract values from navigation metadata,
  backplane metadata, and PDS3 index rows (if available).

For datasets that do not support PDS4 bundle generation, these methods should raise
``NotImplementedError``. See :class:`nav.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISS`
for a complete implementation example.

Adding a New Instrument
-----------------------

To add an instrument, implement a subclass of ``ObsSnapshotInst`` in ``src/nav/obs/`` that provides ``from_file(...)`` and any instrument-specific helpers. Update the instrument registry in ``src/nav/obs/__init__.py`` so datasets can resolve the instrument class.

Example:

.. code-block:: python

   from nav.obs.obs_snapshot_inst import ObsSnapshotInst
   from nav.support.types import PathLike

   class ObsNewInstrument(ObsSnapshotInst):
       def __init__(self, obs, *, config=None, **kwargs):
           super().__init__(obs, config=config, **kwargs)

       @classmethod
       def from_file(cls,
                     path: PathLike,
                     *,
                     config=None,
                     extfov_margin_vu=None,
                     **kwargs):
           # Implement logic to load an image file and return an ObsSnapshotInst
           ...

Adding a New Navigation Model
------------------------------

To implement a new model type:

1. Create a new class in ``src/nav/nav_model/`` inheriting from ``NavModel``.
2. Implement ``create_model(...)`` to generate arrays and annotations.
3. Update ``NavMaster.compute_all_models()`` to construct your model.

Example:

.. code-block:: python

   from nav.nav_model.nav_model import NavModel

   class NavModelNewFeature(NavModel):
       def __init__(self, name, obs, *, config=None):
           super().__init__(name, obs, config=config)

       def create_model(self, *, always_create_model=False, never_create_model=False, create_annotations=True):
           # Implement logic to compute arrays and metadata
           ...

Adding a New Navigation Technique
---------------------------------

To implement a new navigation algorithm:

1. Create a new class in ``src/nav/nav_technique/`` inheriting from ``NavTechnique``.
2. Implement the ``navigate`` method with your algorithm.
3. Update ``NavMaster.navigate()`` to construct and record your technique.

Example:

.. code-block:: python

   from nav.nav_technique.nav_technique import NavTechnique

   class NavTechniqueNewMethod(NavTechnique):
       def __init__(self, nav_master, *, config=None):
           super().__init__(nav_master, config=config)

       def navigate(self):
           # Implement your navigation algorithm
           # Set self._offset, self._uncertainty, and self._confidence
           # Record metadata in self._metadata
           ...
