====================
Extending the System
====================

Adding a new dataset
--------------------

To add a dataset, create a class in ``src/nav/dataset/`` that inherits from
``DataSet`` (or from ``DataSetPDS3`` for archives).  Implement
``_img_name_valid(...)``, the file-yielding methods, and
``add_selection_arguments(...)`` to expose CLI selection flags.  Register the
dataset in ``src/nav/dataset/__init__.py`` so it becomes available to the CLI.

Example:

.. code-block:: python

   from nav.dataset.dataset_pds3 import DataSetPDS3

   class DataSetNewInstrument(DataSetPDS3):
       def __init__(self, *, config=None):
           super().__init__(config=config)

       @staticmethod
       def _img_name_valid(name: str) -> bool:
           return name.startswith("NEW") and name.endswith(".IMG")

The dataset will automatically be available to the CLI once registered.

Implementing PDS4 bundle generation methods
-------------------------------------------

To support PDS4 bundle generation, datasets must implement the abstract
methods on :class:`~nav.dataset.dataset.DataSet`:

* ``pds4_bundle_template_dir()``: returns the absolute path to the template
  directory for PDS4 label generation.  If a relative name is provided in
  config, it should be resolved relative to ``src/pds4/templates/``.
* ``pds4_bundle_name()``: returns the bundle name (e.g.
  ``"<instrument_name>_backplanes_rsfrench2027"``).
* ``pds4_bundle_path_for_image()``: maps an image name to a bundle directory
  path (e.g. ``"1234xxxxxx/123456xxxx"``).  This is a static method.
* ``pds4_path_stub()``: returns the full path stub including directory and
  filename prefix (e.g. ``"1234xxxxxx/123456xxxx/1234567890w"``).
* ``pds4_template_variables()``: returns a dictionary mapping template
  variable names to values for PDS4 label generation.  This should extract
  values from navigation metadata, backplane metadata, and PDS3 index rows
  (if available).
* ``pds4_image_name_to_data_lid()``: converts an image name to a data
  product LID.  Returns a full LID string (e.g.
  ``"urn:nasa:pds:<bundle_name>:data:<image_name>"``).
* ``pds4_image_name_to_data_lidvid()``: converts an image name to a data
  product LIDVID.
* ``pds4_image_name_to_browse_lid()``: converts an image name to a browse
  product LID.
* ``pds4_image_name_to_browse_lidvid()``: converts an image name to a browse
  product LIDVID.

For datasets that do not support PDS4 bundle generation, these methods
should raise ``NotImplementedError``.  See
:class:`~nav.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISS` for a
complete implementation example.

Adding a new instrument
-----------------------

To add an instrument, implement a subclass of ``ObsSnapshotInst`` in
``src/nav/obs/`` that provides ``from_file(...)`` and any instrument-specific
helpers.  Update the instrument registry in ``src/nav/obs/__init__.py`` so
datasets can resolve the instrument class.

.. code-block:: python

   from nav.obs.obs_snapshot_inst import ObsSnapshotInst
   from nav.support.types import PathLike

   class ObsNewInstrument(ObsSnapshotInst):
       def __init__(self, obs, *, config=None, **kwargs):
           super().__init__(obs, config=config, **kwargs)

       @classmethod
       def from_file(
           cls,
           path: PathLike,
           *,
           config=None,
           extfov_margin_vu=None,
           **kwargs,
       ):
           ...

Adding a new NavModel
---------------------

NavModel subclasses self-register via ``__init_subclass__``.  To add a new
predicted-scene generator:

1. Create a class in ``src/nav/nav_model/`` inheriting from
   :class:`~nav.nav_model.nav_model.NavModel` (or from one of the abstract
   bases such as
   :class:`~nav.nav_model.nav_model_body_base.NavModelBodyBase`).
2. Implement ``create_model``, ``to_features(context)``, and
   ``to_annotations(context)``.
3. Override ``instances_for_obs(cls, obs)`` if the model auto-instantiates
   per-observation (one instance per body in FOV, one per planet with
   visible rings, one stars model).  Subclasses that require operator
   parameters (simulated models populated from GUI JSON) inherit the empty
   default; the caller constructs them directly.

.. code-block:: python

   from nav.nav_model.nav_model import NavModel

   class NavModelNewFeature(NavModel):
       def __init__(self, name, obs, *, config=None):
           super().__init__(name, obs, config=config)

       def create_model(self) -> None:
           ...

       def to_features(self, context) -> list[NavFeature]:
           ...

       def to_annotations(self, context) -> Annotations:
           ...

Adding a new NavTechnique
-------------------------

NavTechnique subclasses also self-register via ``__init_subclass__``.

1. Create a class in ``src/nav/nav_technique/`` inheriting from
   :class:`~nav.nav_technique.nav_technique.NavTechnique`.
2. Set the class attributes ``name``, ``accepts_feature_types``, and (if
   relevant) ``requires_prior``.
3. Implement ``is_feasible(features)`` (must read feature metadata only, no
   pixels) and ``navigate(features, context)``.

.. code-block:: python

   from nav.feature.feature_type import NavFeatureType
   from nav.nav_technique.feasibility import NavFeasibilityReport
   from nav.nav_technique.nav_technique import NavTechnique
   from nav.nav_technique.technique_result import NavTechniqueResult

   class NavTechniqueNewMethod(NavTechnique):
       name = 'NavTechniqueNewMethod'
       accepts_feature_types = frozenset({NavFeatureType.STAR})

       def is_feasible(self, features):
           return NavFeasibilityReport(feasible=len(features) >= 1, reason='ok')

       def navigate(self, features, context) -> NavTechniqueResult:
           ...

The technique becomes visible to
:class:`~nav.nav_orchestrator.orchestrator.NavOrchestrator` as soon as the
module is imported.  Glob filters on the orchestrator (``only_techniques``)
let you exclude or single out the new technique without modifying the
registry.

Adding to the image library via the manual-nav dialog
-----------------------------------------------------

The interactive ``ManualNavDialog`` (built by
:class:`~nav.nav_technique.nav_technique_manual.NavTechniqueManual`)
exposes a **Save as Library Entry...** button alongside the OK / Cancel
controls.  Clicking it captures the current dv/du and writes a sidecar
seeded with the auto-fillable fields plus ``TODO_REPLACE_*``
placeholders for the operator-curated ones (scene_tags,
``primary_technique``, notes).  The YAML helper lives in
:mod:`nav.ui.library_entry`.

Operator workflow:

1. Open the manual-nav dialog on the candidate image and pick the
   offset by hand (or accept **Auto**).
2. Click **Save as Library Entry...**.  The save-file dialog suggests
   ``<image_id>.yaml`` as the filename.  Point it at the appropriate
   scene-class directory under
   ``tests/integration/image_library/images/<class>/``.  A companion
   ``<image_id>.png`` capturing the red-image / green-model overlay
   at the chosen ``(dv, du)`` is dropped next to the YAML so future
   reviewers can see the scene at a glance; it is not consumed by any
   test.
3. Open the saved YAML and fill in every ``TODO_REPLACE_*`` value.  An
   unedited template trips
   :func:`tests.integration.sidecar.load_sidecar` so CI fails loudly if
   you forget.
4. Run the structural-invariants test
   (``pytest tests/integration/test_image_library.py``); the per-image
   regression test (``test_autonomous_nav.py``) follows once
   ``PDS3_HOLDINGS_DIR`` is set.

See :doc:`dev_guide_image_library` for the sidecar schema and
the deeper rationale behind the curation policy.
