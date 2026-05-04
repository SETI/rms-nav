==========
Navigation
==========

Navigation is the core competence of RMS-NAV: ingesting one image, predicting what
the scene should look like from SPICE, and reporting the offset that brings the
prediction into alignment with the data.  This chapter is the top-level entry into
every part of the navigation pipeline — from the per-image observation wrapper, to the
predicted-scene NavModels, to the per-feature NavTechniques, to the orchestrator that
runs them, to the per-image PNG overlay and the operator-curated regression library
that anchors confidence calibration.

Each sub-chapter below stands on its own and links to the others.  Start with the
**Navigation Overview** for the architectural sketch and the per-image data flow,
then read **Class Hierarchy** for the inheritance graph and the shared base classes
the rest of the chapter assumes.

.. toctree::
   :maxdepth: 4
   :caption: Architecture

   dev_guide_navigation_overview
   dev_guide_class_hierarchy

.. toctree::
   :maxdepth: 4
   :caption: Pipeline subsystems

   dev_guide_observations
   dev_guide_navigation_models
   dev_guide_techniques
   dev_guide_orchestrator
   dev_guide_uncertainty
   dev_guide_annotations
   dev_guide_rotation

.. toctree::
   :maxdepth: 4
   :caption: Calibration and regression

   dev_guide_image_library
