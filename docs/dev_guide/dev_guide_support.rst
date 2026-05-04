===============
Support Modules
===============

The navigation, reprojection, backplane, and PDS4 pipelines share a small set of
cross-cutting support subsystems — the YAML configuration loader plus the static-data
catalogues it parses, and the project-wide logging contract.  This chapter documents
those subsystems in one place so consumers do not have to chase them through the
per-pipeline pages.

.. toctree::
   :maxdepth: 4

   dev_guide_config_and_static_data
   dev_guide_logging
