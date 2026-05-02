====================
Configuration System
====================

RMS-NAV uses a YAML-based configuration system. The default configuration
files ship under ``src/nav/config_files/`` and are loaded in alphabetical
order; see :doc:`introduction_configuration` for the list of files and for
the full precedence between default files, ``nav_default_config.yaml``,
``--config-file`` overrides, and command-line flags.

Using :class:`~nav.config.config.Config` directly
-------------------------------------------------

The :class:`~nav.config.config.Config` class takes no constructor arguments
and loads the standard files lazily. Additional YAML files can be merged in
via :meth:`~nav.config.config.Config.update_config`:

.. code-block:: python

   from nav.config import Config

   cfg = Config()                       # lazy; reads the standard files on first access
   cfg.update_config('custom.yaml')     # merge in an override file
   print(cfg.offset.correlation_fft_upsample_factor)

A module-level ``DEFAULT_CONFIG`` singleton is also available from
``nav.config`` and is used by most subsystems when no ``config=`` keyword is
supplied.

Configuration sections
----------------------

Configuration YAML is organized into the following top-level sections, each
accessible as an :class:`~nav.support.attrdict.AttrDict` property on the
``Config`` object:

- ``environment`` -- paths (``pds3_holdings_root``, ``nav_results_root``,
  ``backplane_results_root``, ``bundle_results_root``).
- ``general`` -- logging levels and other global settings.
- ``offset`` -- correlation and star-refinement parameters.
- ``stars`` -- star-model and ring-occlusion parameters.
- ``bodies`` -- body rendering parameters.
- ``rings`` -- ring-model parameters (planet shadow removal, fade widths,
  per-planet ``ring_features`` -- see
  :doc:`developer_guide_navigation_models_rings`).
- ``titan`` -- Titan-specific parameters.
- ``bootstrap`` -- bootstrap navigation parameters.
- ``backplanes`` -- the list of body and ring backplanes to generate.
- ``pds4`` -- per-dataset PDS4 template directories and bundle names.

Planetary ring YAML (planet sections, features, fade parameters, and
validation rules) is specified in
:doc:`developer_guide_navigation_models_rings`.
