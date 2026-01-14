=============
Configuration
=============

RMS-NAV uses a hierarchical YAML-based configuration system that allows you to
customize behavior without modifying the source code. Understanding how
configuration files are loaded and how to override settings is important for
effective use of the system.

Configuration Loading Order
============================

The configuration system loads settings in the following order, with later files
overriding earlier ones:

1. **Standard Configuration Files**: All YAML files in the
   ``src/nav/config_files/`` directory are loaded in alphabetical order. These
   files provide default settings for:

   * ``config_01_settings.yaml``: General settings, offset parameters, and body
     defaults
   * ``config_10_satellites.yaml``: Satellite definitions for planets
   * ``config_20_saturn_rings.yaml``: Saturn ring system parameters
   * ``config_30_coiss.yaml``: Cassini ISS instrument-specific settings
   * ``config_31_gossi.yaml``: Galileo SSI instrument-specific settings
   * ``config_32_nhlorri.yaml``: New Horizons LORRI instrument-specific settings
   * ``config_33_vgiss.yaml``: Voyager ISS instrument-specific settings
   * ``config_40_sim.yaml``: Simulated image settings
   * ``config_90_backplanes.yaml``: Backplane generation settings

2. **User Default Configuration**: If present, the file
   ``nav_default_config.yaml`` in the current working directory is loaded. This
   allows you to set personal defaults that apply to all runs.

3. **Command-Line Configuration Files**: Any files specified with the
   ``--config-file`` option are loaded in the order specified. These provide
   the highest priority and can override any previous settings.

Configuration File Structure
============================

Configuration files use YAML format and are organized into sections:

.. code-block:: yaml

   environment:
     nav_results_root: /path/to/results
     pds3_holdings_root: /path/to/pds3

   general:
     log_level_nav_correlate_all: DEBUG

   offset:
     correlation_fft_upsample_factor: 128
     star_refinement_enabled: true

   bodies:
     min_bounding_box_area: 9
     oversample_maximum: 2

Each section can contain multiple settings. When multiple configuration files
define the same setting, the value from the last file loaded takes precedence.

Creating a User Configuration File
===================================

To create your own default configuration:

1. Create a file named ``nav_default_config.yaml`` in your working directory
2. Add only the settings you want to override:

   .. code-block:: yaml

      environment:
        nav_results_root: /my/custom/results/path

      offset:
        correlation_fft_upsample_factor: 256

3. The system will automatically load this file if it exists

Using Command-Line Configuration Overrides
===========================================

You can override configuration on a per-run basis using ``--config-file``:

.. code-block:: bash

   nav_offset coiss N1234567890 --config-file /path/to/special_config.yaml

You can specify multiple configuration files, and they will be loaded in order:

.. code-block:: bash

   nav_offset coiss N1234567890 \
     --config-file base_overrides.yaml \
     --config-file run_specific.yaml

Command-Line Option Overrides
==============================

In addition to configuration files, certain command-line options can override
configuration settings directly. These options take precedence over all
configuration file settings:

Environment Options
-------------------

* ``--pds3-holdings-root PATH``: Overrides the ``PDS3_HOLDINGS_DIR``
  environment variable and any ``environment.pds3_holdings_root`` configuration
  setting. This specifies the root directory or URL for PDS3 holdings.

* ``--nav-results-root PATH``: Overrides the ``NAV_RESULTS_ROOT`` environment
  variable and any ``environment.nav_results_root`` configuration setting. This
  specifies the root directory or URL where navigation results will be written.

Navigation Options
------------------

* ``--nav-models LIST``: Overrides any default model selection. This is a
  comma-separated list of model names or patterns to enable. Valid entries
  include ``stars``, ``rings``, ``titan``, and body-specific entries of the
  form ``body:NAME`` (glob patterns are allowed).

* ``--nav-techniques LIST``: Overrides any default technique selection. This
  is a comma-separated list of navigation techniques to apply. Valid entries
  include ``correlate_all`` and ``manual``.

These command-line options provide the highest priority override mechanism,
taking precedence over all configuration files, including those specified with
``--config-file``.

Example: Combining Configuration Methods
========================================

The following example demonstrates how different configuration methods interact:

1. Default configuration files in ``src/nav/config_files/`` set
   ``offset.correlation_fft_upsample_factor: 128``

2. User's ``nav_default_config.yaml`` overrides it to ``256``

3. Command-line ``--config-file custom.yaml`` overrides it to ``512``

4. The final value used is ``512``

If you also specify ``--nav-models stars,rings`` on the command line, this
overrides any model selection from configuration files, regardless of what's in
the configuration.
