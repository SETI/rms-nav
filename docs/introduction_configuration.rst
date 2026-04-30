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

   * ``config_010_general.yaml``: General settings including all logging levels
   * ``config_020_offset.yaml``: Offset-finding and star refinement parameters
   * ``config_030_stars.yaml``: Star-model and ring-occlusion parameters
   * ``config_040_bodies.yaml``: Body (planet/moon) rendering parameters
   * ``config_050_rings.yaml``: Ring model parameters
   * ``config_060_titan.yaml``: Titan-specific navigation parameters
   * ``config_070_bootstrap.yaml``: Bootstrap navigation parameters (angles in degrees)
   * ``config_100_satellites.yaml``: Satellite definitions for each planet
   * ``config_220_body_shape.yaml``: Per-body shape table (radii, ellipsoid
     residual, crater scale, albedo) consumed by the body NavModel and feature
     extractors
   * ``config_300_jupiter_rings.yaml``: Jupiter ring system parameters
   * ``config_310_saturn_rings.yaml``: Saturn ring system parameters
   * ``config_320_uranus_rings.yaml``: Uranus ring system parameters
   * ``config_330_neptune_rings.yaml``: Neptune ring system parameters
   * ``config_400_inst_coiss.yaml``: Cassini ISS instrument-specific settings
   * ``config_410_inst_gossi.yaml``: Galileo SSI instrument-specific settings
   * ``config_420_inst_nhlorri.yaml``: New Horizons LORRI instrument-specific settings
   * ``config_430_inst_vgiss.yaml``: Voyager ISS instrument-specific settings
   * ``config_440_sim.yaml``: Simulated image settings
   * ``config_510_techniques.yaml``: Per-NavTechnique confidence-formula
     coefficients and runtime tunables (spurious-detection thresholds,
     at-edge tolerances, minimum arc lengths) plus the planet-specific
     ``feature_emission.ring_annulus`` block that decides RING_EDGE vs
     RING_ANNULUS feature emission
   * ``config_900_backplanes.yaml``: Backplane generation settings
   * ``config_950_pds4.yaml``: PDS4 metadata and export settings for generated
     products, overrides for PDS4 label templates and mapping of internal fields
     to PDS4 keys

   The 3-digit numeric prefix is the lexicographic merge order.  Files in the
   ``0xx`` range (000–099) are global / model-shared settings, ``1xx``
   (100–199) are catalogues, ``2xx`` (200–299) are per-target tables (body
   shape), ``3xx`` (300–399) are per-planet ring catalogues, ``4xx``
   (400–499) are per-instrument camera blocks, ``5xx`` (500–599) are
   per-technique tunables, and ``9xx`` (900–999) are downstream-product
   settings.

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
     log_level_model_rings: DEBUG
     log_level_nav_correlate_all: DEBUG

   offset:
     correlation_fft_upsample_factor: 128
     star_refinement_enabled: true

   bodies:
     min_bounding_box_area: 9
     oversample_maximum: 2

Each section can contain multiple settings. When multiple configuration files
define the same setting, the value from the last file loaded takes precedence.

Logging Configuration
---------------------

All logging levels are set in the ``general`` section of ``config_010_general.yaml``.
Each key accepts a standard log-level string: ``DEBUG``, ``INFO``, ``WARNING``,
``ERROR``, or ``CRITICAL``.

**Main logger** (``nav_offset`` -- top-level program events):

* ``general.log_level_main_console`` (default: ``INFO``): Level for output written
  to stdout while the program runs.
* ``general.log_level_main_file`` (default: ``INFO``): Level for the timestamped
  logfile written to ``$NAV_RESULTS_ROOT/logs/nav_offset/``.

**Image logger** (``nav_image`` -- per-image processing events, active only while
an image is being processed):

* ``general.log_level_image_console`` (default: ``INFO``): Level for output written
  to stdout during image processing.
* ``general.log_level_image_file`` (default: ``INFO``): Level for the per-image
  logfile written to ``$NAV_RESULTS_ROOT/logs/{results_path_stub}.log``.

**Navigation model loggers**:

* ``general.log_level_model_bodies`` (default: ``INFO``): Logging level for the
  body (planet and moon) navigation model.
* ``general.log_level_model_stars`` (default: ``INFO``): Logging level for the
  star navigation model.
* ``general.log_level_model_rings`` (default: ``INFO``): Logging level for the
  ring navigation model.

**Navigation technique loggers**:

The autonomous-navigation pipeline routes every per-image technique line
through ``IMAGE_LOGGER``; there is no per-technique log-level knob.  Each
technique opens a ``with self.logger.open(f'TECHNIQUE: {self.name}')``
section so the per-image log file delimits each technique's contribution.
The legacy ``general.log_level_nav_correlate_all`` knob is retained for
backwards compatibility with any user config files that still set it but
the autonomous techniques (``BodyDiscCorrelateNav``, ``BodyBlobNav``,
``BodyLimbNav``, ``BodyTerminatorNav``, ``RingEdgeNav``,
``RingAnnulusNav``, ``StarUniqueMatchNav``, ``StarRefineNav``,
``StarFieldFromCatalogNav``) do not consult it.

**Annotation**:

* ``general.log_level_annotate`` (default: ``ERROR``): Logging level for the
  image annotation step.

Example -- enable verbose output for star and ring models while keeping other
components at the default level:

.. code-block:: yaml

   general:
     log_level_model_stars: DEBUG
     log_level_model_rings: DEBUG

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

Logging Options
---------------

All four logging-level options override the corresponding ``general.*`` config
key for that run. Each accepts a standard log-level string: ``DEBUG``, ``INFO``,
``WARNING``, ``ERROR``, or ``CRITICAL``.

* ``--log-level-main-console LEVEL``: Override ``general.log_level_main_console``
  -- the level at which the main logger writes to stdout.

* ``--log-level-main-file LEVEL``: Override ``general.log_level_main_file``
  -- the level at which the main logger writes to its logfile under
  ``$NAV_RESULTS_ROOT/logs/nav_offset/``.

* ``--log-level-image-console LEVEL``: Override ``general.log_level_image_console``
  -- the level at which the image logger writes to stdout during image processing.

* ``--log-level-image-file LEVEL``: Override ``general.log_level_image_file``
  -- the level at which the image logger writes to the per-image logfile under
  ``$NAV_RESULTS_ROOT/logs/{results_path_stub}.log``.

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
