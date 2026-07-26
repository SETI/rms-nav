=============
Configuration
=============

SpinDoctor uses a hierarchical YAML-based configuration system that allows you to
customize behavior without modifying the source code. Understanding how
configuration files are loaded and how to override settings is important for
effective use of the system.

Configuration Loading Order
============================

SpinDoctor ships with a complete set of built-in defaults, so the system works out
of the box with no configuration on your part. You customize behavior by supplying
your own settings on top of those defaults. Settings are resolved as follows:

1. **Built-in defaults**: SpinDoctor bundles a stack of default configuration files
   that give every setting a sensible value. You do not edit these. (Developers
   who need to know exactly which files ship and what each one holds should see
   :doc:`/dev_guide/dev_guide_config_and_static_data`.)

2. **Exactly one of the following** is loaded on top of the built-in defaults:

   * **Command-line configuration files**: If one or more files are specified
     with the ``--config-file`` option, they are loaded in the order given,
     each overriding the built-in defaults (and, for the same key, any file
     loaded before it).

   * **User default configuration**: Only when no ``--config-file`` option is
     given, a file named ``nav_default_config.yaml`` in the current working
     directory is loaded if it exists. Use it to set personal defaults for
     runs where you do not pass ``--config-file``. Note that passing
     ``--config-file`` replaces this file entirely rather than adding to it;
     to keep your personal defaults in such a run, list
     ``nav_default_config.yaml`` explicitly as the first ``--config-file``
     argument.

3. **Command-line option overrides**: A handful of CLI flags (described under
   `Command-Line Option Overrides`_ below) override the matching configuration
   key directly and take precedence over everything above.

You only ever need to specify the settings you want to change; everything else
falls through to the built-in defaults.

Configuration File Structure
============================

Configuration files use YAML format and are organized into sections:

.. code-block:: yaml

   environment:
     nav_results_root: /path/to/results
     pds3_holdings_root: /path/to/pds3

   general:
     log_level_model_stars: DEBUG
     log_level_model_rings: DEBUG

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

All logging levels live in the ``general`` configuration section. Each key accepts
a standard log-level string: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, or
``CRITICAL``. Set them in your ``nav_default_config.yaml`` or a ``--config-file``
to override the built-in defaults shown below.

**Main logger** (``sd_offset`` -- top-level program events):

* ``general.log_level_main_console`` (default: ``INFO``): Level for output written
  to stdout while the program runs.
* ``general.log_level_main_file`` (default: ``INFO``): Level for the timestamped
  logfile written to ``$NAV_RESULTS_ROOT/logs/sd_offset/``.

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
* ``general.log_level_model_titan`` (default: ``INFO``): Logging level for the
  Titan navigation model.

**Navigation technique loggers**:

The autonomous-navigation pipeline routes every per-image technique line
through ``IMAGE_LOGGER``; there is no per-technique log-level knob.  Each
technique opens a ``with self.logger.open(f'TECHNIQUE: {self.name}')``
section so the per-image log file delimits each technique's contribution.

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

3. The system will automatically load this file if it exists, provided you do
   not pass ``--config-file`` (which replaces it; see below)

Using Command-Line Configuration Overrides
===========================================

You can override configuration on a per-run basis using ``--config-file``:

.. code-block:: bash

   sd_offset coiss N1234567890 --config-file /path/to/special_config.yaml

You can specify multiple configuration files, and they will be loaded in order:

.. code-block:: bash

   sd_offset coiss N1234567890 \
     --config-file base_overrides.yaml \
     --config-file run_specific.yaml

When any ``--config-file`` is given, ``nav_default_config.yaml`` is not loaded
automatically. To keep your personal defaults for that run, pass the file
explicitly as the first ``--config-file`` argument:

.. code-block:: bash

   sd_offset coiss N1234567890 \
     --config-file nav_default_config.yaml \
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
  is a comma-separated list of glob patterns matched against the registered
  technique names: ``BodyBlobNav``, ``BodyDiscCorrelateNav``, ``BodyLimbNav``,
  ``BodyTerminatorNav``, ``RingAnnulusNav``, ``RingEdgeNav``,
  ``StarFieldFromCatalogNav``, ``StarRefineNav``, ``StarUniqueMatchNav``, and
  ``TitanHazeNav``.
  Shell-glob wildcards are allowed (``Star*`` selects the three star
  techniques) and a leading ``!`` excludes matching names (``!Ring*`` runs
  everything except the ring techniques). Interactive manual navigation is
  not selected here; it is invoked with the separate ``--manual`` flag, which
  opens the manual-navigation dialog instead of running the autonomous
  pipeline.

Logging Options
---------------

All four logging-level options override the corresponding ``general.*`` config
key for that run. Each accepts a standard log-level string: ``DEBUG``, ``INFO``,
``WARNING``, ``ERROR``, or ``CRITICAL``.

* ``--log-level-main-console LEVEL``: Override ``general.log_level_main_console``
  -- the level at which the main logger writes to stdout.

* ``--log-level-main-file LEVEL``: Override ``general.log_level_main_file``
  -- the level at which the main logger writes to its logfile under
  ``$NAV_RESULTS_ROOT/logs/sd_offset/``.

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

The following example demonstrates how different configuration methods interact.
Suppose the built-in defaults set
``offset.correlation_fft_upsample_factor: 128``, your
``nav_default_config.yaml`` sets it to ``256``, and ``custom.yaml`` sets it to
``512``:

1. Running ``sd_offset`` with no ``--config-file`` loads
   ``nav_default_config.yaml``, so the final value is ``256``.

2. Running ``sd_offset --config-file custom.yaml`` does not load
   ``nav_default_config.yaml`` at all, so the final value is ``512`` -- and
   every other setting in ``nav_default_config.yaml`` also reverts to its
   built-in default.

3. To combine the two, list both files explicitly:
   ``sd_offset --config-file nav_default_config.yaml --config-file custom.yaml``
   loads them in order, so the final value is ``512`` while the rest of your
   personal defaults still apply.

If you also specify ``--nav-models stars,rings`` on the command line, this
overrides any model selection from configuration files, regardless of what's in
the configuration.
