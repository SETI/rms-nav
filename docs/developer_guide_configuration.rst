=====================================
Developer Guide: Configuration System
=====================================

Configuration System
====================

RMS-NAV uses a YAML-based configuration system. The default configuration files are located
in the ``src/nav/config_files/`` directory.

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
