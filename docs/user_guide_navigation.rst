=====================
Navigation User Guide
=====================

Introduction
============

RMS-NAV is a spacecraft image navigation system designed to analyze images from various space missions and determine precise positional offsets. This guide explains how to use the primary command-line interface in ``main/nav_offset.py`` to navigate images and generate results, and how to invoke the cloud-tasks variant for queue-driven processing.

Purpose of the System
---------------------

The primary purpose of RMS-NAV is to determine the precise pointing of spacecraft instruments by comparing the observed images with theoretical models of what should appear in the field of view. This process, known as "navigation," is crucial for:

1. Validating and correcting spacecraft pointing information
2. Ensuring accurate scientific interpretations of the imagery
3. Creating properly annotated and labeled images for analysis
4. Supporting mission planning and operations

The system works by:

1. Reading spacecraft imagery and metadata
2. Generating theoretical models of stars, planets, moons, and rings
3. Correlating the observed features with the theoretical models
4. Calculating the offset between the expected and actual pointing
5. Producing annotated images and data files with the results

Supported Missions
------------------

RMS-NAV currently supports multiple instruments, organized by dataset names you will pass on the command line. Dataset names are case-insensitive and map to instrument-specific handlers. The complete set is:

* ``coiss`` and ``coiss_pds3`` — Cassini Imaging Science Subsystem
* ``gossi`` and ``gossi_pds3`` — Galileo Solid State Imager
* ``nhlorri`` and ``nhlorri_pds3`` — New Horizons Long Range Reconnaissance Imager
* ``vgiss`` and ``vgiss_pds3`` — Voyager Imaging Science Subsystem

Installation and Setup
======================

Prerequisites
-------------

Before using RMS-NAV, you need to have:

* Python 3.10 or higher
* SPICE toolkit and kernels for the relevant missions
* Dependencies installed from ``requirements.txt``

Environment Setup
-----------------

1. Clone the repository and navigate to the project directory
2. Create and activate a virtual environment (recommended)
3. Install the required packages:

   .. code-block:: bash

      pip install -r requirements.txt

4. Set up SPICE kernels:

   * Download the required SPICE kernels for your mission
   * Set the ``SPICE_PATH`` environment variable:

   .. code-block:: bash

      export SPICE_PATH=/path/to/your/spice/kernels

5. Set up PDS3 data access:

   * For PDS3 formatted datasets (most missions), set the ``PDS3_HOLDINGS_DIR`` environment variable:

   .. code-block:: bash

      export PDS3_HOLDINGS_DIR=/path/to/your/pds3/data

   The PDS3 data should be organized in a standard structure:

   .. code-block:: text

      $PDS3_HOLDINGS_DIR/
      ├── volumes/
      │   └── [volume_set]/
      │       └── [volume]/
      │           └── [data directories]/
      └── metadata/
          └── [volume_set]/
              └── [volume]/
                  ├── [volume]_index.lbl
                  └── [volume]_index.tab

Configuration System
====================

RMS-NAV uses a hierarchical YAML-based configuration system that allows you to customize behavior without modifying the source code. Understanding how configuration files are loaded is important for effective use of the system.

Configuration Loading Order
----------------------------

The configuration system loads settings in the following order, with later files overriding earlier ones:

1. **Standard Configuration Files**: All YAML files in the ``nav/config_files/`` directory are loaded in alphabetical order. These files provide default settings for:
   * ``config_01_settings.yaml``: General settings, offset parameters, and body defaults
   * ``config_10_satellites.yaml``: Satellite definitions for planets
   * ``config_20_saturn_rings.yaml``: Saturn ring system parameters
   * ``config_30_coiss.yaml``: Cassini ISS instrument-specific settings
   * ``config_31_gossi.yaml``: Galileo SSI instrument-specific settings
   * ``config_32_nhlorri.yaml``: New Horizons LORRI instrument-specific settings
   * ``config_33_vgiss.yaml``: Voyager ISS instrument-specific settings
   * ``config_40_sim.yaml``: Simulated image settings
   * ``config_90_backplanes.yaml``: Backplane generation settings

2. **User Default Configuration**: If present, the file ``nav_default_config.yaml`` in the current working directory is loaded. This allows you to set personal defaults that apply to all runs.

3. **Command-Line Configuration Files**: Any files specified with the ``--config-file`` option are loaded in the order specified. These provide the highest priority and can override any previous settings.

Configuration File Structure
----------------------------

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

Each section can contain multiple settings. When multiple configuration files define the same setting, the value from the last file loaded takes precedence.

Creating a User Configuration File
------------------------------------

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
-------------------------------------------

You can override configuration on a per-run basis using ``--config-file``:

.. code-block:: bash

   python main/nav_offset.py coiss N1234567890 --config-file /path/to/special_config.yaml

You can specify multiple configuration files, and they will be loaded in order:

.. code-block:: bash

   python main/nav_offset.py coiss N1234567890 \
     --config-file base_overrides.yaml \
     --config-file run_specific.yaml

Command-Line Interface
======================

Basic Usage
-----------

The main entry point for RMS-NAV is ``main/nav_offset.py``. The basic syntax is:

.. code-block:: bash

   python main/nav_offset.py DATASET_NAME [options]

Where ``DATASET_NAME`` is one of the supported names listed in the "Supported Missions" section. Names are case-insensitive (for example, ``COISS`` and ``coiss`` are equivalent).

Command-Line Arguments
----------------------

The command-line interface groups options by purpose. Environment options control configuration sources and output roots. Navigation options select which models or techniques to run. Output options determine whether to write artifacts locally or to produce a cloud-tasks description instead of processing. Dataset selection options are provided by each dataset type: PDS3 datasets expose volume and image filters. A single profiling toggle is available for performance analysis.

Environment options
^^^^^^^^^^^^^^^^^^^

* ``--config-file PATH`` (repeatable): one or more configuration file paths to override defaults.
* ``--pds3-holdings-root PATH``: root directory or URL for PDS3 holdings, overriding both the ``PDS3_HOLDINGS_DIR`` environment variable and any corresponding configuration setting.
* ``--nav-results-root PATH``: root directory or URL where navigation results will be written, overriding both the ``NAV_RESULTS_ROOT`` environment variable and any corresponding configuration setting.

Navigation options
^^^^^^^^^^^^^^^^^^

* ``--nav-models LIST``: a comma-separated list of model names or patterns to enable. Valid entries include ``stars``, ``rings``, ``titan``, and body-specific entries of the form ``body:NAME`` (glob patterns are allowed).
* ``--nav-techniques LIST``: a comma-separated list of navigation techniques to apply. Valid entries include ``correlate_all`` and ``manual``. Note: You should typically use only one technique at a time, as they serve different purposes.

Output options
^^^^^^^^^^^^^^

* ``--output-cloud-tasks-file PATH``: write a JSON file describing tasks for all selected images suitable for a cloud-tasks queue, and exit without performing navigation.
* ``--dry-run``: print the images that would be processed without performing navigation.
* ``--no-write-output-files``: perform navigation but do not write any output files.

Dataset selection (PDS3 datasets)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For PDS3 datasets (``coiss``, ``coiss_pds3``, ``gossi``, ``gossi_pds3``, ``nhlorri``, ``nhlorri_pds3``, ``vgiss``, ``vgiss_pds3``), the following options control which images are selected. All filters combine with logical AND, and explicit lists restrict the search domain before range filters to improve performance.

* ``img_name`` (positional, repeatable): specific image name(s) to process.
* ``--first-image-num N``: minimum image number (inclusive).
* ``--last-image-num N``: maximum image number (inclusive).
* ``--volumes NAME[,NAME...]`` (repeatable): one or more complete PDS3 volume names; you may pass comma-separated values or specify the option multiple times.
* ``--first-volume NAME``: starting PDS3 volume; only that volume and chronologically later ones are processed.
* ``--last-volume NAME``: ending PDS3 volume; only that volume and chronologically earlier ones are processed.
* ``--image-filespec-csv FILE`` (repeatable): CSV file(s) containing PDS3 file specifications; files must include a header column named ``Primary File Spec`` or ``primaryfilespec``.
* ``--image-file-list FILE`` (repeatable): file(s) containing file specifications or names, one per line; lines beginning with ``#`` are ignored.
* ``--choose-random-images N``: choose a random subset of N images that meet the other criteria.

Miscellaneous
^^^^^^^^^^^^^

* ``--profile`` / ``--no-profile``: enable or disable runtime profiling (default is disabled).

Example Commands
----------------

To process a single Cassini image by specifying its name explicitly and using the default navigation technique:

.. code-block:: bash

   python main/nav_offset.py coiss N1234567890

To process Voyager images within a single PDS3 volume:

.. code-block:: bash

   python main/nav_offset.py vgiss --volumes VGISS_5101

To process a New Horizons image list found in a CSV from PDS using the correlate_all technique:

.. code-block:: bash

   python main/nav_offset.py nhlorri --image-filespec-csv /path/to/nhlorri.csv --nav-techniques correlate_all

To choose ten random Cassini images between two volumes and perform a dry run:

.. code-block:: bash

   python main/nav_offset.py coiss --first-volume COISS_2001 --last-volume COISS_2010 --choose-random-images 10 --dry-run

To generate a cloud-tasks JSON file for images across two Voyager volumes without processing:

.. code-block:: bash

   python main/nav_offset.py vgiss --volumes VGISS_5101 --volumes VGISS_5102 --output-cloud-tasks-file tasks.json

Cloud-tasks entry point
-----------------------

Queue-driven processing is supported by ``main/nav_offset_cloud_tasks.py``. This variant reads tasks from a queue and processes each batch of files described by the task payload. It accepts the same environment options used to derive configuration and results roots and does not include dataset selection flags because the task provides the list of files. Invoke it with:

.. code-block:: bash

   python main/nav_offset_cloud_tasks.py [--config-file PATH] [--nav-results-root PATH]

Each task payload must be a JSON object with the following fields:

* ``dataset_name``: one of the supported dataset names.
* ``arguments``: an object with optional keys ``nav_models`` and ``nav_techniques`` (lists or ``null``).
* ``files``: an array of objects, each containing ``image_file_url``, ``label_file_url``, ``results_path_stub``, and optional ``index_file_row`` metadata.

Inputs and Outputs
==================

Input Files
-----------

The primary input to RMS-NAV is spacecraft imagery. The system supports:

* PDS3 formatted image files (.IMG)
* Associated metadata (labels, SPICE kernels)

The system requires access to:

1. The raw image data
2. SPICE kernels for the appropriate mission and time period
3. Configuration settings (optional, defaults are provided)

Output Files
------------

RMS-NAV generates two types of output files:

Metadata Files (``*_metadata.json``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These JSON files contain the navigation results, including:

* The calculated pointing offset (dv, du)
* Uncertainty estimates (sigma_v, sigma_u)
* Confidence scores
* Metadata about the navigation process
* Status information (success, error, etc.)
* Technique-specific metadata
* Timestamps

Summary PNG Files (``*_summary.png``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are annotated images showing:

* The original image data
* Overlaid model features (stars, rings, bodies)
* Text annotations
* Scale information
* Navigation offset information

Interpreting Results
--------------------

The key information in the results is:

1. **Offset Values**: The u,v pixel offsets that should be applied to the nominal pointing to match the observed features
2. **Correlation Quality**: How well the models matched the observed features
3. **Annotations**: Identifications of specific features in the image
4. **Status**: Whether the navigation was successful, and if not, why

Navigation Techniques
=====================

RMS-NAV supports two navigation techniques that can be selected using the ``--nav-techniques`` command-line option. You should typically use only one technique at a time, as they serve different purposes.

correlate_all
-------------

The ``correlate_all`` technique performs automated navigation by correlating the observed image with a combined theoretical model. This is the primary automated navigation method.

**How it works:**

1. **Model Combination**: The technique first creates a combined model from all available navigation models (stars, bodies, rings, etc.). For each pixel, it selects the model element with the smallest range (closest to the observer), ensuring proper depth ordering.

2. **Correlation**: It uses a pyramid-based correlation algorithm (KPeaks) to find the best match between the observed image and the combined model. The correlation searches for the offset that maximizes the match quality.

3. **Star Refinement** (optional): If star models are available and star refinement is enabled in the configuration, the technique performs a second pass to refine the offset by precisely locating individual stars in the image. This refinement:
   - Searches for each star's position using the instrument's point spread function (PSF)
   - Computes the median offset from all successfully located stars
   - Removes outliers using a robust statistical method
   - Updates the final offset with the refined value

4. **Validation**: The technique validates that the computed offset is within the extended field of view margins. If the offset is outside these bounds, the technique fails.

**Configuration Options:**

The following configuration options in ``config_01_settings.yaml`` control the behavior of ``correlate_all``:

* ``offset.correlation_fft_upsample_factor`` (default: 128): The upsampling factor used in the FFT-based correlation. Higher values provide finer sub-pixel resolution but increase computation time.

* ``offset.star_refinement_enabled`` (default: true): Whether to enable star-based refinement after the initial correlation.

* ``offset.star_refinement_nsigma`` (default: 3): The number of standard deviations used to identify outliers during star refinement.

* ``offset.star_refinement_search_limit`` (default: [2.5, 2.5]): The search radius in pixels (v, u) when locating individual stars.

* ``general.log_level_nav_correlate_all``: Logging level for this technique (can be set to DEBUG, INFO, WARNING, ERROR, or NONE).

**When to use:**

* Use ``correlate_all`` for automated batch processing of images
* Best for images with clear features (stars, planetary bodies, or rings)
* Suitable for both starfield and body-dominated images
* Works well when you need consistent, reproducible results

**Output:**

The technique produces:
* A pixel offset (dv, du) indicating the correction needed
* Uncertainty estimates (sigma_v, sigma_u) based on the correlation quality and star refinement
* Confidence score (typically 1.0 for successful correlations)
* Metadata including correlation quality metrics and star refinement statistics

manual
------

The ``manual`` technique provides an interactive graphical interface for manually adjusting the navigation offset. This is useful when automated techniques fail or when expert judgment is needed.

**How it works:**

1. **Model Combination**: Like ``correlate_all``, the manual technique first creates a combined model from all available navigation models.

2. **Interactive Dialog**: A PyQt6-based dialog window opens showing:
   * The observed image (science data)
   * The combined model overlaid on the image
   * Pan and zoom controls for detailed inspection
   * Offset adjustment controls (dv, du spin boxes)
   * An "Auto" button that runs the same correlation algorithm used by ``correlate_all``

3. **User Interaction**: The user can:
   * Pan and zoom to examine details
   * Manually adjust the offset using spin boxes
   * Click "Auto" to get an automated correlation result as a starting point
   * Accept or cancel the navigation

4. **Result**: If the user accepts, the manually specified offset is used. The confidence is set to 1.0, and uncertainty is set to None (since it's a manual determination).

**Configuration Options:**

The manual technique uses the same correlation settings as ``correlate_all`` when the "Auto" button is clicked:
* ``offset.correlation_fft_upsample_factor``: Controls the precision of the auto-correlation

**When to use:**

* Use ``manual`` when automated techniques fail or produce questionable results
* Useful for images with poor quality, unusual features, or edge cases
* Helpful for expert review and validation of automated results
* Required when running in an environment with a display (X11 or similar)

**Requirements:**

* Requires a graphical display (X11, Wayland, or similar)
* Requires PyQt6 to be installed
* Not suitable for headless batch processing environments

**Output:**

The technique produces:
* A pixel offset (dv, du) as specified by the user
* Confidence score of 1.0 (user-accepted result)
* Uncertainty set to None (manual determination)

Troubleshooting
===============

Common Issues
-------------

If SPICE kernels are missing, ensure that all required kernels are available and that environment variables and configuration files point to valid paths. For PDS3 inputs, verify the files conform to expected formats. In cases where no features are found or correlations are weak, check image quality, adjust the selected models or techniques, or limit processing to images known to contain suitable features. Use ``--dry-run`` to validate selection criteria without performing full processing.

Getting Help
------------

If you encounter persistent issues:

Review logs for detailed errors, consult the developer documentation for architectural context, and provide the command line, log snippets, and representative input data when asking for support.
