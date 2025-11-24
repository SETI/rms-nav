==========
User Guide
==========

Introduction
============

RMS-NAV is a spacecraft image navigation system designed to analyze images from various space missions and determine precise positional offsets. This guide explains how to use the primary command-line interface in ``main/nav_main_offset.py`` to navigate images and generate results, how to invoke the cloud-tasks variant for queue-driven processing, and how to launch the GUI used to design simulated models for testing.

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
* ``sim`` — simulated images described by JSON files

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

Command-Line Interface
======================

Basic Usage
-----------

The main entry point for RMS-NAV is ``main/nav_main_offset.py``. The basic syntax is:

.. code-block:: bash

   python main/nav_main_offset.py DATASET_NAME [options]

Where ``DATASET_NAME`` is one of the supported names listed in the "Supported Missions" section. Names are case-insensitive (for example, ``COISS`` and ``coiss`` are equivalent).

Command-Line Arguments
----------------------

The command-line interface groups options by purpose. Environment options control configuration sources and output roots. Navigation options select which models or techniques to run. Output options determine whether to write artifacts locally or to produce a cloud-tasks description instead of processing. Dataset selection options are provided by each dataset type: PDS3 datasets expose volume and image filters, while the simulated dataset expects one or more JSON files. A single profiling toggle is available for performance analysis.

Environment options
^^^^^^^^^^^^^^^^^^^

* ``--config-file PATH`` (repeatable): one or more configuration file paths to override defaults.
* ``--pds3-holdings-root PATH``: root directory or URL for PDS3 holdings, overriding both the ``PDS3_HOLDINGS_DIR`` environment variable and any corresponding configuration setting.
* ``--nav-results-root PATH``: root directory or URL where navigation results will be written, overriding both the ``NAV_RESULTS_ROOT`` environment variable and any corresponding configuration setting.

Navigation options
^^^^^^^^^^^^^^^^^^

* ``--nav-models LIST``: a comma-separated list of model names or patterns to enable. Valid entries include ``stars``, ``rings``, ``titan``, and body-specific entries of the form ``body:NAME`` (glob patterns are allowed).
* ``--nav-techniques LIST``: a comma-separated list of navigation techniques to apply. Valid entries include ``correlate_all`` and ``manual``.

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

Dataset selection (simulated dataset)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the simulated dataset (``sim``), each dataset value is a path to a JSON file describing an image to be rendered and navigated. Selection options are:

* ``img_path`` (positional, repeatable): one or more paths to JSON files containing simulated images to process.

Miscellaneous
^^^^^^^^^^^^^

* ``--profile`` / ``--no-profile``: enable or disable runtime profiling (default is disabled).

Example Commands
----------------

To process a single Cassini image by specifying its name explicitly and using the default navigation technique:

.. code-block:: bash

   python main/nav_main_offset.py coiss N1234567890

To process Voyager images within a single PDS3 volume:

.. code-block:: bash

   python main/nav_main_offset.py vgiss --volumes VGISS_5101

To process a New Horizons image list found in a CSV from PDS, running both star and manual techniques:

.. code-block:: bash

   python main/nav_main_offset.py nhlorri --image-filespec-csv /path/to/nhlorri.csv --nav-techniques correlate_all,manual

To choose ten random Cassini images between two volumes and perform a dry run:

.. code-block:: bash

   python main/nav_main_offset.py coiss --first-volume COISS_2001 --last-volume COISS_2010 --choose-random-images 10 --dry-run

To generate a cloud-tasks JSON file for images across two Voyager volumes without processing:

.. code-block:: bash

   python main/nav_main_offset.py vgiss --volumes VGISS_5101 --volumes VGISS_5102 --output-cloud-tasks-file tasks.json

Cloud-tasks entry point
-----------------------

Queue-driven processing is supported by ``main/nav_main_offset_cloud_tasks.py``. This variant reads tasks from a queue and processes each batch of files described by the task payload. It accepts the same environment options used to derive configuration and results roots and does not include dataset selection flags because the task provides the list of files. Invoke it with:

.. code-block:: bash

   python main/nav_main_offset_cloud_tasks.py [--config-file PATH] [--nav-results-root PATH]

Each task payload must be a JSON object with the following fields:

* ``dataset_name``: one of the supported dataset names.
* ``arguments``: an object with optional keys ``nav_models`` and ``nav_techniques`` (lists or ``null``).
* ``files``: an array of objects, each containing ``image_file_url``, ``label_file_url``, ``results_path_stub``, and optional ``index_file_row`` metadata.

Simulated model GUI
-------------------

The GUI in ``main/create_simulated_body_model.py`` allows you to build and adjust simulated scenes consisting of one or more planetary bodies and star fields. Launch it with:

.. code-block:: bash

   python main/create_simulated_body_model.py

The interface provides a live preview with pan and zoom, tabs for each model element, and controls to adjust geometry and photometric parameters. You can save a PNG snapshot of the current scene and export or import the JSON parameter set used to describe the model. The rendering uses the same engine as the navigation system (see ``nav.sim.render``), so the generated scenes are suitable for testing navigation algorithms end-to-end.


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

RMS-NAV generates several types of output files:

Offset Files (.off)
^^^^^^^^^^^^^^^^^^^

These files contain the navigation results, including:

* The calculated pointing offset
* Metadata about the navigation process
* Status information
* Timestamps

Overlay Files (.ovr)
^^^^^^^^^^^^^^^^^^^^

These files contain data for visualizing the navigation results, including:

* Star positions and identifications
* Ring feature locations
* Planet and moon positions
* Annotation information

PNG Files (.png)
^^^^^^^^^^^^^^^^

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

Troubleshooting
===============

Common Issues
-------------

If SPICE kernels are missing, ensure that all required kernels are available and that environment variables and configuration files point to valid paths. For PDS3 inputs, verify the files conform to expected formats. In cases where no features are found or correlations are weak, check image quality, adjust the selected models or techniques, or limit processing to images known to contain suitable features. Use ``--dry-run`` to validate selection criteria without performing full processing.

Getting Help
------------

If you encounter persistent issues:

Review logs for detailed errors, consult the developer documentation for architectural context, and provide the command line, log snippets, and representative input data when asking for support.

Backplanes Generation
=====================

Overview
--------

Backplanes are per-pixel geometry products (e.g., longitude, latitude, incidence angle) generated for each image. The system reads prior navigation metadata to apply the image offset, then computes body and ring backplanes, merges them per-pixel, and writes a multi-HDU FITS along with a PDS4 label.

Key properties:

- Output FITS places BODY_ID_MAP as the first image HDU (after the primary HDU).
- Only non-empty backplanes (not all zeros) are included in the FITS and label.
- The list of backplanes is configured under ``backplanes`` in YAML (see config_90_backplanes.yaml).
- For simulations, fake backplanes are synthesized and masks follow simulated body shapes.

Command-Line Interfaces
-----------------------

Two drivers mirror the offset drivers:

- ``main/nav_main_backplanes.py`` (local/CLI)
- ``main/nav_main_backplanes_cloud_tasks.py`` (Cloud Tasks)

Common flags:

- ``--nav-results-root``: Root containing prior navigation results (e.g., ``*_metadata.json``). This was previously referred to as "metadata root".
- ``--backplane-results-root``: Root to write backplane outputs (FITS and PDS4). This was previously "results root".
- Dataset selection flags are the same as in the offset drivers.

Examples
^^^^^^^^

Generate backplanes locally for a dataset:

.. code-block:: bash

    python3 main/nav_main_backplanes.py COISS \
      --nav-results-root /data/nav/results \
      --backplane-results-root /data/nav/backplanes \
      --vol-start COISS_2001 --vol-end COISS_2001 --img-start-num 1454000000 --img-end-num 1454999999

Cloud Tasks variant (arguments come from the queue):

.. code-block:: bash

    python3 main/nav_main_backplanes_cloud_tasks.py \
      --nav-results-root /data/nav/results \
      --backplane-results-root /data/nav/backplanes

Configuration
^^^^^^^^^^^^^

Backplanes are configured in YAML (see ``nav/config_files/config_90_backplanes.yaml``):

- ``backplanes.bodies``: list of backplane entries with ``name``, ``method``, and optional ``units``.
- ``backplanes.rings``: list of ring backplanes; the special ``distance`` entry is used only for per-pixel ordering and is not written as an HDU.
- ``backplanes.target_lids``: optional NAIF ID → LID mapping for PDS4 label target references.

Outputs
^^^^^^^

- FITS: ``<results_path_stub>_backplanes.fits`` with:

  - Primary HDU.
  - BODY_ID_MAP (int32) as the first image HDU.
  - One ``ImageHDU`` per non-empty master backplane array. ``BUNIT`` is set from config when provided.

- PDS4 label: ``<results_path_stub>_backplanes.xml``, generated from a local template (``nav/backplanes/templates/backplanes.lblx``), referencing the output FITS and including target references when configured.

Backplane Viewer GUI
--------------------

Use the interactive GUI to inspect backplane FITS alongside the science image.

Run
^^^

.. code-block:: bash

    python3 main/nav_backplane_viewer.py COISS \
      --nav-results-root /data/nav/results \
      --backplane-results-root /data/nav/backplanes \
      --vol-start COISS_2001 --vol-end COISS_2001 \
      --img-start-num 1454000000 --img-end-num 1454000999

Features
^^^^^^^^

- Image stretch: Blackpoint, whitepoint, and gamma for the grayscale science image.
- Zoom and pan: Same behavior as the simulated body model UI.
- Summary overlay: If ``<results_path_stub>_summary.png`` exists under ``--nav-results-root``, it can be toggled on/off with an alpha control (no stretch or colormap).
- Backplane layers:

  - Lists all FITS image HDUs: ``BODY_ID_MAP`` (int32) plus each backplane (float32).
  - Each backplane can be toggled with a checkbox, assigned transparency 0–1, a colormap, and scaling mode (Absolute or Relative).
  - Relative mode computes min/max using only pixels where ``BODY_ID_MAP != 0`` (numeric zeros are not treated specially).
  - Absolute mode:

    - Longitudes: 0–360°; Latitudes: −90–90°.
    - Incidence/Emission/Phase: 0–180°.
    - Radius: 0 to observed max.
    - Resolution and others: observed min–max.

- Live readout: Shows the science image value at the cursor and, for each backplane row, the current value at the cursor (angles are converted from radians to degrees when applicable).

Notes
^^^^^

- Units: Angular FITS HDUs with ``BUNIT=rad`` are converted to degrees for display and absolute scaling. Heuristics are used for common angle names if units are missing.
- Masking: Backplane visualizations use ``BODY_ID_MAP != 0`` to determine valid pixels for relative scaling; numeric zeros are not treated as masked unless indicated by the body map.
