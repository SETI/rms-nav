==========
User Guide
==========

Introduction
============

RMS-NAV is a spacecraft image navigation system designed to analyze images from various space missions and determine precise positional offsets. This guide explains how to use the ``nav_main_offset.py`` program, which is the primary command-line interface for the RMS-NAV system.

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

RMS-NAV currently supports the following spacecraft missions and instruments:

* **Cassini** - Imaging Science Subsystem (COISS)
* **Voyager** - Imaging Science Subsystem (VGISS)
* **Galileo** - Solid State Imager (GOSSI)
* **New Horizons** - Long Range Reconnaissance Imager (NHLORRI)

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

   python main/nav_main_offset.py INSTRUMENT [options] [file selection options]

Where ``INSTRUMENT`` is one of:

* ``COISS`` - Cassini Imaging Science Subsystem
* ``GOSSI`` - Galileo Solid State Imager
* ``NHLORRI`` - New Horizons Long Range Reconnaissance Imager
* ``VGISS`` - Voyager Imaging Science Subsystem

Command-Line Arguments
----------------------

Navigation Options
^^^^^^^^^^^^^^^^^^

These options control how the navigation process works:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Option
     - Description
   * - ``--force-offset``
     - Force offset computation even if offset file exists
   * - ``--display-offset-results``
     - Graphically display offset results
   * - ``--force-offset-amount U,V``
     - Force the offset to be the specified u,v values
   * - ``--stars-only``
     - Navigate using only stars
   * - ``--allow-stars``/``--no-allow-stars``
     - Include/exclude stars in navigation (default: include)
   * - ``--rings-only``
     - Navigate using only rings
   * - ``--allow-rings``/``--no-allow-rings``
     - Include/exclude rings in navigation (default: include)
   * - ``--moons-only``
     - Navigate using only moons
   * - ``--allow-moons``/``--no-allow-moons``
     - Include/exclude moons in navigation (default: include)
   * - ``--central-planet-only``
     - Navigate using only the central planet
   * - ``--allow-central-planet``/``--no-allow-central-planet``
     - Include/exclude central planet in navigation (default: include)
   * - ``--use-predicted-kernels``/``--no-use-predicted-kernels``
     - Use predicted CK kernels (default: no)
   * - ``--use-gapfill-kernels``/``--no-use-gapfill-kernels``
     - Use gapfill kernels (default: no)
   * - ``--use-kernel KERNEL``
     - Use specified CK kernel(s)
   * - ``--use-cassini-nac-wac-offset``/``--no-use-cassini-nac-wac-offset``
     - Use the computed offset between NAC and WAC frames (default: yes)

Output Options
^^^^^^^^^^^^^^

These options control the output files generated by the system:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Option
     - Description
   * - ``--write-offset-file``/``--no-write-offset-file``
     - Generate an offset file (default: yes)
   * - ``--write-overlay-file``/``--no-write-overlay-file``
     - Generate an overlay file (default: yes)
   * - ``--write-png-file``/``--no-write-png-file``
     - Generate a PNG file (default: yes)
   * - ``--no-write-results``
     - Don't write any output files
   * - ``--png-also-bw``/``--no-png-also-bw``
     - Produce a black and white PNG file along with the color one (default: no)
   * - ``--png-blackpoint VALUE``
     - Set the blackpoint for the PNG file
   * - ``--png-whitepoint VALUE``
     - Set the whitepoint for the PNG file
   * - ``--png-gamma VALUE``
     - Set the gamma for the PNG file
   * - ``--metadata-label-font FONTFILE,SIZE``
     - Set the font for the PNG metadata info
   * - ``--stars-label-font FONTFILE,SIZE``
     - Set the font for star labels
   * - ``--rings-label-font FONTFILE,SIZE``
     - Set the font for ring labels
   * - ``--bodies-label-font FONTFILE,SIZE``
     - Set the font for body labels (moons and central planet)
   * - ``--label-rings-backplane``/``--no-label-rings-backplane``
     - Label backplane longitude and radius on ring images (default: no)
   * - ``--show-star-streaks``/``--no-show-star-streaks``
     - Show star streaks in the overlay and PNG files (default: no)

File Selection Options
^^^^^^^^^^^^^^^^^^^^^^

File Selection Logic
^^^^^^^^^^^^^^^^^^^^

The file selection options can be combined to create complex filtering rules:

1. **Volume-based selection**: Use ``--volumes``, ``--first-volume``, and ``--last-volume`` to select images from specific PDS3 volumes.

2. **Image number selection**: Use ``--first-image-num`` and ``--last-image-num`` to select images within a specific range of image numbers.

3. **Camera type filtering**: Use ``--camera-type`` to select images from a specific camera (e.g., NAC or WAC for Cassini).

4. **File list selection**: Use ``--image-filelist`` or ``--image-pds-csv`` to process images listed in external files.

5. **Processing state filtering**: Use ``--has-offset-file``, ``--has-no-offset-file``, ``--has-png-file``, and ``--has-no-png-file`` to select images based on whether they've been processed before.

6. **Error condition filtering**: Use ``--has-offset-error``, ``--has-offset-spice-error``, and ``--has-offset-nonspice-error`` to select images that had specific errors during previous processing attempts.

7. **Custom filtering**: Use ``--selection-expr`` with a Python expression that evaluates metadata from previous processing attempts.

8. **Random sampling**: Use ``--choose-random-images`` to select a random subset of images that match other criteria.

When multiple selection criteria are specified, they are combined with logical AND - only images that satisfy all criteria will be selected.

File Selection Options
^^^^^^^^^^^^^^^^^^^^^^

These options vary by instrument but generally include:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Option
     - Description
   * - ``image_name``
     - Specific image name(s) to process
   * - ``--image-full-path PATH``
     - Process a single image at the specified path
   * - ``--directory DIR``
     - Process all valid images in the specified directory
   * - ``--first-image-num IMAGE_NUM``
     - The starting image number
   * - ``--last-image-num IMAGE_NUM``
     - The ending image number
   * - ``--camera-type TYPE``
     - Only process images with the given camera type
   * - ``--volumes VOL_NAME``
     - One or more entire PDS3 volumes or volume/range_subdirs
   * - ``--first-volume VOL_NAME``
     - The starting PDS3 volume name
   * - ``--last-volume VOL_NAME``
     - The ending PDS3 volume name
   * - ``--image-pds-csv FILE``
     - A CSV file downloaded from PDS that contains filespecs of images to process
   * - ``--image-filelist FILE``
     - A file that contains image names of images to process
   * - ``--strict-file-order``
     - With --image-filelist or --image-pds-csv, return filename in the order in the file, not numerical order
   * - ``--has-offset-file``
     - Only process images that already have an offset file
   * - ``--has-no-offset-file``
     - Only process images that don't already have an offset file
   * - ``--has-png-file``
     - Only process images that already have a PNG file
   * - ``--has-no-png-file``
     - Only process images that don't already have a PNG file
   * - ``--has-offset-error``
     - Only process images if the offset file exists and indicates a fatal error
   * - ``--has-offset-nonspice-error``
     - Only process images if the offset file exists and indicates a fatal error other than missing SPICE data
   * - ``--has-offset-spice-error``
     - Only process images if the offset file exists and indicates a fatal error from missing SPICE data
   * - ``--selection-expr EXPR``
     - Expression to evaluate to decide whether to reprocess an offset
   * - ``--choose-random-images N``
     - Choose N random images to process within other constraints
   * - ``--show-image-list-only``
     - Only show the list of images that would be processed

Miscellaneous Options
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Option
     - Description
   * - ``--profile``/``--no-profile``
     - Do performance profiling (default: no)

Example Commands
----------------

Process a single Cassini image using only star navigation:

.. code-block:: bash

   python main/nav_main_offset.py COISS --stars-only --image-full-path /path/to/image/N1234567890.IMG

Process all Voyager images in a directory:

.. code-block:: bash

   python main/nav_main_offset.py VGISS --directory /path/to/voyager/images

Process a New Horizons image with custom PNG settings:

.. code-block:: bash

   python main/nav_main_offset.py NHLORRI --image-full-path /path/to/image.IMG --png-blackpoint 100 --png-whitepoint 1000 --png-gamma 1.2

Process a Galileo image using only moon features:

.. code-block:: bash

   python main/nav_main_offset.py GOSSI --moons-only --image-full-path /path/to/image.IMG

Process a Cassini image without creating output files (for testing):

.. code-block:: bash

    python main/nav_main_offset.py COISS --no-write-results --image-full-path /path/to/image.IMG

Using PDS3 Dataset Options
--------------------------

The PDS3 dataset options provide powerful ways to select and filter images from PDS3 archives:

Processing images from a specific volume:

.. code-block:: bash

   python main/nav_main_offset.py VGISS --volumes VGISS_5101 --show-image-list-only

Processing images within a range of volumes:

.. code-block:: bash

   python main/nav_main_offset.py COISS --first-volume COISS_2001 --last-volume COISS_2010

Processing images that need reprocessing (no existing offset file):

.. code-block:: bash

   python main/nav_main_offset.py COISS --has-no-offset-file --volumes COISS_2001

Processing images with specific error conditions:

.. code-block:: bash

   python main/nav_main_offset.py COISS --has-offset-spice-error --volumes COISS_2001

Processing a random sample of images:

.. code-block:: bash

   python main/nav_main_offset.py VGISS --volumes VGISS_5101 --choose-random-images 10

Using a custom selection expression to filter images:

.. code-block:: bash

   python main/nav_main_offset.py COISS --selection-expr "metadata.get('EXPOSURE_DURATION') > 100" --volumes COISS_2001


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

1. **Missing SPICE kernels**: Ensure all required SPICE kernels are available and the SPICE_PATH is set correctly
2. **Image format issues**: Verify that the input image is in the expected PDS3 format
3. **No features found**: Some images may not contain enough features for navigation
4. **Poor correlation**: Check image quality and try different navigation techniques

Getting Help
------------

If you encounter persistent issues:

1. Check the log files for detailed error messages
2. Review the example scripts in the `examples/` directory
3. Consult the developer documentation
4. Contact the development team with:
   - A description of the issue
   - The command line used
   - Log files
   - Sample images (if possible)