==========
Backplanes User Guide
==========

Overview
========

Backplanes are per-pixel geometry products (e.g., longitude, latitude, incidence angle) generated for each image. The system reads prior navigation metadata to apply the image offset, then computes body and ring backplanes, merges them per-pixel, and writes a multi-HDU FITS along with a PDS4 label.

Key properties:

- Output FITS places BODY_ID_MAP as the first image HDU (after the primary HDU).
- Only non-empty backplanes (not all zeros) are included in the FITS and label.
- The list of backplanes is configured under ``backplanes`` in YAML (see config_90_backplanes.yaml).
- For simulations, fake backplanes are synthesized and masks follow simulated body shapes.

Command-Line Interfaces
========================

Two drivers mirror the offset drivers:

- ``main/nav_backplanes.py`` (local/CLI)
- ``main/nav_backplanes_cloud_tasks.py`` (Cloud Tasks)

Common flags:

- ``--nav-results-root``: Root containing prior navigation results (e.g., ``*_metadata.json``). This was previously referred to as "metadata root".
- ``--backplane-results-root``: Root to write backplane outputs (FITS and PDS4). This was previously "results root".
- Dataset selection flags are the same as in the offset drivers.

Examples
--------

Generate backplanes locally for a dataset:

.. code-block:: bash

    python3 main/nav_backplanes.py COISS \
      --nav-results-root /data/nav/results \
      --backplane-results-root /data/nav/backplanes \
      --volumes COISS_2001 --first-image-num 1454000000 --last-image-num 1454999999

Cloud Tasks variant (arguments come from the queue):

.. code-block:: bash

    python3 main/nav_backplanes_cloud_tasks.py \
      --nav-results-root /data/nav/results \
      --backplane-results-root /data/nav/backplanes

Configuration
-------------

Backplanes are configured in YAML (see ``nav/config_files/config_90_backplanes.yaml``):

- ``backplanes.bodies``: list of backplane entries with ``name``, ``method``, and optional ``units``.
- ``backplanes.rings``: list of ring backplanes; the special ``distance`` entry is used only for per-pixel ordering and is not written as an HDU.
- ``backplanes.target_lids``: optional NAIF ID → LID mapping for PDS4 label target references.

Outputs
-------

- FITS: ``<results_path_stub>_backplanes.fits`` with:

  - Primary HDU.
  - BODY_ID_MAP (int32) as the first image HDU.
  - One ``ImageHDU`` per non-empty master backplane array. ``BUNIT`` is set from config when provided.

- PDS4 label: ``<results_path_stub>_backplanes.xml``, generated from a local template (``nav/backplanes/templates/backplanes.lblx``), referencing the output FITS and including target references when configured.

Backplane Viewer GUI
====================

Use the interactive GUI to inspect backplane FITS alongside the science image.

Run
---

.. code-block:: bash

    python3 main/nav_backplane_viewer.py COISS \
      --nav-results-root /data/nav/results \
      --backplane-results-root /data/nav/backplanes \
      --volumes COISS_2001 \
      --first-image-num 1454000000 --last-image-num 1454000999

Features
--------

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
-----

- Units: Angular FITS HDUs with ``BUNIT=rad`` are converted to degrees for display and absolute scaling. Heuristics are used for common angle names if units are missing.
- Masking: Backplane visualizations use ``BODY_ID_MAP != 0`` to determine valid pixels for relative scaling; numeric zeros are not treated as masked unless indicated by the body map.
