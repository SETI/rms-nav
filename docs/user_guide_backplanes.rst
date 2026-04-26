====================
Backplane Generation
====================

Overview
========

Backplanes are per-pixel geometry products (longitude, latitude, incidence
angle, emission angle, phase angle, resolution, etc.) derived from a
navigated image. The system reads prior navigation metadata to apply the
image offset, then computes body and ring backplanes, merges them per-pixel
by distance, and writes a multi-HDU FITS file along with a JSON metadata
file.

Key properties:

- The output FITS places ``BODY_ID_MAP`` as the first image HDU (after the
  primary HDU).
- Backplanes that are entirely zero are omitted from the FITS file.
- The list of backplanes to generate is configured under ``backplanes`` in
  ``src/nav/config_files/config_90_backplanes.yaml``.
- For simulated observations, synthetic backplanes are produced whose masks
  follow the simulated body shapes.

Backplane generation only writes the FITS file and the associated metadata
JSON. PDS4 labels for the backplane products are produced in a later step
by ``nav_create_bundle labels`` (see :doc:`user_guide_pds4_bundle`).

Command-Line Interfaces
========================

Two drivers mirror the offset drivers:

- ``nav_backplanes`` (local/CLI)
- ``nav_backplanes_cloud_tasks`` (Cloud Tasks)

Common flags:

- ``--nav-results-root``: Root containing prior navigation results
  (``*_metadata.json``).
- ``--backplane-results-root``: Root directory for the backplane outputs.
- Dataset selection flags are the same as for ``nav_offset`` (see
  :doc:`user_guide_navigation`).

Examples
--------

Generate backplanes locally for a dataset:

.. code-block:: bash

    nav_backplanes coiss_saturn \
      --nav-results-root /data/nav/results \
      --backplane-results-root /data/nav/backplanes \
      --volumes COISS_2001 --first-image-num 1454000000 --last-image-num 1454999999

To generate a cloud-tasks JSON file for all selected images without actually
generating any backplanes, use ``--output-cloud-tasks-file``:

.. code-block:: bash

    nav_backplanes coiss_saturn \
      --volumes COISS_2001 \
      --output-cloud-tasks-file backplanes_tasks.json

Cloud Tasks variant (file list comes from the queue):

.. code-block:: bash

    nav_backplanes_cloud_tasks \
      --nav-results-root /data/nav/results \
      --backplane-results-root /data/nav/backplanes

Cloud-tasks JSON schema
^^^^^^^^^^^^^^^^^^^^^^^

The file produced by ``--output-cloud-tasks-file`` is a JSON array of task
objects. Each task is:

.. code-block:: json

    {
        "task_id": "<dataset_name>-<label_file_name>-<index>",
        "data": {
            "dataset_name": "<dataset_name>",
            "files": [
                {
                    "image_file_url": "<path or URL to image file>",
                    "label_file_url": "<path or URL to label file>",
                    "results_path_stub": "<relative stub used to name outputs>",
                    "index_file_row": {"<column>": "<value>", "...": "..."}
                }
            ]
        }
    }

Fields:

* ``task_id``: unique string identifier built from the dataset name, the
  first image's label filename, and the enumeration index.
* ``data.dataset_name``: one of the supported dataset names (same value as
  the positional argument to ``nav_backplanes``).
* ``data.files``: one or more file descriptors. Each descriptor has required
  fields ``image_file_url``, ``label_file_url``, ``results_path_stub``, and
  an optional ``index_file_row`` (metadata from the source index file, may
  be ``null``). The ``nav_backplanes_cloud_tasks`` worker accepts no other
  task-level parameters; all other settings come from its own
  ``--config-file``, ``--nav-results-root``, and ``--backplane-results-root``
  CLI flags, which apply to every task the worker handles.

Configuration
-------------

Backplanes are configured under ``backplanes`` in
``src/nav/config_files/config_90_backplanes.yaml``:

- ``backplanes.bodies``: list of body backplane entries. Each entry has
  ``name`` (the FITS HDU name), ``method`` (the ``oops.Backplane`` method to
  call), and optional ``units`` (written to the ``BUNIT`` FITS header).
- ``backplanes.rings``: list of ring backplane entries with the same
  structure. The special ``distance`` entry is used only for per-pixel
  merge ordering and is not written as an HDU.

Outputs
-------

For each processed image, ``nav_backplanes`` writes two files under
``--backplane-results-root``:

- ``<results_path_stub>_backplanes.fits`` containing:

  - A primary HDU.
  - ``BODY_ID_MAP`` (int32) as the first image HDU.
  - One ``ImageHDU`` per non-all-zero backplane array, with ``BUNIT`` set when
    configured.

- ``<results_path_stub>_backplane_metadata.json`` containing per-body
  inventory information and per-backplane ``min``/``max`` statistics
  (consumed by ``nav_create_bundle`` when generating PDS4 labels).

Backplane Viewer GUI
====================

Use the interactive GUI to inspect backplane FITS alongside the science image.

Run
---

.. code-block:: bash

    nav_backplane_viewer coiss_saturn \
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
  - Each backplane can be toggled with a checkbox, assigned transparency 0-1, a colormap, and scaling mode (Absolute or Relative).
  - Relative mode computes min/max using only pixels where ``BODY_ID_MAP != 0`` (numeric zeros are not treated specially).
  - Absolute mode:

    - Longitudes: 0-360 deg; Latitudes: -90-90 deg.
    - Incidence/Emission/Phase: 0-180 deg.
    - Radius: 0 to observed max.
    - Resolution and others: observed min-max.

- Live readout: Shows the science image value at the cursor and, for each backplane row, the current value at the cursor (angles are converted from radians to degrees when applicable).

Notes
-----

- Units: Angular FITS HDUs with ``BUNIT=rad`` are converted to degrees for display and absolute scaling. Heuristics are used for common angle names if units are missing.
- Masking: Backplane visualizations use ``BODY_ID_MAP != 0`` to determine valid pixels for relative scaling; numeric zeros are not treated as masked unless indicated by the body map.
