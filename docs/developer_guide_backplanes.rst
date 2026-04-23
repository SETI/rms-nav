==========
Backplanes
==========

Modules
~~~~~~~

- ``src/backplanes/backplanes.py``: Orchestrates per-image flow

  - Reads prior nav metadata from ``--nav-results-root``
  - Builds ``ObsSnapshot`` with ``extfov_margin_vu=(0, 0)`` and applies ``OffsetFOV``
  - Computes bodies and rings backplanes
  - Merges sources per-pixel by distance
  - Writes the FITS file and a companion metadata JSON via the writer

- ``src/backplanes/backplanes_bodies.py``: Body backplanes

  - For each body in FOV, builds a clipped meshgrid (no oversampling) and evaluates OOPS backplanes
  - Embeds arrays/masks into full-size frames
  - Simulation: synthesizes fake arrays but only within the simulated body mask derived from:
    - ``snapshot.sim_body_mask_map[body_name]`` if present, else
    - ``snapshot.sim_body_index_map`` matched against ``snapshot.sim_body_order_near_to_far``

- ``src/backplanes/backplanes_rings.py``: Ring backplanes

  - Uses full-frame ``snapshot.bp``, evaluates configured ring backplanes
  - Produces per-pixel ``distance`` used for merge ordering

- ``src/backplanes/merge.py``: Per-pixel distance-ordered merge

  - Bodies: body-level scalar distances, broadcast within each body's mask
  - Rings: per-pixel distances
  - BODY_ID_MAP is filled with NAIF IDs; simulation uses deterministic fake IDs when unknown

- ``src/backplanes/writer.py``: Output writer

  - Writes BODY_ID_MAP as the first image HDU
  - Excludes any backplane that is entirely zero from the FITS file
  - Writes a companion ``*_backplane_metadata.json`` containing per-body
    inventory and per-backplane min/max statistics; this JSON is consumed
    by ``nav_create_bundle`` when generating PDS4 labels.  The backplanes
    step itself does not produce any PDS4 labels.

Snapshot Helpers
~~~~~~~~~~~~~~~~

Added methods to ``ObsSnapshot``:

- ``inventory_body_in_fov(inv: dict) -> bool``
- ``inventory_body_in_extfov(inv: dict) -> bool``
- ``clip_rect_fov(u_min, u_max, v_min, v_max) -> tuple[int, int, int, int]``
- ``clip_rect_extfov(u_min, u_max, v_min, v_max) -> tuple[int, int, int, int]``

These unify bounding-box intersection and clipping logic and are used by backplanes and navigation code.

CLI and Roots
~~~~~~~~~~~~~

Backplanes drivers accept two roots:

- ``--nav-results-root``: prior nav results (metadata); used to read ``*_metadata.json``
- ``--backplane-results-root``: destination for new backplane outputs

Offset drivers use:

- ``--nav-results-root`` for navigation outputs (metadata JSON and summary PNG)

Configuration
~~~~~~~~~~~~~

``src/nav/config_files/config_90_backplanes.yaml`` defines:

- ``backplanes.bodies`` and ``backplanes.rings``: each entry has ``name``
  (FITS HDU name), ``method`` (``oops.Backplane`` method name), and
  optional ``units`` (copied to the FITS ``BUNIT`` header card).

Testing
~~~~~~~

There is a smoke test under ``experiments/backplanes/`` using a simulated JSON to ensure backplane generation runs end-to-end. In simulation, per-body masks are respected for fake backplanes to avoid rectangular artifacts.
