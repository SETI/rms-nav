====================
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

Ring Configuration
==================

Planetary ring features are defined in separate YAML files under
``src/nav/config_files/``. The default Saturn configuration is in
``config_20_saturn_rings.yaml``.

YAML Structure
--------------

.. code-block:: yaml

   rings:
     ring_features:
       SATURN:                             # Planet name (must match obs.closest_planet)
         epoch: '2004-01-01 12:00:00'     # Reference epoch for precessing modes
         fade_width_pix: 100.0            # Nominal fade width in pixels for each edge
         min_allowed_fade_width_pix: 2.0  # Minimum fade width before edge is excluded
         min_feature_pixels: 2.0          # Minimum resolvable gap width (pass-3 filter)
         features:                        # Dict of named ring features
           colombo_gap:
             feature_type: GAP
             outer_data:                  # Edge data list (mode 1 = base orbit)
               - mode: 1
                 a: 77870.0              # Semi-major axis in km
                 ae: 100.0              # Amplitude of eccentricity (km)
                 long_peri: 195.0        # Longitude of periapsis (degrees)
                 rate_peri: 0.0          # Precession rate (degrees/year)
                 rms: 2.0               # Edge uncertainty (km, 1-sigma RMS)
           titan_ringlet:
             feature_type: RINGLET
             start_date: '2004-01-01'   # Optional: feature active from this UTC date
             end_date: '2017-09-15'     # Optional: feature active until this UTC date (exclusive)
             inner_data:
               - mode: 1
                 a: 77517.0
                 ae: 3.0
                 long_peri: 0.0
                 rate_peri: 0.0
                 rms: 1.0
             outer_data:
               - mode: 1
                 a: 77871.0
                 ae: 5.0
                 long_peri: 0.0
                 rate_peri: 0.0
                 rms: 2.0
               - mode: 2                # Optional perturbation mode
                 amplitude: 1.5
                 phase: 30.0
                 pattern_speed: 0.5

Planet-Level Parameters
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``epoch``
     - Reference UTC date-time string for evaluating precessing orbital modes.
       All ``long_peri`` angles and ``rate_peri`` precession rates are evaluated
       relative to this epoch. Required.
   * - ``fade_width_pix``
     - Desired fade width in pixels for each rendered edge. The fade spans this
       many pixels everywhere in the image: at the ansae (high resolution) the
       fade covers fewer kilometres; near the ansa edges (low resolution) it
       covers more. Required; must be positive.
   * - ``min_allowed_fade_width_pix``
     - Minimum fade width in pixels. If a neighboring edge would force the
       conflict-adjusted fade below this threshold (at the best resolution along
       the edge), the edge is excluded by the filter. Required; must be positive.
   * - ``min_feature_pixels``
     - Minimum resolvable width in pixels for two-edge features (RINGLETs and
       GAPs where both edges fall within the FOV). Features narrower than
       ``min_feature_pixels * min_resolution`` are excluded by the filter because
       the gap cannot be detected. Required; must be positive.

Feature-Level Parameters
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``feature_type``
     - ``GAP`` or ``RINGLET``. Determines how single-edge features are shaded
       and how the region between a pair of edges is filled.
   * - ``inner_data`` / ``outer_data``
     - List of mode dicts describing the edge orbit. At least one of these must
       be present. Mode 1 is the base orbit (required in the list); higher modes
       are perturbations. See Edge Mode Parameters below.
   * - ``start_date``
     - Optional UTC date string. The feature is active only for observations at
       or after this date.
   * - ``end_date``
     - Optional UTC date string. The feature is active only for observations
       strictly before this date.

Edge Mode Parameters (mode 1 — base orbit)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - ``mode``
     - Must be ``1`` for the base orbit entry.
   * - ``a``
     - Semi-major axis in km. Must be positive.
   * - ``ae``
     - Eccentricity amplitude in km (half of peak-to-peak radial variation).
   * - ``long_peri``
     - Longitude of periapsis at the reference epoch, in degrees.
   * - ``rate_peri``
     - Precession rate of periapsis, in degrees per year.
   * - ``rms``
     - Edge position uncertainty, in km (1-sigma RMS). Used for
       ``NavModelResult.uncertainty``.

Edge Mode Parameters (mode > 1 — perturbation modes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - ``mode``
     - Mode number. Values 2–90 are radial perturbations (supported).
       Values > 90 are inclination modes (stored but silently skipped during
       rendering because the current backplane API does not support them).
   * - ``amplitude``
     - Perturbation amplitude in km.
   * - ``phase``
     - Phase angle at the reference epoch, in degrees.
   * - ``pattern_speed``
     - Pattern speed in degrees per year.

Validation
----------

``RingFeature.from_config()`` validates every feature dictionary immediately when it
is read. Errors raise ``ValueError`` with the feature key in the message. Checks
include:

* ``feature_type`` must be ``"GAP"`` or ``"RINGLET"``.
* At least one of ``inner_data`` / ``outer_data`` must be present.
* Each mode list must contain exactly one mode-1 entry.
* ``a`` must be positive; ``rms`` must be non-negative.
* Date strings must be parseable by ``utc_to_et``.

After all features are loaded, ``validate_no_date_overlaps()`` performs a
cross-feature pass. If two features share overlapping radial extents *and* both have
explicit ``[start_date, end_date)`` windows that overlap in time, a ``ValueError`` is
raised. This catches authoring mistakes where a curator accidentally activates two
conflicting features simultaneously.

Adding a New Planet
-------------------

To configure rings for a new planet (e.g., Uranus):

1. Create ``src/nav/config_files/config_XX_uranus_rings.yaml`` with the structure
   shown above, replacing ``SATURN`` with ``URANUS``.

2. Add a ``!include`` directive (or equivalent) in the main config so that the new
   file is loaded alongside the Saturn file:

   .. code-block:: yaml

      rings:
        ring_features:
          !include config_XX_uranus_rings.yaml

3. Populate ``fade_width_pix``, ``min_allowed_fade_width_pix``,
   ``min_feature_pixels``, and ``epoch`` for the new planet.

4. Add individual features under ``features:`` using the same format as Saturn.

No code changes are required. The orchestrator (``NavModelRings``) reads whichever
planet name appears in ``obs.closest_planet`` and looks it up in
``rings.ring_features`` at runtime.
