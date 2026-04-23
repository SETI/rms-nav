================
Rings
================

:class:`~nav.nav_model.nav_model_rings.NavModelRings` and
:class:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated` build
ring feature models from configuration (or simulation parameters). The
``nav.nav_model.rings`` subpackage holds the domain model: validation,
filtering, and rendering are separated so each concern can be tested in
isolation.

Ring domain model
-----------------

:class:`~nav.nav_model.rings.ring_types.RingFeatureType` is a two-value enum
(``RINGLET``, ``GAP``) that controls how the shaded region is oriented relative
to the known edge when only one edge is present.

:class:`~nav.nav_model.rings.ring_types.RingBaseOrbitMode` and
:class:`~nav.nav_model.rings.ring_types.RingPerturbationMode` are frozen
dataclasses that hold the orbital parameters for an edge.
:class:`~nav.nav_model.rings.ring_types.RingPerturbationMode` exposes an
``is_inclination_mode`` property (mode number > 90) that the rendering path uses
to skip inclination modes that require a different backplane not yet
supported.

:class:`~nav.nav_model.rings.ring_types.RingEdgeData` bundles the base orbit
and zero or more perturbation modes for a single ring edge. Its ``base_radius``
and ``rms`` properties extract the canonical radius and uncertainty from the
base orbit. ``parsed_modes_for_backplane()`` returns the list of ``(mode,
amplitude, phase, pattern_speed)`` tuples that the ``oops`` backplane API
consumes, omitting inclination modes.

:class:`~nav.nav_model.rings.ring_feature.RingFeature` is the core domain
object—a frozen dataclass that owns a single ring feature (ringlet or gap) with
optional inner and outer edges. It is constructed via ``from_config(key, data)``
which validates the YAML dictionary immediately and raises ``ValueError`` on
any malformed input. Derived cached fields (``_start_et``, ``_end_et``) are
computed in ``__post_init__`` using ``object.__setattr__()`` because the
dataclass is frozen. Query methods (``is_visible_at``, ``is_in_radius_range``,
``uncertainty``, ``all_base_radii``, ``uses_fade_for_edge``) allow the filter
to make decisions without coupling to the rendering path. The ``render(context)``
method dispatches to either ``_render_full_ringlet()`` (for two-edge features)
or ``_render_single_edge()`` (for one-edge features) and returns one or two
:class:`~nav.nav_model.rings.ring_render_result.RingRenderResult` instances.

``validate_no_date_overlaps(features)`` is a module-level function that
enforces authoring invariants: if two features with explicit date ranges share
overlapping radii, their date ranges must not overlap. This is checked as a hard
error at YAML load time to catch mistakes before any rendering occurs.

Ring filtering pipeline
-----------------------

:class:`~nav.nav_model.rings.ring_filter.RingFeatureFilter` applies a
four-pass decision pipeline to a list of ``RingFeature`` objects and returns
only the features that should be rendered:

.. list-table:: Filter passes
   :header-rows: 1
   :widths: 10 25 65

   * - Pass
     - Name
     - Decision
   * - 1
     - Date
     - Exclude the feature if the observation time is outside its ``[start_date,
       end_date)`` window. Features without explicit dates are always kept.
   * - 2
     - Radius
     - Exclude the feature if neither edge falls within ``[min_radius,
       max_radius]``. Partially visible features (one edge in range) are kept so
       the renderer can handle the partial case.
   * - 3
     - Resolvability
     - For two-edge features (RINGLETs and GAPs where both edges are in the
       FOV), exclude the feature if the gap width ``outer.base_radius -
       inner.base_radius`` is smaller than ``min_feature_pixels *
       min_resolution`` along the feature. The motivation is that an
       unresolvable gap provides no useful navigational information—shading its
       surroundings without a visible gap would be misleading.
   * - 4
     - Fade conflict
     - For each edge that uses a fade, check whether a neighboring edge's
       ``all_edge_radii`` would push the conflict-adjusted fade width below
       ``min_allowed_fade_width_pix * min_resolution`` at best resolution. If
       so, exclude that edge. A feature whose only remaining edge is excluded is
       dropped entirely. A feature retaining at least one edge is passed
       through with the excluded edge set to ``None`` (a trimmed feature).

The filter logs every exclusion decision at ``DEBUG`` level with the feature
key, pass number, and reason, allowing detailed inspection without polluting
``INFO`` output. Exclusion and trimming are kept in the filter rather than in
the renderer so that each concern can be tested independently.

Ring YAML configuration
-----------------------

Top-level ring model parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These keys sit directly under the ``rings:`` section (not under
``rings.ring_features.<PLANET>``). They are shared across all planets and
control post-render processing that is applied to every feature before the
``NavModelResult`` is constructed.

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - Parameter
     - Default
     - Description
   * - ``remove_planet_shadow``
     - ``true``
     - When ``true``, pixels of each rendered ring feature that fall inside the
       shadow of the nearest planet are zeroed out of the model image and
       removed from the model mask. The shadow boundary is computed once per
       observation via
       ``obs.ext_bp.where_inside_shadow(ring_target, planet.lower())``, where
       *ring_target* is ``<planet>:ring`` (e.g. ``saturn:ring``) and *planet*
       is ``obs.closest_planet``. If the backplane call raises an exception
       (e.g. degenerate illumination geometry), a warning is logged and shadow
       removal is skipped for that observation. Removing shadowed pixels
       prevents the navigator from matching bright model arcs against the dark
       shadow region, which would introduce a systematic offset bias.
   * - ``remove_body_shadows``
     - ``false``
     - Reserved for future implementation. When set, it will zero out ring
       model pixels that lie in the shadows cast by moons. The flag is
       accepted by the configuration parser but has no effect in the current
       release.

Shadow removal is applied **after** rendering and **before**
``NavModelResult`` construction, so the ``model_img`` and ``model_mask``
stored in each result already reflect the masking. At ``INFO`` log level the
orchestrator reports the shadow pixel count:

.. code-block::

   Planet shadow removal: 1284 pixel(s) inside SATURN shadow will be masked

Set ``general.log_level_model_rings`` to ``DEBUG`` for the full backplane call
trace.

Planetary ring features are defined in separate YAML files under
``src/nav/config_files/``. The default Saturn configuration is in
``config_21_saturn_rings.yaml``.

YAML structure
^^^^^^^^^^^^^^

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
                 rate_peri: 0.0          # Precession rate (degrees/day)
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

Planet-level parameters
^^^^^^^^^^^^^^^^^^^^^^^

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

Feature-level parameters
^^^^^^^^^^^^^^^^^^^^^^^^

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
       are perturbations. See edge mode parameters below.
   * - ``start_date``
     - Optional UTC date string. The feature is active only for observations at
       or after this date.
   * - ``end_date``
     - Optional UTC date string. The feature is active only for observations
       strictly before this date.

Edge mode parameters (mode 1 — base orbit)
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
     - Precession rate of periapsis, in degrees per day.
   * - ``rms``
     - Edge position uncertainty, in km (1-sigma RMS). Used for
       ``NavModelResult.uncertainty``.

Edge mode parameters (mode > 1 — perturbation modes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - ``mode``
     - Mode number. Values 2-90 are radial perturbations (supported).
       Values > 90 are inclination modes (stored but silently skipped during
       rendering because the current backplane API does not support them).
   * - ``amplitude``
     - Perturbation amplitude in km.
   * - ``phase``
     - Phase angle at the reference epoch, in degrees.
   * - ``pattern_speed``
     - Pattern speed in degrees per day.

Validation and loading
^^^^^^^^^^^^^^^^^^^^^^

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

Adding a new planet
^^^^^^^^^^^^^^^^^^^

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

No code changes are required. The orchestrator (:class:`~nav.nav_model.nav_model_rings.NavModelRings`) reads whichever
planet name appears in ``obs.closest_planet`` and looks it up in
``rings.ring_features`` at runtime.
