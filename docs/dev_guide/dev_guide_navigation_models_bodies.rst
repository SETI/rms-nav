======
Bodies
======

Body navigation models render the predicted appearance of a planetary
body and emit one
:class:`~nav.feature.feature.NavFeature` per surviving feature type.
Concrete subclasses derive from
:class:`~nav.nav_model.nav_model_body_base.NavModelBodyBase`, which
carries shared annotation helpers (limb-mask extraction and label
placement).

Registered concrete subclasses:

- :class:`~nav.nav_model.nav_model_body.NavModelBody` — catalog-driven
  body navigation; one instance per body whose
  ``inventory_body_in_extfov`` predicate fires.
- :class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated`
  — simulated-image GUI variant; emits a single ``BODY_DISC``
  feature carrying the rendered template.

The catalog-driven body model
-----------------------------

:class:`~nav.nav_model.nav_model_body.NavModelBody` builds an
oversampled meshgrid around the body's bounding box, renders the
Lambert-shaded silhouette, and extracts limb and terminator polylines
from the discrete masks.  Per-body shape parameters come from
:data:`~nav.nav_model.body_shape.BODY_SHAPE_TABLE`.

Per-image quantities the model computes and exposes through
``self._metadata``:

- ``sub_solar_lon_deg`` / ``sub_solar_lat_deg`` —
  sub-solar coordinates.
- ``sub_observer_lon_deg`` / ``sub_observer_lat_deg`` —
  sub-observer coordinates.
- ``phase_angle_deg`` — center-pixel phase angle.
- ``predicted_diameter_px`` — predicted body diameter in pixels.
- ``km_per_pixel_at_limb`` — mean km/px scale across the limb polyline.
- ``visible_lit_fraction`` — fraction of the predicted disc that is
  both lit and inside the sensor FOV.
- ``overflow_fraction`` — fraction of the predicted disc that is
  outside the sensor FOV.

Emission rules
^^^^^^^^^^^^^^

The model emits the right combination of features per the gates in
:mod:`~nav.nav_model.nav_model_body`:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Feature
     - Emission rule
   * - ``LIMB_ARC``
     - Emitted when the polyline has at least one surviving vertex and
       ``limb_uncertainty_px`` (= ``ellipsoid_residual_km /
       km_per_pixel_at_limb``) is at most
       :data:`~nav.nav_model.nav_model_body.LIMB_ARC_MAX_UNCERTAINTY_PX`.
   * - ``BODY_BLOB``
     - Emitted when ``LIMB_ARC`` was rejected by the uncertainty gate
       and the predicted diameter is at least
       ``BodyShape.min_blob_diameter_px``.
   * - ``BODY_DISC``
     - Emitted alongside ``LIMB_ARC`` when ``visible_lit_fraction``
       meets
       :data:`~nav.nav_model.nav_model_body.BODY_DISC_MIN_VISIBLE_LIT_FRACTION`
       and ``overflow_fraction`` is below
       :data:`~nav.nav_model.nav_model_body.BODY_DISC_MAX_OVERFLOW_FRACTION`.
   * - ``TERMINATOR_ARC``
     - Emitted when the terminator polyline has at least
       :data:`~nav.nav_model.nav_model_body.TERMINATOR_MIN_VERTICES`
       surviving vertices and ``sin(phase_angle)`` is at least
       :data:`~nav.nav_model.nav_model_body.TERMINATOR_MIN_PHASE_FACTOR`.

Per-vertex polyline covariance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``LIMB_ARC`` and ``TERMINATOR_ARC`` features the per-vertex
``sigma_normal_per_vertex_px`` follows the design's quadrature-sum:

.. code-block::

   sigma_normal_per_vertex_km = sqrt(
       ellipsoid_residual_km^2
     + crater_scale_km^2
     + (incidence_factor(i) * limb_softness_km)^2
     + spice_orbital_residual_km^2
   )
   sigma_normal_per_vertex_px = sigma_normal_per_vertex_km / km_per_pixel_at_vertex

with ``limb_softness_km = sigma_PSF_px * km_per_pixel_at_vertex``.  The
``incidence_factor`` is capped at
:data:`~nav.feature.constants.MAX_INCIDENCE_FACTOR_CAP`.  Terminator
arcs add an albedo-variation and photometric-model term to the
quadrature sum.  The ``sigma_tangent_per_vertex_px`` is a small
constant (~0.5 px) reflecting polyline-sampling resolution.

Body shape table
----------------

:mod:`nav.nav_model.body_shape` carries per-body shape, albedo, and
SPICE-residual quantities used by the covariance and emission gates.
Each entry is a frozen
:class:`~nav.nav_model.body_shape.BodyShape` dataclass:

- ``ellipsoid_residual_km`` — RMS deviation of the body silhouette from
  the best-fit ellipsoid.
- ``crater_scale_km`` — characteristic per-image limb roughness from
  craters and topography.
- ``albedo_variation`` — fractional disc-brightness variation.
- ``spice_orbital_residual_km`` — SPK ephemeris uncertainty in km.
- ``min_blob_diameter_px`` — predicted disc diameter at which
  ``BODY_BLOB`` is preferred over an unresolved limb.

:func:`~nav.nav_model.body_shape.shape_for_body` performs the
case-insensitive lookup; bodies absent from the table fall back to
:data:`~nav.nav_model.body_shape.DEFAULT_BODY_SHAPE`.
