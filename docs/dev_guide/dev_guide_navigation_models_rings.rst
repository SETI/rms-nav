=====
Rings
=====

Ring navigation models render the predicted appearance of a planet's ring system and emit
either per-edge polylines (the "edges resolve" path) or a per-planet annulus template (the
"edges compress" path). Concrete subclasses derive from
:class:`~nav.nav_model.nav_model_rings_base.NavModelRingsBase`, which carries shared
annotation helpers (per-edge polyline + label rendering).

Registered concrete subclasses:

- :class:`~nav.nav_model.nav_model_rings.NavModelRings` — catalog-driven ring navigation;
  one instance per planet whose ring system has any radius inside the extended FOV.
  Documented at :doc:`dev_guide_navigation_models_ring`.
- :class:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated` —
  simulated-image GUI variant; emits a single
  :data:`~nav.feature.feature_type.NavFeatureType.RING_ANNULUS` feature carrying the
  rendered template. Documented at
  :doc:`dev_guide_navigation_models_ring_simulated`.

The :mod:`nav.nav_model.rings` subpackage holds the catalog-driven ring data model —
validation, filtering, and rendering are separated so each concern can be tested in
isolation (``ring_types``, ``ring_feature``, ``ring_filter``, ``ring_math``,
``ring_render_context``, ``ring_render_result``).

.. toctree::
   :maxdepth: 4

   dev_guide_navigation_models_ring
   dev_guide_navigation_models_ring_simulated
