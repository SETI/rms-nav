======
Bodies
======

Body navigation models render the predicted appearance of a planetary body and emit one
:class:`~nav.feature.feature.NavFeature` per surviving feature type. Concrete subclasses
derive from :class:`~nav.nav_model.nav_model_body_base.NavModelBodyBase`, which carries shared
annotation helpers (limb-mask extraction and label placement).

Registered concrete subclasses:

- :class:`~nav.nav_model.nav_model_body.NavModelBody` — catalog-driven body navigation; one
  instance per body whose :meth:`~nav.obs.obs_snapshot.ObsSnapshot.inventory_body_in_extfov`
  predicate fires. Documented at :doc:`dev_guide_navigation_models_body`.
- :class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated` — simulated-image GUI
  variant; emits a single :data:`~nav.feature.feature_type.NavFeatureType.BODY_DISC` feature
  carrying the rendered template.

Per-body shape, albedo, and SPK-residual quantities consumed by the covariance and emission
gates live in :mod:`nav.nav_model.body_shape`. The runtime lookup
:func:`~nav.nav_model.body_shape.shape_for_body` overlays operator YAML
(``config_220_body_shape.yaml``) onto the hard-coded
:data:`~nav.nav_model.body_shape.BODY_SHAPE_TABLE`, falling back to
:data:`~nav.nav_model.body_shape.DEFAULT_BODY_SHAPE` for entirely unknown bodies.

.. toctree::
   :maxdepth: 4

   dev_guide_navigation_models_body
   dev_guide_navigation_models_body_simulated
