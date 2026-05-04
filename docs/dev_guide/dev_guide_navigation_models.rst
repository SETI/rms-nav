=================
Navigation Models
=================

:class:`~nav.nav_model.nav_model.NavModel` is the abstract base for
predicted-scene generators.  Each subclass implements three methods:

- ``create_model()`` — populate the model's internal state and
  ``metadata`` dict.
- ``to_features(context)`` — return a list of
  :class:`~nav.feature.feature.NavFeature` instances for technique
  consumption.
- ``to_annotations(context)`` — return an
  :class:`~nav.annotation.annotations.Annotations` collection for the
  summary PNG.

Concrete subclasses self-register via ``__init_subclass__``; abstract
bases set ``_abstract = True`` to opt out.  The class method
``instances_for_obs(cls, obs)`` is the per-class hook that
:func:`~nav.nav_model.nav_model.build_models_for_obs` iterates.

Registered concrete models, grouped by feature family:

- **Stars** (:doc:`dev_guide_navigation_models_stars`) —
  :class:`~nav.nav_model.stars.nav_model_stars.NavModelStars` (catalog-driven; one
  instance per observation).  ``NavModelStarsSimulated`` is planned but not yet
  implemented.
- **Bodies** (:doc:`dev_guide_navigation_models_bodies`) —
  :class:`~nav.nav_model.nav_model_body.NavModelBody` (catalog-driven; one instance per
  body whose bounding box overlaps the extended FOV) and
  :class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated`
  (simulated-image GUI variant).
- **Rings** (:doc:`dev_guide_navigation_models_rings`) —
  :class:`~nav.nav_model.nav_model_rings.NavModelRings` (catalog-driven; one instance per
  planet whose ring system is configured and visible) and
  :class:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`
  (simulated-image GUI variant).
- **Titan** (:doc:`dev_guide_navigation_models_titans`) —
  :class:`~nav.nav_model.nav_model_titan.NavModelTitan` (registered placeholder for
  atmospheric-body navigation; no features emitted today).
  ``NavModelTitanSimulated`` is planned but not yet implemented.

Shared annotation helpers live on
:class:`~nav.nav_model.nav_model_body_base.NavModelBodyBase` (body silhouette
+ label rendering) and
:class:`~nav.nav_model.nav_model_rings_base.NavModelRingsBase` (per-edge
polyline + label rendering).

The API surface is summarised under
:doc:`/api_reference/api_nav_model`.

.. toctree::
   :maxdepth: 4

   dev_guide_navigation_models_stars
   dev_guide_navigation_models_bodies
   dev_guide_navigation_models_rings
   dev_guide_navigation_models_titans
