=================
Navigation models
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

Registered concrete models:

- :class:`~nav.nav_model.stars.nav_model_stars.NavModelStars` — catalog-driven
  star navigation; one instance per observation.
- :class:`~nav.nav_model.nav_model_body.NavModelBody` — per-body silhouette
  navigation; one instance per body whose bounding box overlaps the
  extended FOV.
- :class:`~nav.nav_model.nav_model_rings.NavModelRings` — catalog-driven
  ring navigation; one instance per planet whose ring system is
  configured and visible.
- :class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated`
  and :class:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`
  — simulated-image GUI variants that render bodies and rings from
  operator-supplied parameters.
- :class:`~nav.nav_model.nav_model_titan.NavModelTitan` — placeholder for
  atmospheric-body navigation (no features emitted).

Shared annotation helpers live on
:class:`~nav.nav_model.nav_model_body_base.NavModelBodyBase` (body silhouette
+ label rendering) and
:class:`~nav.nav_model.nav_model_rings_base.NavModelRingsBase` (per-edge
polyline + label rendering).  The :mod:`nav.nav_model.rings` subpackage
holds the catalog-driven ring data model — validation, filtering, and
rendering are separated so each concern can be tested in isolation.

The API surface is summarised under
:doc:`/api_reference/api_nav_model`.

.. toctree::
   :maxdepth: 1

   dev_guide_navigation_models_stars
   dev_guide_navigation_models_bodies
   dev_guide_navigation_models_rings
   dev_guide_navigation_models_titan
