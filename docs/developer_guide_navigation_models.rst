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

Today's registered concrete models cover Titan (a stub), simulated
bodies, and simulated rings; the real-scene body and ring models are
unimplemented.  The data-model subpackage :mod:`nav.nav_model.rings`
provides the catalog-driven ring-feature classes shared between the
simulated and real (forthcoming) ring renderers.

The API surface is summarised under
:doc:`api_reference/api_nav_model`.

.. toctree::
   :maxdepth: 1

   developer_guide_navigation_models_stars
   developer_guide_navigation_models_bodies
   developer_guide_navigation_models_rings
   developer_guide_navigation_models_titan
