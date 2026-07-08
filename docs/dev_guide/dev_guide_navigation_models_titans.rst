=====
Titan
=====

Atmospheric-body navigation models render predicted views of bodies whose visible
"limb" is a haze top rather than a solid surface (Titan, Venus, Triton's nitrogen frost
layer at low phase) and emit feature types tailored to that geometry. All concrete
subclasses derive from :class:`~spindoctor.nav_model.nav_model.NavModel` directly; the family
has no shared abstract base because no full implementation is registered.

Registered concrete subclasses:

- :class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` — registered placeholder for
  haze-aware atmospheric-body navigation; emits no features. Documented at
  :doc:`dev_guide_navigation_models_titan`.
- ``NavModelTitanSimulated`` — reserved without an implementation. The simulated-image
  GUI variant would render a haze-bounded disc from operator-supplied parameters.
  Documented at :doc:`dev_guide_navigation_models_titan_simulated`.

Atmospheric-body navigation is conceptually distinct from ellipsoid-limb fitting: the
optical limb is the haze top (which moves with wavelength), phase angle changes the
apparent limb shape because forward-scattered haze brightens differently from
back-scattered haze, and the haze altitude varies with latitude, season, and (on Titan)
year-by-year atmospheric circulation. The placeholder reserves the registry slot for a
haze-aware extractor; while the slot is unfilled, Titan-class scenes navigate against
any other body, ring, or star in the FOV without a Titan-derived contribution.

.. toctree::
   :maxdepth: 4

   dev_guide_navigation_models_titan
   dev_guide_navigation_models_titan_simulated
