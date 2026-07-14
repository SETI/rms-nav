=====
Titan
=====

The Titan navigation model renders a predicted view of Titan, whose visible "limb" is a
haze top rather than a solid surface, and would emit feature types tailored to that
geometry. Titan is handled as a deliberate special case, not as one entry of a general
atmospheric-body class: its atmosphere is unique (transparent at some wavelengths), so what
is true for Titan does not carry over to other thick-atmosphere bodies. The concrete
subclass derives from :class:`~spindoctor.nav_model.nav_model.NavModel` directly; the family
has no shared abstract base because no full implementation is registered.

Registered concrete subclasses:

- :class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` — registered placeholder for
  haze-aware Titan navigation; emits no features. Documented at
  :doc:`dev_guide_navigation_models_titan`.
- ``NavModelTitanSimulated`` — reserved without an implementation. The simulated-image
  GUI variant would render a haze-bounded disc from operator-supplied parameters.
  Documented at :doc:`dev_guide_navigation_models_titan_simulated`.

Titan navigation is conceptually distinct from ellipsoid-limb fitting: the
optical limb is the haze top (which moves with wavelength), phase angle changes the
apparent limb shape because forward-scattered haze brightens differently from
back-scattered haze, and the haze altitude varies with latitude, season, and year-by-year
atmospheric circulation. The placeholder reserves the registry slot for a
haze-aware extractor; while the slot is unfilled, Titan scenes navigate against
any other body, ring, or star in the FOV without a Titan-derived contribution.

.. toctree::
   :maxdepth: 4

   dev_guide_navigation_models_titan
   dev_guide_navigation_models_titan_simulated
