=====
Stars
=====

Star navigation models render the predicted appearance of catalog stars in the field of
view and emit one :class:`~nav.feature.feature.NavFeature` per surviving star. All
concrete subclasses derive from :class:`~nav.nav_model.nav_model.NavModel` directly; the
shared catalog reduction, conflict marking, and photometry helpers live in the
:mod:`nav.nav_model.stars` subpackage.

Registered concrete subclasses:

- :class:`~nav.nav_model.stars.nav_model_stars.NavModelStars` — catalog-driven star
  navigation; one instance per observation. Documented at
  :doc:`dev_guide_navigation_models_star`.
- ``NavModelStarsSimulated`` — reserved without an implementation. The simulated-image
  GUI variant would render stars from operator-supplied parameters. Documented at
  :doc:`dev_guide_navigation_models_star_simulated`.

The :mod:`nav.nav_model.stars` subpackage carries the catalog reduction
(``catalog.py``), per-star body / ring conflict marking (``conflicts.py``), the raw-DN
photometry diagnostic and B-V colour mapping (``predicted_snr.py``), and smear-aware PSF
construction (``smeared_psf.py``). Each helper is independently testable so the per-step
assumptions can be exercised in isolation.

.. toctree::
   :maxdepth: 4

   dev_guide_navigation_models_star
   dev_guide_navigation_models_star_simulated
