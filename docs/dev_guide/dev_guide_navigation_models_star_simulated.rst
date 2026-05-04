==========================================================
Simulated Star Navigation Model (Planned)
==========================================================

Overview
========

``NavModelStarsSimulated`` is the planned simulated-image variant of the star navigation
model.  It will render stars from operator-supplied parameters (per-star pixel position,
magnitude, spectral class, optional smear vector) instead of from a catalog plus SPICE
prediction, then emit one
:data:`~nav.feature.feature_type.NavFeatureType.STAR` :class:`~nav.feature.feature.NavFeature`
per operator-supplied star.  The class is **not yet implemented**; this page reserves the
documentation slot so the toctree under :doc:`dev_guide_navigation_models_stars` mirrors
:doc:`dev_guide_navigation_models_bodies` and :doc:`dev_guide_navigation_models_rings`.

Today the simulated-image driver (``nav_create_simulated_image``) renders operator-supplied
stars directly into the simulated image via :mod:`nav.sim.render`; the per-image
:class:`~nav.obs.obs_inst_sim.ObsInstSim` snapshot carries the operator's star list on
``sim_star_list``.  Downstream, the catalog-driven
:class:`~nav.nav_model.stars.nav_model_stars.NavModelStars` then runs against the simulated
observation the same way it runs against a real one.  The planned
``NavModelStarsSimulated`` will replace that indirection with a direct simulated-image
:class:`~nav.nav_model.nav_model.NavModel` subclass that consumes the operator parameters
without round-tripping through the catalog reduction.

Theory
======

The planned simulated star model will follow the same pattern as the body and ring
simulated models: a controlled-input renderer that paints operator-supplied stars onto an
extended-FOV image plus mask, with operator-known per-star geometry.  The simulated path
is the calibration regime — a developer can probe the star-matching pipeline with a star
field whose true offset, photometry, and smear are known to the pixel.

Restrictions and assumptions
----------------------------

To be specified when the class lands.  Anticipated constraints:

- Operator-supplied stars must carry finite pixel positions inside the extended FOV.
- The simulated stars carry no per-image noise or PSF smearing by default; the operator's
  downstream noise-injection pipeline supplies those.
- Per-star photometric calibration follows the same per-instrument bandpass model used by
  :class:`~nav.nav_model.stars.nav_model_stars.NavModelStars`.

Sources of uncertainty
----------------------

The simulated stars will have no measurement uncertainty by construction.  Downstream
techniques' reported covariance will reflect only the per-star CRLB at the chosen match
position.

Configuration
=============

To be specified when the class lands.  Anticipated sim-params keys:

- ``stars`` — list of dicts each carrying ``center_v``, ``center_u``, ``vmag``,
  ``spectral_class``, optional ``smear_v`` / ``smear_u``.
- Background-star generation knobs already consumed by :mod:`nav.sim.render`
  (``background_stars_num``, ``background_stars_psf_sigma``,
  ``background_stars_distribution_exponent``).

Implementation
==============

To be added when ``NavModelStarsSimulated`` is implemented.  The class will live at
``src/nav/nav_model/stars/nav_model_stars_simulated.py`` and self-register via
``__init_subclass__``; like
:class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated` and
:class:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated` it will *not*
override :meth:`~nav.nav_model.nav_model.NavModel.instances_for_obs`, so the orchestrator's
autonomous registry never builds an instance during real-image runs.

Examples
========

To be added when the class lands.
