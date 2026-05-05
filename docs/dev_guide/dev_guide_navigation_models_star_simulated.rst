==========================================================
Simulated Star Navigation Model
==========================================================

Overview
========

``NavModelStarsSimulated`` is the simulated-image counterpart of
:class:`~nav.nav_model.stars.nav_model_stars.NavModelStars`. It is reserved without an
implementation; the documentation slot exists so the toctree under
:doc:`dev_guide_navigation_models_stars` is parallel with
:doc:`dev_guide_navigation_models_bodies` and
:doc:`dev_guide_navigation_models_rings`.

Star generation in the simulated-image driver (``nav_create_simulated_image``) is handled
by :mod:`nav.sim.render`, which paints operator-supplied stars directly into the
simulated image. The per-image
:class:`~nav.obs.obs_inst_sim.ObsInstSim` snapshot carries the operator's star list on
``sim_star_list`` and the catalog-driven
:class:`~nav.nav_model.stars.nav_model_stars.NavModelStars` runs against the simulated
observation the same way it runs against a real one. A direct simulated-image
``NavModelStarsSimulated`` would consume the operator parameters without round-tripping
through the catalog reduction.

Theory
======

The simulated path is the calibration regime: a developer can probe the star-matching
pipeline with a star field whose true offset, photometry, and smear are known by
construction. The simulated-image counterpart would follow the same pattern as
:class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated` and
:class:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`: a
controlled-input renderer that paints operator-supplied stars onto an extended-FOV image
plus mask, with operator-known per-star geometry.

Restrictions and assumptions
----------------------------

The slot has no implementation, so no algorithmic assumptions apply. A future
implementation would inherit the constraints already documented for
:class:`~nav.nav_model.stars.nav_model_stars.NavModelStars` plus the
operator-supplied-parameter constraints documented for
:doc:`dev_guide_navigation_models_body_simulated`.

Sources of uncertainty
----------------------

The slot reports no uncertainty.

Configuration
=============

The slot consumes no YAML configuration. Background-star generation knobs already used by
:mod:`nav.sim.render` (``background_stars_num``, ``background_stars_psf_sigma``,
``background_stars_distribution_exponent``) live under the ``sim`` configuration block.

Implementation
==============

The slot has no source file. A direct simulated-image counterpart would live at
``src/nav/nav_model/stars/nav_model_stars_simulated.py`` and self-register via
``__init_subclass__``; like
:class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated` and
:class:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated` it would not
override :meth:`~nav.nav_model.nav_model.NavModel.instances_for_obs`, so the orchestrator's
autonomous registry would not build an instance during real-image runs.

Examples
========

The slot has no examples.
