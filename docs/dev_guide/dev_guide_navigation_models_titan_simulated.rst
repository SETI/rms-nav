==========================================================
Simulated Titan Navigation Model (Planned)
==========================================================

Overview
========

``NavModelTitanSimulated`` is the planned simulated-image variant of the Titan
navigation model.  It will render a Titan-class disc from operator-supplied parameters
(haze-top radius, per-filter haze profile, phase / lighting geometry) instead of from
SPICE prediction plus a haze model, then emit a single feature carrying the rendered
template.  The class is **not yet implemented**; this page reserves the documentation
slot so the toctree under :doc:`dev_guide_navigation_models_titans` mirrors
:doc:`dev_guide_navigation_models_bodies`,
:doc:`dev_guide_navigation_models_rings`, and
:doc:`dev_guide_navigation_models_stars`.

The planned class is a logical complement to :class:`~nav.nav_model.nav_model_titan.NavModelTitan`:
the catalog-driven path will need a haze-aware extractor, and the simulated path will
need a controlled-input renderer that lets a developer probe the haze-fit pipeline with
geometry whose true offset is known to the pixel.

Theory
======

The planned simulated Titan model will follow the same pattern as the body and ring
simulated models: a controlled-input renderer that paints an operator-supplied
haze-bounded disc onto an extended-FOV image plus mask, with operator-known per-pixel
geometry.  The simulated path is the calibration regime — it will let a developer
probe the haze-aware navigation pipeline with bodies whose true offset, haze altitude,
and phase are known by construction.

Restrictions and assumptions
----------------------------

To be specified when the class lands.  Anticipated constraints:

- The operator must supply a finite haze-top radius and a finite per-filter haze
  profile (or a sentinel that selects a baseline profile).
- The simulated body carries no per-image noise or PSF smearing by default; the
  operator's downstream noise-injection pipeline supplies those.
- Phase-angle handling will mirror the planned haze-aware
  :class:`~nav.nav_model.nav_model_titan.NavModelTitan` extractor's forward-vs-back
  scattering treatment so the simulated and real paths share their photometric model.

Sources of uncertainty
----------------------

The simulated Titan disc will have no measurement uncertainty by construction.
Downstream techniques' reported covariance will reflect only the correlation-curvature
CRLB at the chosen match position.

Configuration
=============

To be specified when the class lands.  Anticipated sim-params keys:

- ``name`` — body label used in metadata and the summary PNG.
- ``center_v``, ``center_u`` — pixel coordinates of the body centre.
- ``range`` — subject distance in km.
- ``haze_radius`` — the haze-top radius in km.
- ``haze_profile`` — per-filter haze altitude profile (or a sentinel selecting a
  baseline profile).
- ``illumination_angle``, ``phase_angle`` — degrees.

Implementation
==============

To be added when ``NavModelTitanSimulated`` is implemented.  The class will live at
``src/nav/nav_model/nav_model_titan_simulated.py`` and self-register via
``__init_subclass__``; like
:class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated` and
:class:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated` it will *not*
override :meth:`~nav.nav_model.nav_model.NavModel.instances_for_obs`, so the orchestrator's
autonomous registry never builds an instance during real-image runs.

Examples
========

To be added when the class lands.
