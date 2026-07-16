==========================================================
Simulated Titan Navigation Model
==========================================================

Overview
========

``NavModelTitanSimulated`` is the simulated-image counterpart of
:class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan`. It is reserved without an
implementation; the documentation slot exists so Titan's simulated sibling parallels
:doc:`dev_guide_navigation_models_body_simulated`,
:doc:`dev_guide_navigation_models_ring_simulated`, and
:doc:`dev_guide_navigation_models_star_simulated`.

A direct simulated-image counterpart is the logical complement of
:class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan`: the catalog-driven path needs a
haze-aware extractor, and the simulated path needs a controlled-input renderer so a
developer can probe the haze-fit pipeline with geometry whose true offset is known by
construction.

Theory
======

The simulated path is the calibration regime: a controlled-input renderer paints an
operator-supplied haze-bounded disc onto an extended-FOV image plus mask, with
operator-known per-pixel geometry. The renderer would let a developer probe the
haze-aware navigation pipeline with bodies whose true offset, haze altitude, and phase
are known by construction.

Restrictions and assumptions
----------------------------

The slot has no implementation, so no algorithmic assumptions apply. A future
implementation would inherit the constraints already documented for
:class:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated` plus the
haze-profile parameter constraints described in
:doc:`dev_guide_navigation_models_titan`.

Sources of uncertainty
----------------------

The slot reports no uncertainty.

Configuration
=============

The slot consumes no YAML configuration. The anticipated sim-params parallel to
:class:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated` would carry
``name``, ``center_v``, ``center_u``, ``range``, ``haze_radius``, ``haze_profile``,
``illumination_angle``, and ``phase_angle``.

Implementation
==============

The slot has no source file. A direct simulated-image counterpart would live at
``src/spindoctor/nav_model/nav_model_titan_simulated.py`` and self-register via
``__init_subclass__``; like
:class:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated` and
:class:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated` it would
override :meth:`~spindoctor.nav_model.nav_model.NavModel.instances_for_obs` to decline real
observations, so the orchestrator's autonomous registry would not build an instance
during real-image runs.

Examples
========

The slot has no examples.
