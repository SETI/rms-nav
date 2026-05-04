==========================================================
Titan Navigation Model (Placeholder)
==========================================================

Overview
========

:class:`~nav.nav_model.nav_model_titan.NavModelTitan` is a registered placeholder for
atmospheric-body navigation.  Bodies with thick opaque atmospheres (Titan, Venus, Triton's
nitrogen frost layer at low phase) need a fundamentally different algorithm than ellipsoid-
limb fitting: the visible "limb" is the haze top, the haze top varies with wavelength, and
the surface inside is invisible to optical wavelengths.  The model exists in
``NavModel._registry`` so the orchestrator's
:func:`~nav.nav_model.nav_model.build_models_for_obs` driver and the per-image curator know
the placeholder is here, but it never emits features or annotations.

When a Titan-bearing image arrives the orchestrator constructs an instance (when a future
extractor overrides
:meth:`~nav.nav_model.nav_model.NavModel.instances_for_obs`), calls
:meth:`~nav.nav_model.nav_model_titan.NavModelTitan.create_model` (a no-op), calls
:meth:`~nav.nav_model.nav_model_titan.NavModelTitan.to_features` (returns an empty list),
and the technique pipeline runs without any Titan-derived feature.  The image still
navigates against any other body, ring, or star in the FOV.

Theory
======

Atmospheric-body navigation is conceptually distinct from ellipsoid-limb fitting:

- The optical limb is the haze top, not the solid surface.  The haze top moves with
  wavelength (deep red sees a slightly lower altitude than blue), so a single ellipsoidal
  reference shape does not describe every filter.
- Phase angle changes the apparent limb shape because forward-scattered haze brightens
  the near-edge differently from back-scattered haze on the far edge.
- The haze altitude varies with latitude, season, and (on Titan) the year-by-year
  atmospheric circulation; a fixed reference radius is wrong by tens of kilometres
  depending on epoch.

The corresponding algorithm requires a per-filter haze profile, a phase-aware limb-fit cost
function, and per-image refraction modelling.  None of those components is in the current
pipeline; the placeholder reserves the registry slot for the future implementation.

Restrictions and assumptions
----------------------------

The placeholder makes no algorithmic assumptions because it runs no algorithm.  Every
Titan-class body that lands in an extended FOV produces no feature; downstream techniques
do not see any TITAN_LIMB feature today.

Sources of uncertainty
----------------------

The placeholder reports no uncertainty.

Configuration
=============

The placeholder consumes no YAML configuration of its own.  The shared
:class:`~nav.config.config.Config` object is passed in for future use; the placeholder
records ``stub: True`` on its
:attr:`~nav.nav_model.nav_model.NavModel.metadata` dict so the curator surfaces the slot in
the per-image JSON sidecar.

Implementation
==============

Source file: ``src/nav/nav_model/nav_model_titan.py`` —
:class:`~nav.nav_model.nav_model_titan.NavModelTitan`.

Public class :class:`~nav.nav_model.nav_model_titan.NavModelTitan`, base
:class:`~nav.nav_model.nav_model.NavModel`.  Self-registers via ``__init_subclass__`` so
:func:`~nav.nav_model.nav_model.build_models_for_obs` discovers it.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~nav.nav_model.nav_model_titan.NavModelTitan.create_model` — no-op; records
  ``stub: True`` on :attr:`~nav.nav_model.nav_model.NavModel.metadata`.
- :meth:`~nav.nav_model.nav_model_titan.NavModelTitan.to_features` — returns an empty
  list.
- :meth:`~nav.nav_model.nav_model_titan.NavModelTitan.to_annotations` — returns an empty
  :class:`~nav.annotation.annotations.Annotations` collection.

The class does not override
:meth:`~nav.nav_model.nav_model.NavModel.instances_for_obs`; the registry registers the
class but the default base-class implementation returns an empty list, so today the
orchestrator never constructs an instance.  When a haze-aware Titan extractor lands, that
extractor will override
:meth:`~nav.nav_model.nav_model.NavModel.instances_for_obs` to return one instance per
visible Titan-class body.

Examples
========

The Titan placeholder produces no per-image effect today.  A Titan-class scene proceeds
through the orchestrator as if no Titan-bearing model had registered: the
:meth:`~nav.nav_model.nav_model_titan.NavModelTitan.to_features` return value is empty,
:meth:`~nav.nav_model.nav_model_titan.NavModelTitan.to_annotations` return is empty, and
the per-image JSON sidecar's
:attr:`~nav.nav_orchestrator.nav_result.NavResult.model_metadata` contains ``"stub": true``
under the model's name as a record that the slot is reserved.
