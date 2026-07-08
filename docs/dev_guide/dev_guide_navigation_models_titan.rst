==========================================================
Titan Navigation Model
==========================================================

Overview
========

:class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` is a registered placeholder for
atmospheric-body navigation. Bodies with thick opaque atmospheres (Titan, Venus, Triton's
nitrogen frost layer at low phase) need a fundamentally different algorithm than ellipsoid-
limb fitting: the visible "limb" is the haze top, the haze top varies with wavelength, and
the surface inside is invisible to optical wavelengths. The model exists in
``NavModel._registry`` so the orchestrator's
:func:`~spindoctor.nav_model.nav_model.build_models_for_obs` driver and the per-image curator know
the placeholder is here, but it never emits features or annotations.

A Titan-bearing image proceeds through the orchestrator as though no Titan-aware model had
registered: the placeholder's :meth:`~spindoctor.nav_model.nav_model.NavModel.instances_for_obs`
inherits the base-class no-op, so the orchestrator constructs no instance and the technique
pipeline sees no Titan-derived feature. The image still navigates against any other body,
ring, or star in the FOV. A haze-aware extractor would override
:meth:`~spindoctor.nav_model.nav_model.NavModel.instances_for_obs` to opt in.

Theory
======

Atmospheric-body navigation is conceptually distinct from ellipsoid-limb fitting:

- The optical limb is the haze top, not the solid surface. The haze top moves with
  wavelength (deep red sees a slightly lower altitude than blue), so a single ellipsoidal
  reference shape does not describe every filter.
- Phase angle changes the apparent limb shape because forward-scattered haze brightens
  the near-edge differently from back-scattered haze on the far edge.
- The haze altitude varies with latitude, season, and (on Titan) the year-by-year
  atmospheric circulation; a fixed reference radius is wrong by tens of kilometres
  depending on epoch.

The corresponding algorithm requires a per-filter haze profile, a phase-aware limb-fit cost
function, and per-image refraction modelling. None of those components are wired into the
technique pipeline; the placeholder reserves the registry slot for a haze-aware extractor.

Restrictions and assumptions
----------------------------

The placeholder makes no algorithmic assumptions because it runs no algorithm. Every
Titan-class body in an extended FOV produces no feature; downstream techniques receive
no TITAN_LIMB feature.

Sources of uncertainty
----------------------

The placeholder reports no uncertainty.

Configuration
=============

The placeholder consumes no YAML configuration of its own. The shared
:class:`~spindoctor.config.config.Config` object is passed in for future use; the placeholder
records ``stub: True`` on its
:attr:`~spindoctor.nav_model.nav_model.NavModel.metadata` dict so the curator surfaces the slot in
the per-image JSON sidecar.

Implementation
==============

Source file: ``src/spindoctor/nav_model/nav_model_titan.py`` —
:class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan`.

Public class :class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan`, base
:class:`~spindoctor.nav_model.nav_model.NavModel`. Self-registers via ``__init_subclass__`` so
:func:`~spindoctor.nav_model.nav_model.build_models_for_obs` discovers it.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.create_model` — no-op; records
  ``stub: True`` on :attr:`~spindoctor.nav_model.nav_model.NavModel.metadata`.
- :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.to_features` — returns an empty
  list.
- :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.to_annotations` — returns an empty
  :class:`~spindoctor.annotation.annotations.Annotations` collection.

The class does not override
:meth:`~spindoctor.nav_model.nav_model.NavModel.instances_for_obs`; the registry registers the
class but the default base-class implementation returns an empty list, so the
orchestrator never constructs an instance. A haze-aware extractor would override
:meth:`~spindoctor.nav_model.nav_model.NavModel.instances_for_obs` to return one instance per
visible Titan-class body.

Examples
========

The Titan placeholder produces no per-image effect. A Titan-class scene proceeds
through the orchestrator as if no Titan-bearing model had registered: the
:meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.to_features` return value is empty,
:meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.to_annotations` return is empty, and
the per-image JSON sidecar's
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.model_metadata` contains ``"stub": true``
under the model's name as a record that the slot is reserved.
