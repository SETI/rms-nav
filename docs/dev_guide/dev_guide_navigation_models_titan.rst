==========================================================
Titan Navigation Model
==========================================================

Overview
========

:class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` is the atmospheric-body
navigation model. Bodies with thick opaque atmospheres (Titan, and any other member of the
``bodies.atmospheric_bodies`` config list) need a fundamentally different algorithm than
ellipsoid-limb fitting: the visible "limb" is the haze top, the haze top varies with
wavelength, and the surface inside is invisible to optical wavelengths. At high phase such a
body is not even a circle. Ellipsoid disc / limb / terminator navigation is therefore
systematically wrong, not merely noisy, so those features are never emitted for an
atmospheric body.

The model is built and active whenever an atmospheric body is in the extended FOV: the
shape-based :class:`~spindoctor.nav_model.nav_model_body.NavModelBody` skips those bodies, and
this model takes the slot instead. It emits no features, so no technique navigates it;
instead it records, per image, *why* an atmospheric-body scene cannot be navigated. It
exposes the atmospheric body name through an ``atmospheric_body_name`` property that the
orchestrator reads: when an atmospheric body is the frame's only content the pipeline fails
with :attr:`~spindoctor.support.status_reason.NavStatusReason.ATMOSPHERIC_BODY_UNSUPPORTED`
rather than the generic no-features reason. A Titan-bearing image with other content (a
resolved moon, a ring, stars) still navigates against that content; the atmospheric body
simply contributes nothing. A haze-aware extractor would replace the no-result with a real
per-filter limb fit.

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
technique pipeline, so the model records a no-result instead of guessing.

Restrictions and assumptions
----------------------------

The model makes no algorithmic assumptions because it runs no fit. Every atmospheric body in
an extended FOV produces no feature; downstream techniques receive no shape-based or
haze-limb feature for it. When the atmospheric body is the frame's only navigable content the
orchestrator records
:attr:`~spindoctor.support.status_reason.NavStatusReason.ATMOSPHERIC_BODY_UNSUPPORTED`.

Sources of uncertainty
----------------------

The model reports no uncertainty.

Configuration
=============

The set of atmospheric bodies is the ``bodies.atmospheric_bodies`` list in
``src/spindoctor/config_files/config_040_bodies.yaml`` (Titan by default; extend it as other
thick-atmosphere bodies enter the mission set). The model records
``atmospheric_body: <NAME>`` and ``navigable: False`` on its
:attr:`~spindoctor.nav_model.nav_model.NavModel.metadata` dict so the curator surfaces the
refusal in the per-image JSON sidecar.

Implementation
==============

Source file: ``src/spindoctor/nav_model/nav_model_titan.py`` —
:class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan`.

Public class :class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan`, base
:class:`~spindoctor.nav_model.nav_model.NavModel`. Self-registers via ``__init_subclass__`` so
:func:`~spindoctor.nav_model.nav_model.build_models_for_obs` discovers it.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.instances_for_obs` — returns one
  instance per ``bodies.atmospheric_bodies`` member inside the extended FOV.
- :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.create_model` — records
  ``atmospheric_body`` / ``navigable`` metadata and logs the unsupported-navigation reason.
- :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.to_features` — returns an empty
  list.
- :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.to_annotations` — returns an empty
  :class:`~spindoctor.annotation.annotations.Annotations` collection.
- ``atmospheric_body_name`` — read-only property naming the atmospheric body; the orchestrator
  reads it to attribute an otherwise-empty frame to atmospheric-body non-support.

Examples
========

A Titan-only scene fails with status ``failed`` and status_reason
``atmospheric_body_unsupported``: the model builds, logs ``atmospheric body TITAN in FOV:
navigation not supported``, and the per-image JSON sidecar's
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.model_metadata` records
``"atmospheric_body": "TITAN"`` and ``"navigable": false`` under the model's name. A scene with
Titan plus a resolved moon or ring navigates on that other content; Titan contributes no
feature and does not force the atmospheric status.
