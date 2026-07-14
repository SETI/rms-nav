==========================================================
Titan Navigation Model
==========================================================

Overview
========

:class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` is the Titan navigation model.
Titan needs a fundamentally different algorithm than ellipsoid-limb fitting: the visible
"limb" is the haze top, the haze top varies with wavelength, and the surface inside is
invisible to optical wavelengths. At high phase Titan is not even a circle. Ellipsoid disc /
limb / terminator navigation is therefore systematically wrong, not merely noisy, so those
features are never emitted for Titan.

Titan is handled as a deliberate special case, not as one entry of a general atmospheric-body
list. Its atmosphere is unique (transparent at some wavelengths), so what is true for Titan
does not carry over to other thick-atmosphere bodies; Titan is therefore the single hardcoded
special case, named by the ``TITAN_BODY_NAME`` constant in
:mod:`spindoctor.nav_model.nav_model_body`.

The model is built and active whenever Titan is in the extended FOV: the shape-based
:class:`~spindoctor.nav_model.nav_model_body.NavModelBody` skips Titan, and this model takes the
slot instead. It emits no features, so no technique navigates it; instead it records, per
image, *why* a Titan scene cannot be navigated. It exposes a ``titan_in_fov`` property that the
orchestrator reads: when Titan is the frame's only content the pipeline fails with
:attr:`~spindoctor.support.status_reason.NavStatusReason.TITAN_UNSUPPORTED` rather than the
generic no-features reason. A Titan-bearing image with other content (a resolved moon, a ring,
stars) still navigates against that content; Titan simply contributes nothing. A haze-aware
extractor would replace the no-result with a real per-filter limb fit.

Theory
======

Titan navigation is conceptually distinct from ellipsoid-limb fitting:

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

The model makes no algorithmic assumptions because it runs no fit. Titan in an extended FOV
produces no feature; downstream techniques receive no shape-based or haze-limb feature for it.
When Titan is the frame's only navigable content the orchestrator records
:attr:`~spindoctor.support.status_reason.NavStatusReason.TITAN_UNSUPPORTED`.

Sources of uncertainty
----------------------

The model reports no uncertainty.

Configuration
=============

Titan is the single hardcoded special case, named by the ``TITAN_BODY_NAME`` constant in
:mod:`spindoctor.nav_model.nav_model_body`; there is no config list. Titan's atmosphere is
unique (transparent at some wavelengths), so it is handled as a special case rather than the
first entry of a general atmospheric-body list. The model records ``body: TITAN`` and
``navigable: False`` on its :attr:`~spindoctor.nav_model.nav_model.NavModel.metadata` dict so
the curator surfaces the refusal in the per-image JSON sidecar.

Implementation
==============

Source file: ``src/spindoctor/nav_model/nav_model_titan.py`` —
:class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan`.

Public class :class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan`, base
:class:`~spindoctor.nav_model.nav_model.NavModel`. Self-registers via ``__init_subclass__`` so
:func:`~spindoctor.nav_model.nav_model.build_models_for_obs` discovers it.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.instances_for_obs` — returns one
  instance when Titan is inside the extended FOV, else none.
- :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.create_model` — records
  ``body`` / ``navigable`` metadata and logs the unsupported-navigation reason.
- :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.to_features` — returns an empty
  list.
- :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.to_annotations` — returns an empty
  :class:`~spindoctor.annotation.annotations.Annotations` collection.
- ``titan_in_fov`` — read-only bool property, always ``True``; the orchestrator reads it to
  attribute an otherwise-empty frame to Titan non-support.

Examples
========

A Titan-only scene fails with status ``failed`` and status_reason
``titan_unsupported``: the model builds, logs ``Titan in FOV: navigation not supported``, and
the per-image JSON sidecar's
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.model_metadata` records
``"body": "TITAN"`` and ``"navigable": false`` under the model's name. A scene with Titan plus a
resolved moon or ring navigates on that other content; Titan contributes no feature and does
not force the Titan status.
