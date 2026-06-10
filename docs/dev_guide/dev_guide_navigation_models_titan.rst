=======================
Titan Navigation Model
=======================

Overview
========

The Titan navigation model is a registered placeholder for atmospheric-body navigation.  A body
with a thick opaque atmosphere does not present an ellipsoidal hard limb: the visible edge is the
top of the haze, it shifts with wavelength, and the solid surface beneath is invisible.  Fitting
such a body requires a haze-aware limb algorithm with per-filter haze profiles, which is not part
of the current pipeline.  The model therefore predicts no image feature and emits nothing.

The model registers with the orchestrator's auto-discovery registry like any other concrete
navigation model, but it inherits the empty default
:py:meth:`~nav.nav_model.nav_model.NavModel.instances_for_obs` and so is never auto-instantiated
from an observation.  When constructed and run it produces zero features and zero annotations, so
the orchestrator falls through to the other models — stars and rings — on a Titan scene.

Theory
======

There is no algorithm to describe.  The model is an inert placeholder: it implements the required
hooks so the registry stays uniform, but it carries no rendering, no geometry, and no uncertainty
model.  The conceptual reason it is empty is that an opaque-atmosphere limb is photometrically and
geometrically different from an ellipsoidal hard limb, and the haze-aware fit it would need is
unimplemented.

Configuration
=============

The Titan model consumes no configuration in the predicted-scene sense.  A ``titan`` section
exists in ``src/nav/config_files/config_060_titan.yaml`` with a single key,
``atmosphere_height`` (default ``700`` km), reserved for the future haze-aware algorithm; the
current model does not read it.

Implementation
==============

Source file: ``src/nav/nav_model/nav_model_titan.py``.  The public class is
:py:class:`~nav.nav_model.nav_model_titan.NavModelTitan`, a subclass of
:py:class:`~nav.nav_model.nav_model.NavModel`.

:py:meth:`~nav.nav_model.nav_model_titan.NavModelTitan.create_model` records a single stub marker
in metadata and has no other state to populate.
:py:meth:`~nav.nav_model.nav_model_titan.NavModelTitan.to_features` returns an empty list, so the
model emits no :py:class:`~nav.feature.feature_type.NavFeatureType` values.
:py:meth:`~nav.nav_model.nav_model_titan.NavModelTitan.to_annotations` returns an empty
:py:class:`~nav.annotation.annotations.Annotations` collection.

Examples
========

No scene in ``tests/integration/image_library/images/`` is navigated by the Titan model.  On any
observation that includes Titan, the model contributes nothing and the orchestrator relies on the
star and ring models for the offset; see :doc:`dev_guide_navigation_models_stars` and
:doc:`dev_guide_navigation_models_rings` for the features that carry a Titan scene.
