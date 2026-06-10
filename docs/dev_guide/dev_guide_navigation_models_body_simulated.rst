===============================
Simulated Body Navigation Model
===============================

Overview
========

The simulated body navigation model renders a body from operator-supplied geometric parameters
rather than from a SPICE prediction, and emits the rendered disc as a single feature the standard
pipeline can navigate against.  It exists so the simulated-image tooling can compose synthetic
test scenes with a body at a known location, shape, and lighting; the same disc-template feature
the catalog-driven model produces for a real image is produced here from a parameter dictionary.

This model is not auto-instantiated from an observation.  Unlike the catalog-driven
:py:class:`~nav.nav_model.nav_model_body.NavModelBody` described in
:doc:`dev_guide_navigation_models_body`, it inherits the empty default
:py:meth:`~nav.nav_model.nav_model.NavModel.instances_for_obs` and is constructed directly by the
caller that supplies the simulation parameters.  Each constructed instance emits exactly one
``BODY_DISC`` feature.

Theory
======

The body is modelled as a triaxial ellipsoid rendered into the image plane from a small set of
operator-supplied quantities: a centre in pixel coordinates, three semi-axis lengths, a rotation
about the line of sight, a tilt, an illumination angle, and a phase angle.  The renderer shades
the ellipsoid by a Lambert law given the illumination and phase geometry, producing a smooth
brightness profile that falls to zero at the silhouette boundary.  The body mask is the set of
pixels with positive brightness, and the limb is the subset of body pixels adjacent to empty
space.

Because the body is rendered from exact parameters rather than measured from a prediction, there
is no uncertainty model: the rendered disc is treated as a noise-free template with unit
reliability and zero overflow.  Anti-aliasing of the silhouette boundary is always maximal.  The
purpose is to produce a controlled, repeatable scene, so there is no covariance, no phase-bias
correction, and no emission gating to describe beyond the single template emission.

Configuration
=============

This model takes no YAML configuration of its own.  Its geometry comes entirely from the
``sim_params`` dictionary passed to the constructor, whose recognised keys are the body name, the
centre coordinates (``center_v``, ``center_u``), the subject range (``range``), the three
ellipsoid semi-axes (``axis1``, ``axis2``, ``axis3``), the rotation about the line of sight
(``rotation_z``), the body tilt (``rotation_tilt``), the illumination angle
(``illumination_angle``), and the phase angle (``phase_angle``).  Crater and anti-aliasing keys
are accepted but ignored.  The shared label-placement keys it uses for annotations come from the
``bodies`` section documented in :doc:`dev_guide_navigation_models_body`.

Implementation
==============

Source file: ``src/nav/nav_model/nav_model_body_simulated.py``.  The public class is
:py:class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated`, which extends
:py:class:`~nav.nav_model.nav_model_body_base.NavModelBodyBase` and therefore shares the limb-mask
and label-annotation helpers with the catalog-driven body model.

:py:meth:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated.create_model` records
timing metadata and calls the private render path.  The render reads the simulation parameters,
calls the simulated-body renderer to produce a Lambert-shaded ellipsoid image, derives the body
mask as the positive-brightness pixels and the limb mask via the inherited helper, promotes both
into extfov-shaped buffers, and records the predicted centre, subject range, and silhouette
bounding box.

:py:meth:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated.to_features` crops the
rendered image and mask to the bounding box and emits one
:py:class:`~nav.feature.feature.NavFeature` of type
:py:attr:`~nav.feature.feature_type.NavFeatureType.BODY_DISC`, carrying a
:py:class:`~nav.feature.geometry.BodyDiscGeometry` with zero overflow, unit reliability, a
:py:class:`~nav.feature.feature.NavReliabilityBreakdown` with full visible-lit fraction, and
:py:class:`~nav.feature.flags.BodyDiscFlags`.  The single emitted
:py:class:`~nav.feature.feature_type.NavFeatureType` is ``BODY_DISC``.

:py:meth:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated.to_annotations` reuses the
base ``_create_annotations`` helper to draw the body silhouette and label onto the summary image.

Examples
========

The simulated body model does not draw from the real-image corpus under
``tests/integration/image_library/images/``; those scenes are navigated by the catalog-driven
model.  A representative simulated invocation constructs the model with a parameter dictionary
such as a Mimas-like ellipsoid at the frame centre with equal semi-axes, a modest tilt, and a
phase angle of about thirty degrees.
:py:meth:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated.create_model` then renders
the Lambert disc, and
:py:meth:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated.to_features` emits a single
``BODY_DISC`` feature whose template is the rendered postage stamp.  For a worked navigation
against a real body disc, see the ``body_full_fov`` example in
:doc:`dev_guide_navigation_models_body`.
