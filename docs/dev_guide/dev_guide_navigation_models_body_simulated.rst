==========================================================
Simulated Body Navigation Model
==========================================================

Overview
========

:class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated` is the
simulated-image variant of the body navigation model.  It renders a body from
operator-supplied ellipsoid parameters (centre, axes, rotation, lighting) instead of from
SPICE prediction, then emits a single
:data:`~nav.feature.feature_type.NavFeatureType.BODY_DISC` feature carrying the rendered
template.  The simulated GUI driver constructs an instance directly with the operator's
sim parameters; the orchestrator's autonomous registry never builds an instance because the
class does not override
:meth:`~nav.nav_model.nav_model.NavModel.instances_for_obs`.

Theory
======

Simulated body rendering is a controlled-input version of the same silhouette-extraction
pipeline that drives :class:`~nav.nav_model.nav_model_body.NavModelBody`.  The operator
specifies a body in image-plane coordinates (centre and per-axis radii) plus a phase /
lighting geometry, and the renderer paints the corresponding ellipsoidal body onto an
extended-FOV image plus matching mask.

The rendered template is the BODY_DISC feature payload that downstream techniques
(:class:`~nav.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav` is the primary
consumer) navigate against.  The simulated body's geometry is operator-known by
construction, so the simulated path is the calibration regime — it lets a developer probe
the navigation pipeline with bodies whose true offset is known to the pixel.

Restrictions and assumptions
----------------------------

- The operator must supply finite, positive ellipsoid axes.  Degenerate inputs (zero
  radius, negative axes) are rejected by
  ``create_simulated_body``.
- Crater and anti-aliasing keys in the sim-params dict are accepted but ignored; the
  simulated renderer always uses maximum anti-aliasing.
- The simulated body is rendered onto a fixed extfov image without per-instrument noise
  or PSF smearing; the operator's downstream noise-injection pipeline supplies those.

Sources of uncertainty
----------------------

The simulated body has no measurement uncertainty by construction; the rendered template
is an exact ellipsoid.  The downstream technique's reported covariance reflects only the
correlation-curvature CRLB at the chosen NCC peak.

Configuration
=============

The simulated body model consumes no YAML configuration of its own; every parameter comes
in via the per-instance ``sim_params`` dict.  Expected keys:

- ``name`` — body label used in metadata and the summary PNG.
- ``center_v``, ``center_u`` — pixel coordinates of the body centre.
- ``range`` — subject distance in km (defaults to ``+inf``).
- ``axis1``, ``axis2``, ``axis3`` — ellipsoid semi-axes in km.  ``axis3`` defaults to
  ``min(axis1, axis2)``.
- ``rotation_z`` — rotation about the line of sight (degrees).
- ``rotation_tilt`` — tilt of the body (degrees).
- ``illumination_angle`` — degrees.
- ``phase_angle`` — degrees.

Crater and anti-aliasing keys are accepted but ignored.

Implementation
==============

Source file: ``src/nav/nav_model/nav_model_body_simulated.py`` —
:class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated`.

Public class :class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated`, base
:class:`~nav.nav_model.nav_model_body_base.NavModelBodyBase`.  The class does *not*
override :meth:`~nav.nav_model.nav_model.NavModel.instances_for_obs`, so the orchestrator's
:func:`~nav.nav_model.nav_model.build_models_for_obs` driver never constructs an instance
during autonomous runs.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated.create_model` —
  invokes ``create_simulated_body`` to render the simulated body
  image, then computes the limb mask via
  :class:`~nav.nav_model.nav_model_body_base.NavModelBodyBase`'s helper.
- :meth:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated.to_features` — emits
  a single :data:`~nav.feature.feature_type.NavFeatureType.BODY_DISC` feature carrying the
  rendered template plus mask.
- :meth:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated.to_annotations` —
  reuses the shared body annotation helper on
  :class:`~nav.nav_model.nav_model_body_base.NavModelBodyBase` to render body silhouette
  and labels onto the summary PNG.

Inherited :class:`~nav.nav_model.nav_model.NavModel` properties:
:attr:`~nav.nav_model.nav_model.NavModel.name`,
:attr:`~nav.nav_model.nav_model.NavModel.obs`,
:attr:`~nav.nav_model.nav_model.NavModel.metadata`.

Call path
---------

Call path traced through
:meth:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated.create_model`:

1. Open a logged section.  Read the operator-supplied sim parameters off the per-instance
   dict.
2. Convert per-axis rotations and angle parameters from degrees to radians.
3. Call ``create_simulated_body`` with the per-axis radii and
   geometry; the helper returns the rendered simulated body image.
4. Derive the body mask from the rendered image (every non-zero pixel is on the body).
5. Compute the limb mask via
   :class:`~nav.nav_model.nav_model_body_base.NavModelBodyBase`'s shared discrete-mask
   neighbour-shift helper.
6. Promote the rendered image and the masks from sensor-shaped arrays to extfov-shaped
   arrays (zero-padded for the extfov margin).
7. Record the predicted centre, the subject range, and the bounding box on the model's
   internal state for downstream feature emission.

Call path traced through
:meth:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated.to_features`:

1. Crop the rendered template image and mask to the per-instance bounding box.
2. Construct one
   :data:`~nav.feature.feature_type.NavFeatureType.BODY_DISC`
   :class:`~nav.feature.feature.NavFeature` carrying the cropped template image, the
   cropped mask, the predicted centre, the subject range, and a
   :class:`~nav.feature.flags.BodyDiscFlags` with the operator-supplied body name plus
   ``overflow_fov_fraction = 0.0``.
3. Reliability is fixed at ``1.0`` (the simulated body is by construction reliable;
   downstream gates do not drop it).

Examples
========

The simulated body model is consumed by the simulated-image GUI driver
(``nav_create_simulated_image``).  An operator specifies a body — say a Mimas-like
ellipsoid centred at ``(512, 512)`` with semi-axes ``200`` km, illumination angle ``60``
degrees, phase angle ``30`` degrees — and the simulator renders the corresponding
extended-FOV image plus mask.  The downstream
:class:`~nav.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav` correlates the
template against an injected synthetic-noise image and recovers the operator-known
``(0, 0)`` offset (or whatever offset the operator injected) within sub-pixel.  The
operator uses the residual to validate per-instrument plate-scale and PSF assumptions
without a real Cassini observation.
