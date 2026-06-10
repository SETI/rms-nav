==================
Navigation Context
==================

Overview
========

:py:class:`~nav.nav_orchestrator.nav_context.NavContext` is the frozen per-image state object
the orchestrator builds once at the start of a navigation and threads through every feature
extractor and every technique.  It carries the global, feature-agnostic products that the whole
pipeline shares: the extended-FOV image array, the sensor-versus-padding mask, the saturation and
cosmic-ray masks, the robust noise sigma, the image-quality classifier verdict, the shared
image-side derivatives, and the reproducibility envelope.  Everything on it is computed without
knowing where any feature lives in the image, so a single instance can be reused by every model
and technique in the run.

The constructor is the orchestrator's private ``NavOrchestrator._make_context``, which reads the
observation, applies the per-instrument source-image filter, classifies the frame, builds the
masks, computes the derivatives, and assembles the dataclass.  The primary consumers are the
feature extractors (each samples the image array, masks, and noise sigma) and the
distance-transform techniques (each reads the shared gradient magnitude, gradient-vector image,
and edge distance transform rather than recomputing them).  Pass-2 techniques additionally read
the prior offset and covariance that pass 1 installed.

The term *extended FOV* (extfov) denotes the image canvas padded beyond the physical sensor
rectangle, so that a body or ring whose true position is slightly off the sensor still falls
inside the array the models render onto.

Theory
======

The context encodes one structural invariant: every array-shaped member is laid out on the
extended-FOV canvas, not on the physical sensor rectangle.  The image, the sensor mask, the
saturation mask, the cosmic-ray mask, the gradient magnitude, the gradient-vector image, and the
edge distance transform all share that single padded shape.  A boolean sensor mask distinguishes
real detector pixels from the zero-padded border, so any consumer that must restrict a statistic
to physical pixels has the information to do so.

The gradient-vector image stores, at each pixel, the two components of the local image gradient in
the row and column directions.  Consumers that compare a model edge's outward normal against the
observed edge direction read those two components directly; the convention is that the first
channel is the row-direction gradient and the second is the column-direction gradient.

The remaining members are inert: the noise sigma is a single robust scale estimate in the image's
native intensity units, the classifier verdict is a precomputed quality summary, and the
provenance envelope is a fixed record.  The only behavioural rule the container enforces is that
attaching a pass-1 prior produces a new instance rather than mutating the existing one, because the
prior-free pass-1 context and the prior-bearing pass-2 context must coexist within a single
navigation.

Configuration
=============

The context has no configuration block of its own.  The two per-instrument fields it carries are
resolved upstream by :py:func:`~nav.nav_orchestrator.instrument_config.instrument_settings_from_obs`
and copied in at construction; their defaults are module-level constants with no YAML override on
the context itself:

- ``fit_camera_rotation`` — bool, default ``False`` (dimensionless).  Enables 3-DoF technique fits
  that add in-plane camera rotation as a third parameter; when ``False`` techniques produce 2-DoF
  results.
- ``max_rotation_deg`` — float, default ``5.0`` degrees.  Maximum allowed rotation magnitude when
  ``fit_camera_rotation`` is ``True``; a fitted rotation outside the bound is flagged as at-edge.
  Ignored when ``fit_camera_rotation`` is ``False``.

The per-instrument YAML that supplies these two values lives under each camera block in
``config_4N0_inst_*.yaml`` (for example ``config_400_inst_coiss.yaml``); see
:doc:`dev_guide_orchestrator_instrument_config`.

Implementation
==============

Source file: ``src/nav/nav_orchestrator/nav_context.py``.  Public class
:py:class:`~nav.nav_orchestrator.nav_context.NavContext`, a frozen
:py:func:`dataclasses.dataclass` with ``eq=False``.

The required fields are :py:attr:`~nav.nav_orchestrator.nav_context.NavContext.obs` (the
observation snapshot, typed loosely to avoid an import cycle),
:py:attr:`~nav.nav_orchestrator.nav_context.NavContext.image_ext` (the extfov image after the
source-image filter), :py:attr:`~nav.nav_orchestrator.nav_context.NavContext.sensor_mask_ext`
(``True`` on real sensor pixels), :py:attr:`~nav.nav_orchestrator.nav_context.NavContext.image_noise_sigma`
(robust MAD-based noise sigma in native units),
:py:attr:`~nav.nav_orchestrator.nav_context.NavContext.saturation_mask_ext`,
:py:attr:`~nav.nav_orchestrator.nav_context.NavContext.cosmic_ray_mask_ext`,
:py:attr:`~nav.nav_orchestrator.nav_context.NavContext.image_classifier` (the
:py:class:`~nav.nav_orchestrator.image_classifier_result.NavImageClassifierResult` verdict), and
:py:attr:`~nav.nav_orchestrator.nav_context.NavContext.provenance` (the
:py:class:`~nav.nav_orchestrator.provenance.Provenance` envelope).

The shared derivative fields default to ``None`` and are populated once per navigation:
:py:attr:`~nav.nav_orchestrator.nav_context.NavContext.image_gradient_ext` (Sobel-of-Gaussian
magnitude), :py:attr:`~nav.nav_orchestrator.nav_context.NavContext.image_gradient_vu_ext` (the
``(H, W, 2)`` gradient-vector image), and
:py:attr:`~nav.nav_orchestrator.nav_context.NavContext.image_edge_dt_ext` (the signed distance
transform of the thresholded gradient).  The pass-2 prior fields
:py:attr:`~nav.nav_orchestrator.nav_context.NavContext.prior_offset_px` and
:py:attr:`~nav.nav_orchestrator.nav_context.NavContext.prior_covariance_px2` are ``None`` on pass 1.
:py:attr:`~nav.nav_orchestrator.nav_context.NavContext.pre_filter_applied` records the
:py:class:`~nav.support.filters.NavFilterSpec` applied to the source image, or ``None`` when no
filter ran.  The two per-instrument flags are
:py:attr:`~nav.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` and
:py:attr:`~nav.nav_orchestrator.nav_context.NavContext.max_rotation_deg`.

The single public method is
:py:meth:`~nav.nav_orchestrator.nav_context.NavContext.with_prior`.  It is non-mutating: it
validates the supplied offset and covariance, keeps only the top-left 2x2 translation block of the
covariance (a 3x3 rotation-aware covariance is accepted and its rotation prior discarded, because
each pass-2 technique re-derives the rotation prior from its own geometry), marks the kept
covariance read-only, and returns a fresh instance via :py:func:`dataclasses.replace`.  It raises
:py:exc:`ValueError` when the offset is not length-2, is non-finite, or the covariance is not a
finite square 2x2 / 3x3, and :py:exc:`TypeError` when the offset entries are not numeric.  The
class defines no ``__post_init__`` invariant; all validation lives in
:py:meth:`~nav.nav_orchestrator.nav_context.NavContext.with_prior`.

Examples
========

A representative scene is ``body_partial_overflow`` (Cassini ISS narrow-angle frame
``N1484593951_2_CALIB``), in which a body overflows one image edge.  Because the frame is a
calibrated-IF product, the context carries:

- :py:attr:`~nav.nav_orchestrator.nav_context.NavContext.image_ext` shaped on the extended FOV (the
  physical sensor padded by the per-instrument extfov margin), with
  :py:attr:`~nav.nav_orchestrator.nav_context.NavContext.sensor_mask_ext` ``True`` only over the
  physical detector rectangle.
- :py:attr:`~nav.nav_orchestrator.nav_context.NavContext.saturation_mask_ext` all ``False`` — the
  saturation gate is off for calibrated-IF instruments, so the mask is empty.
- :py:attr:`~nav.nav_orchestrator.nav_context.NavContext.image_gradient_ext`,
  :py:attr:`~nav.nav_orchestrator.nav_context.NavContext.image_gradient_vu_ext`, and
  :py:attr:`~nav.nav_orchestrator.nav_context.NavContext.image_edge_dt_ext` populated once, then
  read by the limb and terminator techniques.
- :py:attr:`~nav.nav_orchestrator.nav_context.NavContext.prior_offset_px` ``None`` on pass 1.

When the pass-1 ensemble produces an offset, the orchestrator calls
:py:meth:`~nav.nav_orchestrator.nav_context.NavContext.with_prior` with that offset and its 2x2
covariance, yielding a second context whose
:py:attr:`~nav.nav_orchestrator.nav_context.NavContext.prior_offset_px` holds the ``(dv, du)`` pair
and whose :py:attr:`~nav.nav_orchestrator.nav_context.NavContext.prior_covariance_px2` holds the
read-only translation block.  Every other field is shared unchanged between the two contexts.
