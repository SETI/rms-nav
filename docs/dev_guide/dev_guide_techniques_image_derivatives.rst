=================
Image Derivatives
=================

Overview
========

The image-derivatives module computes the image-side quantities every distance-transform (DT)
navigation technique samples: a gradient-magnitude image, a per-pixel gradient-vector image, and a
truncated edge distance transform. The orchestrator builds all three once per navigation and
attaches them to the per-image state object, the
:py:class:`~nav.nav_orchestrator.nav_context.NavContext`, so that however many limb, terminator, or
ring-edge techniques run, the heavy Gaussian-smooth and Sobel pass executes only once. Each
technique then samples those shared products at its own model polylines rather than recomputing
edges from the raw image.

Theory
======

All three products derive from a single smoothed-gradient pass. The source image is first convolved
with an isotropic Gaussian of standard deviation :math:`\sigma`; the Sobel operator then estimates
the partial derivatives along each axis, giving a signed gradient vector :math:`(g_v, g_u)` at
every pixel. The Gaussian is chosen near the instrument point-spread function: below it the gradient
is dominated by single-pixel noise, above it sharp limbs blur out and lose contrast.

The gradient magnitude is the Euclidean norm

.. math::
    g(v, u) = \sqrt{g_v(v, u)^2 + g_u(v, u)^2}.

Edge pixels are selected by thresholding this magnitude at a multiple of the image noise scale,

.. math::
    g(v, u) > k \cdot \sigma_{\text{noise}},

where :math:`\sigma_{\text{noise}}` is a median-absolute-deviation noise estimate and :math:`k` is
a small constant. The threshold is set high enough to keep single-pixel noise excursions out while
letting genuine limb, terminator, and ring edges through with margin to spare. The thresholding is
deliberately permissive about which edges it keeps: every edge above the threshold becomes a
candidate, and the burden of rejecting edges that point the wrong way is left to each technique's
polarity filter (see :doc:`dev_guide_techniques_dt_fitting`).

A raw magnitude threshold produces a thick ridge several pixels wide. To recover a one-pixel-wide
edge map suitable both for the coarse cross-correlation and the distance transform, the magnitude
ridge is thinned by Canny-style non-maximum suppression. The gradient direction at each candidate is
quantised into four 45-degree sectors (horizontal, vertical, and the two diagonals) using the
standard boundaries at 22.5, 67.5, 112.5, and 157.5 degrees; a candidate survives only if its
magnitude is at least as large as the two neighbours along its own gradient direction. This keeps
the full length of a smooth edge while collapsing it to single-pixel width.

The thinned binary edge map is then turned into a distance transform: at every pixel the value is
the Euclidean distance to the nearest retained edge pixel, computed exactly and then truncated at a
maximum half-width. The truncation bounds the cost a far-away vertex can contribute during the
LM step and bounds the array's working range. When no pixel survives thresholding the distance
transform falls back to a uniformly-saturated array, so downstream consumers always see a fully
defined surface rather than an undefined or empty one.

The gradient-vector image preserves sign, which the polarity filter requires: a limb seen from one
side has a gradient pointing the opposite way from a limb seen from the other side, and the sign is
what distinguishes a correct match from an anti-aligned one.

Configuration
=============

The three constants are bundled into the :py:class:`~nav.nav_orchestrator.image_derivatives.ImageDerivativesConfig`
frozen dataclass, which the orchestrator constructs from these Python module-level defaults; there
is no dedicated YAML stanza for this shared computation, so the configuration surface is just the
three default constants and the dataclass fields that mirror them.

- ``image_gradient_sigma_px`` (``DEFAULT_IMAGE_GRADIENT_SIGMA_PX``) — float, default ``1.2`` px.
  Gaussian sigma applied before the Sobel operator; raising it smooths more and blurs sharp edges,
  lowering it lets single-pixel noise into the gradient.
- ``edge_threshold_k_sigma`` (``DEFAULT_EDGE_THRESHOLD_K_SIGMA``) — float, default ``4.0``
  (dimensionless). Gradient-magnitude threshold in multiples of the noise sigma; raising it keeps
  fewer, stronger edges, lowering it admits weaker ones.
- ``dt_half_width_px`` (``DEFAULT_DT_HALF_WIDTH_PX``) — float, default ``64.0`` px. Cap on the
  truncated distance transform; pixels farther than this from any edge saturate at this value, so
  raising it lets distant vertices contribute a larger cost.

The dataclass ``__post_init__`` raises :py:exc:`ValueError` if any of the three fields is not a
finite positive number.

Implementation
==============

Source file: ``src/nav/nav_orchestrator/image_derivatives.py``. It depends on
:py:func:`scipy.ndimage.gaussian_filter` and :py:func:`scipy.ndimage.sobel` for the smoothing and
gradient pass and on :py:func:`nav.support.filters.apply_filter` with a
:py:class:`~nav.support.filters.NavFilterSpec` of kind
:py:class:`~nav.support.filters.NavFilterKind` ``DISTANCE_TRANSFORM`` to build the truncated DT.

:py:class:`~nav.nav_orchestrator.image_derivatives.ImageDerivativesConfig` carries the three tuning
fields with their defaults and validates them in ``__post_init__``.

Three public functions share one private smoothing core, ``_smooth_and_compute_gradients``, which
validates that the extended-FOV image is 2-D and finite, runs the Gaussian and the two Sobel
passes, and returns ``(gv, gu)``. The private ``_build_edge_dt_from_gradients`` forms the magnitude,
applies the threshold, thins the ridge via ``_directional_nms``, and feeds the binary mask to the
distance-transform filter.

:py:func:`~nav.nav_orchestrator.image_derivatives.build_image_edge_dt` returns the
``(gradient, edge_dt)`` pair: it validates the noise sigma, resolves the config defaults, calls the
smoothing core once, and delegates to ``_build_edge_dt_from_gradients``.
:py:func:`~nav.nav_orchestrator.image_derivatives.compute_image_gradient_vu` returns just the
signed ``(H, W, 2)`` gradient-vector image by stacking the two Sobel outputs.
:py:func:`~nav.nav_orchestrator.image_derivatives.compute_all_image_derivatives` is the entry point
the orchestrator uses: it runs the smoothing core once and returns
``(gradient, edge_dt, gradient_vu)`` together, so the gradient-vector product and the edge-DT
product share the single expensive pass. The three products land on the
:py:class:`~nav.nav_orchestrator.nav_context.NavContext` as the image gradient, the edge distance
transform, and the gradient-vector image that the DT techniques sample (see
:doc:`dev_guide_techniques_dt_fitting` for how the polarity filter consumes the gradient-vector
image).

Examples
========

Per-image cost on a worked scene. On the ``body_full_fov`` scene (Cassini NAC
``N1572105349_1_CALIB``, a fully-lit Dione of predicted diameter about 155 px centred in the frame),
the extended-FOV grid is roughly :math:`1024^2 \approx 1.05 \times 10^6` pixels. A single call to
:py:func:`~nav.nav_orchestrator.image_derivatives.compute_all_image_derivatives` runs one Gaussian
smooth at :math:`\sigma = 1.2` px, two Sobel passes, one magnitude evaluation, one directional
non-maximum-suppression pass that touches each pixel and its eight neighbours, and one exact
distance transform truncated at 64 px — each a single linear sweep over the million-pixel grid.
Because the result is attached to the :py:class:`~nav.nav_orchestrator.nav_context.NavContext`, both
the body-disc and body-limb techniques that run on this image read the same edge DT and gradient
vectors without repeating any of that work.

Threshold behaviour with the noise scale. The edge mask keeps a pixel only when its smoothed
gradient magnitude exceeds :math:`4.0 \times \sigma_{\text{noise}}`. On Dione's bright limb the
gradient magnitude is many noise-sigmas above background, so the limb survives both the threshold
and the non-maximum-suppression thinning as a clean one-pixel-wide arc; the interior of the lit disc
and the dark sky, where the gradient is at the noise floor, fall below the threshold and contribute
the saturated 64 px DT value, exactly the bounded cost a vertex in a featureless region should see.
