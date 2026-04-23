====================================================
Correlation navigation technique
====================================================

:class:`~nav.nav_technique.nav_technique_correlate_all.NavTechniqueCorrelateAll`
aligns image to model using a coarse-to-fine FFT cross-correlation
pyramid, followed by PSF-fit refinement of individual star positions.
This page documents the design choices and numerical details of both
stages.

.. contents::
   :local:
   :depth: 2


Overview
========

The technique runs in two phases:

1. **Pyramid correlation** -- finds the global image-to-model offset
   to sub-pixel precision using a multi-scale masked normalized
   cross-correlation (NCC) surface.
2. **Star position refinement** -- fits a PSF model to each star
   individually to obtain a precise per-star offset, then combines
   those offsets (with outlier rejection) into a final navigated
   offset.


Pyramid correlation
===================

Entry point: :func:`~nav.support.correlate.navigate_with_pyramid_kpeaks`.

Coarse-to-fine levels
---------------------

The image and model are successively downsampled by powers of two,
starting at ``2^(pyramid_levels-1)`` (coarsest) down to ``2^0 = 1``
(full resolution).  Before downsampling, both image and model are
blurred with a Gaussian (``sigma = scale / 2``) to prevent aliasing.

At each level, :func:`~nav.support.correlate.navigate_single_scale_kpeaks`
is called.  The coarsest level searches freely.  Every subsequent
finer level receives the previous level's result as a *prior shift*
so that the prior-penalty term in the quality score steers candidate
selection toward the coarse solution.  This prevents the finer-scale
search from wandering to entirely wrong peaks.

At levels where a prior exists, ``max_peaks`` candidates are evaluated
(rather than just one) so that the prior penalty has multiple
candidates to discriminate between.

Masked NCC surface
------------------

Entry point: :func:`~nav.support.correlate.masked_ncc`.

Given padded ``image``, ``model``, and boolean ``mask`` (all the same
shape), the function returns two surfaces:

``ncc`` -- Pearson-*r* correlation coefficient at every shift:

.. math::

   \text{NCC}(s) = \frac{\sum_{\mathbf{x}} [I({\mathbf{x}+s}) - \bar{I}(s)]\;[M(\mathbf{x}) - \bar{M}]\; W(\mathbf{x})}
                        {\sqrt{\text{SSD}_I(s) \cdot \text{SSD}_M}}

where :math:`W` is the mask, :math:`\bar{I}(s)` is the shift-wise
image mean under the mask, :math:`\bar{M}` is the constant model
mean, and the SSD terms are the corresponding sums-of-squared
deviations.  All shift-wise sums (``sum_iw``, ``sum_i2w``,
``sum_imw``) are computed in O(N log N) via FFT convolution.

``numerator`` -- the mean-subtracted cross-covariance (the NCC
numerator without the denominator normalization).  Returned as a
diagnostic only; it must **not** be used for peak localization
when the template is dense (see below).

**Why two surfaces?**  The NCC (Pearson-*r*) surface is scale
invariant and is always the correct surface for peak localization.
The numerator is related to the NCC by
:math:`\text{num}(s) = \text{NCC}(s) \cdot \sqrt{\text{SSD}_I(s)\, \text{SSD}_M}`
and therefore scales with sqrt of the image variance under the
shifted mask.  For *dense* templates -- body discs, Lambert-shaded
limbs, ring models -- that variance changes systematically with
shift because the fixed body-shaped mask straddles more or less of
the bright/dark boundary at different shifts, producing a
"scale bias" that can move the numerator's peak several pixels
away from the true alignment while the NCC peak is correct.
Peak localization in :func:`~nav.support.correlate.nms_topk`
therefore uses the NCC surface; the numerator is kept as an
internal diagnostic (e.g. for the manual-nav correlation map).

**Numerical stability.**  Epsilon is placed *inside* the square root:

.. code-block:: text

   denom = sqrt(SSD_I * SSD_M + eps)     # eps = 1e-12

This raises the denominator floor to ``sqrt(eps) ~ 1e-6``, preventing
floating-point noise in the numerator (~1e-9) from producing ``|NCC|``
values far outside [-1, 1] at zero-variance shifts (which would occur
if the floor were only ``eps = 1e-12``).

Peak selection and subpixel refinement
---------------------------------------

:func:`~nav.support.correlate.nms_topk` applies non-maximum suppression
on the NCC surface to obtain up to ``max_peaks`` candidate
integer-pixel offsets.  The search is restricted to the extended-FOV
offset range (``max_offset_vu``); this also protects against the
classic NCC failure mode where noise-only shifts produce spurious
\|NCC\| plateaus outside the physically plausible window.

Each candidate is refined to sub-pixel precision by
:func:`~nav.support.correlate.evaluate_candidate` using a localized
upsampled DFT of the correlation spectrum (Guizar-Sicairos 2008).
The PSR quality metric is also computed from the NCC surface so
that the score is scale-invariant and comparable across brightness
levels.

When a prior shift is supplied, candidates far from the prior are
penalized:

.. code-block:: text

   quality_adjusted = quality - prior_weight * distance(candidate, prior)

The highest adjusted-quality candidate is selected as the level
result.

Quality metrics
---------------

Three metrics are supported (configured via ``metric``):

* **PSR** (default) -- ``(peak - mean_sidelobe) / std_sidelobe``
  computed on the NCC surface, excluding a guard-band around the peak.
  Threshold ``quality_thresh`` (default 6.0) gates the final result.
* **PMR** -- ``peak / mean(surface)``.
* **PER** -- ``peak / sqrt(sum(surface^2))``.

Final pass and spurious check
------------------------------

After all pyramid levels, the finest-level prior is passed to a full-
resolution ``navigate_single_scale_kpeaks`` call that evaluates
``max_peaks`` candidates at full resolution.  The result is marked
``spurious`` when either:

* ``quality < quality_thresh``, or
* ``consistency > consistency_tol``, where *consistency* is the
  maximum full-resolution-scaled deviation between level estimates.


Star position refinement
========================

After correlation, each star in the model is refined independently by
fitting a PSF model to a small sub-image centered on the predicted
star location.  The fit is performed by
``psfmodel.PSF.find_position``, which returns the optimized (U, V)
position and a set of quality metrics.

Rejection criteria
------------------

A star is rejected (and marked with a ``conflicts`` flag) when any of
the following conditions is true.  All threshold values are
configurable (see :doc:`developer_guide_configuration`).

+----------------------------+----------------------------------+-----------------------------+
| Conflict flag              | Condition                        | Meaning                     |
+============================+==================================+=============================+
| ``REFINEMENT EDGE``        | Predicted position too close to  | Sub-image would extend      |
|                            | image boundary.                  | outside the frame.          |
+----------------------------+----------------------------------+-----------------------------+
| ``REFINEMENT FAILED``      | ``find_position`` returns        | Optimizer did not           |
|                            | ``None``.                        | converge.                   |
+----------------------------+----------------------------------+-----------------------------+
| ``REFINEMENT NO STAR``     | ``scale <= noise_rms``           | No real signal above the    |
|                            |                                  | noise floor.                |
+----------------------------+----------------------------------+-----------------------------+
| ``REFINEMENT CHI2 LOW``    | ``reduced_chi2 < min_chi2``      | Background polynomial       |
|                            | **and** ``peak_snr < min_snr``   | absorbed the star. A bright |
|                            |                                  | star (high SNR) with low    |
|                            |                                  | chi2 is a good fit, not     |
|                            |                                  | overfitting, so the chi2    |
|                            |                                  | floor is only enforced when |
|                            |                                  | SNR is also low.            |
+----------------------------+----------------------------------+-----------------------------+
| ``REFINEMENT CHI2``        | ``reduced_chi2 > max_chi2``      | PSF model cannot explain    |
|                            |                                  | the data; likely a          |
|                            |                                  | non-stellar source.         |
+----------------------------+----------------------------------+-----------------------------+
| ``REFINEMENT SNR``         | ``peak_snr < min_snr``           | Star too faint to localize  |
|                            |                                  | reliably.                   |
+----------------------------+----------------------------------+-----------------------------+
| ``REFINEMENT POS_ERR``     | ``x_err > max_pos_err`` or       | Fit covariance is too large |
|                            | ``y_err > max_pos_err``          | for a useful position.      |
+----------------------------+----------------------------------+-----------------------------+

Note on ``reduced_chi2``
~~~~~~~~~~~~~~~~~~~~~~~~

``reduced_chi2`` from ``psfmodel`` is computed as ``RSS / DOF``
where ``RSS`` is the sum of squared residuals and ``DOF = n_valid_pixels
- n_params``.  This has units of (pixel value)^2 and is *not*
dimensionless.  For calibrated I/F images with noise_rms ~ 1e-3,
``reduced_chi2`` is typically ~ 1e-6 for a high-SNR star, which is far
below the default ``min_chi2 = 0.10`` floor.  The combined condition
``reduced_chi2 < min_chi2 and peak_snr < min_snr`` avoids incorrectly
rejecting bright, well-fit stars on these grounds.

Combining star offsets
----------------------

The per-star U and V position differences are combined using the
robust median (with MAD-based scatter) after iterative outlier
rejection at ``star_refinement_nsigma`` sigma.  Stars are weighted
by a function of their V magnitude to reduce the influence of
fainter, less reliable measurements.
