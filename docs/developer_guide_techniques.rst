=====================
Navigation Techniques
=====================

A ``NavTechnique`` consumes a subset of the per-image ``NavFeature`` set
plus a per-image ``NavContext`` and produces one
``NavTechniqueResult`` carrying a calibrated translation offset, a
``2x2`` covariance, a ``[0, 1]`` confidence, and a typed diagnostics
dataclass.  The orchestrator's ensemble combine reconciles every
technique's result into a single ``NavResult``.

This page walks the *distance-transform* pipeline that three techniques
share: ``BodyLimbNav``, ``BodyTerminatorNav``, and ``RingEdgeNav``.

DT-based fitting
================

The three DT-based techniques follow the same five-step pipeline:

.. code-block:: text

    +---------------------------+
    |  source image (extfov)    |
    +---------------------------+
                |
                | Gaussian smooth + Sobel
                v
    +---------------------------+
    |  gradient magnitude       |
    |  + gradient (g_v, g_u)    |     <-- once per image, on NavContext
    +---------------------------+
                |
                | threshold > k * noise_sigma
                | + Canny-style directional NMS
                v
    +---------------------------+
    |  thin edge mask           |
    +---------------------------+
                |
                | distance_transform_edt
                v
    +---------------------------+
    |  image_edge_dt_ext        |     <-- once per image, on NavContext
    +---------------------------+
                |
    +-----------|---------------+
    |  per-technique:           |
    |    - render polyline mask |
    |    - coarse_ncc_search    |
    |    - polarity_filter      |
    |    - lm_subpixel_refine   |
    |      (Tukey-reweighted)   |
    |    - information matrix   |
    |      -> covariance        |
    +---------------------------+

Image-side derivatives
----------------------

``nav.nav_orchestrator.image_derivatives.build_image_edge_dt`` is the
helper the orchestrator calls in ``_make_context``.  It Gaussian-smooths
the source image at ``image_gradient_sigma_px`` (default 1.2 px),
computes the Sobel-of-Gaussian gradient vector, takes the magnitude,
thresholds at ``edge_threshold_k_sigma * image_noise_sigma`` (default
``k = 4``), thins via Canny-style directional non-maximum suppression
(comparing each candidate against its two neighbours along the local
gradient direction), and finally takes the truncated distance
transform.  ``compute_image_gradient_vu`` returns the per-pixel
``(g_v, g_u)`` gradient vector image consumed by the polarity filter.

The orchestrator caches all three products on the per-image
``NavContext`` (``image_gradient_ext``, ``image_gradient_vu_ext``,
``image_edge_dt_ext``) so every DT technique reads them rather than
recomputing.

Coarse search and LM refinement
-------------------------------

``nav.nav_technique.dt_fitting`` exposes the five shared helpers:

``coarse_ncc_search``
    Cross-correlates the polyline edge mask against the image edge mask
    over a bounded integer ``(dv, du)`` search window and returns the
    integer-pixel argmax.  Both inputs are binary, so this reduces to
    overlap counting and is exact under the per-shift NCC normaliser.

``polarity_filter``
    Per-vertex gradient-direction agreement test.  At each vertex's
    current position, samples the image gradient vector and dots it with
    the model's polarity normal; vertices with ``dot <= 0`` are rejected
    and assigned an effectively-infinite residual so the Tukey biweight
    zeroes their contribution.  Strict inequality, per the design.

``tukey_biweight_weights``
    Holland-Welsch redescender.  ``w_i = (1 - (r_i / c)^2)^2`` for
    ``|r_i| <= c``, else 0.  ``c = 4.685`` gives 95 % asymptotic
    efficiency on Gaussian residuals when the residuals are scaled by
    their robust scale.

``lm_subpixel_refine``
    Levenberg-Marquardt minimisation of
    ``sum_i w_i * DT(R(theta) (vert_i - pivot) + pivot + (dv, du))^2``
    with iteratively-reweighted Tukey weights.  Damping ``lambda=1e-3``,
    multiplicative update on accept / reject, max 30 iterations,
    termination when the combined step norm
    ``sqrt(d_dv^2 + d_du^2 + (d_theta * pivot_dist)^2)`` drops below
    ``1e-3`` px.  Translation-only by default; rotation enabled with
    ``fit_rotation=True`` and a documented ``pivot_distance_px``.

``information_matrix_to_covariance``
    Returns ``pinvh(J^T diag(w) J, rtol=1e-9)`` so rank-deficient
    Jacobians (from flat ring polylines or other unobservable axes)
    propagate honestly into the per-technique covariance.  Same
    ``rtol`` as the orchestrator's ensemble combine.

Per-technique specifics
=======================

BodyLimbNav
-----------

Accepts ``LIMB_ARC`` features.  ``is_feasible`` returns True when at
least one limb arc has ``visible_arc_px >= LIMB_MIN_ARC_PX`` (30).
Per-vertex weight is ``1 / sigma_normal_per_vertex_px**2``.  Polarity
filtering is enabled — the model's geometric outward normals are
negated before being passed in (the typical bright-body case has the
image gradient pointing into the silhouette, opposite the geometric
outward).  Spurious detection fires when the converged DT RMS exceeds
the larger of ``SPURIOUS_DT_FLOOR_PX = 3`` px and
``5 * min(sigma_normal_per_vertex_px)`` or the Tukey inlier count
drops below ``SPURIOUS_MIN_INLIERS = 6``.

Confidence spec coefficients (placeholders, calibrated against the
image library):

- ``alpha0 = -1``
- ``alpha(visible_limb_arc_fraction) = 3``
- ``alpha(dt_fit_rms_px) = -1.5``
- ``alpha(visible_arc_px / 100, capped at 1) = 0.4``
- hard zero when ``at_edge`` is True

BodyTerminatorNav
-----------------

Accepts ``TERMINATOR_ARC`` features.  Same algorithm as ``BodyLimbNav``
with two terminator-specific changes: each body's per-vertex sigmas are
collapsed to a single per-body scalar (the body's mean), and the
confidence spec carries ``mean_phase_angle_factor`` and
``mean_albedo_penalty`` terms so high-albedo / low-phase scenes
collapse below the rank threshold.

RingEdgeNav
-----------

Accepts ``RING_EDGE`` features.  Polarity is intentionally disabled
because the ring catalog does not yet distinguish bright-ring vs
dark-gap edges (deferred polarity work).  When every input edge has
``is_straight_line=True`` the combined Jacobian is rank-deficient — the
along-edge axis is unobservable — and the technique reports
``is_rank_1=True`` on its diagnostics.  The returned 2x2 covariance is
the M-estimator pseudoinverse, honestly rank-deficient on flat-only
scenes; the ensemble combine fuses it with any orthogonal feature
(star, body limb, body blob) before declaring a final answer.

Logging
=======

Each technique's ``navigate`` body opens a pdslogger section
(``with self.logger.open(f'TECHNIQUE: {self.name}'):``) so the
per-image log file delimits each technique's contribution clearly.
The section logs the consumed feature count, the coarse-search
integer offset, and the converged offset / RMS / inlier count /
confidence on a single INFO line.  The pdslogger
``IMAGE_LOGGER`` is the only logger this codebase uses; the standard
library ``logging`` module is not imported.

See also
========

- :doc:`developer_guide_uncertainty` — derivation of the M-estimator
  information-matrix to covariance step that turns the LM Jacobian at
  convergence into the per-technique 2x2 (or 3x3) covariance reported
  on every ``NavTechniqueResult``.
