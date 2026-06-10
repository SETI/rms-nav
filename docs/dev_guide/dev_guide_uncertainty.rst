===========
Uncertainty
===========

This page derives the M-estimator covariance returned by the DT-based
techniques from first principles and shows where the project-wide
``pinvh`` cutoff comes from.

The cost function
=================

For a polyline of ``N`` vertices ``x_i`` with prior precision
``w_i^prior = 1 / sigma_i^2`` and a parameter vector ``p`` that
shifts and (optionally) rotates the polyline, the DT cost is

.. math::

    C(p) = \sum_i w_i \, r_i(p)^2

where the residual ``r_i(p) = DT(R(theta)(x_i - x_p) + x_p + (dv, du))``
is the bilinearly-sampled distance transform at the shifted vertex
position and ``w_i = w_i^prior \cdot w_i^\mathrm{Tukey}`` combines the
prior precision with the iteratively-reweighted Tukey biweight weight.

At the converged estimate :math:`\hat p`, the gradient

.. math::

    \nabla C(\hat p) = 2 \sum_i w_i \, r_i(\hat p) \, \nabla r_i(\hat p)

vanishes and the local Hessian reduces (under the Gauss-Newton
approximation, dropping the second-order ``r_i \, \nabla^2 r_i``
term that integrates to zero against zero-mean residuals) to

.. math::

    H \approx 2 \sum_i w_i \, \nabla r_i(\hat p) \, \nabla r_i(\hat p)^T
        = 2 \, J^T \, W \, J

with ``J`` the residual Jacobian (rows are
:math:`\nabla r_i(\hat p)^T`) and ``W = diag(w_i)``.

Under the M-estimator interpretation the parameter covariance is the
inverse of the *information matrix* :math:`I = J^T W J` (the factor of
2 cancels in the standard derivation against the residual variance
that the Tukey weights have already absorbed); the dropped
``2 r \nabla^2 r`` term is exactly the variance contribution that
becomes negligible at convergence under low-residual conditions.

The pseudoinverse cutoff
========================

When the model polyline is geometrically rank-deficient (e.g. every
edge in a flat-ring scene is parallel and the along-ring axis is
unobservable), :math:`I` is singular.  Replacing the inverse with the
Hermitian pseudoinverse :math:`I^+` lets the unobservable direction
remain unobservable in the returned covariance: the corresponding
eigenvalue is zero, the marginal variance along that axis is also
zero in :math:`I^+`, and the per-axis sigma reported by the ensemble
correctly flags the rank deficiency rather than silently inverting
floating-point noise.

``scipy.linalg.pinvh(I, rtol=rcond)`` is the implementation.  The
``rtol`` cutoff is set to ``1e-9`` project-wide — the same value the
orchestrator's ensemble combine uses.  A tighter cutoff would silently
treat near-rank-deficient matrices as full-rank, producing garbage
inverse entries; a looser cutoff would prematurely zero observable
directions.  ``1e-9`` is liberal enough to fold near-singular
directions into the null space and conservative enough to preserve
genuine 2-D / 3-D constraints typical of body-limb / ring-edge fits.

Tukey biweight reweighting
==========================

The Holland-Welsch biweight assigns weight

.. math::

    w_i^\mathrm{Tukey} = \left(1 - (r_i / c)^2\right)^2 \quad |r_i| \le c

and zero otherwise, with ``c = 4.685`` selected so the M-estimator
attains 95 % asymptotic efficiency on Gaussian residuals once the
residuals are pre-scaled by their robust scale (the per-vertex
``sigma_i`` in our formulation).  Recomputing the weights after each
accepted LM step (iteratively-reweighted least squares) makes the
Tukey-rejected vertices truly drop out of the information matrix and
keeps the converged covariance an honest expression of the surviving
inliers' precision.

Combining the M-estimator covariance with a prior
=================================================

The per-technique covariance returned to the orchestrator is the
M-estimator pseudoinverse :math:`I^+`.  The orchestrator's ensemble
combines per-technique results in *information form*: invert each
covariance, sum, invert again.  ``scipy.linalg.pinvh`` is used at
every inversion and the same ``rtol`` propagates throughout, so a
single rank-1 ring constraint plus any other rank-2 result yields a
fully-resolved 2-D answer; a single rank-1 ring constraint alone
yields a rank-1 final covariance with the unobservable axis honestly
flagged.
