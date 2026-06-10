======================
Confidence Calibration
======================

Overview
========

Every navigation technique turns its typed diagnostics into a single calibrated confidence value in
:math:`[0, 1]` using one shared formula: a logistic sigmoid of a linear combination of normalised
diagnostic features, gated by optional hard-zero conditions and an optional hard cap. The shape is
identical across techniques; only the coefficients differ, and those coefficients live in YAML so a
technique's calibration can change without touching code. This page documents the evaluator and the
loader that builds a formula spec from configuration; the per-technique coefficient tables live on
the individual technique pages.

Theory
======

A technique reports a small set of scalar diagnostics — peak heights, residual RMS values, inlier
counts, arc fractions — and the confidence formula maps them to a probability-like score. Each
diagnostic :math:`x_i` is first normalised by a per-feature affine transform followed by an optional
clamp:

.. math::
    \tilde{x}_i = \mathrm{clamp}\!\left( \frac{x_i - o_i}{d_i},\; 0,\; \kappa_i \right),

where :math:`o_i` is an offset subtracted before scaling, :math:`d_i` is a non-zero divisor, and
:math:`\kappa_i` is an optional upper cap (when no cap is set the lower clamp at zero is also
skipped). The normalised features enter a linear combination with a constant baseline and per-term
weights, and the sum is squashed by the logistic sigmoid:

.. math::
    s = \alpha_0 + \sum_i \alpha_i\, \tilde{x}_i, \qquad
    \mathrm{confidence} = \sigma(s) = \frac{1}{1 + e^{-s}}.

The sigmoid maps the unbounded linear score into :math:`(0, 1)`; a baseline :math:`\alpha_0` of
zero places a featureless result at confidence 0.5, negative baselines make a technique
conservative, and each weight :math:`\alpha_i` pushes confidence up or down as its diagnostic
improves. A term whose weight is zero is inert: its diagnostic is computed and normalised but
contributes nothing, which lets the wiring exist ahead of being weighted.

Two override mechanisms sit outside the smooth part. A set of hard-zero gates maps boolean
diagnostic attributes to expected values; if any gate's attribute equals its expected value the
formula short-circuits and returns confidence zero regardless of the linear score. This is how a
result flagged spurious or at the edge of its search window is forced to zero rather than merely
discounted. After the sigmoid, an optional hard cap clamps the result down to a ceiling, bounding
how confident a technique is ever allowed to be.

The evaluator can additionally emit a per-step trace recording the sigmoid argument, every term's
raw value, its normalised value, its weight, its signed contribution, which hard-zero gate fired (if
any), and whether the cap was applied. The trace exists so an operator can read off exactly why a
given image scored low or zero. The formula has no internal uncertainty model: it is a
deterministic, calibrated mapping from diagnostics to a score, and its quality depends entirely on
the coefficients supplied for each technique.

Configuration
=============

The coefficients live in ``src/nav/config_files/config_510_techniques.yaml`` under
``techniques.<TechniqueName>``, alongside the per-technique ``tuning`` block. Each technique block
supplies the confidence-formula fields; the loader validates them at config-load time so a malformed
file fails fast at startup rather than mid-image. The recognised per-technique keys are:

- ``alpha0`` — float, required (dimensionless). Constant baseline in the sigmoid argument; raising
  it raises every result's confidence floor.
- ``terms`` — list, default empty. One mapping per linear term; each mapping carries ``feature``
  (the diagnostic attribute name), ``alpha`` (the weight), and optional ``offset``, ``divisor``,
  and ``cap_at`` normalisation keys.
- ``hard_zero_if`` — mapping, default empty. Diagnostic-attribute name to expected boolean; any
  matching condition forces confidence to zero.
- ``hard_cap`` — float or ``null``, default ``null`` (dimensionless, in ``[0, 1]``). Post-sigmoid
  ceiling; when set, the result is clamped down to this value.

The same block also carries a ``tuning`` sub-block of technique-specific runtime numbers, loaded
separately. The four keys above plus ``tuning`` are the only keys a technique block may contain; any
other key is rejected at load time. Because the coefficients vary per technique, the concrete values
are documented in a "Confidence formula" subsection on each technique's own page rather than here.

Implementation
==============

Source files: ``src/nav/nav_technique/confidence.py`` (the formula types and evaluator) and
``src/nav/nav_technique/confidence_config.py`` (the YAML loader).

The formula is described by three frozen dataclasses.
:py:class:`~nav.nav_technique.confidence.ConfidenceTerm` is one linear term, with fields ``feature``,
``alpha``, ``offset``, ``divisor``, and ``cap_at``; its ``__post_init__`` enforces a non-empty
feature name, finite numeric coefficients, a non-zero divisor, and a cap in :math:`[0, 1]`.
:py:class:`~nav.nav_technique.confidence.ConfidenceSpec` is the whole formula, with fields
``alpha0``, ``terms``, ``hard_zero_if``, and ``hard_cap``; its ``__post_init__`` validates types,
defensively copies the gate mapping, and range-checks the cap.

:py:func:`~nav.nav_technique.confidence.evaluate_sigmoid_combination` is the evaluator. It first
walks ``hard_zero_if``, raising :py:exc:`ValueError` if a named attribute is absent from the
diagnostics object and returning zero the moment a gate matches; otherwise it sums ``alpha0`` and
each term's contribution, applying the private ``_normalize`` (offset, divisor, optional clamp) per
term, squashes the sum with the private numerically-stable ``_sigmoid``, and applies the hard cap.
When ``return_breakdown`` is True it also builds a
:py:class:`~nav.nav_technique.confidence.ConfidenceBreakdown` containing the final ``confidence``,
the ``sigmoid_arg``, the ``alpha0`` baseline, a tuple of
:py:class:`~nav.nav_technique.confidence.ConfidenceTermContribution` records (each with ``feature``,
``raw``, ``normalized``, ``alpha``, and ``contribution``), the ``hard_zero`` attribute name that
fired or ``None``, and the ``hard_cap_applied`` flag.

The loader module exposes :py:func:`~nav.nav_technique.confidence_config.load_confidence_spec`, which
reads ``techniques[technique_name]``, rejects unknown block keys, requires ``alpha0``, builds each
term through the private ``_build_term`` helper, validates the gate mapping, and returns a frozen
:py:class:`~nav.nav_technique.confidence.ConfidenceSpec`; any shape or type error raises
:py:exc:`~nav.nav_technique.confidence_config.ConfidenceConfigError`, which names the offending
technique. :py:func:`~nav.nav_technique.confidence_config.load_technique_tuning` reads the same
block's ``tuning`` sub-block and returns a flat ``{key: number}`` mapping with booleans rejected.
The private ``_require_mapping`` and ``_require_finite_float`` helpers back both loaders.

Examples
========

Worked evaluation on the ``one_bright_star_no_body`` scene (Cassini WAC ``W1449079117_1_CALIB``, a
single bright star, Vega, in an otherwise empty FOV). The star-refine technique is the primary on
this image and reports a low confidence tier. The formula for that technique combines the number of
stars used, the median positional error, and the residual scatter into the sigmoid argument; with
only one star available, the inlier-count term cannot accumulate, so the linear score sits low and
the sigmoid returns a small but non-zero confidence — the "low" tier the scene's sidecar records.
No hard-zero gate fires, because the single-star refinement is a valid (if weakly constrained) fit
rather than a spurious one.

Hard-zero short-circuit. On the ``body_partial_overflow`` scene (Cassini NAC
``N1484593951_2_CALIB``), the body-terminator technique returns a degenerate fit (0 of 895 inliers,
no LM iteration). Its spec's hard-zero gate, keyed on the spurious flag, matches, so
:py:func:`~nav.nav_technique.confidence.evaluate_sigmoid_combination` returns confidence zero
immediately, with the breakdown's ``hard_zero`` field naming the gate that fired — independent of
whatever the linear terms would have scored. The body-limb technique on the same image is not gated,
so its sigmoid score stands and it becomes the orchestrator's primary.
