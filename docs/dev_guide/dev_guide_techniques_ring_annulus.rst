==========================================================
Ring Annulus Correlate (RingAnnulusNav)
==========================================================

Overview
========

:class:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav` recovers a single
translation by full-template normalised cross-correlation against a composite annulus
fused from every offered ``RING_ANNULUS`` feature. Per-planet annulus templates are Z-buffer
painted into a single postage stamp (the closer ring system's pixels overwrite the farther
one's), the result is run through the shared pyramid-NCC machinery in
:mod:`nav.support.correlate`, and the chosen peak is returned with a Cramer-Rao-lower-bound
covariance derived from the local correlation curvature.

Multi-planet annulus composites improve disambiguation in the same way as multi-body disc
composites: each annulus contributes its own translational constraint, the geometric
alignment between ring systems removes the "swap planet assignments" ambiguity, and the SNR
of the combined peak grows roughly as :math:`\sqrt{N}` if backgrounds are independent.
Multi-planet scenes are rare but real (the Cassini approach phase imaged Jupiter and Saturn
together; New Horizons imaged Jupiter from Pluto distance).

Feasibility passes when at least one offered ``RING_ANNULUS`` carries a template payload;
feasibility fails when no offered feature has a template (a ring system whose per-edge
polylines were emitted as ``RING_EDGE`` features instead).

Theory
======

The technique fits a per-image translation by maximising the normalised cross-correlation
between the composite annulus template and the observed image.

Composite template construction
-------------------------------

When more than one planet emits a ``RING_ANNULUS`` feature (rare but supported), the per-planet
templates are fused via Z-buffer paint: at each pixel covered by more than one ring system,
the closer planet's template value (and its mask True) overwrite the farther one's so an
in-front ring system occludes an in-behind ring system in the composite. The fused
template carries one combined bounding box, one fused brightness image, and one fused mask;
the orchestrator's
:func:`~nav.feature.composition.compose_template_features` helper does the work.

Cost function
-------------

The technique maximises the masked normalised cross-correlation between the composite
template and the observed image (or a mode-selected gradient of it):

.. math::

    \mathrm{NCC}(\Delta v, \Delta u) =
        \frac{\langle T, I_{\Delta v, \Delta u} \rangle_{M}}
             {\sqrt{\langle T, T \rangle_{M} \cdot \langle I_{\Delta v, \Delta u},
                                          I_{\Delta v, \Delta u} \rangle_{M}}}

over the integer offsets in the per-instrument search window, with the masked inner product
restricted to pixels where the annulus mask is True. Sub-pixel refinement comes from a
quadratic fit to the correlation surface around the integer peak; the fitted curvature
provides the CRLB covariance.

Mode selection (auto / raw / gradient)
--------------------------------------

The shared pyramid evaluates the correlation in two modes — raw vs. raw and gradient vs.
gradient — and ``auto`` picks whichever peak scores higher quality. Raw wins on
broad-brightness-gradient ring geometries (a low-resolution Saturn ring system where the
C-ring is uniformly dim and the A/B rings dominate the per-pixel intensity); gradient wins
when sharp ringlet edges dominate the per-pixel signal.

Search strategy
---------------

The shared pyramid-NCC entry point
(:func:`~nav.support.correlate.navigate_with_pyramid_kpeaks`) runs coarse-to-fine, keeps the
top ``k`` peaks at each level, and reports the per-level consistency. See
:doc:`dev_guide_techniques_dt_fitting` for the per-iteration mechanics that the body-disc
technique shares. Rotation fitting is disabled for ring annuli (the rotation pivot
of a ring system is its planet-centre, which is well outside the rendered template's
support; an outer rotation search would have to rotate the template about a far-off-template
pivot and the NCC peak's rotation curvature is degenerate for symmetric annuli).

Restrictions and assumptions
----------------------------

- The orchestrator must populate
  :attr:`~nav.nav_orchestrator.nav_context.NavContext.image_ext` and
  :attr:`~nav.nav_orchestrator.nav_context.NavContext.sensor_mask_ext`.
- The composite template must have non-empty support in the search window. An empty
  template (every annulus off-frame) collapses the NCC to a constant; the spurious gate
  flags the result.
- The upstream rings model decides whether to emit ``RING_ANNULUS`` or ``RING_EDGE`` features per
  the per-planet ``feature_emission.ring_annulus`` block in
  ``config_510_techniques.yaml`` — the technique only sees the path the model chose.

Sources of uncertainty
----------------------

The reported covariance is the Cramer-Rao lower bound from the local NCC curvature at the
chosen peak. When the chosen peak sits within the at-edge tolerance of any axis bound the
result is flagged
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` and the hard-zero
gate forces confidence to zero. The pyramid consistency check flags peaks that drift across
levels as :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.spurious`.

Configuration
=============

The technique itself has no per-technique ``tuning`` knobs in
``config_510_techniques.yaml`` (the shared pyramid-NCC machinery owns the numeric thresholds
for peak quality, consistency, and top-k count).

Feature-emission tunables (per-planet)
--------------------------------------

The upstream rings model decides whether to emit ``RING_ANNULUS`` or ``RING_EDGE`` features based on
``feature_emission.ring_annulus`` in ``src/nav/config_files/config_510_techniques.yaml``.
The model emits a ``RING_ANNULUS`` template whenever a single ring edge has compressed below the
per-polyline radial-pixel threshold, or when the per-planet km-per-pixel scale exceeds the
per-planet threshold (the entire ring system is below the per-edge resolution limit).

- ``feature_emission.ring_annulus.default.max_radial_px`` — float, default ``5.0`` px.
  Maximum per-edge polyline radial extent below which the rings model emits a ``RING_ANNULUS``
  template instead of per-edge polylines. Per-polyline gate; fires when a single ring edge
  has compressed below this width.
- ``feature_emission.ring_annulus.default.kmpp_threshold`` — float, default ``1000.0`` km/px.
  Ring-radial km/px threshold above which the entire ring system is annulus-class
  regardless of any per-polyline metric.
- ``feature_emission.ring_annulus.planets.SATURN.max_radial_px`` — float, default
  ``5.0`` px.
- ``feature_emission.ring_annulus.planets.SATURN.kmpp_threshold`` — float, default
  ``1000.0`` km/px. Saturn's main rings span ~75,000-140,000 km; at km/px > 1000 the
  entire system is < 65 px wide and individual edges are sub-pixel apart.
- ``feature_emission.ring_annulus.planets.JUPITER.max_radial_px`` — float, default
  ``5.0`` px.
- ``feature_emission.ring_annulus.planets.JUPITER.kmpp_threshold`` — float, default
  ``200.0`` km/px. Jupiter's main ring is ~10x narrower than Saturn's, so the kmpp
  threshold is correspondingly tighter.
- ``feature_emission.ring_annulus.planets.URANUS.max_radial_px`` — float, default
  ``5.0`` px.
- ``feature_emission.ring_annulus.planets.URANUS.kmpp_threshold`` — float, default
  ``300.0`` km/px.
- ``feature_emission.ring_annulus.planets.NEPTUNE.max_radial_px`` — float, default
  ``5.0`` px.
- ``feature_emission.ring_annulus.planets.NEPTUNE.kmpp_threshold`` — float, default
  ``500.0`` km/px.

Per-instrument overrides
------------------------

Per-instrument YAML files in ``src/nav/config_files/config_4N0_inst_*.yaml`` do not
override any ``feature_emission`` keys. The search-window margin used by the
at-edge test comes from the per-instrument
:class:`~nav.nav_orchestrator.instrument_config.InstrumentSettings`.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared
sigmoid combination; see :doc:`dev_guide_techniques_confidence`. The formula spec is
``techniques.RingAnnulusNav`` in the same YAML file and consumes attributes off
:class:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics` plus
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.spurious`.

- :attr:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics.ncc_peak` — alpha = 1.5,
  offset = 0.0, divisor = 6.0, cap at 1.0. PSR-style quality measure of the chosen NCC
  peak. Healthy annulus fits report quality 6 to 15.
- :attr:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics.peak_to_runner_up_ratio` —
  alpha = 0.0, offset = 0.0, divisor = 2.0, cap at 1.0. Ratio of the winning peak's
  quality to the next-best peak's outside the exclusion radius. Carries no weight in
  the configured confidence formula; the wiring is in place so a downstream
  recalibration can tune the alpha.
- :attr:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics.annulus_count` —
  alpha = 0.4, offset = 0.0, divisor = 2.0, cap at 1.0. Number of ``RING_ANNULUS`` features
  fused. Multi-planet scenes saturate at 2 (vs ``body_count``'s 3).

Hard-zero gate: :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.spurious` either firing forces
confidence to zero. The constant baseline is :math:`\alpha_{0} = -2.0`. No post-sigmoid
``hard_cap`` is applied.

Implementation
==============

Source files:

- ``src/nav/nav_technique/nav_technique_ring_annulus.py`` —
  :class:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav` and the per-feature
  filter / composite helpers.
- ``src/nav/feature/composition.py`` —
  :func:`~nav.feature.composition.compose_template_features`.
- ``src/nav/support/correlate.py`` —
  :func:`~nav.support.correlate.navigate_with_pyramid_kpeaks`, the shared pyramid-NCC entry
  point.
- ``src/nav/nav_technique/confidence.py`` — sigmoid-combination evaluator; documented at
  :doc:`dev_guide_techniques_confidence`.
- ``src/nav/nav_technique/diagnostics.py`` —
  :class:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics`; documented at
  :doc:`dev_guide_techniques_diagnostics`.

Public class :class:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`, base
:class:`~nav.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__``.

Class attributes:

- :attr:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.name` —
  ``'RingAnnulusNav'``.
- :attr:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.accepts_feature_types`
  — ``frozenset({RING_ANNULUS})``.
- :attr:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.requires_prior` —
  ``False``.
- :attr:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.confidence_attributes`
  — ``{'at_edge', 'spurious', 'ncc_peak', 'peak_to_runner_up_ratio', 'used_gradient',
  'annulus_count'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.is_feasible` and
:meth:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.navigate`.

Diagnostics
-----------

:class:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics`:

- :attr:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics.ncc_peak` — peak NCC
  quality. Consumed by the confidence formula.
- :attr:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics.peak_to_runner_up_ratio` —
  ratio of winning peak's quality to next-best. Consumed by the confidence formula.
- :attr:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics.annulus_count` — number of
  ``RING_ANNULUS`` features fused.
- :attr:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics.used_gradient` — True when
  ``auto`` mode picked the gradient pass.

Call path
---------

Call path traced through
:meth:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.navigate`:

1. Open a logged section. Filter the offered features to ``RING_ANNULUS`` entries that carry a
   template payload via the private filter helper.
2. Fuse the per-planet templates via
   :func:`~nav.feature.composition.compose_template_features` into a single composite.
3. Read the search-window margin off the observation via
   :func:`~nav.nav_technique.nav_technique.search_window_for_obs`.
4. Run :func:`~nav.support.correlate.navigate_with_pyramid_kpeaks` on the composite
   template against the extfov image. The pyramid returns the chosen peak's
   ``(dv, du)``, the 2x2 CRLB covariance, ``quality``, ``consistency``, ``spurious``,
   ``at_edge``, ``used_gradient``, and the top-``k`` peak telemetry.
5. The covariance shape is always (2, 2); rotation fitting is disabled for ring
   annuli. When
   :attr:`~nav.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true the
   technique embeds the (2, 2) translation block in a (3, 3) covariance via
   :func:`~nav.nav_technique.nav_technique.embed_rotation_unobservable` and reports the
   rotation as unobservable.
6. Apply the at-edge tests against the search-window axis bounds.
7. Build a :class:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics`, evaluate the
   confidence spec via :func:`~nav.nav_technique.confidence.evaluate_sigmoid_combination`,
   log the breakdown, and assemble the
   :class:`~nav.nav_technique.technique_result.NavTechniqueResult`.

The :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.feature_ids` field
preserves every consumed
:attr:`~nav.feature.feature.NavFeature.feature_id` so the orchestrator's curator can
attribute each contribution at audit time.

Examples
========

``ring_only_curved`` (Cassini ISS NAC, image ``N1447064164_1``)
    A high-resolution Saturn-ring scene whose individual ring edges resolve into separable
    polylines. On this scene the rings model emits ``RING_EDGE`` features rather than
    ``RING_ANNULUS``, so :class:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`
    is not feasible. The annulus path fires on lower-resolution / approach-phase scenes
    where the per-planet km/px exceeds the configured threshold.

A multi-planet illustration: a hypothetical fly-by image that catches both Jupiter's main
ring and Saturn's ring system (compressed below their per-planet kmpp thresholds) emits two
``RING_ANNULUS`` features. The technique Z-buffer paints them into a composite (the closer
planet's annulus overwriting the farther one's) and the pyramid NCC's joint geometric
constraint produces a sharper peak than either annulus alone. The
:attr:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics.annulus_count` term in the
confidence formula contributes a positive offset.
