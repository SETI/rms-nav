=======================
Ring Annulus Navigation
=======================

Overview
========

This technique exploits the entire bright annular band of a ring system when the
image is too low-resolution to resolve individual ring edges as separate
polylines.  Instead of fitting edges, it correlates a full synthetic template of
the predicted ring brightness against the image and recovers the translation at
the correlation peak.  Each detectable ring system contributes one template;
multi-planet scenes paint all templates into a single composite by a depth
ordering so the closer ring system's pixels overwrite the farther one's.
Feasibility passes when at least one ring-annulus feature carries a rendered
template payload; it fails when no such feature is present.

Theory
======

When a ring system spans only a few pixels radially, the per-edge gradients
merge into one broad bright band whose internal structure is unresolved.  The
right observable is no longer the position of any one edge but the registration
of the whole brightness pattern.  The technique predicts that pattern as a
template image with an accompanying validity mask and seeks the translation that
maximises the masked normalised cross-correlation between template and image:

.. math::

   (\Delta v, \Delta u) = \arg\max_{(\delta v, \delta u)}
       \operatorname{NCC}\big(I,\; T(\delta v, \delta u)\big),

where :math:`I` is the image, :math:`T` is the masked ring template, and the
correlation is evaluated over the search window set by the pointing-error
envelope.  A coarse-to-fine image pyramid localises the peak, and a
Fourier-domain upsampling step refines it to sub-pixel accuracy.  The correlator
self-selects between raw-intensity and gradient-domain correlation per image:
raw correlation wins on broad smooth brightness gradients (a uniformly dim ring
at low resolution), gradient correlation wins when sharp ringlet edges dominate
the band.

Multi-planet composites tighten the fit the same way multi-body composites do:
each annulus contributes its own translational constraint to the joint peak, the
fixed geometric offset between the two ring systems removes the ambiguity of
swapping their identities, and if the backgrounds are independent the
signal-to-noise of the combined peak grows roughly as :math:`\sqrt{N}` for
:math:`N` annuli.

The reported covariance is derived from the curvature of the correlation surface
about the peak and captures the translational localisation uncertainty only.
The translation correlation surface carries no rotation information, so when a
camera rotation is requested the result is reported as rotation-unobservable:
the rotation parameter is fixed at zero with a sentinel variance large enough
that a downstream pseudo-inverse cleanly drops the rotation contribution.  A
single planar ring band is also weakly constrained along its own bright extent
when that extent is nearly featureless, in which case the correlation peak is
broad and the reported covariance widens accordingly.

Configuration
=============

This technique has no ``tuning`` block in
``src/nav/config_files/config_510_techniques.yaml``; the
``techniques.RingAnnulusNav`` stanza carries only the confidence coefficients
documented below.  The single runtime knob it consumes lives in a different
file: the Fourier upsample factor under ``offset`` in
``src/nav/config_files/config_020_offset.yaml``.

- ``correlation_fft_upsample_factor`` -- int, default ``128`` (count).  The
  Fourier-domain upsampling factor applied when refining the correlation peak to
  sub-pixel accuracy; higher resolves a finer peak position at the cost of a
  larger upsampled grid.  The technique validates that the value is a real
  non-boolean number coercible to an integer in a bounded range, raising
  :py:exc:`ValueError` otherwise, and substitutes an internal default of ``128``
  when the ``offset`` block or the key is absent.

Confidence formula
-------------------

The confidence coefficients live in the ``techniques.RingAnnulusNav`` stanza of
``config_510_techniques.yaml``.  The sigmoid baseline is ``alpha0 = -2.0`` and
hard-zero gates force confidence to zero when ``at_edge`` or ``spurious`` is
true.  See :doc:`dev_guide_techniques_confidence` for the sigmoid mathematics.

- ``ncc_peak`` -- alpha = 1.5, offset = 0, divisor = 6.0, cap at 1.0.  The
  peak-significance quality measure of the chosen correlation peak; healthy
  annulus fits report quality in the 6 to 15 range.
- ``peak_to_runner_up_ratio`` -- alpha = 0.0, offset = 0, divisor = 2.0, cap at
  1.0.  Ratio of the winning peak's quality to the runner-up's; wired into the
  formula with zero weight so a future calibration can activate it without code
  changes.
- ``annulus_count`` -- alpha = 0.4, offset = 0, divisor = 2.0, cap at 1.0.  Number
  of ring systems fused into the composite; the saturation cap of 2 reflects the
  scarcity of multi-planet ring scenes.

Implementation
==============

Source file: ``src/nav/nav_technique/nav_technique_ring_annulus.py``.  The public
class is
:py:class:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`, a
subclass of :py:class:`~nav.nav_technique.nav_technique.NavTechnique`.  Its
``accepts_feature_types`` is the single ``RING_ANNULUS`` feature type, its
``requires_prior`` is ``False`` (it runs in pass 1), and its
``confidence_attributes`` set names ``at_edge``, ``spurious``, ``ncc_peak``,
``peak_to_runner_up_ratio``, ``used_gradient``, and ``annulus_count``.

:py:meth:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.is_feasible`
reads only feature metadata -- never pixels -- and returns feasible when at least
one ``RING_ANNULUS`` feature carries both a template image and a template mask.

:py:meth:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.navigate`:

1. Filters the input features to those carrying a template payload, raising
   :py:exc:`ValueError` when none qualify (the orchestrator gates this via
   ``is_feasible`` first).
2. Composes every eligible template into one composite image and mask sized to
   the extended-FOV shape via
   :py:func:`~nav.feature.composition.compose_template_features`.
3. Sizes the search window from the observation margin via ``search_window_for_obs``,
   reads the upsample factor through the private ``_upsample_factor`` validator,
   and runs the masked correlation pyramid via
   :py:func:`~nav.support.correlate.navigate_with_pyramid_kpeaks` with the
   raw-versus-gradient mode left on ``auto``.
4. Reads the offset, the covariance, and the spurious / at-edge / quality flags
   off the correlator result; when rotation is fit it promotes the ``(2, 2)``
   covariance to the rank-deficient ``(3, 3)`` form via
   ``embed_rotation_unobservable``.
5. Computes the peak-to-runner-up ratio from the correlator's ranked peak list
   (via the module-private ``_peak_to_runner_up_ratio`` helper), evaluates the
   YAML confidence formula via ``evaluate_sigmoid_combination`` wrapped by a
   per-technique context adapter, and logs the per-term breakdown through
   ``log_confidence_breakdown`` (see :doc:`dev_guide_techniques_confidence`).

The result shape branches on whether rotation is fit.  With
``fit_camera_rotation`` false (the default Cassini and New Horizons LORRI
posture) the ``covariance_px2`` is ``(2, 2)`` and both ``rotation_rad`` and
``sigma_rotation_rad`` are ``None``.  With rotation true the covariance is the
rank-deficient ``(3, 3)`` form, ``rotation_rad`` is fixed at ``0.0``, and
``sigma_rotation_rad`` is the rotation-unobservable sentinel returned by
``rotation_unobservable_sigma_rad`` -- the translation correlation carries no
rotation evidence, so the rank-deficient encoding flows through the ensemble
without contaminating other techniques' rotation slots.

The only public methods are ``is_feasible`` and ``navigate``; ``_upsample_factor``
is a private validator.  Every field of
:py:class:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics` is populated:
:py:attr:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics.ncc_peak`,
:py:attr:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics.peak_to_runner_up_ratio`,
and
:py:attr:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics.annulus_count`
feed the confidence formula above, and
:py:attr:`~nav.nav_technique.diagnostics.RingAnnulusDiagnostics.used_gradient`
records whether the auto correlator selected gradient mode.  The return value is
a :py:class:`~nav.nav_technique.technique_result.NavTechniqueResult`.

Examples
========

**ring_only_curved (N1447064164_1_CALIB).** A distant Saturn ring view whose
catalog A, B, and C ring edges all collapse radially below the per-edge
annulus threshold, so the rings model emits one ``RING_ANNULUS`` feature for the
Saturn system rather than separate edge polylines.  Feasibility passes on that
single template, the technique runs one joint correlation against the composite
annulus, and the sidecar records ``primary_technique: RingAnnulusNav`` with
``status: conflicted``.

**ring_only_curved (W1444747627_1_CALIB).** A second distant Saturn ring frame in
the same class that also collapses to a single ``RING_ANNULUS`` template.  Here
the sidecar records ``primary_technique: RingAnnulusNav`` with ``status:
success`` and ``confidence_tier: low``: the single-annulus correlation localises
the offset against the operator ground truth of ``(1.5, -2.5)`` px, and the
low-but-nonzero confidence reflects the weaker constraint of a broad ring band
relative to a sharp edge fit.
