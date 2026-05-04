==========================================================
Manual Navigation (NavTechniqueManual)
==========================================================

Overview
========

:class:`~nav.nav_technique.nav_technique_manual.NavTechniqueManual` is the interactive
navigation technique.  It renders every NavModel's predicted scene into a single composite
overlay (templates, polylines, blob outlines, star markers), opens the
``ManualNavDialog`` PyQt6 widget, and packages the operator's chosen ``(dv, du)`` offset
into a :class:`~nav.nav_technique.technique_result.NavTechniqueResult`.

Manual navigation is not part of the autonomous pipeline — the class sets
``_abstract = True`` so it never appears in
:class:`~nav.nav_technique.nav_technique.NavTechnique`'s discovery registry and the
orchestrator's background driver does not invoke the dialog during automated runs.  An
interactive driver calls
:func:`~nav.nav_technique.nav_technique_manual.run_manual_nav` directly when an operator
wants to navigate an image by hand or override an autonomous result.

Feasibility passes whenever the operator opens the dialog (the technique reports feasible
on any feature set, since manual navigation looks at whatever the scene has rendered);
infeasibility paths exist only as defensive errors when the dialog cannot be opened.

Theory
======

Manual navigation is not algorithmic — there is no cost function, no convergence criterion,
and no search.  The technique renders the predicted scene, the operator inspects it
overlaid on the observed image, and the operator picks an offset that visually matches.

Composite overlay
-----------------

Every renderable feature kind in the scene contributes one element to the overlay:

- Template-bearing features (BODY_DISC, RING_ANNULUS, CARTOGRAPHIC_MODEL) paint their
  predicted brightness images into the composite.
- Polyline-bearing features (LIMB_ARC, TERMINATOR_ARC, RING_EDGE) draw their per-vertex
  polylines as one-pixel-wide strokes.
- BODY_BLOB features draw a circle outline at the predicted-vu position sized by the
  predicted disc diameter.
- STAR features draw a rectangle outline at the predicted-vu position sized by the
  per-feature point-spread-function bounding box.

The composite is a single extended-FOV image plus mask the dialog overlays on top of the
observed image.  The operator drags or types the offset until the overlay aligns with the
observed scene.

Auto-pick option
----------------

The dialog includes an ``Auto`` button that runs the same masked-NCC pyramid the autonomous
correlation-based techniques use; the operator can accept the auto-pick verbatim or
override it.  The auto-pick is a convenience — the technique itself remains operator-driven.

Restrictions and assumptions
----------------------------

- Manual navigation requires PyQt6 and an interactive display.  Background driver runs that
  cannot open a window must not invoke this technique.
- The auto-pick path consults the same NCC pyramid the
  :class:`~nav.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav` and
  :class:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav` techniques use;
  when the scene has no template-bearing feature the auto-pick is unavailable and only the
  manual drag / type path remains.
- Operator precision in the dialog is limited by zoom, eye, and screen pixels; the
  technique reports a per-axis 1 px sigma on the resulting covariance regardless of how the
  operator picked.

Sources of uncertainty
----------------------

The reported covariance is fixed at ``1.0`` px sigma per axis on a diagonal 2x2.  This is a
conservative placeholder — the ensemble never combines a manual result with autonomous
results, so the value only determines what shows up in the per-image JSON metadata.

Configuration
=============

:class:`~nav.nav_technique.nav_technique_manual.NavTechniqueManual` carries no per-technique
``tuning`` block in ``config_510_techniques.yaml`` and no
:attr:`~nav.nav_technique.nav_technique.NavTechnique.confidence_spec`.  The technique opts
out of the autonomous registry (sets ``_abstract = True``) so the config-load validation
walk skips it.  The single Python module-level constant is ``_MANUAL_OFFSET_SIGMA_PX = 1.0``,
the per-axis pixel sigma assigned to every manual pick.

Implementation
==============

Source files:

- ``src/nav/nav_technique/nav_technique_manual.py`` —
  :class:`~nav.nav_technique.nav_technique_manual.NavTechniqueManual` and
  :func:`~nav.nav_technique.nav_technique_manual.run_manual_nav`.
- ``src/nav/feature/composition.py`` —
  :func:`~nav.feature.composition.compose_dialog_overlay`, the helper that builds the
  composite overlay image and mask from the scene's NavFeatures.
- ``src/nav/ui/manual_nav_dialog.py`` — ``ManualNavDialog``, the PyQt6 widget the technique
  invokes.

Public surface (autodocumented at :doc:`/api_reference/api_nav_technique`):

- :class:`~nav.nav_technique.nav_technique_manual.NavTechniqueManual` — the technique.
- :func:`~nav.nav_technique.nav_technique_manual.run_manual_nav` — the standalone
  interactive driver an external caller (a CLI script, an automated regression harness in
  pyqt-aware mode) invokes to open the dialog and obtain a result.

Class attributes:

- :attr:`~nav.nav_technique.nav_technique.NavTechnique.name` — ``'NavTechniqueManual'``.
- :attr:`~nav.nav_technique.nav_technique.NavTechnique.accepts_feature_types` —
  ``frozenset(NavFeatureType)`` — every feature type, since manual navigation overlays the
  whole scene.
- :attr:`~nav.nav_technique.nav_technique.NavTechnique.requires_prior` — ``False``.
- ``_abstract`` — ``True``.  Keeps the technique out of
  ``NavTechnique._registry`` so the orchestrator's
  autonomous driver never invokes it.

The technique declares no
:attr:`~nav.nav_technique.nav_technique.NavTechnique.confidence_spec` and no
:attr:`~nav.nav_technique.nav_technique.NavTechnique.confidence_attributes` — manual results
come back with the operator-implied confidence ``1.0`` and the ensemble does not fuse them
with autonomous results.

Public methods (autodocumented):
:meth:`~nav.nav_technique.nav_technique_manual.NavTechniqueManual.is_feasible` and
:meth:`~nav.nav_technique.nav_technique_manual.NavTechniqueManual.navigate`.

Call path
---------

Call path traced through
:func:`~nav.nav_technique.nav_technique_manual.run_manual_nav`:

1. The driver constructs every registered NavModel's instance for the observation and calls
   :meth:`~nav.nav_model.nav_model.NavModel.create_model` on each so the per-model state is
   populated.
2. Each model's :meth:`~nav.nav_model.nav_model.NavModel.to_features` and
   :meth:`~nav.nav_model.nav_model.NavModel.to_annotations` are invoked to gather the
   per-image features and the merged annotation collection.
3. :func:`~nav.feature.composition.compose_dialog_overlay` builds the composite extfov
   image and mask the dialog overlays on top of the observed image.
4. :class:`~nav.nav_technique.nav_technique_manual.NavTechniqueManual` is constructed with
   the merged annotations.
   :meth:`~nav.nav_technique.nav_technique_manual.NavTechniqueManual.navigate` opens the
   ``ManualNavDialog``.
5. The dialog runs interactively until the operator confirms an offset.  When the operator
   uses ``Auto``, the dialog runs the masked-NCC pyramid against the composite template;
   the result populates the dialog's offset entry and the operator may accept or override.
6. The operator's chosen ``(dv, du)`` plus the per-axis 1 px sigma populate a
   :class:`~nav.nav_technique.technique_result.NavTechniqueResult` with
   :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.confidence` ``1.0``,
   :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.spurious` ``False``,
   :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` ``False``, and a
   diagnostics object capturing whatever the auto-pick scored (a
   :class:`~nav.nav_technique.diagnostics.BodyDiscDiagnostics` when the operator accepted
   the auto-pick; otherwise zero-filled).
7. When the operator saves a sidecar, the merged annotations populate a labelled summary
   PNG alongside the JSON.

Examples
========

**Operator overrides an auto-pick.**  An operator opens the dialog on a Cassini fly-by
image where :class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav` reported a
``spurious=True`` result.  The composite overlay shows the predicted limb and disc; the
operator clicks ``Auto`` and the masked-NCC pyramid lands on a peak the operator can see is
near-correct but a sub-pixel off.  The operator drags the overlay one pixel and confirms
the offset.  The technique returns a
:class:`~nav.nav_technique.technique_result.NavTechniqueResult` with the operator's offset
and confidence ``1.0``; the summary PNG saves alongside the manual sidecar for reviewer
audit.

**Star-only scene with no auto-pick.**  An operator navigates a dense star field where the
autonomous :class:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`
reported ``status=failed``.  The composite overlay shows the predicted catalog stars as
rectangle outlines.  The operator manually drags the overlay until visible stars align with
the predicted boxes and confirms.  The dialog's ``Auto`` button is greyed out (the scene has
no template-bearing feature for the masked-NCC pyramid to consume); only the manual path is
available.

**Headless backend rejection.**  A CI runner invokes
:func:`~nav.nav_technique.nav_technique_manual.run_manual_nav` without a display server.
The PyQt6 import fails and the driver raises before the dialog opens.  The autonomous
pipeline never invokes manual navigation, so this failure is confined to the interactive
driver path.
