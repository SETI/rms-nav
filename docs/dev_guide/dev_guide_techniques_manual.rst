=================
Manual Navigation
=================

Overview
========

This technique is the interactive escape hatch: it composes every renderable
predicted feature -- body discs, ring annuli, cartographic models, limb and
terminator and ring-edge polylines, body-blob outlines, and star markers -- into
a single overlay image and lets a human operator pick the ``(dv, du)`` offset by
hand in a dialog.  An auto button inside the dialog runs the same masked
correlation pyramid the autonomous correlation techniques use, so the operator
can accept the auto pick or override it.  It is not part of the autonomous
pipeline -- it is kept out of the technique auto-discovery registry and invoked
directly by an interactive driver.  Feasibility passes whenever at least one
feature paints something into the composite overlay; it fails when nothing is
renderable.

Theory
======

Manual navigation carries no fitting algorithm of its own.  It renders the
predicted scene -- the union of every feature's template silhouette, polyline
vertices, blob outline, or star marker -- as an overlay registered to the
extended field of view, and displays it over the observed image so a human can
judge the registration directly.  The operator translates the overlay until the
predicted features align with what the image shows, and the chosen translation is
the reported offset.  An optional in-dialog correlation gives a starting
suggestion, but the authoritative result is the operator's confirmed pick.

Because the offset is set by eye, its uncertainty is governed by display zoom,
screen pixel pitch, and operator precision rather than by any measurement model.
A fixed conservative per-axis pixel sigma is assigned to a confirmed pick.  A
confirmed pick is treated as fully confident, since a human has visually verified
the registration; a cancelled dialog yields a zero-confidence spurious result so
the surrounding driver drops it.

Configuration
=============

This technique has no configuration: it has no ``tuning`` block and no confidence
formula in ``src/nav/config_files/config_510_techniques.yaml``, and its
``confidence_attributes`` set is empty.  The single behavioural constant -- the
per-axis pixel sigma assigned to a confirmed pick -- is a module-level constant
(``_MANUAL_OFFSET_SIGMA_PX``) in the source, with no YAML override path; it only
determines what covariance appears in the written metadata, because the ensemble
never combines a manual result with autonomous results.

Implementation
==============

Source files: ``src/nav/nav_technique/nav_technique_manual.py``, the dialog in
:py:mod:`nav.ui.manual_nav_dialog`, and the overlay composer in
:py:mod:`nav.feature.composition`.  The public class is
:py:class:`~nav.nav_technique.nav_technique_manual.NavTechniqueManual`, a subclass
of :py:class:`~nav.nav_technique.nav_technique.NavTechnique`.  It sets the private
``_abstract`` flag to ``True`` so it stays out of the auto-discovery registry, its
``accepts_feature_types`` is the full set of feature types (manual navigation
looks at whatever the scene rendered), and its ``requires_prior`` is ``False``.

:py:meth:`~nav.nav_technique.nav_technique_manual.NavTechniqueManual.is_feasible`
counts renderable features -- template-bearing features with a template image and
mask, polyline-bearing features with non-empty vertices, body-blob geometries, and
star geometries -- and returns feasible when at least one is renderable.

:py:meth:`~nav.nav_technique.nav_technique_manual.NavTechniqueManual.navigate`
composes the overlay image and mask via
:py:func:`~nav.feature.composition.compose_dialog_overlay`, ensures a Qt
application exists, opens the dialog, and runs it modally.  The result shape
branches on the operator's choice: a cancelled or empty pick returns a spurious
:py:class:`~nav.nav_technique.technique_result.NavTechniqueResult` with zero
confidence, a zero offset, an identity covariance, and a
:py:class:`~nav.nav_technique.diagnostics.ManualNavDiagnostics` whose
:py:attr:`~nav.nav_technique.diagnostics.ManualNavDiagnostics.operator_accepted`
field is ``False``; a confirmed pick returns a non-spurious result carrying the
operator's offset, a covariance of the per-axis pixel sigma squared, full
confidence, and ``operator_accepted`` ``True``.

The module-level function
:py:func:`~nav.nav_technique.nav_technique_manual.run_manual_nav` is the
single-observation entry point: it builds the same models, context, and features
the autonomous orchestrator would, opens the dialog with the autonomous
reliability gate bypassed (the operator visually overrides the gate's decisions),
and on a confirmed pick wraps the result in a full
:py:class:`~nav.nav_orchestrator.nav_result.NavResult` -- provenance, image
classifier, feature inventory, and annotations populated identically to the
autonomous pipeline -- so callers can write the same metadata and preview outputs.
It returns ``None`` when the operator cancels or when the composed overlay paints
no pixels into the extended-FOV mask.

Examples
========

**multi_body (corpus class).** A scene with several bodies in the field renders a
template silhouette per body plus any limb and terminator polylines into the
overlay.  Feasibility passes because multiple features are renderable; the
operator sees every predicted silhouette over the image and picks the offset that
registers them.  A confirmed pick returns a non-spurious result with
``operator_accepted`` ``True`` and the per-axis pixel-sigma covariance.

**ring_only_curved (N1447064164_1_CALIB).** A distant Saturn ring frame whose
rings collapse to a single ``RING_ANNULUS`` template paints that template into the
overlay, so feasibility passes on the one renderable feature.  An operator can
register the ring band by hand here even though the autonomous run on this frame
records ``status: conflicted`` -- manual navigation does not depend on the
autonomous confidence outcome.
