===========
Annotations
===========

Overview
========

The :mod:`nav.annotation` subsystem produces the per-image summary-PNG
overlay.  Every :class:`~nav.nav_model.nav_model.NavModel` exposes a
``to_annotations(context)`` method that returns a fresh
:class:`~nav.annotation.annotations.Annotations` collection holding the
labels and graphical primitives for that one model's contribution
(body silhouettes, ring polylines, star markers, etc.).  The
orchestrator merges every model's collection into
:attr:`~nav.nav_orchestrator.nav_result.NavResult.annotations` via
:meth:`~nav.annotation.annotations.Annotations.add_annotations`; the
top-level driver (:func:`~nav.navigate_image_files.navigate_image_files`)
then calls
:meth:`~nav.annotation.annotations.Annotations.combine` to render the
final RGB overlay and composites it onto the contrast-stretched source
image before writing the PNG.

Class hierarchy
===============

The subsystem is intentionally narrow: four shipping types and no
abstract base.

- :class:`~nav.annotation.annotation.Annotation` — a single typed
  primitive (line, polyline, marker, text).  Each instance carries an
  RGBA overlay array, an overlay color, an optional avoid-mask used
  by the label placer, and a list of attached
  :class:`~nav.annotation.annotation_text_info.AnnotationTextInfo`
  entries.
- :class:`~nav.annotation.annotation_text_info.AnnotationTextInfo` —
  text payload + placement parameters (anchor location, arrow style,
  font color).  The placement constants
  (:data:`~nav.annotation.annotation_text_info.TEXTINFO_TOP`,
  ``TEXTINFO_BOTTOM_LEFT``, ``TEXTINFO_LEFT_ARROW``, etc.) name the
  twelve supported anchor / arrow combinations.
- :class:`~nav.annotation.annotation_text_info.TextLocInfo` — the
  label placer's per-text resolution result (chosen pixel position,
  arrow tail, fitness score).
- :class:`~nav.annotation.annotations.Annotations` — collection of
  :class:`~nav.annotation.annotation.Annotation` instances.  Subclass
  of :class:`~nav.support.nav_base.NavBase` so it inherits ``config``
  and ``logger``.

Pipeline
========

Annotation flow during a single navigation run:

1. **Per-model emission.**  Each
   :class:`~nav.nav_model.nav_model.NavModel` builds its
   :class:`~nav.annotation.annotations.Annotations` collection in
   ``to_annotations(context)``.  Concrete bodies / rings / stars use
   the shared helpers on
   :class:`~nav.nav_model.nav_model_body_base.NavModelBodyBase` and
   :class:`~nav.nav_model.nav_model_rings_base.NavModelRingsBase` so
   the silhouette / polyline rendering is consistent across models.
2. **Orchestrator merge.**  The orchestrator's
   ``_collect_annotations`` step builds an empty
   :class:`~nav.annotation.annotations.Annotations` and calls
   :meth:`~nav.annotation.annotations.Annotations.add_annotations`
   for each surviving model's contribution.  The merged collection
   lands on
   :attr:`~nav.nav_orchestrator.nav_result.NavResult.annotations`.
3. **Final composition.**
   :func:`~nav.navigate_image_files.navigate_image_files` calls
   :meth:`~nav.annotation.annotations.Annotations.combine` with the
   navigation offset.  ``combine`` runs the label placer (consulting
   each :class:`~nav.annotation.annotation.Annotation`'s avoid-mask),
   shifts every overlay by the offset, and produces a single RGB
   overlay array.  The driver composites that overlay over the
   contrast-stretched source image and writes the PNG.

Reading the per-image PNG therefore tells the operator three things in
one image: what was in the source (background), what each NavModel
predicted (overlay), and where the orchestrator placed the predictions
relative to the data (the offset shift applied at composition time).

Configuration
=============

The :mod:`nav.annotation` subsystem itself consumes no YAML
configuration.  The label / colour knobs live in the per-NavModel
config blocks instead:

- ``bodies.label_*``, ``bodies.outline_thicken``, ``bodies.min_text_area``
  in ``src/nav/config_files/config_040_bodies.yaml`` (consumed via
  :class:`~nav.nav_model.nav_model_body_base.NavModelBodyBase`).
- ``rings.label_*`` in ``src/nav/config_files/config_050_rings.yaml``
  (consumed via
  :class:`~nav.nav_model.nav_model_rings_base.NavModelRingsBase`).
- ``stars.label_*`` and ``stars.label_star_color`` in
  ``src/nav/config_files/config_030_stars.yaml`` (consumed via
  :class:`~nav.nav_model.stars.nav_model_stars.NavModelStars`).

Each per-NavModel page documents the relevant subset:
:doc:`dev_guide_navigation_models_body`,
:doc:`dev_guide_navigation_models_ring`, and
:doc:`dev_guide_navigation_models_star`.

Per-model annotation contributions
==================================

- **Body** —
  :meth:`~nav.nav_model.nav_model_body.NavModelBody.to_annotations`
  emits a body silhouette outline plus a body-name label with leader
  arrow.  The label placer scans the limb for a clear gap and falls
  back to a coarse grid when no per-limb candidate fits.
- **Ring** —
  :meth:`~nav.nav_model.nav_model_rings.NavModelRings.to_annotations`
  emits one polyline per surviving ring edge plus a per-planet label.
- **Star** —
  :meth:`~nav.nav_model.stars.nav_model_stars.NavModelStars.to_annotations`
  emits a per-star marker plus an optional per-star magnitude label.
- **Simulated body / rings** — the simulated NavModels reuse their
  catalog-driven counterparts' annotation helpers, so a simulated
  scene's PNG overlay is visually indistinguishable from a real scene's.
- **Titan** — the placeholder
  :meth:`~nav.nav_model.nav_model_titan.NavModelTitan.to_annotations`
  returns an empty collection.

API reference
=============

The autodocumented API surface is at :doc:`/api_reference/api_annotation`.
