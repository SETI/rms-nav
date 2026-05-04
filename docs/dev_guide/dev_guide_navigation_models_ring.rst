==========================================================
Ring Navigation Model
==========================================================

Overview
========

:class:`~nav.nav_model.nav_model_rings.NavModelRings` is the catalog-driven ring navigation
model.  For each planet whose ring system has any radius inside the extended FOV the model
renders the per-ring-edge silhouette from the catalog, runs a four-pass
``RingFeatureFilter`` to drop edges that are not separable / detectable on this image, and
emits either a :data:`~nav.feature.feature_type.NavFeatureType.RING_EDGE` per surviving edge
(the "edges resolve" path) or a single
:data:`~nav.feature.feature_type.NavFeatureType.RING_ANNULUS` per planet (the "edges
compress" path) when individual edges fall below the resolvability threshold.

The orchestrator constructs one model instance per planet whose ring system overlaps the
extfov.  A simulated-image sibling
(:class:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`) renders rings
from operator-supplied parameters instead of the catalog; both classes share annotation
helpers on :class:`~nav.nav_model.nav_model_rings_base.NavModelRingsBase`.

Theory
======

The ring model is a per-planet edge renderer plus a feature-emission gate that decides
whether to ship per-edge polylines or a composite annulus template.

Edge rendering
--------------

For each catalog-defined ring edge the model:

1. Builds an oversampled meshgrid around the predicted ring's projected bounding box.
2. Queries the per-pixel ring radius backplane and the ring longitude backplane.
3. Marks the discrete pixel set whose radius lies on the edge's catalog radius.
4. Walks the pixel set to produce a polyline of vertices, each with an outward radial
   normal estimated from the local radius gradient.

Per-edge feature filtering
--------------------------

A four-pass filter removes edges that cannot contribute a useful constraint:

- **Catalog presence.**  Edges absent from the catalog or with non-finite radius are
  dropped.
- **Visibility.**  Edges whose predicted bounding box has no overlap with the extfov are
  dropped.
- **Resolvability.**  Edges whose maximum radial pixel extent compresses below
  ``feature_emission.ring_annulus.<planet>.max_radial_px`` are flagged for the annulus path.
- **Detectability.**  Edges whose per-pixel signal-to-noise falls below the per-instrument
  detection threshold are dropped.

Curvature classification
------------------------

For each surviving polyline the model fits a best-fit straight line and measures the
maximum perpendicular deviation.  An edge whose deviation exceeds
``curvature_threshold_pixels`` (or the configured fraction of the edge's length) is flagged
``is_curved``; otherwise the edge is straight-line and the per-edge constraint is rank-1
along the radial direction.  The :doc:`dev_guide_techniques_ring_edge` page describes how
the downstream technique handles the all-straight case.

Per-vertex covariance
---------------------

Each polyline vertex carries a per-vertex radial sigma derived from:

- The catalog-side ``rms`` (the catalog's reported radial uncertainty for the edge).
- The optical PSF sigma converted to kilometres at the ring radius.
- An additional photometric softness term scaled by the per-edge surface-brightness
  contrast against the background.

The along-edge sigma is set to a small constant matching the polyline-sampling resolution.

Annulus template
----------------

When the per-planet km/px scale exceeds the configured threshold (or any single ring edge
compresses below the per-polyline radial-pixel threshold), the model emits a single
RING_ANNULUS feature carrying a rendered template image of the entire ring system
(every ring radius painted at the catalog brightness contrast) plus the matching mask.
The template's bounding box is the union of the per-edge bounding boxes; the template
brightness at each pixel is the sum of the per-ring-edge contributions.

Restrictions and assumptions
----------------------------

- The catalog must provide a per-ring-edge radius and an RMS uncertainty.  Rings missing
  either field are dropped silently.
- The model assumes the per-image SPICE pose is good enough that the predicted ring
  geometry is approximately correct.  A wrong pose shifts every edge polyline by the same
  pose error; the downstream DT fit recovers the offset.
- The detectability filter assumes the per-instrument calibration converts the catalog
  surface brightness into a per-pixel signal correctly.  When the calibration is wrong
  (e.g. on a calibrated-IF instrument with a stale CALIB pipeline), the detectability
  test may include or exclude wrong edges.

Sources of uncertainty
----------------------

The per-vertex sigma values capture the catalog-side RMS and the optical PSF; they do not
capture a per-image radial bias from a wrong epoch ring solution, nor a longitude-dependent
brightness modulation that would shift the apparent edge position non-uniformly around the
ring.  Edges flagged ``is_curved`` carry full-rank locally-observable information; edges
flagged straight-line are rank-1 along radial only.

Configuration
=============

The model's runtime knobs live in ``rings`` in
``src/nav/config_files/config_050_rings.yaml`` plus per-planet ring catalogues in
``src/nav/config_files/config_3N0_*_rings.yaml`` (one per planet).  The per-planet feature-
emission tunables under ``feature_emission.ring_annulus`` in
``src/nav/config_files/config_510_techniques.yaml`` decide the RING_EDGE vs. RING_ANNULUS
path; see :doc:`dev_guide_techniques_ring_annulus` for those.

The ``rings`` block carries general per-model knobs (oversample factor, label rendering,
detection thresholds).  The per-planet ring catalogues carry the per-edge radii and RMS
values; format is one ``ring_features`` mapping per planet, keyed by edge name.

Per-instrument overrides
------------------------

Per-instrument YAML files in ``src/nav/config_files/config_4N0_inst_*.yaml`` may override
detection thresholds.  See the per-instrument source for the full list.

Implementation
==============

Source files:

- ``src/nav/nav_model/nav_model_rings.py`` —
  :class:`~nav.nav_model.nav_model_rings.NavModelRings`, the four-pass filter, the
  per-edge sampler, and the annulus-template builder.
- ``src/nav/nav_model/nav_model_rings_base.py`` —
  :class:`~nav.nav_model.nav_model_rings_base.NavModelRingsBase`, abstract shared base
  carrying the ring annotation pipeline.
- ``src/nav/nav_model/rings/`` — the rings subpackage with the validation, filtering, and
  rendering helpers (``ring_types``, ``ring_feature``, ``ring_filter``, ``ring_math``,
  ``ring_render_context``, ``ring_render_result``).

Public class :class:`~nav.nav_model.nav_model_rings.NavModelRings`, base
:class:`~nav.nav_model.nav_model_rings_base.NavModelRingsBase`.  Self-registers via
``__init_subclass__``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~nav.nav_model.nav_model_rings.NavModelRings.instances_for_obs` — class method
  that returns one instance per planet whose ring system has any radius inside the
  extended FOV.
- :meth:`~nav.nav_model.nav_model_rings.NavModelRings.create_model` — populates the model
  state by rendering each per-edge silhouette, running the four-pass filter, and emitting
  per-vertex polyline data plus an optional annulus template.
- :meth:`~nav.nav_model.nav_model_rings.NavModelRings.to_features` — runs the per-edge
  emission gates and constructs zero or more
  :class:`~nav.feature.feature.NavFeature` instances (RING_EDGE per surviving edge, or
  RING_ANNULUS per planet when the annulus path fires).
- :meth:`~nav.nav_model.nav_model_rings.NavModelRings.to_annotations` — emits per-edge
  polylines and per-planet labels for the summary PNG.

Inherited :class:`~nav.nav_model.nav_model.NavModel` properties:
:attr:`~nav.nav_model.nav_model.NavModel.name`,
:attr:`~nav.nav_model.nav_model.NavModel.obs`,
:attr:`~nav.nav_model.nav_model.NavModel.metadata`.

Call path
---------

Call path traced through
:meth:`~nav.nav_model.nav_model_rings.NavModelRings.create_model`:

1. Open a logged section.  Look up the per-planet ring catalogue from the configured
   ``ring_features`` mapping.  Each entry carries a name, a radius, an RMS, and a per-edge
   surface-brightness profile.
2. Build an oversampled meshgrid around the predicted ring's projected bounding box and
   query the per-pixel ring radius and longitude backplanes.
3. For each catalog edge, mark the pixel set whose ring radius lies within the per-edge
   tolerance and run the four-pass filter.
4. For each surviving edge, walk the pixel set to produce a per-vertex polyline (position,
   radial normal, per-vertex sigma).  Classify the polyline curvature via the best-fit
   straight-line residual.
5. Decide the per-planet emission path: when the per-planet km/px scale exceeds the
   configured threshold, or when any single edge compresses below the per-polyline radial-
   pixel threshold, render a RING_ANNULUS template; otherwise emit per-edge polylines.

Call path traced through
:meth:`~nav.nav_model.nav_model_rings.NavModelRings.to_features`:

1. **Annulus path.**  Construct one
   :data:`~nav.feature.feature_type.NavFeatureType.RING_ANNULUS` feature per planet
   carrying the rendered annulus template plus the per-planet bounding box.
2. **Per-edge path.**  For each surviving edge, construct one
   :data:`~nav.feature.feature_type.NavFeatureType.RING_EDGE` feature carrying the
   :class:`~nav.feature.geometry.RingEdgePolyline` (vertices, normals, per-vertex sigmas)
   plus a per-edge :class:`~nav.feature.flags.RingEdgeFlags` with the catalog edge name and
   the curvature classification.

Examples
========

``ring_only_curved`` (Cassini ISS NAC, image ``N1447064164_1``)
    A high-resolution Saturn-ring scene whose individual catalog edges resolve into
    separable polylines.  The ring model emits multiple
    :data:`~nav.feature.feature_type.NavFeatureType.RING_EDGE` features (the F-ring outer
    edge, the A-ring outer edge, gaps, ringlets); the per-planet km/px on this scene is
    well below the annulus threshold so the annulus path does not fire.  Curvature
    classification flags the edges curved (each surviving polyline arcs noticeably across
    the FOV); the rank of the joint :class:`~nav.nav_technique.nav_technique_ring_edge.RingEdgeNav`
    fit is full-rank because the curvature lifts the rank-1 degeneracy.  See
    :doc:`dev_guide_techniques_ring_edge`.

A second illustrative scenario: a low-resolution approach-phase Saturn image whose km/px
exceeds the per-planet kmpp threshold.  The ring model emits a single RING_ANNULUS feature
per planet carrying the rendered annulus template; the
:class:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav` consumes it via the
shared pyramid-NCC machinery.
