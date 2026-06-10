======================
Ring Navigation Model
======================

Overview
========

The ring navigation model predicts the appearance of a planet's ring system and emits the ring
edges, or a composite ring-system template, that ring techniques consume.  For each ring feature
in the per-planet catalog that survives a four-pass selection filter, it renders the feature into
the image plane from the planet's ring backplane and emits either a per-edge polyline carrying
``RING_EDGE`` or, when the edges compress to too few pixels to trace individually, a single
composite ``RING_ANNULUS`` template for the whole system.

The orchestrator builds one
:py:class:`~nav.nav_model.nav_model_rings.NavModelRings` per planet whose ring system has any
radius inside the extended field of view and whose ring catalog is configured;
:py:meth:`~nav.nav_model.nav_model_rings.NavModelRings.instances_for_obs` returns a single
instance for the closest planet when those conditions hold, and an empty list otherwise.  That one
model emits one ``RING_EDGE`` feature per traceable surviving ring edge, plus at most one
``RING_ANNULUS`` feature collapsing every annulus-eligible edge into one composite.

Theory
======

A planet's rings lie in a flat plane.  Seen obliquely they project to nested ellipses; seen
edge-on they collapse to a line.  Each named ring edge is the locus of a fixed ring-plane radius,
possibly perturbed by orbital modes.  The image-plane curve of an edge is found by evaluating the
ring-radius field over the field of view and tracing the iso-radius contour at the edge's radius.
Where the rings are well resolved, the curve is a smooth arc whose curvature alone constrains the
pointing offset perpendicular to the local tangent; where the rings are compressed to a few pixels
of radial extent, individual edges blur together and only the brightness profile of the whole
system carries usable information.

The selection filter applies four passes to the catalog features.  A date pass drops any feature
whose active date window excludes the observation time.  A radius pass drops any feature whose
edges all fall outside the visible radial range.  A resolvability pass drops any two-edge feature
whose width is smaller than a configured number of pixels at the local resolution, since an
unresolvable gap shaded onto the model would mislead the navigator.  A fade-conflict pass trims an
edge whose soft-edged fade would be squeezed below a minimum width by a neighbouring edge, and
drops a feature left with no edges.

Each surviving feature is rendered to a brightness image and an edge mask.  The edge mask is one
pixel wide; sampling its set pixels yields an ordered polyline, and the discrete gradient across
the mask gives the radial normal at each vertex.  The polyline's radial extent is the spread of
its vertices projected onto the mean normal, and its straightness is the maximum perpendicular
deviation from the best-fit line, found by a singular-value decomposition of the centred vertex
cloud.

Two regimes decide how a rendered edge is emitted.  A per-edge regime keeps the polyline as a
``RING_EDGE`` when its radial extent exceeds a threshold and it is not straight.  An annulus regime
applies otherwise, and also unconditionally when the system-wide kilometres-per-pixel resolution
exceeds a per-planet threshold so that even a nominally traceable edge spans only a handful of
pixels.  Annulus-eligible renderings are unioned into one composite brightness template and mask
for the whole planet, emitted as a single ``RING_ANNULUS``.

Per-edge uncertainty is the catalog edge's radial root-mean-square residual projected to pixels
through the local radial resolution, carried as the across-edge sigma at every vertex; the
along-edge sigma is a fixed sub-pixel value reflecting sampling resolution.  A straight edge is
flagged so the technique-side fit knows it provides a rank-one constraint only.  The reliability of
a ring edge scales a catalog-default value by the visible-arc fraction and one minus the
shadow-occluded fraction, with a multiplier for straight edges; the reliability of an annulus
scales with the number of constituent edges and a sigmoid of its radial extent.  The reported
uncertainty captures the catalog radial residual and the sampling resolution; it does not model
correlated errors along an edge or the unmodelled-mode contribution beyond what the catalog
residual encodes.

To avoid paying for a dense ring backplane on images where the rings are not usefully visible, a
cheap coarse-grid radius evaluation runs first and short-circuits the render when no ray
intersects the ring plane or when every sampled radius lies beyond the catalog's outermost
feature.

Configuration
=============

The ring model reads shared parameters from the ``rings`` section of
``src/nav/config_files/config_050_rings.yaml``, the per-planet ring catalogs from the
``rings.ring_features`` blocks in ``config_300_jupiter_rings.yaml`` through
``config_330_neptune_rings.yaml``, and the emission thresholds from the
``feature_emission.ring_annulus`` block in ``config_510_techniques.yaml``.

The ``rings`` section keys are:

- ``model_source`` — string, default ``ephemeris`` (dimensionless).  Selects the SPICE-ephemeris
  ring model.
- ``fiducial_feature_threshold`` — int, default ``3`` (count).  Minimum fiducial feature count used
  by downstream ring fitting.
- ``fiducial_rms_gain`` — float, default ``2`` dimensionless.  Gain applied to the catalog radial
  residual when weighting fiducial edges.
- ``fiducial_min_feature_width`` — float, default ``2`` px.  Minimum fiducial feature width used
  downstream.
- ``one_sided_feature_width`` — float, default ``30.0`` px.  Shading width applied to a single-edge
  feature.
- ``fiducial_ephemeris_width`` — float, default ``100`` px.  Ephemeris fiducial width used
  downstream.
- ``min_curvature_low_confidence`` — list, default ``[0.0, 0.5]``.  Curvature and confidence pair for
  the low-confidence ring tier.
- ``min_curvature_high_confidence`` — list, default ``[0.17, 1.0]``.  Curvature and confidence pair
  for the high-confidence ring tier.
- ``curvature_to_reduce_features`` — float, default ``1.5707963267948966`` rad.  Curvature above
  which the feature set is reduced.
- ``curvature_reduced_features`` — int, default ``1`` (count).  Feature count retained when curvature
  forces reduction.
- ``emission_fiducial_threshold`` — float, default ``0.75`` dimensionless.  Emission-angle fiducial
  threshold used downstream.
- ``emission_use_threshold`` — float, default ``0.2`` dimensionless.  Emission-angle usability
  threshold used downstream.
- ``remove_planet_shadow`` — bool, default ``true`` (dimensionless).  When true, ring pixels inside
  the planet's shadow are zeroed from the rendered model and mask so the navigator does not match
  bright model arcs against the dark shadow.
- ``remove_body_shadows`` — bool, default ``false`` (dimensionless).  Accepted by the parser; ring
  pixels in moon shadows are not removed.
- ``label_font`` — string, default ``liberation2/LiberationMono-Bold.ttf``.  Font file for ring
  labels.
- ``label_font_size`` — int, default ``18`` pt.  Ring label font size.
- ``label_font_color`` — RGB triple, default ``[255, 0, 0]``.  Ring label text colour.
- ``label_mask_enlarge`` — int, default ``10`` px.  Dilation radius of the ring mask used as a
  label-avoidance region.
- ``label_horiz_gap`` — int, default ``7`` px.  Horizontal gap between an edge and a left/right label
  arrow head.
- ``label_vert_gap`` — int, default ``5`` px.  Vertical gap between an edge and a top/bottom label
  arrow head.
- ``label_limb_color`` — RGB triple, default ``[255, 0, 0]``.  Colour of the drawn ring-edge overlay.

Ring-annulus emission
---------------------

The emission thresholds live under ``feature_emission.ring_annulus`` in
``config_510_techniques.yaml`` with a ``default`` block and per-planet overrides; each block sets:

- ``max_radial_px`` — float, default ``5.0`` px.  Per-edge radial extent at or below which the model
  emits an annulus template instead of a per-edge polyline.  Larger values push more edges into the
  annulus regime.
- ``kmpp_threshold`` — float, default ``1000.0`` km/px (Saturn; ``200.0`` Jupiter, ``300.0`` Uranus,
  ``500.0`` Neptune).  System-wide radial resolution at or above which every surviving feature is
  forced into the annulus regime.  Lower values trigger the annulus path at higher resolution.

Per-planet ring catalogs
------------------------

Each ``rings.ring_features.<PLANET>`` block sets the reference ``epoch``, the edge-fade widths
(``fade_width_pix``, ``min_allowed_fade_width_pix``), the resolvability floor
(``min_feature_pixels``), and a ``features`` mapping of named ring features.  Each feature names a
``feature_type`` (``GAP`` or ``RINGLET``), one or both edge orbits (``inner_data`` / ``outer_data``
as lists of orbital modes), and optional ``start_date`` / ``end_date`` activity windows.  See
:py:class:`~nav.nav_model.rings.ring_feature.RingFeature` and the ring domain types in
``ring_types`` for the full schema.

Implementation
==============

Source files: ``src/nav/nav_model/nav_model_rings.py`` (the concrete model and its feature
builders), ``src/nav/nav_model/nav_model_rings_base.py`` (the shared edge-annotation helper), and
the ``rings`` subpackage — ``ring_types.py``, ``ring_feature.py``, ``ring_filter.py``,
``ring_math.py``, ``ring_render_context.py``, and ``ring_render_result.py``.

The public class is :py:class:`~nav.nav_model.nav_model_rings.NavModelRings`, a subclass of
:py:class:`~nav.nav_model.nav_model_rings_base.NavModelRingsBase` (itself a
:py:class:`~nav.nav_model.nav_model.NavModel`).

:py:meth:`~nav.nav_model.nav_model_rings.NavModelRings.create_model` records timing metadata, runs
the render, and logs a summary.  The render identifies the closest planet, parses and validates
the per-planet catalog into :py:class:`~nav.nav_model.rings.ring_feature.RingFeature` objects,
checks cross-feature date overlaps, runs the cheap coarse-grid visibility pre-check, evaluates the
dense ring-radius backplane to find the visible radial range, builds a
:py:class:`~nav.nav_model.rings.ring_filter.RingFeatureFilter` and applies the four passes,
computes the subject range and the optional planet-shadow mask, and renders each surviving feature
through :py:meth:`~nav.nav_model.rings.ring_feature.RingFeature.render` into a
:py:class:`~nav.nav_model.rings.ring_render_result.RingRenderResult`.  The rendered images, masks,
uncertainties, and per-edge info are stored for emission.

:py:meth:`~nav.nav_model.nav_model_rings.NavModelRings.to_features` resolves the per-planet
emission thresholds, decides the system-wide annulus gate, and walks the rendered edges.  For each
edge it samples the polyline and normals, measures the radial extent and straightness, and either
emits a :py:class:`~nav.feature.geometry.RingEdgePolyline` carrying
:py:attr:`~nav.feature.feature_type.NavFeatureType.RING_EDGE` with
:py:class:`~nav.feature.flags.RingEdgeFlags`, or accumulates the rendering for the annulus.  At the
end, accumulated annulus renderings are unioned into one composite
:py:class:`~nav.feature.geometry.RingAnnulusGeometry` carrying
:py:attr:`~nav.feature.feature_type.NavFeatureType.RING_ANNULUS` with
:py:class:`~nav.feature.flags.RingAnnulusFlags`.  The two emitted
:py:class:`~nav.feature.feature_type.NavFeatureType` values are therefore ``RING_EDGE`` and
``RING_ANNULUS``; the design emits at most one ``RING_ANNULUS`` per planet per scene.

:py:meth:`~nav.nav_model.nav_model_rings.NavModelRings.to_annotations` draws the per-edge polyline
overlays and ring labels via the base ``_create_edge_annotations`` helper.

The reliability of a ``RING_EDGE`` is the catalog default scaled by the visible-arc fraction and
one minus the shadow-occluded fraction, with a straight-line multiplier; the reliability of a
``RING_ANNULUS`` scales the catalog default by the constituent-edge count and a sigmoid of the
radial extent.  Each emitted feature carries a
:py:class:`~nav.feature.feature.NavReliabilityBreakdown`.

Examples
========

**ring_only_curved** (``N1447064164_1_CALIB``, ``W1444747627_1_CALIB``).  Distant Saturn ring views
in which the catalog A, B, and C ring edges all compress radially below the annulus radial
threshold, so the model collapses every surviving edge into one composite ``RING_ANNULUS`` feature
for the Saturn ring system rather than emitting per-edge polylines.  The sidecars' expected primary
technique is the ring-annulus navigator running one joint correlation against the composite
template.  A nearer, well-resolved Saturn ring view in which the principal edges each span more
than the radial threshold and curve across the field would instead emit one ``RING_EDGE`` per
traceable edge.
