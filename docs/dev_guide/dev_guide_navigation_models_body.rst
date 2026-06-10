=====================
Body Navigation Model
=====================

Overview
========

The body navigation model renders the predicted appearance of one planetary body from its
SPICE prediction and emits the image features that body techniques consume: the lit limb arc,
the terminator arc, the full disc as a correlation template, and an unresolved-body centroid
blob.  It works by rendering an oversampled Lambert silhouette over the body's predicted
bounding box, extracting the discrete limb and terminator masks, sampling them into per-vertex
polylines, and then applying a set of emission gates that decide which features carry enough
geometric information to be worth navigating against.

The orchestrator builds one
:py:class:`~nav.nav_model.nav_model_body.NavModelBody` per body whose predicted bounding box
overlaps the extended field of view.  The extended field of view (extfov) is the image sensor
grid plus a configurable margin on every side, so a body that is partly off the sensor but
whose silhouette still touches the margin gets an instance.  The per-image instance set is
produced by
:py:meth:`~nav.nav_model.nav_model_body.NavModelBody.instances_for_obs`, which asks the
observation for an inventory of the closest planet and its configured satellites and keeps every
entry whose bounding box passes the observation's in-extfov predicate.

Theory
======

A body's silhouette against the sky is the projection of its triaxial ellipsoid onto the image
plane.  Where the line of sight grazes the ellipsoid, the surface incidence angle (the angle
between the local outward normal and the direction to the Sun) passes through the geometry that
defines two distinct curves: the geometric limb, where the silhouette meets empty space, and the
terminator, where the lit hemisphere meets the unlit hemisphere.  Only the lit portion of the
limb produces a brightness gradient an image-side edge detector can lock onto; the unlit limb
fades into dark space and carries no usable signal, so the rendered limb curve is restricted to
its lit vertices.

The render proceeds on an oversampled grid so the silhouette boundary is anti-aliased.  The
oversampling factor per axis is chosen so the predicted body spans a fixed number of fine cells,
capped at a maximum.  The fine incidence-angle field is averaged back down to the sensor grid.
A pixel belongs to the body when its incidence angle is defined; it is lit when the incidence is
below ninety degrees.  The geometric limb is the set of body-side pixels adjacent to empty space;
the terminator is the set of lit pixels adjacent to an unlit pixel.

Each extracted curve is sampled into an ordered list of vertices.  At every vertex the outward
normal is computed as the discrete gradient of the region mask (the body silhouette for the limb,
the lit mask for the terminator), pointing from the interior toward the exterior.  Using the
region gradient rather than the one-pixel ridge gradient avoids the sign ambiguity of a
single-pixel-wide ridge.

The per-vertex normal uncertainty drives whether a limb arc is usable.  It combines, in
quadrature, the body's root-mean-square departure from a best-fit ellipsoid, a characteristic
crater-and-topography scale, an incidence-dependent limb-softness term (the point-spread sigma
in kilometres, amplified near grazing incidence where the brightness falls off slowly across the
limb), and the ephemeris position uncertainty projected onto the limb plane.  Dividing the
resulting kilometre uncertainty by the local kilometres-per-pixel scale yields a per-vertex
sigma in pixels.  When the representative limb uncertainty exceeds a fixed threshold, the limb is
too soft to fit and the model falls back to a blob.

The emission gates encode which feature carries information for a given geometry:

- The limb arc is emitted when the representative limb uncertainty is at or below the threshold
  and at least one lit limb vertex survives.
- A blob is emitted instead when the predicted disc diameter is at least a floor (which a
  per-body shape entry can raise but not lower) and the limb is too soft for an arc.  The blob
  carries only a brightness-weighted centroid and a bounding box.  At non-zero phase the measured
  brightness centroid sits at the centroid of the lit hemisphere rather than at the geometric
  centre, so the predicted centroid is computed as the brightness-weighted mean of the rendered
  body; this folds the systematic phase offset into the prediction so the recovered navigation
  offset is pure pointing error.
- The disc template is emitted alongside the limb arc when the body fits inside the sensor with
  enough of its lit side visible and little enough of its area off-frame.  The visible-lit
  fraction is measured over the whole predicted disc, not the lit hemisphere alone, so it retains
  discriminating power for a fully-in-frame body.
- The terminator arc is emitted when its polyline has at least a minimum vertex count and the
  sine of the phase angle is above a floor; below a few degrees of phase the terminator is
  photometrically indistinguishable from the limb.

The blob centroid covariance combines a photon-noise-limited centroid sigma with an irregularity
sigma that scales with the body's fractional shape residual and a phase factor running from one
at full phase to three at full crescent: the unlit hemisphere can hide an equal amount of shape
irregularity, so the centroid bias the ellipsoidal model cannot remove grows with phase.  The
reported limb and terminator polylines carry per-vertex along-curve and across-curve sigmas; the
along-curve sigma is a fixed sub-pixel value reflecting the sampling resolution, while the
across-curve sigma is the quadrature uncertainty described above.  The covariance does not model
correlated errors between neighbouring vertices, nor the unmodelled-pose bias of an irregular
body beyond the scalar irregularity term on the blob.

Configuration
=============

The body model reads its parameters from the ``bodies`` section of
``src/nav/config_files/config_040_bodies.yaml``.  The model itself consumes the render-control,
photometry, and bounding-box keys; the remaining keys configure the shared label-placement
helper on the annotation base and the reprojection seed thresholds.

- ``min_bounding_box_area`` — int, default ``9`` px.  Minimum predicted bounding-box area before
  the body is flagged size-ok in diagnostics; smaller bodies are still rendered.
- ``min_emission_ring_body`` — int, default ``20`` px.  Minimum body extent used by the
  ring/body interaction logic; consumed downstream of the model.
- ``oversample_edge_limit`` — int, default ``512`` (count).  Target fine-cell budget per axis; the
  per-axis oversample factor is this divided by the predicted body extent, raising anti-aliasing
  quality for small bodies.
- ``oversample_maximum`` — int, default ``2`` (count).  Hard cap on the per-axis oversample factor,
  bounding render cost for tiny bodies.
- ``curvature_threshold_frac`` — float, default ``0.02`` dimensionless.  Fractional curvature
  threshold used by downstream limb-curvature logic.
- ``curvature_threshold_pixels`` — int, default ``20`` px.  Absolute curvature threshold companion
  to the fractional one.
- ``limb_incidence_threshold`` — float, default ``1.53588974175501`` rad.  Incidence angle beyond
  which limb vertices are treated as grazing.
- ``limb_incidence_frac`` — float, default ``0.4`` dimensionless.  Fraction of the limb permitted to
  exceed the grazing-incidence threshold.
- ``surface_bumpiness`` — mapping, per-body km values.  Characteristic topographic relief per body,
  keyed by SPICE name.
- ``geometric_albedo`` — mapping, per-body dimensionless albedos.  Multiplied into the rendered
  Lambert model when ``use_albedo`` is set.
- ``use_lambert`` — bool, default ``true`` (dimensionless).  When true the silhouette is shaded by
  the Lambert law; when false it is a flat mask.  Lambert shading sharpens the disc template and
  biases the blob centroid toward the lit hemisphere.
- ``use_albedo`` — bool, default ``false`` (dimensionless).  When true the per-body geometric albedo
  scales the rendered brightness.
- ``min_reproj_seed_area`` — int, default ``40000`` px.  Minimum body area to seed a reprojection
  mosaic; consumed by the reprojection stage, not the model.
- ``min_reproj_candidate_area`` — int, default ``2500`` px.  Minimum body area to be a reprojection
  candidate; reprojection stage only.
- ``reproj_lon_resolution`` — float, default ``0.017453292519943295`` rad.  Longitude grid step for
  reprojection.
- ``reproj_lat_resolution`` — float, default ``0.017453292519943295`` rad.  Latitude grid step for
  reprojection.
- ``reproj_latlon_type`` — string, default ``centric`` (dimensionless).  Planetocentric vs.
  planetographic coordinate convention for reprojection.
- ``reproj_lon_direction`` — string, default ``east`` (dimensionless).  Longitude sign convention for
  reprojection.
- ``min_text_area`` — float, default ``0.003`` dimensionless.  Minimum fractional body area before a
  label is drawn.
- ``label_mask_enlarge`` — int, default ``10`` px.  Dilation radius of the body mask used as a
  label-avoidance region.
- ``label_limb_color`` — RGB triple, default ``[255, 0, 0]``.  Colour of the drawn limb overlay.
- ``label_font`` — string, default ``liberation2/LiberationMono-Bold.ttf``.  Font file for body
  labels.
- ``label_font_size`` — int, default ``18`` pt.  Body label font size.
- ``label_font_color`` — RGB triple, default ``[255, 0, 0]``.  Body label text colour.
- ``label_horiz_gap`` — int, default ``7`` px.  Horizontal gap between the limb and a left/right
  label arrow head.
- ``label_vert_gap`` — int, default ``5`` px.  Vertical gap between the limb and a top/bottom label
  arrow head.
- ``label_scan_v`` — int, default ``1`` px.  Vertical scan granularity when searching the limb for
  label anchors.
- ``label_grid_v`` — int, default ``10`` px.  Coarse vertical grid spacing for fallback label
  placement.
- ``label_grid_u`` — int, default ``10`` px.  Coarse horizontal grid spacing for fallback label
  placement.
- ``outline_thicken`` — int, default ``0`` px.  Extra thickness applied to the drawn limb outline.

Per-body shape and SPICE-residual values are resolved separately by
:py:func:`~nav.nav_model.body_shape.shape_for_body`, which merges the per-body entries in
``src/nav/config_files/config_220_body_shape.yaml`` over the hard-coded
:py:data:`~nav.nav_model.body_shape.BODY_SHAPE_TABLE` profiles and the
:py:data:`~nav.nav_model.body_shape.DEFAULT_BODY_SHAPE` fallback.  Every numeric field in the
shipped shape YAML is currently ``null``, so each body resolves to its hard-coded profile.

The feature-emission and gating thresholds are module-level constants rather than YAML keys:
:py:data:`~nav.nav_model.nav_model_body.LIMB_ARC_MAX_UNCERTAINTY_PX` (``3.0``),
:py:data:`~nav.nav_model.nav_model_body.BODY_BLOB_MIN_DIAMETER_PX` (``8.0``),
:py:data:`~nav.nav_model.nav_model_body.BODY_DISC_MIN_VISIBLE_LIT_FRACTION` (``0.4``),
:py:data:`~nav.nav_model.nav_model_body.BODY_DISC_MAX_OVERFLOW_FRACTION` (``0.3``),
:py:data:`~nav.nav_model.nav_model_body.TERMINATOR_MIN_VERTICES` (``8``),
:py:data:`~nav.nav_model.nav_model_body.TERMINATOR_MIN_PHASE_FACTOR` (``0.05``), and
:py:data:`~nav.nav_model.nav_model_body.BODY_POSITION_SLOP_FRAC` (``0.05``).

Implementation
==============

Source files: ``src/nav/nav_model/nav_model_body.py`` (the concrete model and its feature
builders), ``src/nav/nav_model/nav_model_body_base.py`` (the shared limb-mask and
label-annotation helpers), and ``src/nav/nav_model/body_shape.py`` (per-body shape resolution).

The public class is :py:class:`~nav.nav_model.nav_model_body.NavModelBody`, which extends
:py:class:`~nav.nav_model.nav_model_body_base.NavModelBodyBase` (itself a subclass of
:py:class:`~nav.nav_model.nav_model.NavModel`).  The base supplies
``_compute_limb_mask_from_body_mask`` and the body-label annotation pipeline.

The instance set comes from
:py:meth:`~nav.nav_model.nav_model_body.NavModelBody.instances_for_obs`: it reads the closest
planet and configured satellites, calls the observation's inventory, and constructs one model per
body whose inventory entry passes the in-extfov predicate.

:py:meth:`~nav.nav_model.nav_model_body.NavModelBody.create_model` records timing metadata, calls
the private render path, and logs a geometry summary.  The render computes the sub-solar and
sub-observer longitudes and latitudes, the phase angle and its sine factor, and the subject
range; inflates the inventory bounding box by the slop fraction and clips it into the extfov;
builds the oversampled Lambert backplane; downsamples the incidence field; derives the body, lit,
geometric-limb, lit-limb, and terminator masks; promotes the local arrays to extfov shape; and
samples the limb and terminator polylines.  Visible-lit fraction, overflow fraction, and the mean
kilometres-per-pixel at the limb are computed during the render and stored for the gates.

:py:meth:`~nav.nav_model.nav_model_body.NavModelBody.to_features` resolves the body shape, then
applies the gates in order.  It computes the representative limb uncertainty (the shape ellipsoid
residual divided by the mean limb scale), emits a :py:class:`~nav.feature.geometry.LimbPolyline`
carrying ``LIMB_ARC`` when the uncertainty clears the threshold and lit vertices survive, or
otherwise a :py:class:`~nav.feature.geometry.BodyBlobGeometry` carrying ``BODY_BLOB`` when the
diameter clears the blob floor.  When a limb arc was emitted and the disc gates pass it also emits
a :py:class:`~nav.feature.geometry.BodyDiscGeometry` carrying ``BODY_DISC``, and when the
terminator gates pass it emits a :py:class:`~nav.feature.geometry.TerminatorPolyline` carrying
``TERMINATOR_ARC``.  The four emitted :py:class:`~nav.feature.feature_type.NavFeatureType` values
are therefore ``LIMB_ARC``, ``BODY_BLOB``, ``BODY_DISC``, and ``TERMINATOR_ARC``; a single image
emits at most one of ``LIMB_ARC`` / ``BODY_BLOB`` plus optionally ``BODY_DISC`` and
``TERMINATOR_ARC``.

:py:meth:`~nav.nav_model.nav_model_body.NavModelBody.to_annotations` reuses the base
``_create_annotations`` helper to draw the limb overlay and body label on the summary image.

The reliability gates are scalar functions of the rendered geometry.  The limb-arc reliability is
a sigmoid of the visible-arc fraction and the surviving vertex count; the terminator reliability
multiplies a sigmoid of the visible-arc fraction and the albedo variation by the phase factor;
the disc reliability is the visible-lit fraction times one minus the overflow fraction times a
diameter sigmoid; and the blob reliability is a sigmoid of the per-pixel signal-to-noise and the
diameter, capped at ``0.4``.  Each emitted feature also carries a
:py:class:`~nav.feature.feature.NavReliabilityBreakdown` recording the individual contributions,
and a flags block (:py:class:`~nav.feature.flags.LimbArcFlags`,
:py:class:`~nav.feature.flags.TerminatorArcFlags`,
:py:class:`~nav.feature.flags.BodyDiscFlags`, or
:py:class:`~nav.feature.flags.BodyBlobFlags`).  The blob flags surface the phase angle and a
combined ``phase_irregularity_factor`` so the technique-side confidence can down-weight irregular
high-phase blobs.

Examples
========

**body_full_fov** (``N1572105349_1_CALIB``).  Dione fills the centre of the frame, mostly lit
with a sliver of terminator in the upper left; the sidecar records a predicted diameter of about
155 px, an overflow fraction of ``0.0``, and a visible-lit fraction of ``0.97``.  The render emits
a ``LIMB_ARC`` and, because the disc fits with ample lit fraction and no overflow, a
``BODY_DISC``.  The limb-arc reliability is dragged below the downstream LIMB_ARC gate on this
fully-lit geometry, so the body-limb technique does not consume the arc; the disc template is the
feature that drives navigation here.

**below_resolution_body** (``N1777325846_1_CALIB``).  Mimas spans roughly 20 px in the lower-left
corner at about 72 degrees phase.  The predicted limb uncertainty exceeds the limb-arc threshold,
so the model emits a single ``BODY_BLOB`` carrying the brightness-weighted centroid rather than a
``LIMB_ARC``; the sidecar's expected primary technique is the blob navigator.

**body_partial_overflow** (``N1484593951_2_CALIB``).  A large Rhea continues off the upper-right
corner; the sidecar records an overflow fraction of ``0.222``.  Because that sits below the
``0.3`` disc-overflow gate the model emits both a ``LIMB_ARC`` and a ``BODY_DISC``, and the limb
arc is the feature that the downstream technique fits successfully.

**high_phase_terminator** (``N1597846115_2_CALIB``).  A high-phase crescent with a terminator arc
and no other features in the frame.  The phase sine clears the terminator floor and the polyline
clears the minimum vertex count, so the model emits a ``TERMINATOR_ARC`` alongside the lit
``LIMB_ARC``.

**multi_body** (``N1487595731_1_CALIB``).  Dione and Rhea both appear and overlap at about 90
degrees phase, so
:py:meth:`~nav.nav_model.nav_model_body.NavModelBody.instances_for_obs` produces two
:py:class:`~nav.nav_model.nav_model_body.NavModelBody` instances, each emitting its own limb,
disc, and terminator features.
