==========================================================
Simulated Ring Navigation Model
==========================================================

Overview
========

:class:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated` is the
simulated-image variant of the ring navigation model. It renders a ring system from
operator-supplied edge radii and shading parameters instead of from a SPICE-driven
catalog, then emits two feature kinds:

- a :data:`~nav.feature.feature_type.NavFeatureType.RING_ANNULUS` carrying the rendered
  template, for :class:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`;
- one :data:`~nav.feature.feature_type.NavFeatureType.RING_EDGE` per rendered edge (inner
  and outer) -- a per-vertex polyline with outward radial normals -- for
  :class:`~nav.nav_technique.nav_technique_ring_edge.RingEdgeNav`.

The model overrides :meth:`~nav.nav_model.nav_model.NavModel.instances_for_obs` to build
one instance per ring of a simulated observation; the parent
:class:`~nav.nav_model.nav_model_rings.NavModelRings` declines simulated observations, so
the autonomous registry routes simulated frames here.

Theory
======

Simulated ring rendering paints a stack of catalog-shaped ring edges onto an extended-FOV
image plus mask using image-plane coordinates and operator-supplied shading. The
:class:`~nav.nav_model.rings.ring_feature.RingFeature` data model carries the per-edge
metadata exactly as in the catalog-driven path; only the rendering pipeline differs
(pixel-space here vs. backplane-based for the real model).

The rendered template is the
:data:`~nav.feature.feature_type.NavFeatureType.RING_ANNULUS` feature payload the
correlation technique navigates against. The per-edge RING_EDGE polyline extends the
simulated ring to the edge-fitting technique: each rendered edge mask
(:func:`~nav.sim.sim_ring.compute_border_atop_simulated`) is sampled into a vertex
polyline with outward radial normals and a curvature-based ``is_straight_line`` flag, and
:class:`~nav.nav_technique.nav_technique_ring_edge.RingEdgeNav` fits two curved arcs to
constrain the offset in both axes. The simulated ring's geometry is operator-known by
construction, so the simulated path is the calibration regime -- a developer can probe the
ring pipelines with rings whose true offset is known to the pixel.

Restrictions and assumptions
----------------------------

- The operator must supply finite per-edge radii and a finite shading distance.
- The simulated rings carry no per-image noise or PSF smearing; the operator's downstream
  noise-injection pipeline supplies those.
- The ring system is rendered in image-plane coordinates: edges are concentric ellipses
  about the operator-supplied centre. More elaborate orbit geometry (precession, nodal
  regression) requires the catalog-driven path.

Sources of uncertainty
----------------------

The simulated annulus has no measurement uncertainty by construction. The downstream
technique's reported covariance reflects only the correlation-curvature CRLB at the chosen
NCC peak.

Configuration
=============

The simulated ring model consumes no YAML configuration of its own; every parameter comes
in via the per-instance ``sim_params`` dict. Expected keys:

- ``name`` — ring-system label used in metadata.
- ``feature_type`` — ``RINGLET`` (bright ring) or ``GAP`` (dark gap).
- ``center_v``, ``center_u`` — pixel coordinates of the ring centre.
- ``range`` — subject distance in km.
- ``shading_distance`` — controls the per-edge surface-brightness falloff.
- ``inner_data``, ``outer_data`` — lists of dicts each carrying ``mode``, ``a``, ``rms``,
  ``ae``, ``long_peri``, and ``rate_peri`` keys (catalog-shaped per-edge entries).

Implementation
==============

Source file: ``src/nav/nav_model/nav_model_rings_simulated.py`` —
:class:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`.

Public class :class:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`,
base :class:`~nav.nav_model.nav_model_rings_base.NavModelRingsBase`. The class overrides
:meth:`~nav.nav_model.nav_model.NavModel.instances_for_obs` to build one instance per ring
of a simulated observation; the parent
:class:`~nav.nav_model.nav_model_rings.NavModelRings` returns an empty list for a simulated
observation, so the orchestrator's
:func:`~nav.nav_model.nav_model.build_models_for_obs` driver routes simulated frames to
this subclass.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.create_model` —
  invokes :func:`~nav.sim.sim_ring.render_ring` to render the simulated ring stack, then
  computes the bounding box and reliability scalars.
- :meth:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.to_features` —
  emits the RING_ANNULUS plus one RING_EDGE per rendered edge (inner / outer).
- :meth:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.to_annotations`
  — reuses the shared ring annotation helper on
  :class:`~nav.nav_model.nav_model_rings_base.NavModelRingsBase` to render per-edge
  polylines and labels onto the summary PNG.

Inherited :class:`~nav.nav_model.nav_model.NavModel` properties:
:attr:`~nav.nav_model.nav_model.NavModel.name`,
:attr:`~nav.nav_model.nav_model.NavModel.obs`,
:attr:`~nav.nav_model.nav_model.NavModel.metadata`.

Call path
---------

Call path traced through
:meth:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.create_model`:

1. Open a logged section. Read the operator-supplied sim parameters off the per-instance
   dict and the ring-system name.
2. Build :class:`~nav.nav_model.rings.ring_feature.RingFeature` instances from the
   per-edge sim parameter lists. The :class:`~nav.nav_model.rings.ring_feature.RingFeature`
   data model is shared with the catalog-driven path so the downstream emission path is
   identical.
3. Call :func:`~nav.sim.sim_ring.render_ring` with the per-edge data, the operator-supplied
   centre, and the shading parameters. The helper returns the rendered ring-system image
   plus mask.
4. Compute the per-edge brightness contrast and the per-pixel ``border_atop`` masking via
   :func:`~nav.sim.sim_ring.compute_border_atop_simulated`.
5. Promote the rendered image plus mask from sensor-shaped arrays to extfov-shaped arrays.
6. Record the predicted centre, the subject range, and the bounding box on the model's
   internal state for downstream feature emission.

Call path traced through
:meth:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.to_features`:

1. Construct one
   :data:`~nav.feature.feature_type.NavFeatureType.RING_ANNULUS`
   :class:`~nav.feature.feature.NavFeature` carrying the rendered template image, the
   mask, the predicted centre, and a
   :class:`~nav.feature.flags.RingAnnulusFlags` with the operator-supplied ring-system name.
2. For each rendered edge, recompute its 1-pixel edge mask via
   :func:`~nav.sim.sim_ring.compute_border_atop_simulated`, sample it into a vertex
   polyline with outward radial normals, classify it straight or curved, and append a
   :data:`~nav.feature.feature_type.NavFeatureType.RING_EDGE` carrying a
   :class:`~nav.feature.geometry.RingEdgePolyline`.
3. Reliability on each feature is fixed at ``1.0``.

Examples
========

The simulated ring model is consumed by the simulated-image GUI driver
(``nav_create_simulated_image``). An operator specifies a Saturn-like ring stack —
inner edge radii ``[74,490, 91,980, 122,170]`` km, outer edge radii
``[91,980, 117,580, 136,780]`` km, shading distance set per the per-image illumination
geometry — and the simulator renders the corresponding extended-FOV image plus annulus
mask. The downstream
:class:`~nav.nav_technique.nav_technique_ring_annulus.RingAnnulusNav` correlates the
template against an injected synthetic-noise image and recovers the operator-known
``(0, 0)`` offset within sub-pixel. The operator uses the residual to validate
per-instrument ring-rendering assumptions without a real ring image.
