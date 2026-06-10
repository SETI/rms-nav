===============================
Simulated Ring Navigation Model
===============================

Overview
========

The simulated ring navigation model renders a ring system from operator-supplied parameters
rather than from a SPICE prediction, and emits the rendered system as a single composite template
feature.  It exists so the simulated-image tooling can compose synthetic ring scenes with edges at
known radii and a known centre; the resulting template feature is the same ``RING_ANNULUS`` kind
the catalog-driven model produces when ring edges are unresolvable.

This model is not auto-instantiated from an observation.  It inherits the empty default
:py:meth:`~nav.nav_model.nav_model.NavModel.instances_for_obs` and is constructed directly by the
caller that supplies the simulation parameters.  Each constructed instance emits exactly one
``RING_ANNULUS`` feature.

Theory
======

The simulated ring system is described by a centre in pixel coordinates and one or two edge
orbits, each given by a base radius and optional orbital modes.  The renderer shades the region
between the inner and outer edges solidly, mimicking the behaviour of the catalog-driven model
where the space between resolved edges is treated as opaque absent further information.  The result
is a brightness image and a mask of the shaded ring pixels.

Because the system is rendered from exact parameters, there is no uncertainty model: the template
is treated as a noise-free image with unit reliability.  The data-model classes that describe the
edges are shared with the catalog-driven ring path; only the image generation differs, working
directly in pixel space here rather than through a ring backplane.

Configuration
=============

This model takes no YAML configuration of its own beyond the shared label settings.  Its geometry
comes from the ``sim_params`` dictionary passed to the constructor, whose recognised keys are the
ring name and feature type, the centre coordinates (``center_v``, ``center_u``), the subject range
(``range``), a shading distance (``shading_distance``), and the ``inner_data`` / ``outer_data``
edge-orbit lists (each a list of dicts with ``mode``, ``a``, ``rms``, ``ae``, ``long_peri``, and
``rate_peri`` keys).  The render-time log level is read from the shared ``rings`` configuration
section documented in :doc:`dev_guide_navigation_models_rings`.

Implementation
==============

Source file: ``src/nav/nav_model/nav_model_rings_simulated.py``.  The public class is
:py:class:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`, which extends
:py:class:`~nav.nav_model.nav_model_rings_base.NavModelRingsBase` and therefore shares the
edge-annotation helper with the catalog-driven ring model.

:py:meth:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.create_model` records
timing metadata and calls the private render path.  The render adapts the simulation parameters
into a feature config, parses it into a
:py:class:`~nav.nav_model.rings.ring_feature.RingFeature`, renders the solid-shaded ring image into
an extfov-shaped buffer at the requested centre, derives the ring mask from the non-zero pixels,
and records the predicted centre, subject range, and bounding box.

:py:meth:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.to_features` emits one
:py:class:`~nav.feature.feature.NavFeature` of type
:py:attr:`~nav.feature.feature_type.NavFeatureType.RING_ANNULUS`, carrying a
:py:class:`~nav.feature.geometry.RingAnnulusGeometry`, the rendered image and mask as the template,
unit reliability, a :py:class:`~nav.feature.feature.NavReliabilityBreakdown` with full visible-arc
fraction, and :py:class:`~nav.feature.flags.RingAnnulusFlags` whose constituent edge count reflects
how many of the inner and outer edges are present.  The single emitted
:py:class:`~nav.feature.feature_type.NavFeatureType` is ``RING_ANNULUS``.

:py:meth:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.to_annotations` rebuilds
the inner and outer edge masks in pixel space and draws their polyline overlays and labels via the
base ``_create_edge_annotations`` helper.

Examples
========

The simulated ring model does not draw from the real-image corpus under
``tests/integration/image_library/images/``; those ring scenes are navigated by the catalog-driven
model.  A representative simulated invocation constructs the model with a ringlet feature whose
inner and outer edges sit a few hundred kilometres apart about a frame-centre projection.
:py:meth:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.create_model` then renders
the solid-shaded annulus, and
:py:meth:`~nav.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.to_features` emits a
single ``RING_ANNULUS`` feature carrying the rendered template.  For a worked navigation against a
real compressed ring system, see the ``ring_only_curved`` example in
:doc:`dev_guide_navigation_models_rings`.
