======
Bodies
======

Body navigation models render the predicted appearance of a planetary body
into a per-feature template plus optional polyline geometries (limb,
terminator).  Concrete subclasses derive from
:class:`~nav.nav_model.nav_model_body_base.NavModelBodyBase`, which carries
shared annotation helpers (limb-mask extraction and label placement).

Today's only registered concrete body model is
:class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated`,
used by the simulated-image GUI to compose synthetic test scenes from
operator-supplied geometric parameters; it emits a single
``BODY_DISC`` :class:`~nav.feature.feature.NavFeature` carrying the
rendered template.

The real-scene body model is unimplemented; its concrete subclass will
emit a mix of ``LIMB_ARC``, ``TERMINATOR_ARC``, ``BODY_DISC``, and
``BODY_BLOB`` features depending on per-image visibility, lighting, and
shape uncertainty.
