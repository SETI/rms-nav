=====
Stars
=====

The star NavModel emits one
:class:`~nav.feature.feature.NavFeature` per catalog star predicted to
fall in extended FOV, each carrying a ``StarGeometry`` payload, a
predicted-SNR-driven reliability score, and a ``StarFlags`` block (smear
length, in-body-silhouette, in-saturation-or-cosmic, etc.).  Two
techniques consume STAR features:
``StarFieldFromCatalogNav`` (similarity-invariant triplet pattern match)
and ``StarUniqueMatchNav`` (catalog-uniqueness 1- or 2-star match).

The real-scene star NavModel is unimplemented; the catalog-reduction
helpers (aberration, proper motion, multi-catalog precedence,
incremental search, body / ring conflict marking) and the
``SCLASS_TO_B_MINUS_V`` lookup are preserved in git history under the
deleted ``nav.nav_model.nav_model_stars`` module and will be lifted into
a per-component sub-package as the new model lands.
