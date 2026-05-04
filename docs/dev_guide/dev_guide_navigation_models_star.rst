==========================================================
Star Navigation Model
==========================================================

Overview
========

:class:`~nav.nav_model.stars.nav_model_stars.NavModelStars` is the catalog-driven star
navigation model.  For each observation the model reduces the configured catalog set into a
deduplicated star list, computes per-star predicted SNR using the per-instrument PSF and the
catalog-side B-V colour, flags stars whose predicted positions overlap a body silhouette or
ring annulus, and emits one
:data:`~nav.feature.feature_type.NavFeatureType.STAR` :class:`~nav.feature.feature.NavFeature`
per star whose predicted SNR clears the floor and whose conflict flags allow it.

The orchestrator builds exactly one instance of this model per observation regardless of
scene content; star navigation is the universal fallback when other navigation paths fail.

Theory
======

The star model orchestrates four cooperating steps: catalog reduction, conflict marking,
SNR prediction, and per-star feature emission.

Catalog reduction
-----------------

Multiple catalogs are merged into a single deduplicated star list, processed in priority
order (most-precise first).  When two stars from different catalogs match within a
configurable angular tolerance and visual-magnitude tolerance, the higher-priority catalog's
entry survives.  Pairs of stars whose magnitudes are too close together to disambiguate
visually are dropped from both catalogs (the autonomous match would be unable to attribute a
detection to one or the other).

Conflict marking
----------------

Each predicted star is checked against:

- Body silhouettes — when a star's predicted position lies inside any body's predicted
  silhouette mask, the star is flagged ``in_body_silhouette``.
- Ring annuli — when a star's predicted position lies inside any planet's ring system
  (defined by the per-planet radial bounds in the YAML config), the star is flagged
  ``in_ring_annulus``.
- Saturation / cosmic-ray masks — populated at navigate time from the per-image masks.

Conflict-flagged stars stay in the model's list (so the curator surfaces them in the
sidecar) but are excluded from the autonomous matching path by the upstream
``usable_stars`` filter consulted by every star technique.

Predicted SNR
-------------

The per-star SNR prediction combines:

- The catalog-side magnitude and spectral class (mapped to a B-V colour).
- The per-instrument quantum-efficiency-weighted bandpass and exposure.
- The per-instrument optical PSF, including motion smear when the per-image smear vector
  exceeds the anisotropic-smear minimum.

A smear-aware PSF kernel is built from the per-image smear-vector pixels, and the predicted
SNR is the integrated signal divided by the per-pixel noise sigma scaled by the kernel's
support.  Below the configured magnitude floor the predicted SNR is too small to support
detection and the star drops out of the emission set.

Per-feature uncertainty
-----------------------

Each emitted STAR feature carries a Cramer-Rao lower bound covariance derived from the
predicted SNR and the per-instrument PSF.  The CRLB scales as :math:`1 / \mathrm{SNR}` for
position; high-SNR stars have sub-pixel sigma, low-SNR stars have several-pixel sigma.

Restrictions and assumptions
----------------------------

- Predicted SNR depends on a calibrated per-instrument bandpass and exposure model.  When
  the per-instrument calibration is wrong, the predicted SNR is wrong, and the emission
  gate may include or exclude the wrong stars.
- Catalog reduction assumes the per-catalog sky-coverage epoch is recent enough that proper
  motion brings the star to within the duplicate-detection tolerance of its true epoch
  position.  Stars with very high proper motion may move out of catalog match between
  catalogs.
- Ring-occlusion checks consult the per-planet radial bounds in the YAML; non-listed
  planets contribute no ring-occlusion flag.

Sources of uncertainty
----------------------

The per-star CRLB covariance reflects the photon-noise-limited centroid uncertainty.  It
does not capture systematic biases from a misaligned camera distortion model, from
unmodelled background flux, or from PSF smear that exceeds the per-instrument
``max_smear`` cap (above which the model raises an error during construction rather than
emit unreliable predictions).  Star features whose predicted position lies inside a
saturation or cosmic-ray mask are flagged on the
:class:`~nav.feature.flags.StarFlags` so the orchestrator's reliability gate can drop them.

Configuration
=============

The model's runtime knobs live in ``stars`` in
``src/nav/config_files/config_030_stars.yaml``.

- ``catalogs`` — list[str], default ``[ucac4, tycho2, ybsc]``.  Catalog priority order
  (most-precise first).  Catalog files are loaded from the configured catalog roots.
- ``body_conflict_margin`` — int, default ``5`` px.  Margin around body silhouettes for
  the body-conflict flag.
- ``default_star_class`` — str, default ``'G0'``.  Spectral class assigned to catalog stars
  with no per-entry class.
- ``stellar_aberration`` — bool, default ``true``.  Apply stellar aberration to predicted
  positions.
- ``proper_motion`` — bool, default ``true``.  Apply proper motion to predicted positions.
- ``max_stars`` — int, default ``100``.  Maximum number of predicted stars retained per
  observation.
- ``max_movement_steps`` — int, default ``50``.  Maximum sub-exposure steps used by the
  smear-vector computation.
- ``label_font`` — str.  Font used for star labels in the summary PNG.
- ``label_font_size`` — int, default ``18`` px.  Label font size.
- ``label_font_color`` — list[int], default ``[255, 0, 0]`` (RGB).  Label font color.
- ``label_star_color`` — list[int], default ``[255, 0, 0]`` (RGB).  Color of the star marker
  drawn on the summary PNG.
- ``duplicate_ra_dec_threshold_arcsec`` — float, default ``5`` arcsec.  Tolerance for the
  per-catalog duplicate detection step.
- ``duplicate_vmag_threshold`` — float, default ``3`` mag.  Magnitude tolerance for the
  duplicate detection step.
- ``overlapping_vmag_threshold`` — float, default ``2`` mag.  Below this magnitude
  difference, two visually-overlapping stars are dropped from both catalogs.
- ``calibrated_data`` — bool, default ``true``.  Whether the per-image data is in
  calibrated I/F units (vs. raw DN).
- ``float_psf_sigma`` — bool, default ``false``.  When true, the per-instrument PSF sigma is
  treated as a fit parameter; when false, the per-instrument value is used verbatim.
- ``search_multipliers`` — list[float], default ``[0.25, 0.5, 0.75, 1.0]``.  Multipliers on
  the per-instrument SPICE pointing-error envelope used by the star matcher's coarse
  search.
- ``perform_photometry`` — bool, default ``true``.  Whether to run per-star photometric
  validation.
- ``try_without_photometry`` — bool, default ``false``.  Whether to attempt a fallback
  match path with photometry disabled.
- ``min_stars_low_confidence`` — list (count, confidence), default ``[3, 0.75]``.  Minimum
  star count and confidence level for the low-confidence match path.
- ``min_stars_high_confidence`` — list (count, confidence), default ``[6, 1.0]``.  Minimum
  star count and confidence level for the high-confidence match path.
- ``min_confidence`` — float, default ``0.9``.  Minimum confidence level for the match to
  succeed.
- ``psf_gain`` — list (DN, gain), default ``[5000, 4]``.  PSF integrated gain mapping for
  flux estimation.
- ``max_smear`` — float, default ``100`` (dimensionless).  Maximum smear length above which
  the model rejects the image.
- ``min_vmag`` — float, default ``5.0`` mag.  Minimum visual magnitude (i.e. brightest)
  considered.
- ``max_vmag`` — float, default ``15.0`` mag.  Maximum visual magnitude (i.e. dimmest)
  considered for prediction.
- ``vmag_increment`` — float, default ``0.5`` mag.  Magnitude bin width for the per-bin
  star-list build.
- ``max_star_dn`` — float, default ``100000.0`` DN.  Above this, stars are too bright to
  use (saturation regime).
- ``min_dn_force_one_star`` — float, default ``25000.0`` DN.  Below this DN, the
  unique-bright single-star path will not fire even on a uniquely bright catalog star.
- ``star_body_conflict_margin`` — int, default ``3`` px.  Smaller-than-body conflict margin
  used when the per-star centroid is close to the body silhouette boundary.
- ``too_bright_dn`` — float, default ``1000`` DN.  Threshold above which a star is "very
  bright" for the per-star photometric tests.
- ``too_bright_factor`` — float, default ``1`` (dimensionless).  Multiplier on the
  too-bright threshold (reserved tuning slot).
- ``ring_occlusion_enabled`` — bool, default ``true``.  Whether to flag stars whose
  predicted positions lie inside a planet's ring system.
- ``ring_occlusion_radii_km`` — dict[str, list[list[float]]].  Per-planet ring annular
  bounds (km) used by the occlusion check.  Saturn entry covers C, B, and A rings; Uranus
  covers the main ring system; Neptune covers Galle through Adams rings.

Per-instrument overrides
------------------------

Per-instrument YAML files in ``src/nav/config_files/config_4N0_inst_*.yaml`` may override
catalog selection or photometric parameters; see the per-instrument source for the full
list.

Implementation
==============

Source files:

- ``src/nav/nav_model/stars/nav_model_stars.py`` —
  :class:`~nav.nav_model.stars.nav_model_stars.NavModelStars`.
- ``src/nav/nav_model/stars/catalog.py`` — multi-catalog reduction helpers.
- ``src/nav/nav_model/stars/conflicts.py`` — body / ring conflict marking.
- ``src/nav/nav_model/stars/predicted_snr.py`` —
  :func:`~nav.nav_model.stars.predicted_snr.predicted_snr` and
  :func:`~nav.nav_model.stars.predicted_snr.psf_sigma_px` plus
  ``SCLASS_TO_B_MINUS_V``.
- ``src/nav/nav_model/stars/smeared_psf.py`` —
  :func:`~nav.nav_model.stars.smeared_psf.compute_smear_vector_px` and
  :func:`~nav.nav_model.stars.smeared_psf.smear_length_px`.

Public class :class:`~nav.nav_model.stars.nav_model_stars.NavModelStars`, base
:class:`~nav.nav_model.nav_model.NavModel`.  Self-registers via ``__init_subclass__``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~nav.nav_model.stars.nav_model_stars.NavModelStars.instances_for_obs` — always
  returns ``[NavModelStars('stars', obs)]`` (one instance per observation).
- :meth:`~nav.nav_model.stars.nav_model_stars.NavModelStars.create_model` — runs catalog
  reduction, computes the smear vector, marks conflicts, and populates the per-star
  predicted SNR and CRLB covariance.
- :meth:`~nav.nav_model.stars.nav_model_stars.NavModelStars.to_features` — emits one
  :data:`~nav.feature.feature_type.NavFeatureType.STAR` feature per surviving catalog star.
- :meth:`~nav.nav_model.stars.nav_model_stars.NavModelStars.to_annotations` — emits
  per-star markers and labels for the summary PNG.

Inherited read-only properties on :class:`~nav.nav_model.nav_model.NavModel`:
:attr:`~nav.nav_model.nav_model.NavModel.name`,
:attr:`~nav.nav_model.nav_model.NavModel.obs`,
:attr:`~nav.nav_model.nav_model.NavModel.metadata`.

Call path
---------

Call path traced through
:meth:`~nav.nav_model.stars.nav_model_stars.NavModelStars.create_model`:

1. Open a logged section.  Read the per-image observation epoch and configured catalog
   list.
2. Compute the per-image motion smear vector via
   :func:`~nav.nav_model.stars.smeared_psf.compute_smear_vector_px`.  When the smear length
   exceeds ``max_smear`` the model raises so the orchestrator skips star navigation rather
   than emit unreliable predictions.
3. Reduce the configured catalogs into a deduplicated star list via
   :func:`~nav.nav_model.stars.catalog.reduce_catalogs`.  The function returns a list of
   mutable star records carrying RA/Dec, magnitude, spectral class, parallax, proper
   motion, and per-catalog provenance.
4. Mark body / ring conflicts on each star via
   :func:`~nav.nav_model.stars.conflicts.mark_body_and_ring_conflicts`.  The function
   consults the per-image inventory and the YAML's ``ring_occlusion_radii_km`` to set
   ``in_body_silhouette`` and ``in_ring_annulus`` flags.
5. Compute per-star predicted SNR via
   :func:`~nav.nav_model.stars.predicted_snr.predicted_snr` using the per-instrument PSF
   sigma and the per-star spectral class (mapped to B-V colour via
   ``SCLASS_TO_B_MINUS_V``).
6. Drop stars whose predicted SNR falls below the per-instrument detection floor; record
   the survivors on ``self._stars``.

Call path traced through
:meth:`~nav.nav_model.stars.nav_model_stars.NavModelStars.to_features`:

1. For each surviving star, build a :class:`~nav.feature.geometry.StarGeometry` from the
   predicted position and the per-feature CRLB covariance.
2. Build a :class:`~nav.feature.flags.StarFlags` carrying the predicted SNR, the body /
   ring conflict flags, and the saturation / cosmic-ray-mask flags read off the per-image
   masks.
3. Construct one :data:`~nav.feature.feature_type.NavFeatureType.STAR`
   :class:`~nav.feature.feature.NavFeature` per star and return the list.

Examples
========

``one_bright_star_no_body`` (Cassini ISS WAC, image ``W1449079117_1``)
    Single bright star (Vega) in an otherwise empty FOV.  The star model emits one STAR
    feature for Vega with high predicted SNR and no body / ring conflict flags.  The
    pass-1 :class:`~nav.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`
    consumes the feature in its 1-star path and reports the operator-verified offset
    :math:`(\Delta v, \Delta u) = (3.06, -0.02)` px.  Other catalog stars in the WAC's
    extended FOV are below the magnitude floor and produce no STAR feature.

``star_dominated`` (Cassini ISS WAC, image ``W1580760393_1``)
    Dense star field with no body in FOV.  The star model emits dozens of STAR features
    above the magnitude floor.  The pass-1
    :class:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` consumes
    the cohort and reports the operator-verified offset
    :math:`(\Delta v, \Delta u) = (-2.68, -3.68)` px via the triplet-hash matcher.

``below_resolution_body`` (Cassini ISS NAC, image ``N1777325846_1``)
    Mimas in the lower left at ~20 px diameter at phase 72 degrees.  The star model emits
    STAR features for catalog stars in the FOV; stars whose predicted positions fall
    inside the predicted Mimas silhouette are flagged ``in_body_silhouette`` and the
    upstream ``usable_stars`` filter drops them so
    the star techniques do not consume them.
