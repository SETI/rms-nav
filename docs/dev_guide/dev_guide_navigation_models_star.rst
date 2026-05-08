==========================================================
Star Navigation Model
==========================================================

Overview
========

:class:`~nav.nav_model.stars.nav_model_stars.NavModelStars` is the catalog-driven star
navigation model. For each observation the model reduces the configured catalog set into a
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
order (most-precise first). When two stars from different catalogs match within a
configurable angular tolerance and visual-magnitude tolerance, the higher-priority catalog's
entry survives. Pairs of stars whose magnitudes are too close together to disambiguate
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

The per-star predicted SNR is the matched-filter integrated signal-to-noise ratio at the
star's predicted pixel position, given the per-instrument PSF and the per-image robust
noise estimate. Three quantities feed the formula:

- The integrated in-band signal :math:`S_{\mathrm{DN}}` derived from the catalog
  V-magnitude with a per-instrument-per-filter ``mag_offset``:

  .. math::

      S_{\mathrm{DN}} = 2.512^{-(V_{\mathrm{mag}} + \delta_{\mathrm{mag}} - 4)}

  with :math:`\delta_{\mathrm{mag}}` resolved from
  :class:`~nav.nav_orchestrator.instrument_config.InstrumentSettings` per the per-image
  filter combo. The reference magnitude (4) is the calibration zero-point chosen so that
  a fourth-magnitude star produces 1 DN of integrated in-band signal at the per-instrument
  reference sensitivity.

- The per-pixel noise sigma :math:`\sigma_{\mathrm{DN}}` in DN units, obtained by
  rescaling the per-image robust noise estimate
  :attr:`~nav.nav_orchestrator.nav_context.NavContext.image_noise_sigma` through the
  per-instrument ``signal_dn_to_image_unit_scale``. For raw-DN instruments the scale is
  ``1.0`` and :math:`\sigma_{\mathrm{DN}}` is the per-pixel sigma directly; for
  calibrated-IF instruments (e.g. Cassini ISS ``_CALIB.IMG``) the scale is the per-camera
  DN-to-I/F factor (typically of order :math:`10^{-7}`) so the per-pixel noise gets
  converted back to a DN-equivalent for SNR formation. Without this conversion the SNR
  for every catalog star would collapse to :math:`\sqrt{S_{\mathrm{DN}}}` on calibrated
  images and the reliability gate would drop them all.

- The PSF noise-equivalent aperture :math:`N_{\mathrm{ap}}`. Treating every PSF as a 2-D
  Gaussian for SNR purposes, the noise-equivalent area is

  .. math::

      N_{\mathrm{ap}} = 4 \pi \, \sigma_{\mathrm{PSF}}^{2}

  in pixels, with :math:`\sigma_{\mathrm{PSF}}` the per-axis-averaged Gaussian sigma
  returned by :func:`~nav.nav_model.stars.predicted_snr.psf_sigma_px`.

The integrated SNR is the standard matched-filter form
(:func:`~nav.nav_model.stars.predicted_snr.predicted_snr`):

.. math::

    \mathrm{SNR} =
        \frac{S_{\mathrm{DN}}}
             {\sqrt{S_{\mathrm{DN}} + \sigma_{\mathrm{DN}}^{2} \, N_{\mathrm{ap}}}}.

The numerator is the matched-filter signal; the denominator is the quadrature combination
of shot noise (:math:`S_{\mathrm{DN}}`, the source's own Poisson contribution) and
background-plus-read noise (:math:`\sigma_{\mathrm{DN}}^{2} N_{\mathrm{ap}}`, the
per-pixel noise variance summed over the PSF support). For bright stars
:math:`\mathrm{SNR} \approx \sqrt{S_{\mathrm{DN}}}` (shot-noise-limited); for faint stars
:math:`\mathrm{SNR} \approx S_{\mathrm{DN}} / (\sigma_{\mathrm{DN}} \sqrt{N_{\mathrm{ap}}})`
(background-limited).

When the per-image smear length exceeds the anisotropic-smear threshold the PSF is
replaced by a smear-aware kernel built from the per-image smear vector via
:func:`~nav.nav_model.stars.smeared_psf.compute_smear_vector_px`; the kernel has elongated
support and a correspondingly larger :math:`N_{\mathrm{ap}}`. Below the per-instrument
detection floor (``min_vmag`` / SNR cutoff) the predicted SNR is too small to support
detection and the star drops out of the emission set.

Per-feature uncertainty
-----------------------

Each emitted ``STAR`` feature carries a Cramer-Rao lower bound covariance derived from the
predicted SNR and the per-instrument PSF. For a 2-D Gaussian PSF the CRLB on each
position component reduces to

.. math::

    \sigma_{\mathrm{pos}} = \frac{\sigma_{\mathrm{PSF}}}{\mathrm{SNR}},

so high-SNR stars have sub-pixel position sigma and low-SNR stars have several-pixel
sigma. The per-axis variances populate the diagonal of the
:attr:`~nav.feature.feature.NavFeature.position_cov_px` covariance; the off-diagonal is
zero except when the smear-aware kernel is anisotropic, in which case the cross-axis
correlation reflects the smear orientation.

Restrictions and assumptions
----------------------------

- Predicted SNR depends on a calibrated per-instrument bandpass and exposure model. When
  the per-instrument calibration is wrong, the predicted SNR is wrong, and the emission
  gate may include or exclude the wrong stars.
- Catalog reduction assumes the per-catalog sky-coverage epoch is recent enough that proper
  motion brings the star to within the duplicate-detection tolerance of its true epoch
  position. Stars with very high proper motion may move out of catalog match between
  catalogs.
- Ring-occlusion checks consult the per-planet radial bounds in the YAML; non-listed
  planets contribute no ring-occlusion flag.

Sources of uncertainty
----------------------

The per-star CRLB covariance reflects the photon-noise-limited centroid uncertainty. It
does not capture systematic biases from a misaligned camera distortion model, from
unmodelled background flux, or from PSF smear that exceeds the per-instrument
``max_smear`` cap (above which the model raises an error during construction rather than
emit unreliable predictions). Star features whose predicted position lies inside a
saturation or cosmic-ray mask are flagged on the
:class:`~nav.feature.flags.StarFlags` so the orchestrator's reliability gate can drop them.

Configuration
=============

The model's runtime knobs live in ``stars`` in
``src/nav/config_files/config_030_stars.yaml``.

- ``catalogs`` — list[str], default ``[ucac4, tycho2, ybsc]``. Catalog priority order
  (most-precise first). Catalog files are loaded from the configured catalog roots.
- ``body_conflict_margin`` — int, default ``5`` px. Margin around body silhouettes for
  the body-conflict flag.
- ``default_star_class`` — str, default ``'G0'``. Spectral class assigned to catalog stars
  with no per-entry class.
- ``stellar_aberration`` — bool, default ``true``. Apply stellar aberration to predicted
  positions.
- ``proper_motion`` — bool, default ``true``. Apply proper motion to predicted positions.
- ``max_stars`` — int, default ``100``. Maximum number of predicted stars retained per
  observation.
- ``max_movement_steps`` — int, default ``50``. Maximum sub-exposure steps used by the
  smear-vector computation.
- ``label_font`` — str. Font used for star labels in the summary PNG.
- ``label_font_size`` — int, default ``18`` px. Label font size.
- ``label_font_color`` — list[int], default ``[255, 0, 0]`` (RGB). Label font color.
- ``label_star_color`` — list[int], default ``[255, 0, 0]`` (RGB). Color of the star marker
  drawn on the summary PNG.
- ``duplicate_ra_dec_threshold_arcsec`` — float, default ``5`` arcsec. Tolerance for the
  per-catalog duplicate detection step.
- ``duplicate_vmag_threshold`` — float, default ``3`` mag. Magnitude tolerance for the
  duplicate detection step.
- ``overlapping_vmag_threshold`` — float, default ``2`` mag. Below this magnitude
  difference, two visually-overlapping stars are dropped from both catalogs.
- ``calibrated_data`` — bool, default ``true``. Whether the per-image data is in
  calibrated I/F units (vs. raw DN).
- ``float_psf_sigma`` — bool, default ``false``. When true, the per-instrument PSF sigma is
  treated as a fit parameter; when false, the per-instrument value is used verbatim.
- ``search_multipliers`` — list[float], default ``[0.25, 0.5, 0.75, 1.0]``. Multipliers on
  the per-instrument SPICE pointing-error envelope used by the star matcher's coarse
  search.
- ``perform_photometry`` — bool, default ``true``. Whether to run per-star photometric
  validation.
- ``try_without_photometry`` — bool, default ``false``. Whether to attempt a fallback
  match path with photometry disabled.
- ``min_stars_low_confidence`` — list (count, confidence), default ``[3, 0.75]``. Minimum
  star count and confidence level for the low-confidence match path.
- ``min_stars_high_confidence`` — list (count, confidence), default ``[6, 1.0]``. Minimum
  star count and confidence level for the high-confidence match path.
- ``min_confidence`` — float, default ``0.9``. Minimum confidence level for the match to
  succeed.
- ``psf_gain`` — list (DN, gain), default ``[5000, 4]``. PSF integrated gain mapping for
  flux estimation.
- ``max_smear`` — float, default ``100`` (dimensionless). Maximum smear length above which
  the model rejects the image.
- ``min_vmag`` — float, default ``5.0`` mag. Minimum visual magnitude (i.e. brightest)
  considered.
- ``max_vmag`` — float, default ``15.0`` mag. Maximum visual magnitude (i.e. dimmest)
  considered for prediction.
- ``vmag_increment`` — float, default ``0.5`` mag. Magnitude bin width for the per-bin
  star-list build.
- ``max_star_dn`` — float, default ``100000.0`` DN. Above this, stars are too bright to
  use (saturation regime).
- ``min_dn_force_one_star`` — float, default ``25000.0`` DN. Below this DN, the
  unique-bright single-star path will not fire even on a uniquely bright catalog star.
- ``star_body_conflict_margin`` — int, default ``3`` px. Smaller-than-body conflict margin
  used when the per-star centroid is close to the body silhouette boundary.
- ``too_bright_dn`` — float, default ``1000`` DN. Threshold above which a star is "very
  bright" for the per-star photometric tests.
- ``too_bright_factor`` — float, default ``1`` (dimensionless). Multiplier on
  ``too_bright_dn`` (reserved tuning slot).
- ``ring_occlusion_enabled`` — bool, default ``true``. Whether to flag stars whose
  predicted positions lie inside a planet's ring system.
- ``ring_occlusion_radii_km`` — dict[str, list[list[float]]]. Per-planet ring annular
  bounds (km) used by the occlusion check. Saturn entry covers C, B, and A rings; Uranus
  covers the main ring system; Neptune covers Galle through Adams rings.

Module-level emission constants
-------------------------------

The detection-side thresholds and the catalog magnitude-bin grid are Python module-level
constants in the :mod:`nav.nav_model.stars` subpackage and are not exposed as YAML knobs.
Tests and downstream tools read the canonical values via these symbols.

- :data:`~nav.nav_model.stars.detection.DAOPHOT_DEFAULT_DETECTION_SIGMA` — float,
  ``4.0`` (in ``image_noise_sigma``). Threshold for matched-filter peaks. Below 4 sigma
  the cosmic-ray-driven false-positive rate dominates real detections; the autonomous
  detector rejects peaks below this level.
- :data:`~nav.nav_model.stars.detection.DAOPHOT_DEFAULT_SHARPNESS_MIN` — float, ``0.2``
  (dimensionless). Minimum DAOPHOT sharpness for a real star. Sharpness below 0.2 is
  dominated by single-pixel hot spikes whose wing contribution is too small to be a
  star.
- :data:`~nav.nav_model.stars.detection.DAOPHOT_DEFAULT_SHARPNESS_MAX` — float, ``1.0``
  (dimensionless). Maximum DAOPHOT sharpness for a real star. Sharpness above 1.0
  indicates an extended source (galaxy / blended pair) whose central pixel does not
  dominate.
- :data:`~nav.nav_model.stars.detection.DAOPHOT_DEFAULT_ROUNDNESS_BOUND` — float,
  ``1.0`` (dimensionless). Maximum :math:`|\mathrm{roundness}|` for a real star,
  computed as the per-axis Gaussian-marginal asymmetry; values outside
  :math:`[-1, 1]` point at a CCD bloom or one-axis trail rather than a smear-oriented
  PSF.
- :data:`~nav.nav_model.stars.catalog.CATALOG_MAGNITUDE_BINS` — tuple[float, ...]. The
  coarse magnitude grid the catalog reducer pulls stars by. Stars are pulled bin by bin
  until the configured ``max_stars`` budget is hit, which avoids the worst case of
  pulling every dim star in a degree-square chunk of UCAC4.

Per-instrument overrides
------------------------

Per-instrument YAML files in ``src/nav/config_files/config_4N0_inst_*.yaml`` may override
catalog selection or photometric parameters; see the per-instrument source for the full
list.

Implementation
==============

Source files:

- ``src/nav/nav_model/stars/nav_model_stars.py`` —
  :class:`~nav.nav_model.stars.nav_model_stars.NavModelStars` plus the per-star
  CRLB-covariance helper, the smear-resolver, and the per-summary-PNG annotation
  builder.
- ``src/nav/nav_model/stars/catalog.py`` —
  :mod:`nav.nav_model.stars.catalog` multi-catalog reduction helpers and
  :data:`~nav.nav_model.stars.catalog.CATALOG_MAGNITUDE_BINS`.
- ``src/nav/nav_model/stars/conflicts.py`` —
  :mod:`nav.nav_model.stars.conflicts` body / ring conflict marking.
- ``src/nav/nav_model/stars/predicted_snr.py`` —
  :func:`~nav.nav_model.stars.predicted_snr.predicted_snr` and
  :func:`~nav.nav_model.stars.predicted_snr.psf_sigma_px` plus the
  ``SCLASS_TO_B_MINUS_V`` spectral-class-to-colour table.
- ``src/nav/nav_model/stars/smeared_psf.py`` —
  :func:`~nav.nav_model.stars.smeared_psf.compute_smear_vector_px` and
  :func:`~nav.nav_model.stars.smeared_psf.smear_length_px`.
- ``src/nav/nav_model/stars/detection.py`` —
  :mod:`nav.nav_model.stars.detection` matched-filter source detection plus the four
  :data:`~nav.nav_model.stars.detection.DAOPHOT_DEFAULT_DETECTION_SIGMA`,
  :data:`~nav.nav_model.stars.detection.DAOPHOT_DEFAULT_SHARPNESS_MIN`,
  :data:`~nav.nav_model.stars.detection.DAOPHOT_DEFAULT_SHARPNESS_MAX`, and
  :data:`~nav.nav_model.stars.detection.DAOPHOT_DEFAULT_ROUNDNESS_BOUND` constants
  documented above. Consumed by the star techniques
  rather than by the model itself; documented here because its constants set the
  detection floor every star technique applies.

Public class :class:`~nav.nav_model.stars.nav_model_stars.NavModelStars`, base
:class:`~nav.nav_model.nav_model.NavModel`. Self-registers via ``__init_subclass__``.

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

Annotation helpers
------------------

:class:`~nav.nav_model.stars.nav_model_stars.NavModelStars` derives directly from
:class:`~nav.nav_model.nav_model.NavModel` and carries its own annotation helpers
(unlike the body and ring families, which share helpers via an abstract base).

- ``_build_annotations`` — builds the per-star
  :class:`~nav.annotation.annotations.Annotations` collection: a rectangle outline at
  every predicted star position sized by its
  :attr:`~nav.feature.flags.StarFlags.psf_size`, plus per-star labels carrying the
  catalog name and visual magnitude. Stars flagged with a body / ring conflict are
  skipped (they are surfaced in the per-image metadata for reviewer awareness but not
  drawn). Consumes the ``label_*`` and ``label_star_color`` keys documented above.
- ``_extfov_indices`` — converts a star's predicted ``(u, v)`` to extfov-frame indices
  for the rectangle drawer.
- The per-star label string is built by the module-level ``_star_label`` helper, which
  picks one of the four arrow directions
  (:data:`~nav.annotation.annotation_text_info.TEXTINFO_TOP_ARROW` /
  :data:`~nav.annotation.annotation_text_info.TEXTINFO_BOTTOM_ARROW` /
  :data:`~nav.annotation.annotation_text_info.TEXTINFO_LEFT_ARROW` /
  :data:`~nav.annotation.annotation_text_info.TEXTINFO_RIGHT_ARROW`) based on the
  star's distance from the frame edge.

Per-image metadata
------------------

:meth:`~nav.nav_model.stars.nav_model_stars.NavModelStars.create_model` populates
:attr:`~nav.nav_model.nav_model.NavModel.metadata` with the following entries for the
curator to surface in the per-image JSON sidecar:

- ``start_time`` / ``end_time`` / ``elapsed_time_sec`` — wall-clock timing for the model
  build.
- ``star_count`` — int, number of stars that survived catalog reduction, conflict
  marking, and the predicted-SNR floor.
- ``stars`` — list[dict], one entry per surviving star. Each entry carries
  ``catalog_name``, ``unique_number``, ``pretty_name``, ``ra_deg``, ``dec_deg``,
  ``vmag``, ``u``, ``v``, ``move_u``, ``move_v``, ``spectral_class``, and ``conflicts``
  (the comma-separated body- / ring-occlusion flag string built from the per-star
  conflict marking step).

Conflict-flagged stars stay in the metadata so a reviewer can see which stars were
predicted but excluded; the upstream ``usable_stars`` filter consulted by every star
technique drops them from the autonomous matching path.

Call path
---------

Call path traced through
:meth:`~nav.nav_model.stars.nav_model_stars.NavModelStars.create_model`:

1. Open a logged section. Read the per-image observation epoch and configured catalog
   list.
2. Compute the per-image motion smear vector via
   :func:`~nav.nav_model.stars.smeared_psf.compute_smear_vector_px`. When the smear length
   exceeds ``max_smear`` the model raises so the orchestrator skips star navigation rather
   than emit unreliable predictions.
3. Reduce the configured catalogs into a deduplicated star list via
   :func:`~nav.nav_model.stars.catalog.reduce_catalogs`. The function returns a list of
   mutable star records carrying RA/Dec, magnitude, spectral class, parallax, proper
   motion, and per-catalog provenance.
4. Mark body / ring conflicts on each star via
   :func:`~nav.nav_model.stars.conflicts.mark_body_and_ring_conflicts`. The function
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
    Single bright star (Vega) in an otherwise empty FOV. The star model emits one STAR
    feature for Vega with high predicted SNR and no body / ring conflict flags. The
    pass-1 :class:`~nav.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`
    consumes the feature in its 1-star path and reports the operator-verified offset
    :math:`(\Delta v, \Delta u) = (3.06, -0.02)` px. Other catalog stars in the WAC's
    extended FOV are below the magnitude floor and produce no STAR feature.

``star_dominated`` (Cassini ISS WAC, image ``W1580760393_1``)
    Dense star field with no body in FOV. The star model emits dozens of STAR features
    above the magnitude floor. The pass-1
    :class:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` consumes
    the cohort and reports the operator-verified offset
    :math:`(\Delta v, \Delta u) = (-2.68, -3.68)` px via the triplet-hash matcher.

``below_resolution_body`` (Cassini ISS NAC, image ``N1777325846_1``)
    Mimas in the lower left at ~20 px diameter at phase 72 degrees. The star model emits
    STAR features for catalog stars in the FOV; stars whose predicted positions fall
    inside the predicted Mimas silhouette are flagged ``in_body_silhouette`` and the
    upstream ``usable_stars`` filter drops them so
    the star techniques do not consume them.
