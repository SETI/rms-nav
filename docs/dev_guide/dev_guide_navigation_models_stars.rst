======================
Star Navigation Model
======================

Overview
========

The star navigation model predicts the pixel positions of catalog stars in the field of view and
emits one point feature per usable star.  It reduces the configured star catalogs into a
deduplicated list, projects each star into image coordinates including stellar aberration and
proper motion, flags stars occluded by a body silhouette or an opaque ring annulus, and emits a
``STAR`` feature for every star within the per-image limiting magnitude.  Star techniques then
match these predicted points against centroided sources in the real image to recover the pointing
offset.

The orchestrator builds exactly one
:py:class:`~nav.nav_model.stars.nav_model_stars.NavModelStars` per observation;
:py:meth:`~nav.nav_model.stars.nav_model_stars.NavModelStars.instances_for_obs` always returns a
single instance.  That one model emits as many ``STAR`` features as there are catalog stars inside
the extended field of view that pass the magnitude and smear gates.

Theory
======

A star is a point source at a known celestial position.  Predicting where it lands on the sensor
requires three corrections.  Proper motion advances the catalog position to the observation epoch.
Stellar aberration shifts the apparent direction by the spacecraft's velocity relative to the
solar system barycentre.  The field-of-view projection then maps the corrected direction to a
pixel, and the per-exposure pointing motion gives a smear vector along which the star trails.

Stars are pulled from several catalogs in order of decreasing astrometric precision.  Within each
catalog the search walks coarse magnitude bins from bright to faint until a star budget is
reached.  Across catalogs, a star whose position and magnitude match one already kept from a more
precise catalog is dropped as a duplicate.  Stars whose point-spread support would spill off the
edge of the extended field of view, or whose smear motion would carry them off the edge during the
exposure, are culled.  When two stars are close enough that their point-spread supports overlap,
the fainter one is flagged; if they are also of similar brightness, both are flagged, because each
blinds the centroid fit of the other.

Whether a star is usable is decided purely by brightness.  A star is kept when its catalog
magnitude is at or brighter than the per-image limiting magnitude, which the observation derives
from its noise level and exposure.  This is a magnitude gate, not a measured signal-to-noise gate:
no image pixel values enter the decision, so it carries no dependence on any conversion between
raw detector counts and calibrated image units.

The centroiding precision of each kept star is still expressed as an effective signal-to-noise,
synthesised from how far the star sits below the limiting magnitude.  A star exactly at the limit
is assigned a reference value; every additional magnitude of brightness headroom multiplies the
effective signal-to-noise by one Pogson flux ratio per magnitude.  This magnitude-margin
effective signal-to-noise drives two quantities.  First, the per-star centroid covariance, which
follows the Cramer-Rao lower bound for a smeared Gaussian source: the across-smear standard
deviation is the point-spread sigma divided by the square root of the effective signal-to-noise,
and the along-smear standard deviation adds the smear length's variance contribution in quadrature
before the same division.  The covariance ellipse is rotated so its major axis aligns with the
smear vector; below a minimum smear length the ellipse collapses to isotropic.  Second, the
reliability scalar, a sigmoid of the effective signal-to-noise centred a few magnitudes above the
limit, multiplied by hard-zero factors for body occlusion, ring occlusion, and saturation or
cosmic-ray contamination.

The reported covariance captures the photon-noise-limited centroiding precision of an isolated,
well-modelled star.  It does not model the additional error from a blended neighbour, an
imperfect point-spread model, or residual smear-model error; those cases are instead handled by
excluding the star from the feature list or by zeroing its reliability.

Configuration
=============

The star model reads its parameters from the ``stars`` section of
``src/nav/config_files/config_030_stars.yaml``.

- ``catalogs`` — list, default ``[ucac4, tycho2, ybsc]``.  Catalog search order from most to
  least astrometrically precise; earlier catalogs win on duplicate conflicts.
- ``body_conflict_margin`` — int, default ``5`` px.  Margin around a body silhouette within which a
  star is treated as body-occluded.
- ``default_star_class`` — string, default ``G0`` (dimensionless).  Spectral class assumed for stars
  with no catalog temperature, used to synthesise a colour.
- ``stellar_aberration`` — bool, default ``true`` (dimensionless).  When true each catalog position
  is aberration-corrected into the spacecraft frame before projection.
- ``proper_motion`` — bool, default ``true`` (dimensionless).  When true catalog positions are
  advanced to the observation epoch by their proper-motion vectors.
- ``max_stars`` — int, default ``100`` (count).  Budget on the reduced star list; the catalog walk
  stops and the final list is capped at this count.
- ``max_movement_steps`` — int, default ``50`` (count).  Cap on the number of smear-integration
  samples along a star trail.
- ``label_font`` — string, default ``liberation2/LiberationMono-Bold.ttf``.  Font file for star
  labels.
- ``label_font_size`` — int, default ``18`` pt.  Star label font size.
- ``label_font_color`` — RGB triple, default ``[255, 0, 0]``.  Star label text colour.
- ``label_star_color`` — RGB triple, default ``[255, 0, 0]``.  Colour of the drawn star-box overlay.
- ``duplicate_ra_dec_threshold_arcsec`` — float, default ``5`` arcsec.  Angular separation below
  which two catalog stars are candidate duplicates.
- ``duplicate_vmag_threshold`` — float, default ``3`` mag.  Magnitude difference below which two
  matched-position stars are treated as the same star.
- ``overlapping_vmag_threshold`` — float, default ``2`` mag.  Magnitude difference below which two
  visually overlapping stars are both flagged rather than only the fainter one.
- ``calibrated_data`` — bool, default ``true`` (dimensionless).  Marks the image as calibrated for
  the star pipeline's downstream photometry helpers.
- ``float_psf_sigma`` — bool, default ``false`` (dimensionless).  Whether to fit a floating
  point-spread sigma during detection.
- ``search_multipliers`` — list, default ``[0.25, 0.5, 0.75, 1.0]``.  Search-radius multipliers used
  by downstream star matching.
- ``perform_photometry`` — bool, default ``true`` (dimensionless).  Whether downstream detection
  performs aperture photometry.
- ``try_without_photometry`` — bool, default ``false`` (dimensionless).  Whether to retry matching
  without photometry on failure.
- ``min_stars_low_confidence`` — list, default ``[3, 0.75]``.  Star-count and confidence pair for the
  low-confidence match tier.
- ``min_stars_high_confidence`` — list, default ``[6, 1.0]``.  Star-count and confidence pair for the
  high-confidence match tier.
- ``min_confidence`` — float, default ``0.9`` dimensionless.  Minimum match confidence accepted by
  the star techniques.
- ``psf_gain`` — list, default ``[5000, 4]``.  Gain pair used by the point-spread model during
  detection.
- ``max_smear`` — float, default ``100`` px.  Smear length above which a star is unfittable and
  dropped from the feature list.
- ``min_vmag`` — float, default ``5.0`` mag.  Bright end of the catalog magnitude search window.
- ``max_vmag`` — float, default ``15.0`` mag.  Faint end of the catalog magnitude search window.
- ``vmag_increment`` — float, default ``0.5`` mag.  Magnitude step used when walking the catalog
  search window.
- ``max_star_dn`` — float, default ``100000.0`` DN.  Detector-count ceiling above which a star is
  treated as saturated by downstream photometry.
- ``min_dn_force_one_star`` — float, default ``25000.0`` DN.  Detector-count floor that forces a
  single-star match path downstream.
- ``star_body_conflict_margin`` — int, default ``3`` px.  Margin used by the star-versus-body
  conflict check.
- ``too_bright_dn`` — float, default ``1000`` DN.  Detector-count level above which a star is treated
  as too bright for the standard centroid.
- ``too_bright_factor`` — float, default ``1`` dimensionless.  Scale applied in the too-bright
  handling path.
- ``ring_occlusion_enabled`` — bool, default ``true`` (dimensionless).  When true, stars whose
  predicted pixel falls inside a configured opaque ring annulus are flagged ring-occluded.
- ``ring_occlusion_radii_km`` — mapping, per-planet annulus pairs.  Inner/outer radius pairs
  (kilometres) of the opaque ring annuli used by the occlusion check.

The star model also reads ``min_predicted_snr`` from this section with a default of ``0.0`` when
the key is absent; it scales the reliability-breakdown contribution score but does not act as a
gate.  The key is not present in the shipped YAML.

Implementation
==============

Source files: the orchestrator ``src/nav/nav_model/stars/nav_model_stars.py`` and the helper
modules in the ``stars`` subpackage — ``catalog.py`` (multi-catalog reduction, aberration, proper
motion, projection, dedup), ``conflicts.py`` (body and ring occlusion marking),
``predicted_snr.py`` (point-spread and raw-detector-count photometry helpers), ``smeared_psf.py``
(smear vector and smear-aware point-spread rendering), and ``detection.py`` (a DAOPHOT-style
detector used by downstream techniques, not by this model).

The public class is :py:class:`~nav.nav_model.stars.nav_model_stars.NavModelStars`, a subclass of
:py:class:`~nav.nav_model.nav_model.NavModel`.

:py:meth:`~nav.nav_model.stars.nav_model_stars.NavModelStars.create_model` records timing
metadata, reduces the catalogs via
:py:func:`~nav.nav_model.stars.catalog.reduce_catalogs`, computes the per-image smear vector via
:py:func:`~nav.nav_model.stars.smeared_psf.compute_smear_vector_px`, marks body and ring conflicts
via :py:func:`~nav.nav_model.stars.conflicts.mark_body_and_ring_conflicts`, and stores the star
count and per-star summaries in metadata.  The reduced list is exposed through the
:py:attr:`~nav.nav_model.stars.nav_model_stars.NavModelStars.stars` property.

:py:meth:`~nav.nav_model.stars.nav_model_stars.NavModelStars.to_features` walks the reduced list.
For each star it applies the magnitude gate against the observation's limiting magnitude, computes
the magnitude-margin effective signal-to-noise, applies the smear gate against ``max_smear``,
projects the star into extfov coordinates, reads the body, ring, saturation, and cosmic-ray flags,
builds the anisotropic Cramer-Rao covariance, and emits one
:py:class:`~nav.feature.feature.NavFeature` of type
:py:attr:`~nav.feature.feature_type.NavFeatureType.STAR` carrying a
:py:class:`~nav.feature.geometry.StarGeometry`, the covariance, a reliability scalar, a
:py:class:`~nav.feature.feature.NavReliabilityBreakdown`, and a
:py:class:`~nav.feature.flags.StarFlags` block.  The single emitted
:py:class:`~nav.feature.feature_type.NavFeatureType` is ``STAR``.  The covariance collapses to
isotropic below :py:data:`~nav.feature.constants.MIN_ANISOTROPIC_SMEAR_PX`.

:py:meth:`~nav.nav_model.stars.nav_model_stars.NavModelStars.to_annotations` draws a box overlay
plus a name-and-magnitude label for each unconflicted star on the summary image.

The reliability gate is the sigmoid of the effective signal-to-noise multiplied by hard-zero
factors: a star flagged as body-occluded, ring-occluded, or saturated/cosmic-contaminated gets
reliability zero regardless of its brightness.  Stars fainter than the limiting magnitude or with
no catalog magnitude are skipped before reaching the gate, as are stars whose smear exceeds the
configured maximum.

The :py:mod:`~nav.nav_model.stars.predicted_snr` module is retained for its reusable photometry
helpers — :py:func:`~nav.nav_model.stars.predicted_snr.psf_sigma_px`,
:py:func:`~nav.nav_model.stars.predicted_snr.psf_aperture_pixels`,
:py:func:`~nav.nav_model.stars.predicted_snr.integrated_signal_dn`, and the spectral-class colour
re-export — and its :py:func:`~nav.nav_model.stars.predicted_snr.predicted_snr` formula is kept as
a raw-detector-count diagnostic.  It is not consulted by the magnitude gate.

Examples
========

**one_bright_star_no_body** (``W1449079117_1_CALIB``).  A single bright star (Vega) in a wide-angle
field through the red filter, with no body present.  The catalog reduction keeps the one star, it
clears the magnitude gate comfortably, and the model emits a single ``STAR`` feature with high
effective signal-to-noise and an anisotropic centroid covariance aligned to the per-image smear
vector.  The sidecar's expected primary technique is the prior-refinement star navigator.

**star_dominated** (``W1580760393_1_CALIB``).  A dense star field in a wide-angle clear-filter
frame.  The reduction returns many catalog stars, each projected and dedup-checked; the model
emits one ``STAR`` feature per star that clears the magnitude and smear gates, with visually
overlapping pairs carrying the overlap conflict flag that zeroes their reliability.  The sidecar's
expected primary technique is the catalog star-field pattern matcher.
