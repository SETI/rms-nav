=====
Stars
=====

:class:`~nav.nav_model.stars.nav_model_stars.NavModelStars` emits one
:class:`~nav.feature.feature.NavFeature` per catalog star predicted to
fall in the extended FOV.  Each feature carries a
:class:`~nav.feature.geometry.StarGeometry` payload, an anisotropic
Cramer-Rao centroid covariance, a predicted-SNR-driven reliability
score, and a :class:`~nav.feature.flags.StarFlags` block with
``saturated``, ``smear_length_px``, ``in_body_silhouette``, and
``in_saturation_or_cosmic_mask`` fields.  Three techniques consume STAR
features: ``StarFieldFromCatalogNav`` (similarity-invariant triplet
pattern match for ≥ 3 stars), ``StarUniqueMatchNav`` (catalog-uniqueness
1- or 2-star match), and ``StarRefineNav`` (pass-2 refinement on a
prior offset).

The :mod:`nav.nav_model.stars` package is split by responsibility:

.. list-table:: Star NavModel modules
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Responsibility
   * - :mod:`nav.nav_model.stars.catalog`
     - Stellar aberration via
       :func:`~nav.nav_model.stars.catalog.aberrate_star`, proper-motion
       evaluation through
       :func:`~nav.nav_model.stars.catalog.select_radec_list`, and
       multi-catalog reduction (UCAC4 / Tycho-2 / YBSC) through
       :func:`~nav.nav_model.stars.catalog.reduce_catalogs`.  Catalog
       precedence and per-catalog deduplication thresholds come from
       ``config.stars``.
   * - :mod:`nav.nav_model.stars.conflicts`
     - Body intercept and ring-occlusion checks through
       :func:`~nav.nav_model.stars.conflicts.mark_body_and_ring_conflicts`.
       Per-planet ring annuli are read from
       ``config.stars.ring_occlusion_radii_km`` and validated by
       :func:`~nav.nav_model.stars.conflicts.parse_ring_occlusion_annuli`.
   * - :mod:`nav.nav_model.stars.predicted_snr`
     - Per-star integrated SNR estimate using ``obs.star_psf()``,
       :class:`~nav.nav_orchestrator.nav_context.NavContext.image_noise_sigma`,
       and the ``SCLASS_TO_B_MINUS_V`` spectral-class colour lookup.
       Accepts an optional per-camera-per-filter ``mag_offset`` that
       converts the catalog V-band magnitude to the instrument's
       bandpass.
   * - :mod:`nav.nav_model.stars.smeared_psf`
     - Smear-aware PSF rendering via ``psf.eval_rect(movement=...)`` and
       per-image smear-vector computation from the SPICE pointing
       brackets.
   * - :mod:`nav.nav_model.stars.detection`
     - DAOPHOT-style source detector (matched filter, local maxima,
       Gaussian centroid fit, saturated-star annular moment, CCD bloom
       column detection, sharpness / roundness shape cuts).  Used by
       downstream techniques that sweep the image for centroidable
       stars when the catalog match is ambiguous; not run during
       ``to_features``.
   * - :mod:`nav.nav_model.stars.nav_model_stars`
     - Thin orchestrator implementing the
       :class:`~nav.nav_model.nav_model.NavModel` ABC.  ``create_model``
       reduces the catalogs and marks conflicts; ``to_features`` emits
       one ``STAR`` feature per usable catalog star;
       ``to_annotations`` builds star-box overlays plus name and
       magnitude labels.

Position covariance
-------------------

The 2x2 ``position_cov_px`` for a STAR feature is the Cramer-Rao Lower
Bound of the centroid estimate, as specified in Part 1's "Position
covariance per feature type" section:

.. code-block::

   sigma_along_smear  = sqrt(L^2 / 12 + sigma_PSF^2) / sqrt(SNR)
   sigma_across_smear = sigma_PSF / sqrt(SNR)

where ``L = hypot(move_v, move_u)`` is the smear length in pixels.  The
covariance is rotated so the major axis aligns with the smear vector.
When the smear length is below
:data:`~nav.feature.constants.MIN_ANISOTROPIC_SMEAR_PX`, the
covariance collapses to an isotropic ``(sigma_PSF / sqrt(SNR))^2 * I``.

Reliability and gating
----------------------

The reliability is a sigmoid of the predicted SNR multiplied by hard
zero terms for body or ring occlusion, saturation, and cosmic-ray hits.
Stars whose smear length exceeds ``stars.max_smear`` are skipped from
the feature list entirely (the smear-aware PSF cannot fit a usable
centroid).  Stars whose predicted SNR is below
``stars.min_predicted_snr`` (default ``0`` keeps all) are likewise
skipped.

The
:class:`~nav.feature.feature.NavReliabilityBreakdown` block on each
emitted feature carries the per-component contributions
(``predicted_snr``, ``in_body_silhouette``, ``in_saturation_or_cosmic``,
``smear_length_ok``).
