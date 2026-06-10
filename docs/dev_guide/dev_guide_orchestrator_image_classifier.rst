================
Image Classifier
================

Overview
========

The image classifier is the orchestrator's quick-fail front gate.  It looks at the whole sensor
area of an incoming image -- never at any predicted feature position -- and assigns it to one of a
small closed set of classes.  Most of the "bad" classes terminate navigation in milliseconds with a
clear reason, before any model renders or any technique runs.  The classifier reads three cheap
statistics (saturated-pixel fraction, missing-data fraction, and a robust noise estimate), applies
per-instrument thresholds, and returns a verdict that rides on every
:py:class:`~nav.nav_orchestrator.nav_result.NavResult` so a downstream reader always sees which
class the input fell into.

Theory
======

The classifier is global: it computes summary statistics over the sensor pixels and decides on
those alone.  Three quantities drive the decision.  The saturated fraction is the share of sensor
pixels at or above the saturation threshold.  The missing fraction is the share of sensor pixels
equal to the instrument's missing-data sentinel (detected as an exact equality, or as a
not-a-number test when the sentinel is itself not-a-number, as it is for calibrated reflectance
imagery).  The noise level is a robust estimate based on the median absolute deviation of the
sensor pixels, which resists outliers from a handful of bright sources.

The outcome is decided by an ordered cascade.  A blank check runs first: if the maximum sensor
value is below a floor, the image is blank and nothing further is evaluated -- this order matters,
because a near-zero frame whose missing-data marker is also zero would otherwise be mislabelled as
missing-data-dominated.  Next, if the saturated fraction exceeds its cap, the image is fully
overexposed.  Next, if the missing fraction exceeds its cap, the image is missing-data-dominated.
If none of those fire, the image is clean.  A clean image may still carry advisory flags that do not
change its class: a partial-dropout flag when the missing fraction sits above an advisory floor but
below the dominated cap, and a noisy flag when the robust noise estimate exceeds its threshold.  One
further class, corrupt, is not produced by the statistics at all; the caller sets it when reading
the image file itself raised.

The classifier makes no geometric assumptions and reads no SPICE state, so it is fast and pure.  Its
limitation is exactly that globality: it cannot distinguish a usable bright target from saturation
spread across the frame beyond what the fraction caps encode, and it deliberately leaves
finer-grained dropout patterns to the downstream extractors.

Configuration
=============

The classifier's thresholds are carried on
:py:class:`~nav.nav_orchestrator.image_classifier.ImageQualityThresholds`, a frozen dataclass.  The
orchestrator builds an instance per image from the per-instrument
``image_quality_thresholds`` block in ``config_4N0_inst_*.yaml`` (for example
``src/nav/config_files/config_400_inst_coiss.yaml``), normalising DN-keyed and I/F-keyed values to
the right units, unless an explicit override was passed to the orchestrator constructor.  The fields
and their dataclass defaults:

- ``saturation_threshold_dn`` -- float, default ``4095.0`` DN.  Pixels at or above this count as
  saturated; for calibrated-I/F cameras it is set to infinity so the saturation gate never fires.
- ``missing_data_marker_dn`` -- float, default ``0.0`` DN.  Pixels exactly equal to this are
  missing data; for calibrated-I/F cameras it is the not-a-number sentinel.
- ``max_saturation_frac_clean`` -- float, default ``0.80`` (dimensionless).  Above this saturated
  fraction the image is fully overexposed; lower values refuse overexposed frames sooner.
- ``max_missing_frac_clean`` -- float, default ``0.30`` (dimensionless).  Above this missing
  fraction the image is missing-data-dominated; lower values refuse dropout-heavy frames sooner.
- ``partial_dropout_min_frac`` -- float, default ``0.05`` (dimensionless).  At or above this
  missing fraction (but below the dominated cap) the partial-dropout advisory flag is raised.
- ``blank_max_dn`` -- float, default ``5.0`` DN.  If the maximum sensor value is below this, the
  image is blank; higher values reject dim frames more aggressively.
- ``noisy_threshold`` -- float, default ``10.0`` DN.  Above this robust noise level the noisy
  advisory flag is raised; the image stays clean.

For per-instrument overrides, the orchestrator reads ``image_quality_thresholds`` from the matching
``config_4N0_inst_*.yaml`` camera block.  The raw-DN cameras supply ``blank_max_dn``,
``saturation_threshold_dn``, ``noisy_threshold_dn``, and the two fraction caps; the calibrated-I/F
cameras supply I/F-keyed ``blank_max_if`` and ``noisy_threshold_if`` and omit any saturation
threshold (the saturation gate is off for calibrated reflectance imagery).  See
:doc:`dev_guide_orchestrator_instrument_config` for how those keys are read and normalised.

Implementation
==============

Source files: ``src/nav/nav_orchestrator/image_classifier.py`` and
``src/nav/nav_orchestrator/image_classifier_result.py``.

The public class is :py:class:`~nav.nav_orchestrator.image_classifier.NavImageClassifier`, a
dataclass holding its :py:class:`~nav.nav_orchestrator.image_classifier.ImageQualityThresholds`.
Its single public method,
:py:meth:`~nav.nav_orchestrator.image_classifier.NavImageClassifier.classify`, takes the image, an
optional sensor mask, and an optional pre-computed missing fraction.  It validates the inputs
(raising :py:exc:`TypeError` for a non-array or non-2-D image and :py:exc:`ValueError` for a
mismatched, empty, or all-false sensor mask), restricts the statistics to the sensor pixels,
computes the saturated and missing fractions and the robust noise sigma via
:py:func:`~nav.support.noise_estimate.estimate_image_noise_sigma`, and walks the blank /
overexposed / dominated / clean cascade, appending advisory flags on the clean branch.  When the
orchestrator supplies a pre-computed missing fraction (from the true missing mask, before the
not-a-number sentinel is sanitised for the finite-only derivative path), the classifier trusts it
rather than re-deriving it.

The verdict is a :py:class:`~nav.nav_orchestrator.image_classifier_result.NavImageClassifierResult`,
a frozen dataclass with fields
:py:attr:`~nav.nav_orchestrator.image_classifier_result.NavImageClassifierResult.image_class`,
:py:attr:`~nav.nav_orchestrator.image_classifier_result.NavImageClassifierResult.saturation_frac`,
:py:attr:`~nav.nav_orchestrator.image_classifier_result.NavImageClassifierResult.missing_frac`,
:py:attr:`~nav.nav_orchestrator.image_classifier_result.NavImageClassifierResult.noise_sigma`,
:py:attr:`~nav.nav_orchestrator.image_classifier_result.NavImageClassifierResult.max_dn`, and
:py:attr:`~nav.nav_orchestrator.image_classifier_result.NavImageClassifierResult.flags`.  The class
literal (``clean``, ``blank``, ``fully_overexposed``, ``mostly_missing_data``, ``corrupt``) and the
flag literal (``partial_dropout``, ``noisy``) are package-private type aliases in the result module.
The orchestrator maps the four hard-failure classes to the matching
:py:class:`~nav.support.status_reason.NavStatusReason` through its ``_HARD_FAILURE_TO_REASON`` table;
see :doc:`dev_guide_orchestrator_orchestrator`.

Examples
========

The ``body_partial_overflow`` scene (Cassini NAC ``N1484593951_2_CALIB``) is a clean calibrated
frame: large Rhea is partially off-frame with a good limb, the saturated fraction is zero (the
saturation gate is off for calibrated-I/F input), the missing fraction is below the partial-dropout
advisory floor, and the noise level is low.  The verdict is
:py:attr:`~nav.nav_orchestrator.image_classifier_result.NavImageClassifierResult.image_class` equal
to ``clean`` with no advisory flags, so the orchestrator proceeds to feature extraction rather than
short-circuiting.  A frame whose maximum value fell below the per-instrument blank floor would
instead return ``blank`` and trip the
:py:attr:`~nav.support.status_reason.NavStatusReason.NO_SIGNAL_IN_IMAGE` hard-failure gate before
any model ran.
