===================
Instrument Settings
===================

Overview
========

:py:class:`~nav.nav_orchestrator.instrument_config.InstrumentSettings` is the frozen, already-resolved
bundle of per-instrument values the orchestrator needs at navigate time: the data-unit kind, the
saturation DN (when applicable), the missing-data sentinel, the image-quality thresholds, and the
camera-rotation flags.  It exists so the orchestrator can branch on a resolved data-unit kind and
hand the classifier a ready-made threshold set without re-reading raw YAML.

The constructor is the free function
:py:func:`~nav.nav_orchestrator.instrument_config.instrument_settings_from_obs`, which reads the
per-camera config mapping carried on the observation (``obs.inst_config``, populated when the image
was loaded) and resolves it into the dataclass.  The consumer is the orchestrator's
``NavOrchestrator._make_context``, which uses the resolved data-unit kind to choose the
saturation-mask policy and the marker-value sanitisation, and forwards the thresholds and rotation
flags into the :py:class:`~nav.nav_orchestrator.nav_context.NavContext` it builds.

Theory
======

The settings encode one branching convention: the data-unit kind.  An instrument exposes its
pixels either in raw analog-to-digital counts or in calibrated incidence-corrected reflectance.
The raw-count kind carries a meaningful full-well saturation level and DN-keyed quality thresholds.
The calibrated-reflectance kind does not: the same physical full-well count maps to a different
reflectance value for every combination of exposure time, filter, and gain, so a single
reflectance saturation threshold would be meaningless.  For the calibrated kind the saturation gate
is therefore disabled outright, and the classifier is handed an infinite saturation threshold so
the overexposed fraction is always zero and the fully-overexposed early-out cannot fire; the
reflectance-keyed blank and noise thresholds are used in place of the count-keyed ones.

The missing-data sentinel differs by kind as well: raw instruments mark dropped pixels with a
numeric value (typically zero), while calibrated instruments use a not-a-number sentinel.  All
threshold values that survive resolution are required to be finite, because they are compared
directly against pixel intensities downstream and a non-finite threshold would silently corrupt
the comparison.

Configuration
=============

:py:func:`~nav.nav_orchestrator.instrument_config.instrument_settings_from_obs` reads three blocks
from the per-camera section of ``config_4N0_inst_*.yaml`` (for example
``config_400_inst_coiss.yaml``).  The keys it consumes are:

- ``data_units`` — string, one of ``raw_dn`` or ``calibrated_if`` (dimensionless).  Selects the
  saturation-mask policy, the sentinel kind, and which threshold variants apply.
- ``fit_camera_rotation`` — bool, default ``false`` (dimensionless).  Enables 3-DoF technique fits
  that add in-plane camera rotation as a third parameter.
- ``max_rotation_deg`` — float, default ``5.0`` degrees.  Maximum rotation magnitude when rotation
  is fitted; rejected if non-finite or not positive.

Under the ``noise`` block:

- ``saturation_dn`` — float, required for ``raw_dn`` instruments.  Per-instrument full-well DN used
  to build the saturation mask; not read for ``calibrated_if`` instruments.
- ``marker_value`` — numeric or the string ``NaN``, default ``0`` for ``raw_dn`` and ``NaN`` for
  ``calibrated_if``.  The missing-data sentinel; higher specificity here tightens which pixels are
  treated as dropped.

Under the ``image_quality_thresholds`` block:

- ``saturation_threshold_dn`` — float, required for ``raw_dn``.  Count above which a pixel counts
  as overexposed for the clean-fraction gate; unread for ``calibrated_if`` (an infinite threshold
  is substituted, and declaring ``saturation_threshold_if`` is rejected).
- ``blank_max_dn`` / ``blank_max_if`` — float, required.  Intensity below which the frame is judged
  blank; the ``_dn`` variant applies to ``raw_dn``, the ``_if`` variant to ``calibrated_if``.
- ``noisy_threshold_dn`` / ``noisy_threshold_if`` — float, required.  Noise-sigma level above which
  the frame is judged noisy; the ``_dn`` variant applies to ``raw_dn``, the ``_if`` variant to
  ``calibrated_if``.
- ``max_missing_frac_clean`` — float, default ``0.30`` (fraction).  Maximum missing fraction a
  frame may carry and still be ranked clean; lowering it tightens the clean gate.
- ``max_overexposed_frac_clean`` — float, default ``0.80`` (fraction).  Maximum overexposed
  fraction a frame may carry and still be ranked clean.
- ``partial_dropout_min_frac`` — float, default ``0.05`` (fraction).  Missing fraction above which
  the partial-dropout class fires.

When ``obs.inst_config`` is absent (simulated or test obs without per-instrument wiring) the
function returns ``raw_dn`` defaults with no saturation mask and the standard
:py:class:`~nav.nav_orchestrator.image_classifier.ImageQualityThresholds` defaults.  These keys are
overridden per instrument by editing the corresponding ``config_4N0_inst_*.yaml`` camera block.

Implementation
==============

Source file: ``src/nav/nav_orchestrator/instrument_config.py``.  Public class
:py:class:`~nav.nav_orchestrator.instrument_config.InstrumentSettings`, a frozen
:py:func:`dataclasses.dataclass`.  The module also defines the ``DataUnits`` literal alias for the
two recognised data-unit kinds.

The public fields are :py:attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.data_units`,
:py:attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.saturation_dn` (``None`` for
calibrated-IF), :py:attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.marker_value`,
:py:attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.thresholds` (an
:py:class:`~nav.nav_orchestrator.image_classifier.ImageQualityThresholds`),
:py:attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.fit_camera_rotation`, and
:py:attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.max_rotation_deg`.  The
dataclass defines no ``__post_init__`` and no classmethods; it is a plain resolved container.

The public entry point :py:func:`~nav.nav_orchestrator.instrument_config.instrument_settings_from_obs`
reads ``obs.inst_config``, returns the untyped ``raw_dn`` default when it is absent, and otherwise
validates that the mapping carries a recognised ``data_units`` value and a
``image_quality_thresholds`` block.  On the ``raw_dn`` path it requires a ``noise`` block and reads
the count-keyed thresholds; on the ``calibrated_if`` path it substitutes an infinite saturation
threshold, reads the reflectance-keyed thresholds, and rejects any explicit reflectance saturation
threshold.  Validation raises :py:exc:`ValueError` for missing or unrecognised fields and
:py:exc:`TypeError` when a block has the wrong shape.  The module's private helpers — the
marker-value coercion and the required-finite-float reader — perform the per-field parsing and
finiteness checks.

Examples
========

For a Cassini ISS narrow-angle raw frame, ``obs.inst_config`` carries ``data_units: raw_dn``, a
``noise`` block with ``saturation_dn: 4095`` and ``marker_value: 0``, and an
``image_quality_thresholds`` block with ``saturation_threshold_dn: 4095``, ``blank_max_dn: 5.0``,
and ``noisy_threshold_dn: 10.0``.  The resolved
:py:class:`~nav.nav_orchestrator.instrument_config.InstrumentSettings` then has
:py:attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.data_units` equal to
``'raw_dn'``, :py:attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.saturation_dn`
equal to ``4095.0``, and
:py:attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.marker_value` equal to
``0.0``; the orchestrator builds a real saturation mask from the full-well DN.

For the matching calibrated frame in ``body_partial_overflow`` (``N1484593951_2_CALIB``),
``obs.inst_config`` carries ``data_units: calibrated_if``, a ``noise`` block with
``marker_value: NaN``, and reflectance-keyed ``blank_max_if`` / ``noisy_threshold_if`` thresholds.
The resolved settings have
:py:attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.saturation_dn` equal to
``None`` and a :py:attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.thresholds`
whose saturation threshold is infinite, so the orchestrator emits an empty saturation mask and the
overexposed early-out cannot fire.
