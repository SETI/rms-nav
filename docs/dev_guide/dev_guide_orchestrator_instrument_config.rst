==========================================================
Per-Instrument Settings (InstrumentSettings)
==========================================================

Overview
========

:class:`~nav.nav_orchestrator.instrument_config.InstrumentSettings` is the frozen
dataclass that carries per-instrument runtime parameters the orchestrator needs to navigate
an observation.  The companion function
:func:`~nav.nav_orchestrator.instrument_config.instrument_settings_from_obs` reads the
per-camera ``inst_config`` mapping that an
:class:`~nav.obs.obs_inst.ObsInst` populates from
``config_4N0_inst_*.yaml`` and returns a populated dataclass.  Every per-instrument
behavioural branch in :class:`~nav.nav_orchestrator.orchestrator.NavOrchestrator` ultimately
reads off this dataclass.

Theory
======

Per-instrument configuration is split into two layers:

- The static per-instrument YAML (``config_400_inst_coiss.yaml``,
  ``config_410_inst_gossi.yaml``, ``config_420_inst_nhlorri.yaml``,
  ``config_430_inst_vgiss.yaml``) carries the slow-moving parameters: data unit
  convention, saturation DN, classifier thresholds, camera rotation flags, and the
  DN-to-image-unit scale.
- The per-image observation snapshot
  (:class:`~nav.obs.obs_snapshot_inst.ObsSnapshotInst`) carries the fast-moving
  parameters: PSF sigma, midtime, extfov margin.  These come from the per-image instrument
  spec rather than the YAML.

:func:`~nav.nav_orchestrator.instrument_config.instrument_settings_from_obs` reads the
slow-moving keys off the per-camera YAML block and returns a populated
:class:`~nav.nav_orchestrator.instrument_config.InstrumentSettings`.  The orchestrator
calls it once per observation in ``_make_context`` and the result is consumed by:

- :class:`~nav.nav_orchestrator.image_classifier.NavImageClassifier` (via the
  ``thresholds`` field).
- The orchestrator's saturation-mask construction (via ``saturation_dn`` / ``data_units``).
- The orchestrator's pre-filter selection (per-instrument ``source_image_filter`` block).
- The :class:`~nav.nav_orchestrator.nav_context.NavContext` rotation flags
  (``fit_camera_rotation``, ``max_rotation_deg``).
- The star navigation model's predicted-SNR formula (via
  ``signal_dn_to_image_unit_scale``).

Restrictions and assumptions
----------------------------

- The per-instrument YAML must declare the ``data_units`` field; missing values raise
  :exc:`ValueError` at config-load time so the failure surfaces during process startup.
- Required numeric fields are coerced through a ``_required_float`` helper that rejects
  non-numeric, non-finite, or boolean values.
- The marker-value field accepts the literal string ``"NaN"`` for calibrated-IF
  instruments where the missing-data sentinel is :class:`float` ``NaN``; numeric values
  are coerced through :class:`float`.
- The dataclass is frozen; new instances are returned per observation rather than
  mutated.

Sources of uncertainty
----------------------

The dataclass carries no uncertainty.

Configuration
=============

The per-instrument YAML schema consumed by
:func:`~nav.nav_orchestrator.instrument_config.instrument_settings_from_obs`:

- ``data_units`` — str, one of ``'raw_dn'`` or ``'calibrated_if'``.  ``raw_dn`` exposes
  pixels in raw counts; ``calibrated_if`` exposes pixels in calibrated I/F reflectance.
- ``noise.saturation_dn`` — float, optional.  Per-instrument full-well DN.  Required for
  ``raw_dn`` instruments; ``None`` for ``calibrated_if`` (where saturation cannot be
  identified post-CALIB).
- ``noise.marker_value`` — int / float / str.  Missing-data sentinel value.  For raw
  instruments typically 0; for calibrated-IF the literal ``"NaN"`` (or ``null``) becomes
  :class:`float` ``NaN``.
- ``image_quality_thresholds`` — block consumed by
  :class:`~nav.nav_orchestrator.image_classifier.ImageQualityThresholds`.  See
  :doc:`dev_guide_orchestrator_image_classifier` for the field-by-field schema.
- ``camera_rotation.fit_camera_rotation`` — bool, default ``False``.  When ``True`` every
  technique adds in-plane camera rotation as a third parameter.  Cassini ISS / NHLORRI
  default to ``False``; VGISS / GOSSI default to ``True``.
- ``camera_rotation.max_rotation_deg`` — float, default ``5.0`` deg.  Maximum allowed
  rotation magnitude when ``fit_camera_rotation`` is ``True``.
- ``signal_dn_to_image_unit_scale`` — float, default ``1.0``.  Per-camera scale factor
  converting an integrated DN signal into the image's native units.  ``1.0`` for
  ``raw_dn`` instruments; the per-camera CALIB-pipeline scale factor for
  ``calibrated_if`` instruments (typically of order :math:`10^{-7}` for Cassini ISS
  CALIB).

Implementation
==============

Source file: ``src/nav/nav_orchestrator/instrument_config.py`` —
:class:`~nav.nav_orchestrator.instrument_config.InstrumentSettings`,
:func:`~nav.nav_orchestrator.instrument_config.instrument_settings_from_obs`, and the
``DataUnits`` Literal alias.

Public surface (autodocumented at :doc:`/api_reference/api_nav_orchestrator`):

- :class:`~nav.nav_orchestrator.instrument_config.InstrumentSettings` — frozen dataclass.
  Public fields:

  - :attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.data_units` —
    ``'raw_dn'`` or ``'calibrated_if'``.
  - :attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.saturation_dn` —
    float or ``None``.  Saturation DN; ``None`` for calibrated-IF.
  - :attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.marker_value` —
    float.  Missing-data sentinel.
  - :attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.thresholds` —
    :class:`~nav.nav_orchestrator.image_classifier.ImageQualityThresholds` for the
    classifier.
  - :attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.fit_camera_rotation`
    — bool.
  - :attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.max_rotation_deg` —
    float.
  - :attr:`~nav.nav_orchestrator.instrument_config.InstrumentSettings.signal_dn_to_image_unit_scale`
    — float.  Default ``1.0``.

- :func:`~nav.nav_orchestrator.instrument_config.instrument_settings_from_obs` — reads the
  per-camera YAML mapping off ``obs.inst_config`` and returns a populated
  :class:`~nav.nav_orchestrator.instrument_config.InstrumentSettings`.

The module's private helpers ``_coerce_marker_value`` and ``_required_float`` validate
the YAML values; the public function is the only entry point external callers use.

Examples
========

**Cassini ISS NAC.**  ``config_400_inst_coiss.yaml`` declares
``data_units: raw_dn``, ``noise.saturation_dn: 4095.0``, ``noise.marker_value: 0``,
``camera_rotation.fit_camera_rotation: false``,
``camera_rotation.max_rotation_deg: 5.0``,
``signal_dn_to_image_unit_scale: 1.0``.
:func:`~nav.nav_orchestrator.instrument_config.instrument_settings_from_obs` returns::

    InstrumentSettings(
        data_units='raw_dn',
        saturation_dn=4095.0,
        marker_value=0.0,
        thresholds=ImageQualityThresholds(...),
        fit_camera_rotation=False,
        max_rotation_deg=5.0,
        signal_dn_to_image_unit_scale=1.0,
    )

**Voyager ISS.**  ``config_430_inst_vgiss.yaml`` declares
``camera_rotation.fit_camera_rotation: true`` and
``camera_rotation.max_rotation_deg: 10.0``.  The orchestrator's per-image NavContext
inherits ``fit_camera_rotation=True`` and every technique runs the 3-DoF path, so
:class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav` reports a 3x3 covariance
on Voyager imagery.

**Cassini ISS CALIB pipeline.**  When the operator runs the calibrated-IF pipeline,
``data_units: calibrated_if``, ``noise.saturation_dn: null``, ``noise.marker_value: NaN``,
``signal_dn_to_image_unit_scale`` carries the per-camera CALIB-pipeline scale (~``5e-7``).
The orchestrator's saturation-mask helper returns an empty mask (saturation cannot be
identified post-CALIB), and the star navigation model's predicted-SNR formula multiplies
its DN-keyed catalog signal by ``signal_dn_to_image_unit_scale`` to bring it into
calibrated-IF units before forming the SNR ratio.
