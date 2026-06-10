==================
Image Library
==================

The image library is the operator-curated regression suite at
``tests/integration/image_library/``.  Each entry is a YAML *sidecar*
that records:

* the image's mission / camera / filter combo,
* an opaque ``pds3://`` URL resolved through ``PDS3_HOLDINGS_DIR``,
* the operator-verified ground-truth offset and its 1sigma uncertainty,
* the expected ``status``, ``confidence_tier``, and primary technique,
* and the set of techniques that must (or must not) run on the scene.

Two test layers consume it: a fast structural-invariants test
(:file:`tests/integration/test_image_library.py`) and a slow per-image
regression test (:file:`tests/integration/test_autonomous_nav.py`),
gated by the ``integration`` pytest marker.

Layout
======

The directory tree IS the registry — there is no ``manifest.yaml``::

   tests/integration/image_library/
     images/
       body_mostly_offscreen/
         W1521598221_1_CALIB.yaml
       ring_only_curved/
         N1601122100_1.yaml
       ...

Adding a sidecar to the right scene-class directory enrolls it in CI
automatically; removing a sidecar stops its tests.  The set of
scene-class subdirectories is checked against the master list in
:data:`tests.integration.sidecar.DECLARED_SCENE_CLASSES`, so
typos like ``body_overflow`` vs ``body_partial_overflow`` fail loudly
at collection time.

Sidecar schema (schema_version 1)
=================================

.. code-block:: yaml

   schema_version: 1
   image_id: W1521598221_1_CALIB
   mission: CASSINI_ISS               # CASSINI_ISS | VOYAGER_ISS | GOSSI | NHLORRI
   camera: WAC                        # NAC | WAC | SSI | NA | WA | LORRI
   filter_combo: 'CL+VIO'             # canonicalized: filters sorted, '+'-joined
   image_url: 'pds3://volumes/COISS_2xxx/COISS_2021/.../W1521598221_1_CALIB.IMG'

   scene_tags: [body_mostly_offscreen, rhea]
                                      # First tag is the primary class;
                                      # must match the directory name.

   ground_truth:
     offset_dv_px: 12.5
     offset_du_px: -3.25
     offset_uncertainty_px: 1.0       # 1sigma; the test's tolerance budget
     source: operator_verified
     operator: rfrench
     verified_date: 2026-04-28
     ui_version: 'rms-nav 0.1.dev0'
     notes: |
       Hand-verified limb fit, no rings in the FOV.

   expected:
     status: success                  # success | failed | conflicted
     confidence_tier: high            # high | medium | low | failed
     primary_technique: BodyLimbNav
     techniques_must_run: [BodyLimbNav]
     techniques_must_skip: [StarFieldFromCatalogNav]

The full validator lives in :mod:`tests.integration.sidecar`; malformed
fields raise :class:`~tests.integration.sidecar.SidecarValidationError`
at collection time.

Adding a new entry
==================

The recommended path is the manual-navigation dialog's
**Save as Library Entry...** button:

1. Run the manual-nav dialog on the candidate image (e.g. via the
   ``NavTechniqueManual`` interactive driver).
2. Pick the offset by hand (or accept the **Auto** result).
3. Click **Save as Library Entry...**.  A file-save dialog suggests
   ``<image_id>.yaml`` as the filename — point it at the right
   scene-class directory under
   ``tests/integration/image_library/images/<class>/``.
4. Open the saved YAML and replace every ``TODO_REPLACE_*`` placeholder
   (scene_tags, primary_technique, notes, etc.).
5. Re-run ``pytest tests/integration/test_image_library.py`` to check
   the schema; iterate until it passes.
6. With ``PDS3_HOLDINGS_DIR`` set, run
   ``pytest tests/integration/test_autonomous_nav.py -k <image_id>`` to
   check the offset assertion against the live orchestrator.

Tolerance regimes
=================

.. list-table::
   :header-rows: 1
   :widths: 50 20 30

   * - Source
     - Typical uncertainty (px)
     - Use when
   * - ``operator_verified``, sharp limb / bright stars
     - 1.0
     - majority of cases
   * - ``operator_verified``, soft features / star-poor
     - 2.0
     - high-phase / soft-edge / faint-star

The CI test tolerance is ``offset_uncertainty_px + 0.5 px`` slack on
each axis.  ``confidence_tier`` mismatches always fail (no slack — tier
is part of the calibration target).

Cross-image inference is forbidden: every sidecar's ground-truth offset
must come from manually navigating *that* image.  Spacecraft attitude
drift is non-linear at sub-second time scales, so any "between two
anchors" interpolation is unsafe at pixel precision.

Regression baselines
====================

In addition to the per-sidecar tolerance test, a separate baseline
layer records the *exact* rounded ``(offset_dv_px, offset_du_px,
confidence)`` triple in
``tests/integration/baselines/<image_id>.json``.  Comparison is
exact-equal on rounded values (``offset`` to 4 decimals, ``confidence``
to 3); the baseline schema deliberately omits
``pipeline_run_iso8601`` because that is the only provenance field that
is *not* byte-identical between identical runs.  Baseline updates
require explicit operator review on the PR.
