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
   mission: COISS                     # COISS | VGISS | GOSSI | NHLORRI
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
     status: ok                       # ok | failed | conflicted
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

1. Run the manual-nav dialog on the candidate image with the
   ``nav_offset [args] --manual`` CLI flag, where ``[args]`` are the
   selection / dataset / config flags that pin the run down to a
   single image (e.g. dataset id, an image-list file, ``--config`` for
   a non-default bundle).
2. Pick the offset by hand (or accept the **Auto** result).
3. Click **Save as Library Entry...**.  A file-save dialog suggests
   ``<image_id>.yaml`` as the filename — point it at the right
   scene-class directory under
   ``tests/integration/image_library/images/<class>/``.  The dialog
   also drops a companion ``<image_id>.png`` next to the YAML showing
   the red-image / green-model overlay at the chosen ``(dv, du)``;
   it's an orientation aid for future reviewers and is not consumed
   by any test.
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

Regression baselines
====================

In addition to the per-sidecar tolerance test, a separate baseline
layer records the *exact* rounded ``(offset_dv_px, offset_du_px,
confidence)`` triple in
``tests/integration/baselines/<image_id>.json``.  Comparison is
exact-equal on rounded values (``offset`` to 4 decimals, ``confidence``
to 3); the baseline schema deliberately omits
``pipeline_run_iso8601`` because that is the only provenance field that
is *not* byte-identical between identical runs.

What checks the baselines
-------------------------

Two tests under ``tests/integration/test_baselines.py``:

- ``test_every_baseline_cites_a_sidecar`` — runs in the fast suite (no
  holdings needed).  Asserts that every ``baselines/<image_id>.json``
  has a matching sidecar at
  ``image_library/images/*/<image_id>.yaml``, and that the file's stem
  matches the baseline's ``image_id`` field.  Catches the common drift
  where a sidecar is renamed or deleted but its baseline lingers.
- ``test_regression_baseline_exact_match`` — gated by the
  ``integration`` pytest marker and skipped when ``PDS3_HOLDINGS_DIR``
  is unset.  Parametrized one case per ``(baseline, sidecar)`` pair;
  runs the orchestrator against the real holdings, calls
  :meth:`tests.integration.baseline.Baseline.from_run` to round the
  fresh outputs, and asserts ``actual == expected`` (exact equality on
  all four keys).  The failure message tells the operator to update
  the JSON in the same PR if the diff is intended.

Plus a handful of round-trip / serialisation unit tests on
``Baseline.from_run`` and ``Baseline.to_json`` that pin the rounding
rule and confirm byte-stable JSON (sorted keys, trailing newline).

How a baseline is created or updated
------------------------------------

Use the ``nav_update_baselines`` CLI (registered in
``[project.scripts]``; runs from a project checkout).  It refuses to
run without ``PDS3_HOLDINGS_DIR`` set.

.. code-block:: bash

   nav_update_baselines --image-id <image_id>      # one image
   nav_update_baselines --image-id A --image-id B  # hand-picked batch
   nav_update_baselines --all                      # every sidecar
   nav_update_baselines --all --dry-run            # preview only

For each image the tool runs ``navigate_image_files`` against the live
holdings, rounds the result via
:meth:`tests.integration.baseline.Baseline.from_run`, compares against
any existing baseline, and reports one of:

* ``CREATE`` — no on-disk baseline; new file written.
* ``UPDATE`` — baseline drifted; old → new diff printed and the file
  overwritten.
* ``UNCHANGED`` — bytes match; file untouched (mtime preserved).
* ``FAILED`` — orchestrator returned no offset, or the requested
  ``--image-id`` matched no sidecar.

The exit code is ``0`` when every selected image succeeded (regardless
of write/update/unchanged), ``1`` when at least one ``FAILED`` line
was emitted, ``2`` on argument-parsing errors or when
``PDS3_HOLDINGS_DIR`` is unset.

Sidecars must land first — the
``test_every_baseline_cites_a_sidecar`` invariant refuses orphan
baselines.  Baseline updates always require explicit operator review
on the PR; the CLI is the mechanical step, but the human review of
the resulting diff (does the new offset still overlay the limb?  does
the new confidence still match ``expected.confidence_tier``?) is what
keeps the regression layer trustworthy.
