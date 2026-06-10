=============
Image Library
=============

The image library is an operator-curated set of spacecraft images, each paired with a
manually verified ground-truth offset and a block of expected-outcome targets. It lives at
``tests/integration/image_library/`` and drives two layers of the integration test suite:
a fast structural check that every entry conforms to the sidecar schema, and a
holdings-backed regression that runs the autonomous orchestrator against each image and
scores the result. The library is also the calibration input that the per-technique
confidence formulas are tuned against, so each entry simultaneously documents a real scene
and pins down the result the pipeline is expected to produce on it.

Every entry consists of three artifacts that an operator produces in order: a *sidecar*
YAML describing the image and the expected outcome, an optional companion PNG overlay, and
a *regression baseline* JSON recording the exact rounded navigation output. The sidecar is
the load-bearing record; the baseline is mechanically derived from a live navigation run.

The Directory As Registry
=========================

The library has no manifest file. The directory tree under
``tests/integration/image_library/images/`` *is* the registry: dropping a sidecar at
``images/<scene_class>/<image_id>.yaml`` enrolls that image automatically. Test discovery
walks the tree with :meth:`~tests.integration.sidecar.LibraryRoot.discover_sidecar_paths`,
which globs ``images/*/*.yaml`` and returns the sorted result, so a relocated, renamed, or
deleted sidecar changes the enrolled set with no other bookkeeping.

The on-disk layout is::

   tests/integration/
       sidecar.py                       schema + validator + discovery
       baseline.py                      regression-baseline schema
       test_image_library.py            structural invariants (fast)
       test_autonomous_nav.py           per-image regression (holdings)
       test_baselines.py                baseline regression (holdings)
       update_baselines.py              baseline refresh tool
       image_library/
           images/
               <scene_class>/
                   <image_id>.yaml      sidecar
                   <image_id>.png       overlay (optional, not read by tests)
                   README.txt           per-class scene-selection guide
       baselines/
           <image_id>.json              rounded regression baseline

The :class:`~tests.integration.sidecar.LibraryRoot` dataclass computes every path from the
location of ``sidecar.py``, so the library is portable with the checkout and needs no
configured root. ``LibraryRoot.images`` resolves the sidecar tree and ``LibraryRoot.baselines``
resolves the sibling ``baselines/`` directory.

The Sidecar Schema
==================

A sidecar is a YAML mapping validated by :func:`~tests.integration.sidecar.load_sidecar`,
which parses the file and checks every field against the schema before returning a frozen
:class:`~tests.integration.sidecar.Sidecar`. The validator is hand-rolled so a malformed
entry fails at collection time with the offending file named, rather than producing a
confusing per-image failure later. The current schema is ``schema_version: 1``; any other
value is rejected.

Top-Level Fields
----------------

``schema_version``
   Integer. Must equal ``1``.

``image_id``
   Non-empty string. Opaque identifier, conventionally the PDS3 product ID with any
   pipeline suffix retained (for example ``N1572105349_1_CALIB`` for a Cassini calibrated
   product). The filename stem must equal this value: ``<image_id>.yaml``.

``mission``
   One of the mission codes in :data:`~tests.integration.sidecar.ALLOWED_MISSIONS`:
   ``COISS``, ``VGISS``, ``GOSSI``, ``NHLORRI``. These are the dataset names registered in
   :mod:`nav.dataset` upper-cased, so a sidecar's mission maps unambiguously onto a CLI
   invocation such as ``nav_offset coiss``. The regression test maps each code to its
   :class:`~nav.obs.obs_snapshot_inst.ObsSnapshotInst` subclass.

``camera``
   One of :data:`~tests.integration.sidecar.ALLOWED_CAMERAS`: ``NAC``, ``WAC``, ``SSI``,
   ``NA``, ``WA``, ``LORRI``.

``filter_combo``
   Non-empty string. The filters applied for the exposure, sorted alphabetically and joined
   with ``+`` (for example ``CL1+IR1`` or ``CL+CL``).

``image_url``
   Non-empty string. Opaque URL locating the image file. The ``pds3://`` scheme is resolved
   relative to ``PDS3_HOLDINGS_DIR`` at test time; ``https://``, ``gs://``, and ``file://``
   URLs are passed through unchanged.

``scene_tags``
   Non-empty list of unique strings. The first entry is the *primary scene tag* and must
   both name a declared scene class and equal the basename of the containing directory.
   Subsequent entries are free-form reviewer annotations, typically a body name and a
   morphology qualifier such as ``crescent`` or ``ansa``.

``exposure_time_sec``
   Optional. Finite positive float giving the exposure duration in seconds, taken from
   ``obs.texp``. May be omitted or null.

``image_datetime_utc``
   Optional. Non-empty string holding the observation midtime as a UTC ISO 8601 timestamp,
   derived from ``obs.midtime``. Informational metadata; not a navigation input. May be
   omitted.

The ``ground_truth`` Block
--------------------------

A required mapping validated into :class:`~tests.integration.sidecar.GroundTruth`, carrying
the operator's manually picked offset and its provenance.

``offset_dv_px`` / ``offset_du_px``
   Finite floats. The verified pixel offset. By convention the *predicted* position
   ``(v, u)`` plus the offset ``(dv, du)`` equals the *actual* position in the image.

``offset_uncertainty_px``
   Finite float, strictly greater than zero. The operator's per-axis one-sigma uncertainty.
   The regression test admits an offset within ``offset_uncertainty_px + 0.5`` pixels of the
   ground truth on each axis.

``source``
   Must equal ``operator_verified``, the only value in
   :data:`~tests.integration.sidecar.ALLOWED_GT_SOURCES`. Every offset comes from manually
   navigating that specific image.

``operator``
   Non-empty string naming the person who picked the offset.

``verified_date``
   A calendar date (``YYYY-MM-DD``) parsed by the YAML loader as a real date.

``ui_version``
   Non-empty string giving the ``rms-nav`` version at the time of verification.

``notes``
   Optional string holding a human-readable rationale for the verification.

The ``expected`` Block
----------------------

A required mapping validated into :class:`~tests.integration.sidecar.Expected`. These are
the targets the regression test scores the autonomous run against.

``status``
   One of :data:`~tests.integration.sidecar.ALLOWED_STATUSES`: ``ok``, ``failed``, or
   ``conflicted``. ``ok`` marks a navigable scene whose offset is the load-bearing output.
   ``failed`` marks an unnavigable scene and is reserved for the ``negative_cases``
   directory. ``conflicted`` marks a scene where two disjoint ensembles disagree by more
   than their combined uncertainty, which makes the orchestrator hard-set the rank to
   ``conflicted``.

``confidence_tier``
   One of :data:`~tests.integration.sidecar.ALLOWED_TIERS`: ``high``, ``medium``, ``low``,
   ``failed``, or ``conflicted``. The tier is a calibration target rather than a description
   of any momentary pipeline behavior. The validator cross-checks it against ``status``:
   ``failed`` pairs only with ``status: failed`` and ``conflicted`` pairs only with
   ``status: conflicted``, in both directions.

``primary_technique``
   Non-empty string naming the ``NavTechnique`` subclass expected to win pass one. The
   regression test selects the winner as the highest-confidence per-technique result, with
   ties broken by ``(-confidence, technique_name)`` ascending so the outcome is independent
   of registration order.

``techniques_must_run``
   Optional list of technique names. Every name listed must appear in the orchestrator's
   per-technique results. Defaults to an empty list.

``techniques_must_skip``
   Optional list of technique names. No name listed may appear in the per-technique results;
   use this to assert that the feasibility gate rejects a technique outright. Defaults to an
   empty list. A name cannot appear in both ``techniques_must_run`` and
   ``techniques_must_skip``.

The ``camera_rotation_expected`` Block
--------------------------------------

An optional mapping validated into
:class:`~tests.integration.sidecar.CameraRotationExpected`, meaningful only when the
per-camera config enables ``fit_camera_rotation``. It carries ``rotation_deg`` and
``uncertainty_deg``, each a finite number or null.

Scene Classes
=============

Each primary scene tag, and therefore each ``images/`` subdirectory, must be one of the
classes in :data:`~tests.integration.sidecar.DECLARED_SCENE_CLASSES`. The structural test
asserts that every subdirectory present is a member of this set, so a typo such as
``body_overflow`` fails loudly. The declared classes are:

``star_dominated``
   At least three catalog stars predicted detectable, with no body silhouette.

``body_full_fov``
   A regular ellipsoidal body filling most of the field of view with its full limb in frame.

``body_partial_overflow``
   A regular body roughly 70 to 90 percent in frame with a visible limb arc.

``body_mostly_offscreen``
   A regular body mostly off frame, with only a limb arc visible.

``body_irregular``
   An irregular body in the blob-uncertainty regime, where the limb is too uncertain for a
   limb fit.

``multi_body``
   Two or more separable, non-occluding bodies in the field of view.

``ring_only_curved``
   A ring edge with measurable curvature and no body in frame.

``ring_only_flat``
   A ring edge whose fit is rank-one, with no body in frame.

``ring_plus_body``
   A ring edge together with at least one moon.

``stars_plus_body``
   A body together with at least three visible catalog stars.

``one_bright_star_no_body``
   Exactly one unambiguous star, with no body and no rings.

``two_bright_stars_no_body``
   Exactly two unambiguous stars, with no body and no rings.

``faint_stars``
   A field whose catalog stars all fall below the detection signal-to-noise threshold.

``scattered_light``
   A frame dominated by a stray-light gradient.

``high_phase_terminator``
   A crescent geometry at phase angle greater than 90 degrees.

``below_resolution_body``
   A body whose diameter is below the resolution gate.

``negative_cases``
   A deliberately unnavigable scene, paired with ``status: failed``.

Each class directory carries a ``README.txt`` describing in detail what scenes belong there,
what to avoid, and which mission archives are likely to hold a clean example.

Regression Baselines
====================

A baseline is a JSON file at ``tests/integration/baselines/<image_id>.json`` recording the
exact rounded headline output of the last approved navigation run. It is validated into a
:class:`~tests.integration.baseline.Baseline` carrying four fields:

.. code-block:: json

   {
     "image_id": "N1597846115_2_CALIB",
     "offset_dv_px": 299.0010,
     "offset_du_px": -130.9985,
     "confidence": 0.871
   }

The offsets are rounded to four decimals and the confidence to three, matching
:data:`~tests.integration.baseline.OFFSET_DECIMALS` and
:data:`~tests.integration.baseline.CONFIDENCE_DECIMALS`. Rounding makes the comparison an
exact equality on stable values; a pipeline-run timestamp is deliberately excluded because
it is the one field that differs between otherwise identical runs.

Baselines are mechanical and generated from live navigation results by the developer tool
``update_baselines.py``, invoked from the project checkout as a module:

.. code-block:: bash

   python -m tests.integration.update_baselines --all
   python -m tests.integration.update_baselines --image-id N1597846115_2_CALIB
   python -m tests.integration.update_baselines --all --dry-run

The tool requires ``PDS3_HOLDINGS_DIR`` so the orchestrator can fetch each image. For every
selected sidecar it runs :func:`~nav.navigate_image_files.navigate_image_files`, rounds the
result through :meth:`~tests.integration.baseline.Baseline.from_run`, and reports each image
as ``CREATE``, ``UPDATE``, ``UNCHANGED``, or ``FAILED``. With ``--dry-run`` it computes the
result but writes nothing. ``test_baselines.py`` then asserts that every baseline cites an
existing sidecar and, when holdings are available, that a fresh run reproduces the recorded
triple exactly.

Validation And Adding An Image
==============================

Structural validation runs in the fast suite without holdings access.
``test_image_library.py`` asserts that the ``images/`` directory exists, that every
subdirectory is a declared scene class, that every sidecar parses and validates, that each
primary scene tag matches its containing directory, that the filename stem matches
``image_id``, and that no two sidecars share an ``image_id``. The same module exercises the
schema's field-level and cross-field rules directly against in-memory YAML.

The holdings-backed regression in ``test_autonomous_nav.py`` parametrizes one case per
discovered sidecar. For each image it runs the orchestrator and asserts the result against
the sidecar: ``status`` and ``confidence_tier`` match exactly, an ``ok`` offset lands within
``offset_uncertainty_px + 0.5`` pixels per axis of the ground truth, the highest-confidence
technique equals ``primary_technique``, and the ``techniques_must_run`` and
``techniques_must_skip`` sets are honored. The module is gated by the ``integration`` marker
and skips when ``PDS3_HOLDINGS_DIR`` is unset.

To add an image, an operator works through the following steps:

#. Pick a candidate image and choose the scene class whose expected primary technique it
   exercises, consulting the per-class ``README.txt`` when a candidate sits between two
   classes.
#. Open the manual-navigation dialog on the image with ``nav_offset <selection> --manual``,
   using whatever selection flags pin the run to a single image.
#. Align the overlay by hand to sub-pixel precision against limbs, terminators, ring edges,
   or star centroids, and accept the resulting offset.
#. Save the entry with "Save as Library Entry", targeting
   ``images/<scene_class>/<image_id>.yaml``. The dialog also writes a companion
   ``<image_id>.png`` overlay next to the sidecar as an orientation aid; no test reads it.
#. Edit the saved YAML to fill in every field using the schema above, replacing each
   placeholder.
#. Validate the schema without holdings::

      pytest tests/integration/test_image_library.py -k <image_id>

#. With holdings available, run the per-image regression and seed the baseline::

      pytest -m integration tests/integration/test_autonomous_nav.py -k <image_id>
      python -m tests.integration.update_baselines --image-id <image_id>

The sidecar schema and discovery live in ``tests/integration/sidecar.py``; the baseline
schema lives in ``tests/integration/baseline.py``. For the workflow context that surrounds
library curation and calibration, see
:doc:`/user_guide_image_library`.
