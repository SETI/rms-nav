================
Image Navigation
================

Introduction
============

SpinDoctor is a spacecraft image navigation system designed to analyze images from various space missions and determine precise positional offsets. This guide explains how to use the primary command-line interface exposed by the ``sd_offset`` script to navigate images and generate results, and how to invoke the cloud-tasks variant for queue-driven processing.

Purpose of the System
---------------------

The primary purpose of SpinDoctor is to determine the precise pointing of spacecraft instruments by comparing the observed images with theoretical models of what should appear in the field of view. This process, known as "navigation," is crucial for:

1. Validating and correcting spacecraft pointing information
2. Ensuring accurate scientific interpretations of the imagery
3. Creating properly annotated and labeled images for analysis
4. Supporting mission planning and operations

The system works by:

1. Reading spacecraft imagery and metadata
2. Generating theoretical models of stars, planets, moons, and rings
3. Correlating the observed features with the theoretical models
4. Calculating the offset between the expected and actual pointing
5. Producing annotated images and data files with the results

Supported Missions
------------------

SpinDoctor currently supports multiple instruments, organized by dataset names you will pass on the command line. Dataset names are case-insensitive and map to instrument-specific handlers. The complete set is:

* ``coiss`` and ``coiss_pds3`` — Cassini Imaging Science Subsystem (all volumes)
* ``coiss_cruise`` and ``coiss_cruise_pds3`` — Cassini Imaging Science Subsystem (Cruise volumes 1001-1009)
* ``coiss_saturn`` and ``coiss_saturn_pds3`` — Cassini Imaging Science Subsystem (Saturn volumes 2001-2116)
* ``gossi`` and ``gossi_pds3`` — Galileo Solid State Imager
* ``nhlorri`` and ``nhlorri_pds3`` — New Horizons Long Range Reconnaissance Imager
* ``vgiss`` and ``vgiss_pds3`` — Voyager Imaging Science Subsystem
* ``sim`` — simulated images

Installation and Setup
======================

See :doc:`/introduction_overview` for package installation with ``pip`` or
``pipx``.

Environment Setup
-----------------

In addition to installing the package, the following external resources are
needed at runtime.

**SPICE kernels.**  Download the SPICE kernels required for your mission and
set ``SPICE_PATH`` to the directory that contains them:

.. code-block:: bash

   export SPICE_PATH=/path/to/your/spice/kernels

**PDS3 holdings.**  For PDS3 datasets (all currently supported missions), set
``PDS3_HOLDINGS_DIR`` to the root of a PDS3 holdings tree (or pass
``--pds3-holdings-root`` on the command line):

.. code-block:: bash

   export PDS3_HOLDINGS_DIR=/path/to/your/pds3/data

The holdings tree follows the layout used by the PDS Ring-Moon Systems Node::

   $PDS3_HOLDINGS_DIR/
       volumes/
           <volume_set>/
               <volume>/
                   <data directories>/
       metadata/
           <volume_set>/
               <volume>/
                   <volume>_index.lbl
                   <volume>_index.tab

Remote holdings are supported: ``PDS3_HOLDINGS_DIR`` and
``--pds3-holdings-root`` accept any URL understood by ``filecache.FCPath``
(for example ``https://pds-rings.seti.org/holdings``).

Configuration System
====================

SpinDoctor uses a hierarchical YAML-based configuration system. For detailed
information about the configuration system, including its structure, default
YAML files, and how to override settings using configuration files and
command-line options, see :doc:`/introduction_configuration`.

Command-Line Interface
======================

Basic Usage
-----------

The main entry point for SpinDoctor is the ``sd_offset`` script installed via ``pyproject.toml``. The basic syntax is:

.. code-block:: bash

   sd_offset DATASET_NAME [options]

Where ``DATASET_NAME`` is one of the supported names listed in the "Supported Missions" section. Names are case-insensitive (for example, ``COISS`` and ``coiss`` are equivalent).

Command-Line Arguments
----------------------

The command-line interface groups options by purpose. Environment options control configuration sources and output roots. Navigation options select which models or techniques to run. Output options determine whether to write artifacts locally or to produce a cloud-tasks description instead of processing. Dataset selection options are provided by each dataset type: PDS3 datasets expose volume and image filters. A single profiling toggle is available for performance analysis.

Environment options
^^^^^^^^^^^^^^^^^^^

* ``--config-file PATH`` (repeatable): one or more configuration file paths to
  override defaults. See :doc:`/introduction_configuration` for details.

* ``--pds3-holdings-root PATH``: root directory or URL for PDS3 holdings,
  overriding both the ``PDS3_HOLDINGS_DIR`` environment variable and any
  corresponding configuration setting.

* ``--nav-results-root PATH``: root directory or URL where navigation results
  will be written, overriding both the ``NAV_RESULTS_ROOT`` environment variable
  and any corresponding configuration setting.

Navigation options
^^^^^^^^^^^^^^^^^^

* ``--nav-models LIST``: a comma-separated glob-pattern list selecting which
  ``NavModel`` instances run.  Names follow the ``stars`` /
  ``body:NAME`` / ``rings:PLANET`` convention.  Defaults to ``*``.  See
  :ref:`selecting-models-and-techniques` for the full syntax (globs,
  ``!`` exclusion, prefix-only shorthand).

* ``--nav-techniques LIST``: a comma-separated glob-pattern list selecting
  which registered ``NavTechnique`` subclasses run.  Defaults to ``*``.
  See :ref:`selecting-models-and-techniques` for the full syntax and the
  list of shipping technique class names.

Output options
^^^^^^^^^^^^^^

* ``--output-cloud-tasks-file PATH``: write a JSON file describing tasks for all selected images suitable for a cloud-tasks queue, and exit without performing navigation.
* ``--dry-run``: print the images that would be processed without performing navigation.
* ``--no-write-output-files``: perform navigation but do not write any output files.

Dataset selection (PDS3 datasets)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For PDS3 datasets (``coiss``, ``coiss_pds3``, ``coiss_cruise``, ``coiss_cruise_pds3``, ``coiss_saturn``, ``coiss_saturn_pds3``, ``gossi``, ``gossi_pds3``, ``nhlorri``, ``nhlorri_pds3``, ``vgiss``, ``vgiss_pds3``), the following options control which images are selected. All filters combine with logical AND, and explicit lists restrict the search domain before range filters to improve performance.

* ``img_name`` (positional, repeatable): specific image name(s) to process.
* ``--first-image-num N``: minimum image number (inclusive).
* ``--last-image-num N``: maximum image number (inclusive).
* ``--volumes NAME[,NAME...]`` (repeatable): one or more complete PDS3 volume names; you may pass comma-separated values or specify the option multiple times.
* ``--first-volume NAME``: starting PDS3 volume; only that volume and chronologically later ones are processed.
* ``--last-volume NAME``: ending PDS3 volume; only that volume and chronologically earlier ones are processed.
* ``--image-filespec-csv FILE`` (repeatable): CSV file(s) containing PDS3 file specifications; files must include a header column named ``Primary File Spec`` or ``primaryfilespec``.
* ``--image-file-list FILE`` (repeatable): file(s) containing file specifications or names, one per line; lines beginning with ``#`` are ignored.
* ``--choose-random-images N``: choose a random subset of N images that meet the other criteria.

Miscellaneous
^^^^^^^^^^^^^

* ``--profile`` / ``--no-profile``: enable or disable runtime profiling (default is disabled).

Logging options
^^^^^^^^^^^^^^^

All four options accept a standard log-level string (``DEBUG``, ``INFO``, ``WARNING``,
``ERROR``, or ``CRITICAL``) and override the corresponding ``general.*`` configuration
key for that run. For full details and the config-file equivalents see
:doc:`/introduction_configuration`.

* ``--log-level-main-console LEVEL``: stdout level for the main logger (overrides
  ``general.log_level_main_console``; default ``INFO``).

* ``--log-level-main-file LEVEL``: logfile level for the main logger written to
  ``$NAV_RESULTS_ROOT/logs/sd_offset/`` (overrides ``general.log_level_main_file``;
  default ``INFO``).

* ``--log-level-image-console LEVEL``: stdout level for the image logger, active
  only while each image is being processed (overrides
  ``general.log_level_image_console``; default ``INFO``).

* ``--log-level-image-file LEVEL``: level for the per-image logfile written to
  ``$NAV_RESULTS_ROOT/logs/{results_path_stub}.log`` (overrides
  ``general.log_level_image_file``; default ``INFO``).

Example Commands
----------------

To process a single Cassini image by specifying its name explicitly and using the default navigation technique:

.. code-block:: bash

   sd_offset coiss N1234567890

To process Voyager images within a single PDS3 volume:

.. code-block:: bash

   sd_offset vgiss --volumes VGISS_5101

To process a New Horizons image list found in a CSV from PDS, restricting the
run to the body-limb and ring-edge DT techniques:

.. code-block:: bash

   sd_offset nhlorri --image-filespec-csv /path/to/nhlorri.csv \
       --nav-techniques 'BodyLimbNav,RingEdgeNav'

To choose ten random Cassini images between two volumes and perform a dry run:

.. code-block:: bash

   sd_offset coiss --first-volume COISS_2001 --last-volume COISS_2010 --choose-random-images 10 --dry-run

To generate a cloud-tasks JSON file for images across two Voyager volumes without processing:

.. code-block:: bash

   sd_offset vgiss --volumes VGISS_5101 --volumes VGISS_5102 --output-cloud-tasks-file tasks.json

Cloud-tasks entry point
-----------------------

Queue-driven processing is supported by ``sd_offset_cloud_tasks``. This variant reads tasks from a queue and processes each batch of files described by the task payload. It accepts the same environment options used to derive configuration and results roots and does not include dataset selection flags because the task provides the list of files. Invoke it with:

.. code-block:: bash

   sd_offset_cloud_tasks [--config-file PATH] [--nav-results-root PATH]

Cloud-tasks JSON schema
^^^^^^^^^^^^^^^^^^^^^^^

The file produced by ``--output-cloud-tasks-file`` is a JSON array of task
objects. Each task is:

.. code-block:: json

    {
        "task_id": "<dataset_name>-<label_file_name>-<index>",
        "data": {
            "dataset_name": "<dataset_name>",
            "arguments": {
                "nav_models": ["body:*", "rings", "stars"],
                "nav_techniques": ["*"]
            },
            "files": [
                {
                    "image_file_url": "<path or URL to image file>",
                    "label_file_url": "<path or URL to label file>",
                    "results_path_stub": "<relative stub used to name outputs>",
                    "index_file_row": {"<column>": "<value>", "...": "..."},
                    "extra_params": {"<key>": "<value>"}
                }
            ]
        }
    }

Fields:

* ``task_id``: unique string identifier built from the dataset name, the
  first image's label filename, and the enumeration index.
* ``data.dataset_name``: one of the supported dataset names.
* ``data.arguments``: an object with optional keys ``nav_models`` and
  ``nav_techniques`` (each a list of strings, or ``null``).
* ``data.files``: one or more file descriptors with required fields
  ``image_file_url``, ``label_file_url``, and ``results_path_stub``, and
  optional ``index_file_row`` (metadata from the source index file, may be
  ``null``) and ``extra_params`` (arbitrary key/value dictionary forwarded
  to the task implementation; optional, may be ``null`` or omitted).

.. _selecting-models-and-techniques:

Selecting models and techniques
===============================

``sd_offset`` runs every applicable navigation model and every feasible
navigation technique by default.  Two glob-pattern filters narrow that
set: ``--nav-models`` selects which :class:`~spindoctor.nav_model.nav_model.NavModel`
instances run, ``--nav-techniques`` selects which
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique` subclasses run.
The same syntax applies in three places:

* ``sd_offset --nav-models LIST --nav-techniques LIST`` on the CLI.
* ``sd_offset_cloud_tasks`` task JSON, under
  ``data.arguments.nav_models`` and ``data.arguments.nav_techniques``
  (each a list of strings).
* :class:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator` programmatic
  use, via the ``only_models=`` and ``only_techniques=`` keyword arguments.

The two filters share their pattern syntax; only the *names* they match
differ.  Filtering is purely additive over the existing registry — it does
not register new models or techniques, so an entry that does not exist on
this build of ``rms-spindoctor`` simply does not match.

Pattern syntax
--------------

Patterns are gitignore-style fnmatch globs evaluated against the
candidate name.  A single string or a comma-separated list (CLI) /
list-of-strings (JSON, Python) is accepted; the orchestrator splits on
commas and trims whitespace.

Inclusion patterns
^^^^^^^^^^^^^^^^^^

* A literal name matches that name only:
  ``BodyLimbNav`` matches the technique class
  :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` and
  nothing else.
* ``*`` matches any sequence of characters; ``?`` matches a single
  character; ``[abc]`` matches any character from the set.  Standard
  Python ``fnmatch`` semantics apply.
* The default ``'*'`` matches every candidate.

Exclusion patterns
^^^^^^^^^^^^^^^^^^

* A leading ``!`` marks an *exclusion* pattern: matches against the
  remaining glob are removed from the result.
  ``--nav-techniques '!StarFieldFromCatalogNav'`` runs every registered
  technique except that one.
* When every pattern in the list begins with ``!`` (a pure-exclusion
  list), an implicit ``'*'`` inclusion is added so the result is
  "everything except the excluded names".  ``--nav-models '!body:MIMAS'``
  is therefore equivalent to ``--nav-models '*,!body:MIMAS'``.
* When at least one inclusion pattern is present, only the listed
  inclusions plus their non-excluded matches survive.
  ``'body:*,!body:MIMAS'`` runs every body model except Mimas.

Multiple patterns
^^^^^^^^^^^^^^^^^

* On the CLI, comma-separate patterns inside a single argument:
  ``--nav-models 'body:MIMAS,rings:SATURN,stars'``.
* In JSON or Python, supply a list of strings:
  ``["body:MIMAS", "rings:SATURN", "stars"]``.
* The list is order-independent: a candidate name is kept iff it
  matches at least one inclusion pattern and no exclusion pattern.

Model names
-----------

The catalog-driven models register under these per-instance names:

* ``stars`` — :class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars`
  (one instance per observation; no namespace).
* ``body:NAME`` —
  :class:`~spindoctor.nav_model.nav_model_body.NavModelBody` (one instance per
  body whose bounding box overlaps the extended FOV).  The ``NAME``
  portion is the upper-case SPICE body name
  (``body:MIMAS``, ``body:DIONE``, ``body:SATURN``).
* ``rings:PLANET`` —
  :class:`~spindoctor.nav_model.nav_model_rings.NavModelRings` (one instance
  per planet whose ring system has any radius inside the extended FOV;
  Saturn, Uranus, and Neptune today).

Two convenience normalizations apply to model patterns:

* The ``VALUE`` part of ``prefix:VALUE`` is upper-cased automatically,
  so ``body:saturn`` matches ``body:SATURN``.
* A bare prefix without a colon and without glob characters
  (``body``, ``rings``) is auto-expanded to ``prefix:*``, matching
  every namespaced model under that prefix.  ``stars`` (which has no
  namespace) continues to match itself directly.

Both normalizations preserve the leading ``!`` exclusion marker.
``--nav-models 'body'`` is therefore shorthand for "every body model";
``--nav-models '!body'`` excludes every body model.

Technique names
---------------

Techniques register under their class name.  The shipping concrete
techniques are:

* Body family —
  :class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav`,
  :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav`,
  :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`,
  :class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav`.
* Ring family —
  :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`,
  :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`.
* Star family —
  :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`,
  :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`,
  :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav`.

The star field matcher re-centroids each matched star with a point-spread-function fit
when the star is faint, and keeps the simpler brightness-weighted centroid when the star
is bright enough that its noise has already fallen below the PSF fit's residual bias.
This makes the star field the most accurate technique on a well-exposed field. The
brightness at which it switches is the configurable
``techniques.StarFieldFromCatalogNav.tuning.psf_refine_snr_max`` knob in
``config_510_techniques.yaml`` (set the whole step off with ``psf_refine_enabled: 0``).

:class:`~spindoctor.nav_technique.nav_technique_manual.NavTechniqueManual` is
the interactive driver and is not part of the autonomous registry; it
cannot be invoked by ``--nav-techniques``.

Multiple feasible techniques run in parallel and the orchestrator
combines their results via the ensemble step; ``--nav-techniques`` is
not a "pick one technique" knob the way the legacy pipeline was — it
restricts the candidate set the orchestrator considers.

Examples
--------

.. code-block:: bash

   # Run every model and every technique (the default).
   sd_offset coiss N1234567890

   # Mimas only — drop every other body and the ring/star models.
   sd_offset coiss N1234567890 --nav-models 'body:MIMAS'

   # Every body, plus rings, but no stars.
   sd_offset coiss N1234567890 --nav-models 'body:*,rings'

   # Every model except Mimas (auto-expanded ``'*'`` inclusion).
   sd_offset coiss N1234567890 --nav-models '!body:MIMAS'

   # Two specific DT-based techniques only.
   sd_offset nhlorri LOR_0034851733 \
       --nav-techniques 'BodyLimbNav,RingEdgeNav'

   # Every technique except the catalog star matcher.
   sd_offset coiss N1234567890 \
       --nav-techniques '!StarFieldFromCatalogNav'

   # Body and ring families only (every body / ring technique, no stars).
   sd_offset coiss N1234567890 \
       --nav-techniques 'Body*,Ring*'

Inputs and Outputs
==================

Input Files
-----------

The primary input to SpinDoctor is spacecraft imagery. The system supports:

* PDS3 formatted image files (.IMG)
* Associated metadata (labels, SPICE kernels)

The system requires access to:

1. The raw image data
2. SPICE kernels for the appropriate mission and time period
3. Configuration settings (optional, defaults are provided)

Output Files
------------

SpinDoctor generates two types of output files:

Metadata Files (``*_metadata.json``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These JSON files contain the navigation results, including:

* The calculated pointing offset (dv, du)
* Uncertainty estimates (sigma_v, sigma_u)
* Confidence scores
* Metadata about the navigation process
* Status information (success, error, etc.)
* Technique-specific metadata
* Timestamps

Summary PNG Files (``*_summary.png``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are annotated images showing:

* The original image data
* Overlaid model features (stars, rings, bodies)
* Text annotations
* Scale information
* Navigation offset information

Interpreting Results
--------------------

The key information in the results is:

1. **Offset Values**: The u,v pixel offsets that should be applied to the nominal pointing to match the observed features
2. **Correlation Quality**: How well the models matched the observed features
3. **Annotations**: Identifications of specific features in the image
4. **Status**: Whether the navigation was successful, and if not, why

Simulated Images
================

SpinDoctor includes an image simulator used to test and validate the navigation
pipeline. It is not needed for navigating real data, but a simulated frame can be
navigated through the same pipeline by passing the ``sim`` dataset name and a path
to a JSON parameter file:

.. code-block:: bash

   sd_offset sim /path/to/simulated_image.json

The simulator, its scene formats, and the ``sd_create_simulated_image`` GUI are
documented for developers in the :doc:`/dev_guide/dev_guide_simulator` chapter.
See also :doc:`user_guide_simulated_images`.

Navigation Techniques
=====================

The autonomous-navigation pipeline runs every registered ``NavTechnique``
whose feasibility check passes on the surviving feature set, then combines
the per-technique offsets via the orchestrator's precision-weighted
ensemble.  Use ``--nav-techniques`` to restrict which techniques run; the
default ``*`` runs all of them.

The algorithmic detail (DT pipeline, Levenberg-Marquardt refinement,
information-matrix covariance) lives in
:doc:`/dev_guide/dev_guide_techniques` and
:doc:`/dev_guide/dev_guide_techniques_dt_fitting`; this page summarises
what each technique does and which scenes it applies to.

Implemented techniques
----------------------

``BodyLimbNav``
^^^^^^^^^^^^^^^

Translation fit on a body's lit limb.  Consumes every ``LIMB_ARC`` feature
emitted by ``NavModelBody``, concatenates their per-vertex polylines, and
runs a coarse-NCC plus Levenberg-Marquardt refinement against the image
edge-distance transform.  Tukey biweight reweighting rejects outlier
vertices; the M-estimator information matrix at the converged solution
yields the result's covariance.  Multi-body inputs sharpen the fit by
``sqrt(N_bodies)``.

Best for: scenes where one or more bodies show a visible limb arc
(typical Cassini ISS Mimas / Enceladus / Tethys / Dione / Rhea encounter
images).  Feasibility threshold: at least one limb arc with at least
30 surviving polyline vertices.

``BodyTerminatorNav``
^^^^^^^^^^^^^^^^^^^^^

Same shape as ``BodyLimbNav`` on ``TERMINATOR_ARC`` features, with two
differences: each body's per-vertex sigmas collapse to one per-body scalar
(the body's mean sigma), and the confidence formula carries
phase-angle-factor and albedo-penalty terms.  Best for crescent
geometries where the terminator runs through bright, nearly-uniform
hemispheres.

``RingEdgeNav``
^^^^^^^^^^^^^^^

DT-based fit on every ``RING_EDGE`` feature.  Polarity prediction is
intentionally disabled today (the ring catalog does not yet flag
polarity_predictable, deferred work).  When every input edge is
straight-line the technique reports ``is_rank_1=True`` and returns an
honest rank-deficient covariance; the ensemble combine fuses it with any
orthogonal-axis result (a star, body limb, body blob) before declaring a
final answer.

Best for: scenes containing bright ring edges (typical Cassini ISS
Saturn-rings imagery).

``RingAnnulusNav``
^^^^^^^^^^^^^^^^^^

Pyramid-NCC fit on every ``RING_ANNULUS`` feature.  ``RING_ANNULUS``
features are emitted by the rings model in two regimes: when adjacent
ring edges compress radially below the per-planet
``feature_emission.ring_annulus.max_radial_px`` threshold in
``config_510_techniques.yaml`` (individual edges no longer separable),
and when the per-planet km/px threshold fires on a low-resolution
ring scene where the entire ring system spans only a handful of
pixels.  In either case the rings model collapses every surviving
ring into a single composite annulus per planet.  Multi-planet scenes
(rare) emit one ``RING_ANNULUS`` per ring system; the technique fuses
them via Z-buffer paint and runs one joint NCC.
``use_gradient='auto'`` self-selects raw vs gradient mode per image.

Best for: low-resolution ring scenes where ``RingEdgeNav`` cannot
separate individual edges (distant Cassini ring views; potential
NHLORRI Pluto/Charon ring geometries).

``NavTechniqueManual``
^^^^^^^^^^^^^^^^^^^^^^

Interactive PyQt6 dialog that composes every template-bearing feature
into a single ext-FOV overlay and lets the operator pick the offset by
hand.  Not part of the autonomous registry; opt into it from the normal
``sd_offset`` driver with the ``--manual`` flag, which requires the
selection to resolve to exactly one image:

.. code-block:: bash

   echo W1521598221_1_CALIB > /tmp/img_list.txt
   sd_offset coiss --manual --image-file-list /tmp/img_list.txt

The driver loads the image, runs the orchestrator's ``prepare`` step
(image classifier + NavModels + features + reliability gate), opens the
dialog, and prints the chosen ``offset_dv_px`` / ``offset_du_px`` to
stdout.  Exit code is ``2`` if the dialog is cancelled or no
template-bearing features are available.  The dialog's **Save as
Library Entry...** button is the recommended path for adding a sidecar
to the operator-curated test image library; see
:doc:`/dev_guide/dev_guide_image_library`.

Programmatic equivalent (one obs in, ``NavTechniqueResult`` out):

.. code-block:: python

   from spindoctor.nav_technique import run_manual_nav

   result = run_manual_nav(obs)

Filtering examples
------------------

Run only the ring-edge technique:

.. code-block:: bash

   sd_offset coiss N1234567890 --nav-techniques RingEdgeNav

Run every technique except ``BodyTerminatorNav``:

.. code-block:: bash

   sd_offset coiss N1234567890 --nav-techniques '!BodyTerminatorNav'

Run both DT body techniques together:

.. code-block:: bash

   sd_offset coiss N1234567890 --nav-techniques 'BodyLimbNav,BodyTerminatorNav'

Output
------

Every technique that runs contributes one entry to
``NavResult.per_technique`` carrying the per-technique offset, 2x2
covariance, calibrated confidence, and a typed ``*Diagnostics``
dataclass.  The orchestrator's ensemble combine reconciles those
entries into a single ``NavResult.offset_px`` and ``confidence_rank``;
both numbers land in the per-image ``_metadata.json``.

Navigation Models
=================

A *navigation model* is SpinDoctor's prediction of what the image *should*
look like at the spacecraft's nominal pointing.  Three model families
ship out of the box: stars, planetary bodies, and planetary rings.
Each contributes one or more *features* (typed predictions with their
own per-feature uncertainty) to the navigator.  You can restrict which
families run by passing ``--nav-models`` on the command line; valid
entries are ``stars``, ``rings``, and body-specific entries of the form
``body:NAME`` (glob patterns are allowed).

Star Navigation Model
---------------------

The star model builds a deduplicated catalog of stars expected to fall
inside the field of view, applies stellar aberration and proper motion
to bring each catalog position into the spacecraft frame at observation
time, and emits one feature per usable star.

**Catalog precedence.**  Catalogs are searched in the order configured
in ``config_03_stars.yaml`` under ``stars.catalogs`` (default
``[ucac4, tycho2, ybsc]``).  Stars present in more than one catalog are
deduplicated using the RA / DEC and V-magnitude thresholds in the same
file.

**Per-star detectability.**  Each star is gated by its catalog visual
magnitude against the per-observation limiting magnitude
``obs.star_max_usable_vmag()``, which depends on the per-instrument
sensitivity and the exposure time.  Stars fainter than the limiting
magnitude (or with no catalog magnitude) are dropped.

**Smear.**  When the spacecraft attitude rate is non-zero during the
exposure, stars smear into trails.  The model computes the per-image
smear vector from the SPICE pointing brackets and uses
``psfmodel.eval_rect(movement=...)`` to render a smear-aware kernel
when a downstream technique needs one.  Stars whose smear length
exceeds ``stars.max_smear`` are dropped (the centroid is unfittable).

**Body and ring conflicts.**  Each star's predicted pixel is checked
against an ``oops`` body intercept and a per-planet opaque ring
annulus (configured under ``stars.ring_occlusion_radii_km``).  Stars
that fall behind a body or inside an opaque ring annulus are tagged
with a ``BODY:`` or ``RING:`` conflict string and excluded from
matching.  Body intercepts win over ring intercepts.

**Configuration.**  Most user-tunable parameters live in
``config_03_stars.yaml``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Effect
   * - ``stars.catalogs``
     - Catalog search order; default ``[ucac4, tycho2, ybsc]``.
   * - ``stars.max_stars``
     - Maximum number of stars retained per image (default 100).
   * - ``stars.max_smear``
     - Smear length in pixels above which a star is dropped (default
       100).
   * - ``stars.min_vmag`` / ``stars.max_vmag``
     - Magnitude window applied to the per-instrument
       ``star_min_usable_vmag`` / ``star_max_usable_vmag`` floor.
   * - ``stars.proper_motion``
     - Apply proper motion at ``obs.midtime`` (default true).
   * - ``stars.stellar_aberration``
     - Apply stellar aberration (default true).
   * - ``stars.ring_occlusion_enabled``
     - Toggle the ring-annulus occlusion check (default true).
   * - ``stars.ring_occlusion_radii_km``
     - Per-planet list of opaque ``[inner_km, outer_km]`` annuli.

Body Navigation Model
---------------------

For every body whose predicted bounding box overlaps the extended
field of view, the body model renders an oversampled Lambert-shaded
silhouette, extracts the limb and terminator polylines, and emits a
mix of feature types depending on resolution, lighting, and shape
quality:

- ``LIMB_ARC`` — emitted when the limb position is well-determined
  (per-vertex normal sigma below the ``LIMB_ARC_MAX_UNCERTAINTY_PX``
  cap).  Carries a polyline of vertex coordinates and per-vertex
  anisotropic sigmas.
- ``BODY_BLOB`` — emitted instead of ``LIMB_ARC`` when the limb is too
  uncertain to fit but the predicted body diameter is above the
  body-specific blob threshold.  Carries only a centroid and bounding
  box.
- ``BODY_DISC`` — emitted alongside ``LIMB_ARC`` when the body fits
  well inside the FOV (overflow below 30 %, lit-and-visible fraction
  at least 40 %).  Carries the rendered template for full-disc
  correlation.
- ``TERMINATOR_ARC`` — emitted when the terminator polyline has at
  least 8 vertices and the phase-angle factor (``sin(phase_angle)``)
  is above 0.05.

**Per-body shape data.**  ``ellipsoid_residual_km``, ``crater_scale_km``,
``albedo_variation``, ``spice_orbital_residual_km``, and
``min_blob_diameter_px`` come from the static body-shape table.  These
quantities drive the per-vertex polyline sigmas and the BODY_BLOB
emission threshold.  For bodies absent from the table a conservative
generic-icy-moon profile is used.

**Configuration.**  ``config_04_bodies.yaml`` exposes:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Effect
   * - ``bodies.min_bounding_box_area``
     - Minimum predicted body bbox area (px squared) below which
       silhouette rendering is skipped.
   * - ``bodies.oversample_edge_limit``
     - Anti-aliasing oversample limit for the silhouette render.
   * - ``bodies.oversample_maximum``
     - Hard cap on the per-axis oversample factor.
   * - ``bodies.use_lambert``
     - Use Lambert shading (default true) vs. flat-disc rendering.
   * - ``bodies.use_albedo`` / ``bodies.geometric_albedo``
     - Apply per-body geometric albedo when computing brightness.

The bodies considered for navigation are the planet returned by
``obs.closest_planet`` plus the satellites configured under
``planets.satellites``.

Ring Navigation Model
---------------------

The ring navigation model generates theoretical brightness profiles
for planetary ring edges and emits one feature per surviving edge.
Two top-level options in ``config_05_rings.yaml`` control whether
ring pixels in shadow are excluded from the model before navigation.

For each surviving ring feature the model emits one of:

- ``RING_EDGE`` — a per-vertex polyline of edge coordinates with
  per-vertex radial sigma derived from the catalog ``rms`` divided by
  the radial km-per-pixel scale.  When the polyline is straight
  (deviation from a best-fit line below 1 px) the ``is_straight_line``
  flag is set so techniques can handle the rank-1 covariance.
- ``RING_ANNULUS`` — a multi-edge composite template emitted when the
  surviving polyline compresses radially below 5 px (the edges are
  not separable at the image scale).

Per-edge feature definitions live in ``config_2X_<planet>_rings.yaml``
under ``rings.ring_features.<PLANET>.features``.  See "Ring YAML
configuration" in the developer guide for the full schema.

Planet shadow removal
^^^^^^^^^^^^^^^^^^^^^

When a planet casts a shadow across part of its own ring system, those ring
arcs appear dark in the image. If the model still shows those arcs as bright,
the navigator will try to align a bright model against a dark image region,
which introduces a systematic pointing error.

The ``rings.remove_planet_shadow`` option (default ``true``) instructs the
ring model to zero out all ring pixels that fall inside the planet's own
shadow:

.. code-block:: yaml

   rings:
     remove_planet_shadow: true   # default

When active, the ring model logs the number of masked pixels at ``INFO`` level:

.. code-block::

   Planet shadow removal: 1284 pixel(s) inside SATURN shadow will be masked

If the shadow geometry cannot be computed for a particular observation (for
example, because the illumination geometry is degenerate), a warning is logged
and the full unmasked ring model is used instead. Navigation proceeds
normally; no output files are suppressed.

To disable shadow removal entirely -- for example, to compare navigation
quality with and without the mask -- set the option to ``false`` in a
``--config-file`` override:

.. code-block:: yaml

   rings:
     remove_planet_shadow: false

Body shadow removal (future)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``rings.remove_body_shadows`` option (default ``false``) is reserved for a
future enhancement that will remove ring pixels shadowed by moons. Setting it
to ``true`` has no effect in the current release.

.. code-block:: yaml

   rings:
     remove_body_shadows: false   # default; not yet implemented

Troubleshooting
===============

Common Issues
-------------

If SPICE kernels are missing, ensure that all required kernels are available and that environment variables and configuration files point to valid paths. For PDS3 inputs, verify the files conform to expected formats. In cases where no features are found or correlations are weak, check image quality, adjust the selected models or techniques, or limit processing to images known to contain suitable features. Use ``--dry-run`` to validate selection criteria without performing full processing.

Getting Help
------------

If you encounter persistent issues:

Review logs for detailed errors, consult the developer documentation for architectural context, and provide the command line, log snippets, and representative input data when asking for support.
