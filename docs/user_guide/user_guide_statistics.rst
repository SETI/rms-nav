Navigation Statistics
=====================

The statistics system turns the per-image metadata JSON files written by the
navigation pipeline into a local SQLite database and a deterministic report
(Markdown text plus PNG charts). It is the standing quality check on a
production run: success and failure rates, which techniques and models carry
the load, offset distributions, how well the techniques agree with one
another, and whether the confidence tiers behave as designed.

Two commands cooperate over the database and are cron-friendly (each is a
single non-interactive invocation):

.. code-block:: bash

    # Ingest one or more navigation-results roots into ./nav_stats.sqlite3
    sd_stats_ingest /data/nav-offset-results --db nav_stats.sqlite3

    # Generate report.md + charts for any slice of the database
    sd_stats_report --db nav_stats.sqlite3 --output-dir stats_report
    sd_stats_report --db nav_stats.sqlite3 --instrument coiss \
        --start-date 2005-03-01 --end-date 2005-03-01 --output-dir day_report

Ingestion
---------

``sd_stats_ingest ROOT [ROOT ...] [--db PATH]`` scans each root recursively
for ``*_metadata.json`` files (the documents
:func:`~spindoctor.navigate_image_files.navigate_image_files` writes under
``nav_results_root``) and loads them into the database. Roots may be
local directories or any URL the project's ``filecache`` layer accepts, so
cloud-hosted results ingest the same way as local ones.

Ingestion is idempotent: the database holds one row per image (keyed by image
name), and re-ingesting the same or an updated metadata file replaces that
image's row and its child rows rather than duplicating them. Malformed files
are logged and skipped; the exit status is nonzero only when nothing at all
was ingested.

Every ingestible document must carry ``observation.image_name`` and
``observation.instrument`` (the pipeline records both in every metadata
document it writes). A document missing either field is logged, skipped, and
counted as an error, exactly like a file that fails to parse.

The database is disposable, and there is no schema migration: opening a
database whose ``images`` table does not match the current column set raises
an error naming the mismatched columns. The remedy is always the same --
delete the database file and re-run ``sd_stats_ingest`` (the source of truth
is the metadata documents, so nothing is lost).

Database schema
---------------

The database is plain SQLite; opening it directly (``sqlite3`` command-line
shell, Python's ``sqlite3`` module, pandas, a GUI browser) is a supported way
to answer questions the standard report does not. Three tables hold the data;
``techniques`` and ``feature_sources`` reference ``images`` by
``image_name`` with ``ON DELETE CASCADE``, and re-ingesting an image replaces
its row and children.

``images`` — one row per image, keyed by ``image_name``:

.. list-table::
   :header-rows: 1
   :widths: 22 10 68

   * - Column
     - Type
     - Meaning
   * - ``image_name``
     - TEXT
     - Image filename (primary key), e.g. ``N1454725799_1_CALIB.IMG``.
   * - ``instrument``
     - TEXT
     - ``coiss`` / ``vgiss`` / ``gossi`` / ``nhlorri`` / ``sim``, from the
       metadata document's ``observation.instrument`` field (required; a
       document without it is skipped as an ingest error).
   * - ``image_path``
     - TEXT
     - Absolute path of the source image at navigate time.
   * - ``image_et``
     - REAL
     - Observation midtime, TDB seconds past J2000.
   * - ``image_date``
     - TEXT
     - UTC calendar date ``YYYY-MM-DD`` derived from ``image_et``; drives
       the ``--start-date`` / ``--end-date`` report filters.
   * - ``status``
     - TEXT
     - Navigation outcome: ``success``, ``failed``, or ``error``.
   * - ``status_reason``
     - TEXT
     - Failure reason (e.g. ``no_features_extracted``,
       ``missing_spice_data``); NULL for successes.
   * - ``offset_dv``, ``offset_du``
     - REAL
     - Fused pointing offset in pixels (V then U); NULL when navigation
       found no offset.
   * - ``sigma_dv``, ``sigma_du``
     - REAL
     - Per-axis 1-sigma uncertainty of the fused offset, pixels.
   * - ``confidence``
     - REAL
     - Fused confidence in ``[0, 1]``.
   * - ``confidence_rank``
     - TEXT
     - Confidence tier label assigned by the ensemble.
   * - ``n_techniques``
     - INTEGER
     - Number of per-technique results recorded for the image.
   * - ``excluded_from_consensus``
     - TEXT
     - JSON list of technique names the ensemble excluded as outliers
       (``[]`` when none).
   * - ``image_class``
     - TEXT
     - Image-classifier verdict (e.g. ``clean``).
   * - ``noise_sigma``
     - REAL
     - Image-classifier noise estimate.
   * - ``image_shape_v``, ``image_shape_u``
     - INTEGER
     - Pixel dimensions of the image data (V then U), from the metadata
       document's ``observation.image_shape`` field; NULL when the image
       never loaded (e.g. a read error).
   * - ``run_start``, ``run_end``
     - TEXT
     - UTC ISO8601 run start and end of the navigation of this image, from
       the metadata document's ``timing`` section; start is captured before
       the image load and end after navigation (or at error time).
   * - ``elapsed_s``
     - REAL
     - Wall-clock seconds between ``run_start`` and ``run_end``.
   * - ``config_hash``
     - TEXT
     - sha256 of the fully-resolved configuration used for the run.
   * - ``git_sha``
     - TEXT
     - Short git SHA of the navigating code (``-dirty`` suffix when the
       tree had uncommitted changes).
   * - ``pipeline_run``
     - TEXT
     - UTC ISO8601 timestamp of the navigation run.
   * - ``source_file``
     - TEXT
     - Path or URL of the ingested metadata document.

``techniques`` — one row per technique result per image:

.. list-table::
   :header-rows: 1
   :widths: 22 10 68

   * - Column
     - Type
     - Meaning
   * - ``image_name``
     - TEXT
     - Foreign key into ``images``.
   * - ``technique_name``
     - TEXT
     - Technique class name (e.g. ``BodyLimbNav``, ``StarUniqueMatchNav``).
   * - ``offset_dv``, ``offset_du``
     - REAL
     - The technique's own offset estimate, pixels.
   * - ``sigma_dv``, ``sigma_du``
     - REAL
     - Per-axis 1-sigma from the technique's covariance.
   * - ``confidence``
     - REAL
     - The technique's calibrated confidence in ``[0, 1]``.
   * - ``spurious``
     - INTEGER
     - 1 when the technique flagged its own result as spurious.
   * - ``at_edge``
     - INTEGER
     - 1 when the fit landed at the edge of its search space.
   * - ``source_names``
     - TEXT
     - JSON list of body / ring / catalog names the technique used.
   * - ``diagnostics``
     - TEXT
     - The technique's full diagnostics dataclass as a JSON object.

``feature_sources`` — per image, feature counts grouped by source:

.. list-table::
   :header-rows: 1
   :widths: 22 10 68

   * - Column
     - Type
     - Meaning
   * - ``image_name``
     - TEXT
     - Foreign key into ``images``.
   * - ``feature_type``
     - TEXT
     - Feature type (e.g. ``BODY_DISC``, ``STAR``, ``RING_EDGE``).
   * - ``source_model``
     - TEXT
     - NavModel family that produced the features (``body``, ``rings``,
       ``stars``).
   * - ``source_name``
     - TEXT
     - Body, ring, or catalog name (e.g. ``IAPETUS``, ``UCAC4``).
   * - ``n_features``
     - INTEGER
     - Features of this type/source extracted for the image.
   * - ``n_gated``
     - INTEGER
     - How many of them the reliability gate removed.

Indexes exist on ``images(image_date)``, ``images(instrument)``, and the
``image_name`` foreign keys.

Querying the database directly
------------------------------

Success rate per instrument, from the ``sqlite3`` shell:

.. code-block:: sql

    SELECT instrument,
           COUNT(*) AS images,
           AVG(status = 'success') AS success_rate
    FROM images
    GROUP BY instrument;

The ten largest fused offsets, with their confidence tier:

.. code-block:: sql

    SELECT image_name, offset_dv, offset_du, confidence_rank
    FROM images
    WHERE status = 'success'
    ORDER BY MAX(ABS(offset_dv), ABS(offset_du)) DESC
    LIMIT 10;

JSON columns unpack with SQLite's built-in JSON functions -- for example,
counting how often each technique was excluded from the consensus:

.. code-block:: sql

    SELECT excluded.value AS technique, COUNT(*) AS images
    FROM images, json_each(images.excluded_from_consensus) AS excluded
    GROUP BY excluded.value
    ORDER BY COUNT(*) DESC;

The same database loads straight into pandas:

.. code-block:: python

    import pandas as pd
    import sqlite3

    conn = sqlite3.connect('nav_stats.sqlite3')
    images = pd.read_sql_query('SELECT * FROM images', conn)
    per_technique = pd.read_sql_query(
        'SELECT t.*, i.instrument, i.image_date '
        'FROM techniques t JOIN images i USING (image_name)',
        conn,
    )
    # Median per-technique confidence by instrument:
    print(per_technique.groupby(['instrument', 'technique_name'])['confidence'].median())

Reporting
---------

``sd_stats_report [--db PATH] [--output-dir DIR] [--instrument NAME]
[--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--min-image NAME]
[--max-image NAME] [--top-n N] [--filelists] [--suspect-fraction F]
[--csv]`` writes ``report.md`` and its charts into the output directory.
All filters combine and apply to every section; dates are inclusive UTC
image dates, so a single day's run is ``--start-date D --end-date D``.
``--min-image`` / ``--max-image`` bound the numeric portion of the image
name (the first digit run in the basename, so ``--min-image N1454725799``
and ``--min-image 1454725799`` are equivalent); both bounds are inclusive
and either may be given alone. The same inputs always produce the same
numbers and the same charts.

Three options control drill-down output:

- ``--top-n N`` makes each categorical section (failure reasons, failure
  taxonomy, ensemble exclusions, suspect offsets) list up to N example
  image names per category, caps the suspect-offset and worst-BOTSIM-pair
  tables at N rows, and lists the N slowest images.
- ``--filelists`` writes one plain-text file per category (one image name
  per line, the full list rather than the top N) into the ``filelists/``
  subdirectory of the output directory, ready to feed back into re-runs
  and triage scripts.
- ``--csv`` writes ``images.csv`` next to ``report.md``: one row per image
  with every ``images`` column plus per-image technique and feature-source
  counts, for pandas or spreadsheet analysis.

The report contains:

- **Success / failure counts** with a breakdown of failure reasons.
- **Failure taxonomy by image content** — failed images classified from
  their recorded feature inventory (``stars-only``, ``single-body``,
  ``multi-body``, ``rings-only``, ``body+rings``, ``no-features``), with a
  per-category failure-reason breakdown and a per-body table of how often
  each named body appears in failed versus successful images (a body with
  a high failure share points at a modeling problem for that body).
- **Technique usage** — runs, non-spurious runs, and mean confidence per
  technique.
- **Model and source usage** — which bodies, rings, and star catalogs
  appeared, in how many images, and how many of their features survived the
  reliability gate.
- **Offset statistics** — mean, median, standard deviation, minimum, and
  maximum of the fused V and U offsets over successful images, with
  histograms, plus the same statistics grouped by (instrument, image size).
- **Suspect offsets** — successful images whose fused offset reaches at
  least ``--suspect-fraction`` (default 0.9) of the instrument's per-axis
  maximum expected pointing offset (the configured ``extfov_margin_vu``
  search margin; for Cassini ISS the NAC/WAC margin chosen from the image
  name) on either axis. An offset pinned near the search boundary may be a
  correlation artifact, so these images deserve operator review. When a
  limit cannot be resolved for an image (unknown instrument, no recorded
  image shape), the report says so rather than silently skipping it.
- **BOTSIM pair consistency (Cassini ISS)** — BOTSIM observations shutter
  the NAC and WAC simultaneously and the image names share one
  spacecraft-clock count. One WAC pixel is ten NAC pixels, so a consistent
  pair satisfies NAC offset = 10 x WAC offset per axis; the section reports
  the count, median, and 95th percentile of the ``NAC - 10 x WAC``
  residuals over pairs where both frames navigated successfully, and (with
  ``--top-n``) the worst pairs. This is an end-to-end accuracy check that
  needs no ground truth.
- **Cross-technique agreement** — for every technique pair, the median and
  95th-percentile Euclidean distance between their offsets on images where
  both produced non-spurious results.
- **Confidence calibration** — per confidence tier, the distribution of each
  image's maximum cross-technique disagreement. Without ground truth,
  agreement between independent techniques is the production proxy for
  accuracy (the calibrated anchor is the simulation campaign; see
  :doc:`/dev_guide/dev_guide_techniques_confidence`): a healthy pipeline shows
  high-tier images agreeing tightly and disagreement growing toward the low
  tier.
- **Ensemble outlier exclusions** — how often the ensemble excluded a
  technique from the consensus, and which techniques.
- **Run-time statistics** — minimum, maximum, mean, median, standard
  deviation, and total of the per-image wall-clock run times, a run-time
  histogram, and (with ``--top-n``) the slowest images. The section is
  omitted when no ingested document carries timing data.
