Navigation Statistics
=====================

The statistics system turns the per-image metadata JSON files written by the
navigation pipeline into a results index and a deterministic report (Markdown
text plus PNG charts). It is the standing quality check on a production run:
success and failure rates, which techniques and models carry the load, offset
distributions, how well the techniques agree with one another, and whether the
confidence tiers behave as designed.

Two commands cooperate over the index and are cron-friendly (each is a single
non-interactive invocation):

.. code-block:: bash

    # Read a navigation-results root into a local index
    sd_stats_ingest --nav-results-root /data/nav-offset-results \
        --results-db sqlite:////data/nav-offset-results/index.sqlite3

    # Generate report.md + charts for any slice of the index
    sd_stats_report --results-db sqlite:////data/nav-offset-results/index.sqlite3 \
        --output-dir stats_report
    sd_stats_report --results-db sqlite:////data/nav-offset-results/index.sqlite3 \
        --instrument coiss --start-date 2005-03-01 --end-date 2005-03-01 \
        --output-dir day_report

The index is named by a connection URL rather than a file path, so the same two
commands work against a local file and against a shared server:

.. code-block:: text

    sqlite:////data/nav-offset-results/index.sqlite3
    postgresql+psycopg://user@dbhost/spindoctor

A ``sqlite:`` URL names a **local filesystem path** and nothing else: the
database library opens it directly, so it cannot live on a cloud store, and the
URL carries no query string. PostgreSQL is the option for sharing one index
across machines; its driver ships as an optional extra, installed with
``pip install rms-spindoctor[postgres]``.

Both commands take the URL from ``--results-db``, the ``environment.results_db``
configuration variable, or the ``NAV_RESULTS_DB`` environment variable, in that
order of precedence.

Ingestion
---------

``sd_stats_ingest`` walks each navigation-results root recursively for
``*_metadata.json`` files (the documents
:func:`~spindoctor.navigate_image_files.navigate_image_files` writes under
``nav_results_root``) and loads them into the index. Roots may be local
directories or any URL the project's ``filecache`` layer accepts, so
cloud-hosted results ingest the same way as local ones.

**The roots are navigation-results roots**, resolved the way every other
program resolves one: ``--nav-results-root`` (repeatable), then the
``environment.nav_results_root`` configuration variable, then the
``NAV_RESULTS_ROOT`` environment variable. Each row records the root it came
from and its path under that root, and every consumer looks a row up by that
pair. Pointing ingest at a subdirectory of a results root would produce
identifiers no consumer's lookup can match. A root that is not a location is
refused before anything is walked, and an empty one is such a root: with
``--nav-results-root "$ROOT"`` and ``ROOT`` unset the program stops rather than
ingesting whatever directory it was started from under a name nobody chose.

**Ingestion is incremental.** One recursive listing per root collects both the
metadata documents and the summary PNGs beside them, and carries each file's
size and modification time along with it. A file whose recorded size and
modification time still match the listing, and beside which the walk sees the
summary PNG the index already recorded, is not read at all, so a second pass
over an unchanged root costs one listing and nothing else. This holds for
files that could not be ingested as well: a file that is not a navigation
document is recorded as such, with the same two metrics, and is skipped for as
long as it does not change. ``--force`` re-reads everything; so does a storage
backend whose listing reports neither size nor modification time, which ingest
warns about rather than silently skipping.

**Those two metrics are everything a listing supplies about a file.** A document
rewritten in place that kept both of them is therefore one the ingest cannot
tell from the document it already read: it is skipped, and its row goes on
recording what the earlier document said, however many passes complete
afterwards. A tree restored by a copy that preserves modification times, a
document patched and then stamped back from a sibling, and a backend reporting
one modification time for two writes of equal length all produce that; an
ordinary re-navigation writes a different length at a later time and does not.
``sd_stats_ingest --force`` re-reads every document and is what puts such a row
right. Reading each file to find out whether it needs reading is the retrieval
the skip exists to avoid, which is why the remedy is a flag rather than a finer
comparison.

**A recorded refusal outlives the version of SpinDoctor that made it.** The
record says what was wrong with the file, not which build read it, so a document
refused by one build is skipped by every later one, including a build that has
since learned to read it. After upgrading SpinDoctor, run ``sd_stats_ingest
--force`` once over each root to re-offer everything it refused.

**Ingestion is idempotent.** The index holds one row per image, and re-ingesting
the same or an updated document replaces that image's row and its child rows
rather than duplicating them.

**A document that leaves the tree takes its row with it.** No row for an image
is what every consumer reads as "this image was never navigated", so a row is
only allowed to survive while the document behind it does. Each pass deletes
the rows of one root whose documents the walk no longer found, and reports how
many. That is done on the strength of a complete listing of the root and no
other: a root the walk could not list at all -- a mistyped path, an unmounted
share -- has nothing removed, and its ingest run is deliberately left
unfinished, so every consumer reports it as a root nobody has ingested rather
than answering "not navigated" for every image under it. A root that exists and
is empty completes normally.

**Ingestion is never automatic.** No batch driver runs it as a side effect. The
index is a snapshot of the tree as of the last ingest: there is no staleness
detection and no automatic refresh, so an operator who navigates more images and
wants them visible runs ingest again.

Every ingestible document must carry ``observation.image_name`` and
``observation.instrument``, every container it declares -- ``observation``,
``navigation_result`` and the objects and lists inside it, ``timing`` -- must
hold what the schema says, and each of its ``per_technique`` entries must carry
a ``technique_name`` of its own, since that name is what the index stores the
entry under (the pipeline writes all of this in every metadata document). A
results tree also holds ``*_metadata.json`` files that are not per-image
navigation documents at all; each is counted as an error for its own file, the
run continues, and the closing summary tallies the failures by reason and names
one file per reason, so several hundred files that were never navigation results
read as exactly that rather than as a broken ingest. A directory the walk cannot
list -- one this user may not read, a share that stopped answering -- costs the
files under it and nothing else: the pass continues over the rest of the root
and removes no row from it.

**A directory the walk did not list is counted, and absence under it is not an
answer.** The closing summary reports how many directories a pass did not
enumerate, and each pass records the number on its ``ingest_runs`` row. Two
things put a directory in that count: one the walk could not list, and one it
had already walked under another name, which is what a link pointing back into
a tree produces. Such a pass still completes, and the rows it wrote are as good
as any other pass's -- but under one of those directories the index holds no
rows at all, and **no row there means the walk never looked, not that the image
was never navigated**. Read a nonzero count as a question about the tree before
reading any absence beneath it as a result. A pass whose count is zero listed
every directory of the root, and absence means what it says everywhere.

**The exit status says whether the pass completed, not what it found.**
``sd_stats_ingest`` exits 0 when every named root was walked, whatever mix of
documents was read, skipped and refused, and 1 when the run could not complete:
no index or no results root could be resolved, a named root is not a location
that can be read, the index could not be opened, or a root could not be listed
at all. A scheduled invocation therefore reads the same status from the same
tree every time, and a status of 1 always means something needs fixing rather
than that a tree happens to hold no results.

The index is disposable, and there is no schema migration. It carries the column
set version that wrote it, and opening one stamped with a different version
fails naming both versions. The remedy is always the same -- delete the database
and re-run ``sd_stats_ingest`` (the source of truth is the metadata documents,
so nothing is lost).

Ingesting over a queue of workers
---------------------------------

A root of a few hundred thousand documents is one listing followed by that many
independent reads, so the reads can be spread over a queue. Three steps do it,
and the middle one is where the work happens:

.. code-block:: bash

    # 1. List each root, remove the rows of documents that have left it, and
    #    write out the shares.
    sd_stats_ingest --nav-results-root /data/nav-offset-results \
        --results-db postgresql+psycopg://user@dbhost/spindoctor \
        --output-cloud-tasks-file ingest_tasks.json

    # 2. Run the workers over those tasks, however the queue is driven. Each
    #    worker writes its results into an event log.
    sd_stats_ingest_cloud_tasks \
        --results-db postgresql+psycopg://user@dbhost/spindoctor ...

    # 3. Add the workers' tallies up and record them against each root.
    sd_stats_ingest --nav-results-root /data/nav-offset-results \
        --results-db postgresql+psycopg://user@dbhost/spindoctor \
        --complete-cloud-tasks-file events.log

**Workers on one machine can share a SQLite index**; workers on several cannot.
A ``sqlite:`` URL names a local file, so a run spread across machines connects
to PostgreSQL instead. Several worker processes on one machine writing to one
local file is supported, and needs no merge step -- there is one file.

**Only step 1 creates the index.** A worker opens an index that already exists
and fails if it does not, because a worker that created one would answer a
mistyped URL by building an empty index beside the real one, and every consumer
would then read absence of a row as "this image was never navigated".

**Only step 1 removes a row.** Deleting the rows of documents that have left the
tree is allowed on the strength of a complete listing of the root, and step 1 is
where the one listing of the pass happens; a worker holds a share and knows
nothing about the stubs outside it, so a worker that removed rows would remove
its peers'. Nothing a worker is about to write can be removed in step 1 either:
every file a worker is handed came from that listing, and only stubs the listing
did **not** hold are removed.

**A root is not readable until step 3.** Its ingest run stays unfinished from
step 1 onwards, so every consumer reports it as a root nobody has ingested while
the workers are still writing -- rather than answering from whichever shares
have landed.

**Step 3 refuses to finish a root its tasks did not cover.** Step 1 records how
many files it found; step 3 adds up how many the tasks ingested, skipped and
refused, counting each task's report once. If the tasks account for fewer files
than the listing found -- a task that failed, timed out, or was never run -- the
root is named, its run is left unfinished, and ``sd_stats_ingest`` exits 1. Re-run
the outstanding tasks and run step 3 again over a log holding the re-run results;
a task re-run over a share it already ingested reads nothing, because its files
match what the index records, so it reports them as skipped. A task that reports
twice is still one task: the later report stands in for the earlier one, and a
share reported twice never covers for a share that never ran. An account that
runs past the listing is refused the same way: with each task counted once the
sum can only exceed the listing on a report belonging somewhere else.

**Step 3 counts a task's result only for the root it was written under.** A
result names the run it belongs to and the root it wrote its rows under, and both
have to match. A run number is only unique inside the index that minted it, so a
task file run after its index was deleted and rebuilt -- the remedy for a
schema-version mismatch -- names a run of whatever was built next. Its shares add
up correctly and their rows are somewhere else entirely, and a root stamped on
them would hold nothing at all. Such results are counted and named in the summary
rather than credited; the root they were meant for is left unfinished, and step 1
over that root is what starts it again.

**Step 3 needs every task's result in the log it reads.** It reads one event log
and counts what is in it, so a root whose tasks are spread over several logs is
completed from the concatenation of them::

    cat worker-*.events.log > all-events.log

A task appearing in more than one of them is counted once, under the last report
of it in the file. Order therefore decides between two reports of one task that
disagree -- a task that failed on one worker and succeeded on another -- so a
concatenation that puts the failure last leaves the root unfinished, which the
same log concatenated the other way completes. Both outcomes are safe; neither
stamps a root whose documents were not read. A log naming only some of a root's
tasks leaves that root unfinished, which is the same outcome as tasks that never
ran and is corrected the same way.

**Step 3 refuses a root whose listing was never recorded.** A root that step 1
could not list -- mistyped, or an unmounted share -- gets no tasks and no record
of what it holds, and step 3 will not finish it: there is nothing for its tasks
to be measured against, and a root completed on that basis would report every
image under it as never navigated. Correct the root and run step 1 again.

``--force`` belongs to step 1, and is refused in step 3, which reads no document.
A pass whose shares must ignore what the index records is one whose fan-out was
run with ``--force``.

The tasks file is a JSON array in the shape a ``cloud_tasks`` queue loads. Each
entry has a ``task_id`` and a ``data`` object carrying ``run_id`` (the ingest
run the share belongs to), ``root_url`` (the normalized results root),
``force``, ``has_file_metrics`` (whether the listing reported a size and
modification time for every file), and ``files`` -- one object per document,
with its ``results_path_stub``, ``mtime_ns``, ``size_bytes`` and
``has_summary_png``. Every one of those comes from the single listing, so no
worker stats a file or checks for one.

Step 3 reads the ``cloud_tasks`` event log, which is JSON Lines with one event
per line; the ``task_completed`` events carry what each worker returned, under
the ``task_id`` the task ran as. Lines that are not events are counted and
reported rather than refused, because an event log being appended to while it is
read ends in a partial line.

The closing summary of step 3 is the summary a single-process ingest writes:
files seen, ingested, skipped and refused, with the refusals tallied by reason
and one example file per reason. The reasons come back in the task results,
since a worker has no run log to write them in. Every file a share could not
read is named in its task result too, and the ones refused for something about
the document are recorded in the index's ``failed_files`` table as well.

Index schema
------------

Opening the index directly is a supported way to answer questions the standard
report does not: the ``sqlite3`` command-line shell or ``psql``, Python's
:mod:`sqlite3` module or ``psycopg``, pandas, or a GUI browser. Six tables hold
the data.

An image is identified by the pair ``(root_url, results_path_stub)``:
``root_url`` is the navigation-results root in normalized form, and
``results_path_stub`` is the image's path under it with the ``_metadata.json``
suffix removed, for example
``COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB``. Two volumes may
hold images with the same basename, so the pair rather than the name is what
keys a row. ``techniques`` and ``feature_sources`` reference ``images`` by that
pair with ``ON DELETE CASCADE``.

``images`` -- one row per image:

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Column
     - Type
     - Meaning
   * - ``root_url``
     - TEXT
     - Navigation-results root this image was ingested from, normalized to
       one absolute form with no trailing separator. Half of the primary key.
   * - ``results_path_stub``
     - TEXT
     - The image's path under that root, without the ``_metadata.json``
       suffix. The other half of the primary key.
   * - ``volume``
     - TEXT
     - First path segment of the stub, e.g. ``COISS_2001``. NULL for a stub
       with no separator, which is what the simulated dataset produces.
   * - ``image_name``
     - TEXT
     - Image filename, e.g. ``N1454725799_1_CALIB.IMG``.
   * - ``instrument``
     - TEXT
     - ``coiss`` / ``vgiss`` / ``gossi`` / ``nhlorri`` / ``sim``, from the
       metadata document's ``observation.instrument`` field (required; a
       document without it is skipped as an ingest error).
   * - ``camera``
     - TEXT
     - The camera that took the image (``NAC`` / ``WAC`` / ``SSI`` /
       ``LORRI``), from the metadata document's ``observation.camera``
       field. Offset statistics group by this: pointing error belongs to
       the camera, not the spacecraft. NULL when the document carries no
       camera; it is never inferred from the image name.
   * - ``image_path``
     - TEXT
     - Absolute path of the source image at navigate time.
   * - ``image_et``
     - DOUBLE
     - Observation midtime, TDB seconds past J2000. Taken from the
       navigation provenance, or -- for an image that never loaded, which
       has no provenance -- from the ``observation.image_et`` the navigator
       read out of the PDS3 index. An image whose navigation died for want
       of a SPICE kernel is therefore still placed in time.
   * - ``image_date``
     - TEXT
     - UTC calendar date ``YYYY-MM-DD`` derived from ``image_et``; drives
       the ``--start-date`` / ``--end-date`` report filters.
   * - ``status``
     - TEXT
     - Navigation outcome: ``success``, ``failed``, ``conflicted``, or
       ``error`` (the last from image-load failures); ``unknown`` when the
       metadata document carries no status at all.
   * - ``status_error``
     - TEXT
     - The fatal error that ended the run, e.g. ``missing_spice_data``.
       Stored verbatim: the SPICE-error image selection matches this token
       exactly.
   * - ``status_reason``
     - TEXT
     - The navigator's own explanation of the outcome: successes hold ``ok``
       or ``rank_1_only``, failures hold the failure reason (e.g.
       ``no_features_extracted``). A different vocabulary from
       ``status_error``, in its own column; a failure is described by
       whichever of the two the document carried.
   * - ``offset_dv``, ``offset_du``
     - DOUBLE
     - Fused pointing offset in pixels (V then U), stored exactly as the
       document's top-level ``offset`` holds it. This is the number every
       consumer applies. NULL when navigation found no offset.
   * - ``sigma_dv``, ``sigma_du``
     - DOUBLE
     - Per-axis 1-sigma uncertainty of the fused offset, pixels.
   * - ``covariance_vv``, ``covariance_vu``, ``covariance_uu``
     - DOUBLE
     - The 2x2 offset block of the fused covariance, pixels squared. For a
       twist-fitted result the rotation row and column are deliberately not
       indexed.
   * - ``sigma_along_unobservable_px``
     - DOUBLE
     - Uncertainty along the direction the scene could not constrain.
   * - ``rotation_deg``, ``sigma_rotation_deg``
     - DOUBLE
     - Fitted camera twist and its uncertainty, present only where twist was
       fitted.
   * - ``confidence``
     - DOUBLE
     - Fused confidence in ``[0, 1]``, from the document's top-level
       ``confidence``.
   * - ``confidence_rank``
     - TEXT
     - Confidence tier label assigned by the ensemble.
   * - ``n_techniques``
     - INTEGER
     - Number of per-technique results recorded for the image.
   * - ``excluded_from_consensus``
     - JSON
     - Technique names the ensemble excluded as outliers (``[]`` when none).
   * - ``image_class``
     - TEXT
     - Image-classifier verdict (e.g. ``clean``).
   * - ``noise_sigma``
     - DOUBLE
     - Image-classifier noise estimate.
   * - ``image_shape_v``, ``image_shape_u``
     - INTEGER
     - Pixel dimensions of the image data (V then U), from the metadata
       document's ``observation.image_shape`` field; NULL when the image
       never loaded.
   * - ``run_start``, ``run_end``
     - TEXT
     - UTC ISO8601 run start and end of the navigation of this image, from
       the metadata document's ``timing`` section; start is captured before
       the image load and end after navigation (or at error time).
   * - ``elapsed_s``
     - DOUBLE
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
   * - ``image_number``
     - BIGINT
     - Numeric portion of the image name (the first digit run in the
       basename). What the ``--min-image`` / ``--max-image`` filters compare.
   * - ``has_summary_png``
     - BOOLEAN
     - Whether the ingest walk saw a ``_summary.png`` beside the document.
   * - ``start_et``, ``stop_et``, ``exposure_s``
     - DOUBLE
     - Shutter open and close epochs and the exposure between them.
   * - ``sclk_start``, ``sclk_midtime``, ``sclk_stop``
     - TEXT
     - The same three instants as spacecraft-clock strings.
   * - ``camera_frame_id``, ``ck_frame_id``
     - INTEGER
     - SPICE frame identifiers of the camera and of the C-kernel a corrected
       attitude targets.
   * - ``cmatrix``, ``cmatrix_original``
     - JSON
     - Corrected and as-flown camera attitude, nine floats row-major.
       ``cmatrix`` is absent where the navigation fitted a camera rotation.
       Absent means SQL NULL here and in every JSON column, so
       ``WHERE cmatrix IS NOT NULL`` selects the images that carry one; an
       empty list or object, where one appears, is a value rather than an
       absence.
   * - ``source_file``
     - TEXT
     - Path or URL of the ingested metadata document.
   * - ``mtime_ns``, ``size_bytes``
     - BIGINT
     - Modification time and size of that document as the ingest listing
       reported them. A later pass compares these to decide whether the
       document has to be read again.

``techniques`` -- one row per technique result per image:

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Column
     - Type
     - Meaning
   * - ``root_url``, ``results_path_stub``
     - TEXT
     - Foreign key into ``images``. Unique together with
       ``technique_name``: a technique reports once for an image.
   * - ``technique_name``
     - TEXT
     - Technique class name (e.g. ``BodyLimbNav``, ``StarUniqueMatchNav``).
   * - ``offset_dv``, ``offset_du``
     - DOUBLE
     - The technique's own offset estimate, pixels.
   * - ``sigma_dv``, ``sigma_du``
     - DOUBLE
     - Per-axis 1-sigma from the technique's covariance.
   * - ``confidence``
     - DOUBLE
     - The technique's calibrated confidence in ``[0, 1]``.
   * - ``spurious``
     - BOOLEAN
     - True when the technique flagged its own result as spurious.
   * - ``at_edge``
     - BOOLEAN
     - True when the fit landed at the edge of its search space.
   * - ``source_names``
     - JSON
     - Body / ring / catalog names the technique used.
   * - ``diagnostics``
     - JSON
     - The technique's full diagnostics dataclass as a JSON object.

``feature_sources`` -- per image, feature counts grouped by source:

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Column
     - Type
     - Meaning
   * - ``root_url``, ``results_path_stub``
     - TEXT
     - Foreign key into ``images``. Unique together with ``feature_type``,
       ``source_model`` and ``source_name``.
   * - ``feature_type``
     - TEXT
     - Feature type (e.g. ``BODY_DISC``, ``STAR``, ``RING_EDGE``).
   * - ``source_model``
     - TEXT
     - :class:`~spindoctor.nav_model.NavModel` family that produced the
       features (``body``, ``rings``, ``stars``, ``titan``).
   * - ``source_name``
     - TEXT
     - Body, ring, or catalog name (e.g. ``IAPETUS``, ``UCAC4``).
   * - ``n_features``
     - INTEGER
     - Features of this type/source extracted for the image.
   * - ``n_gated``
     - INTEGER
     - How many of them the reliability gate removed.

``ingest_runs`` records one row per ingest pass over one root: the root, when
the pass started and finished (``finished_utc`` is NULL while it is running),
how many files it saw, ingested, skipped as unchanged, could not read and
removed, how many directories it did not list (``directories_missed``), and the
schema version it wrote. A root whose newest row has no finish time, or which
has no row at all, has not been fully ingested, and a consumer says so rather
than reading absence of rows as "nothing was navigated". A finished row whose
``directories_missed`` is nonzero covers the root apart from those directories:
the rows it wrote are good, and absence of a row under one of them means the
walk never looked rather than that the image was never navigated.

``failed_files`` records one row per file that is not a current-schema
navigation document: the root and stub that identify it, the reason it was
refused, and the size and modification time it had when it was read. It is what
lets a second pass skip it. It is deliberately not an ``images`` row, because a
file with no usable data must not answer the question ``images`` exists to
answer.

``schema_meta`` holds a single row stamping the database with the column-set
version that created it.

Indexes exist on ``images(results_path_stub)``, ``images(image_date)``,
``images(instrument)``, and ``ingest_runs(root_url)``, plus the uniqueness
constraints on the two child tables.

Querying the index directly
---------------------------

Success rate per instrument:

.. code-block:: sql

    SELECT instrument,
           COUNT(*) AS images,
           AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END) AS success_rate
    FROM images
    GROUP BY instrument;

The ten largest fused offsets, with their confidence tier:

.. code-block:: sql

    SELECT image_name, offset_dv, offset_du, confidence_rank
    FROM images
    WHERE status = 'success'
    ORDER BY GREATEST(ABS(offset_dv), ABS(offset_du)) DESC
    LIMIT 10;

SQLite has no ``GREATEST``; its two-argument ``MAX`` does the same thing:

.. code-block:: sql

    ORDER BY MAX(ABS(offset_dv), ABS(offset_du)) DESC

The failure reason of every non-successful image, whichever of the two
vocabularies described it:

.. code-block:: sql

    SELECT COALESCE(status_reason, status_error) AS reason,
           instrument,
           COUNT(*) AS images
    FROM images
    WHERE status != 'success'
    GROUP BY reason, instrument
    ORDER BY images DESC;

JSON columns unpack with each backend's own JSON functions -- for example,
counting how often each technique was excluded from the consensus. On SQLite:

.. code-block:: sql

    SELECT excluded.value AS technique, COUNT(*) AS images
    FROM images, json_each(images.excluded_from_consensus) AS excluded
    GROUP BY excluded.value
    ORDER BY COUNT(*) DESC;

and on PostgreSQL, where the column is ``jsonb``:

.. code-block:: sql

    SELECT technique, COUNT(*) AS images
    FROM images,
         jsonb_array_elements_text(images.excluded_from_consensus) AS technique
    GROUP BY technique
    ORDER BY COUNT(*) DESC;

A child table joins to its image on the pair that keys one:

.. code-block:: sql

    SELECT t.technique_name, i.instrument, AVG(t.confidence)
    FROM techniques t
    JOIN images i
      ON i.root_url = t.root_url
     AND i.results_path_stub = t.results_path_stub
    WHERE NOT t.spurious
    GROUP BY t.technique_name, i.instrument;

The index loads straight into pandas from either backend:

.. code-block:: python

    import pandas as pd
    import sqlalchemy

    engine = sqlalchemy.create_engine('sqlite:////data/nav-offset-results/index.sqlite3')
    images = pd.read_sql_query('SELECT * FROM images', engine)
    per_technique = pd.read_sql_query(
        'SELECT t.*, i.instrument, i.image_date '
        'FROM techniques t JOIN images i USING (root_url, results_path_stub)',
        engine,
    )
    # Median per-technique confidence by instrument:
    print(per_technique.groupby(['instrument', 'technique_name'])['confidence'].median())

Reporting
---------

``sd_stats_report [--results-db URL] [--root ROOT] [--output-dir DIR]
[--instrument NAME] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
[--min-image NAME] [--max-image NAME] [--top-n N] [--filelists]
[--suspect-fraction F] [--csv]`` writes ``report.md`` and its charts into the
output directory. All filters combine and apply to every section; dates are
inclusive UTC image dates, so a single day's run is ``--start-date D
--end-date D``. ``--min-image`` / ``--max-image`` bound the numeric portion of
the image name (the first digit run in the basename, so ``--min-image
N1454725799`` and ``--min-image 1454725799`` are equivalent); both bounds are
inclusive and either may be given alone. The same inputs always produce the same
numbers and the same charts.

``--root`` restricts the report to one ingested navigation-results root and may
be given more than once; with none given the report covers every root the index
holds. A report legitimately spans roots where a per-image lookup never does.
Naming a root the index has not fully ingested is an error rather than an empty
report.

This is the one program for which the index is not optional: it has no
file-reading mode, and it fails naming ``--results-db`` when no index is
configured.

Three options control drill-down output:

- ``--top-n N`` makes each categorical section (failure reasons, failure
  taxonomy, ensemble exclusions, suspect offsets) list up to N example
  image names per category and instrument, caps the suspect-offset and
  worst-BOTSIM-pair tables at N rows, and lists the N slowest images.
- ``--filelists`` writes one plain-text file per category and instrument
  (one image name per line, the full list rather than the top N) into the
  ``filelists/`` subdirectory of the output directory, ready to feed back
  into re-runs and triage scripts.
- ``--csv`` writes ``images.csv`` next to ``report.md``: one row per image
  with every ``images`` column in schema order -- ``root_url`` and
  ``results_path_stub`` through ``mtime_ns`` and ``size_bytes`` -- plus
  ``n_technique_rows``, ``n_feature_sources``, ``n_features`` and ``n_gated``
  aggregates, for pandas or spreadsheet analysis. Rows end with a single
  newline on every platform, and a JSON column that holds nothing is an empty
  cell.

The first two write *image names* rather than file names -- ``N1454725799``
rather than ``N1454725799_1_CALIB.IMG`` -- because that is the token the
datasets' ``--image-filelist`` option selects on. The filelists are
directly consumable by it: one name per line, with a leading ``#`` comment
naming the category.

Every image count in the report carries its percentage -- ``5 (3.2%)``.
Counts are broken down by instrument: a table of counts gets one column per
instrument plus a total column, where an instrument column's percentage is
of that instrument's images and the total column's is of all selected
images, so each column sums to 100% on its own. Tables of *statistics*
rather than counts (offsets, run time, per-body shares, cross-technique
agreement) carry an instrument column instead, a total being meaningless
for a mean or a standard deviation. Bar charts are stacked, one segment per
instrument, with a fixed color per instrument across every chart.

The report contains:

- **Images selected** -- per instrument: how many images, the first and last
  image, and the first and last available date. Image numbers only compare
  within one instrument, so the bounds are never pooled across instruments.
  The date bounds are found independently of the image ordering, so a
  single image with no recorded epoch at either end of the number range
  cannot hide the instrument's real time span.
- **Success / failure counts** with a breakdown of failure reasons. The
  reason table carries each reason's status, so errors (SPICE-related or
  not) are visible alongside outright navigation failures.
- **Failure taxonomy by image content** -- failed images classified from
  their recorded feature inventory (``stars-only``, ``single-body``,
  ``multi-body``, ``rings-only``, ``body+rings``, ``no-features``), with a
  per-category failure-reason breakdown and a per-body table of how often
  each named body appears in failed versus successful images (a body with
  a high failure share points at a modeling problem for that body).
- **Technique usage** -- the images each technique ran on, plus a per-
  technique, per-instrument detail table of non-spurious runs and mean
  confidence.
- **Model and source usage** -- which bodies, rings, and star catalogs
  appeared, in how many images, and how many of their features survived the
  reliability gate.
- **Offset statistics** -- mean, median, standard deviation, minimum, and
  maximum of the fused V and U offsets over successful images, grouped by
  camera, with one histogram per camera, plus the same statistics grouped
  by (instrument, camera, image size). Distributions are never pooled
  across cameras: one Cassini WAC pixel is ten NAC pixels, so a pooled
  distribution would describe neither camera.
- **Suspect offsets** -- successful images whose fused offset reaches at
  least ``--suspect-fraction`` (default 0.9) of the instrument's per-axis
  maximum expected pointing offset (the configured ``extfov_margin_vu``
  search margin; for Cassini ISS the NAC/WAC margin chosen from the image
  name) on either axis. An offset pinned near the search boundary may be a
  correlation artifact, so these images deserve operator review. When a
  limit cannot be resolved for an image (unknown instrument, no recorded
  image shape), the report says so rather than silently skipping it.
- **BOTSIM pair consistency (Cassini ISS)** -- BOTSIM observations shutter
  the NAC and WAC simultaneously and the image names share one
  spacecraft-clock count. One WAC pixel is ten NAC pixels, so a consistent
  pair satisfies NAC offset = 10 x WAC offset per axis; the section reports
  the count, median, and 95th percentile of the ``NAC - 10 x WAC``
  residuals over pairs where both frames navigated successfully, and (with
  ``--top-n``) the worst pairs. This is an end-to-end accuracy check that
  needs no ground truth.
- **Cross-technique agreement** -- for every technique pair, the median and
  95th-percentile Euclidean distance between their offsets on images where
  both produced non-spurious results.
- **Confidence calibration** -- per confidence tier, the distribution of each
  image's maximum cross-technique disagreement. The tiers always read
  ``high`` / ``medium`` / ``low`` / ``failed`` / ``conflicted``, so a tier
  with no images reads as an explicit zero rather than a missing row.
  Without ground truth,
  agreement between independent techniques is the production proxy for
  accuracy (the calibrated anchor is the simulation campaign; see
  :doc:`/dev_guide/dev_guide_techniques_confidence`): a healthy pipeline shows
  high-tier images agreeing tightly and disagreement growing toward the low
  tier.
- **Ensemble outlier exclusions** -- how often the ensemble excluded a
  technique from the consensus, and which techniques.
- **Run-time statistics** -- per instrument (and pooled, when more than one
  instrument is selected): minimum, maximum, mean, median, standard
  deviation, and total of the per-image wall-clock run times, a run-time
  histogram, and (with ``--top-n``) the slowest images. The section is
  omitted when no ingested document carries timing data.
