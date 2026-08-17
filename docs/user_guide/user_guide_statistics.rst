Navigation Statistics
=====================

``sd_stats_report`` turns a results index into a deterministic report: Markdown
text plus PNG charts. It is the standing quality check on a production run --
success and failure rates, which techniques and models carry the load, offset
distributions, how well the techniques agree with one another, and whether the
confidence tiers behave as designed -- and it is cron-friendly, being a single
non-interactive invocation:

.. code-block:: bash

    sd_stats_report --results-db sqlite:////data/nav-offset-results/index.sqlite3 \
        --output-dir stats_report
    sd_stats_report --results-db sqlite:////data/nav-offset-results/index.sqlite3 \
        --instrument coiss --start-date 2005-03-01 --end-date 2005-03-01 \
        --output-dir day_report

**The report reads a results index and nothing else.** The index is a database
built from the navigation results tree by a separate pass,
``sd_stats_ingest``, and :doc:`user_guide_results_index` is where it is
documented: how it is named and resolved, how to build and rebuild one, the
tables it holds, and how to query it directly for the questions this report
does not answer.

.. code-block:: bash

    sd_stats_ingest --nav-results-root /data/nav-offset-results \
        --results-db sqlite:////data/nav-offset-results/index.sqlite3
    sd_stats_report --results-db sqlite:////data/nav-offset-results/index.sqlite3

An index is optional here as it is everywhere else, and the option behaves as it
does everywhere else: named on the command line, in the
``environment.results_db`` configuration variable or in ``NAV_RESULTS_DB``, in
that order, with the literal ``none`` meaning no index.

Without one the navigation results tree is read:

.. code-block:: bash

    sd_stats_report --nav-results-root /data/nav-offset-results

**What that costs is one full read of every document under the root.** Every
number in the report comes from a query, so a report over a tree is a report over
an index of that tree: the run ingests the tree into a temporary index of its own,
reports from that, and throws the index away. The report is identical either way,
because the same statements answer it. But the read is exactly the cost an index
exists to remove --- a Cassini-scale root is several hundred thousand documents,
and on a cloud root each one is a paid round trip --- so for anything but a local
tree and a single report, build an index with ``sd_stats_ingest`` first and name
it.

The roots to read come from ``--nav-results-root`` (repeatable), then the
``nav_results_root`` configuration variable, then ``NAV_RESULTS_ROOT``, as they
do for the ingest. What the pass could not read is reported the way the ingest
reports it: files that are not navigation documents are counted and tallied by
reason, with one example each, so a report over a tree says what it covered
rather than quietly covering less. A root that cannot be listed at all fails the
run.

``sd_stats_report [--results-db URL] [--nav-results-root ROOT] [--root ROOT]
[--output-dir DIR]
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
report. It selects among the roots one index holds, so it is a different thing
from ``--nav-results-root`` and is refused when no index is named.

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
