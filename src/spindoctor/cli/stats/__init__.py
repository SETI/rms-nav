"""Navigation statistics system: metadata ingestion and reporting.

Two programs cooperate over the results index, the database that
:mod:`spindoctor.results_index` defines:

- ``sd_stats_ingest`` (:mod:`spindoctor.cli.stats.ingest`, with the column
  mapping in :mod:`spindoctor.cli.stats.ingest_rows`) walks one or more
  navigation-results roots -- local paths or any URL accepted by ``FCPath`` --
  reads every ``*_metadata.json`` written by ``navigate_image_files``, and
  writes one row per image keyed by its root and its results path stub.  A
  document whose recorded size and modification time still match the tree is
  not read again, so a second pass over an unchanged root costs one listing.
- ``sd_stats_report`` (:mod:`spindoctor.cli.stats.report`, with section
  builders in :mod:`spindoctor.cli.stats.report_sections` and shared helpers
  in :mod:`spindoctor.cli.stats.report_common`) queries the index and emits a
  deterministic report (Markdown text plus PNG charts): success/failure counts
  with failure reasons, a failure taxonomy by scene content with per-body
  failure shares, technique and model usage, offset statistics (overall and by
  instrument/image size), a suspect-offset screen against the configured search
  limits, Cassini BOTSIM NAC/WAC pair consistency, cross-technique agreement,
  confidence-tier calibration against observed agreement, and run-time
  statistics.  Optional drill-down flags list example images per category,
  write per-category filelists, and export a one-row-per-image CSV.

Both are cron-friendly single commands and work over any date range and any
instrument.  Both take the index as a connection URL, so the same pair of
commands works against a local SQLite file and against a PostgreSQL server.
"""

from spindoctor.cli.stats.ingest import ingest_metadata_files
from spindoctor.cli.stats.report import main_report

__all__ = ['ingest_metadata_files', 'main_report']
