"""Navigation statistics system: metadata ingestion and reporting.

Two programs cooperate over a local SQLite database:

- ``sd_stats_ingest`` (:mod:`spindoctor.cli.stats.ingest`) walks one or more
  navigation-results roots (local paths or any URL accepted by ``FCPath``),
  parses every ``*_metadata.json`` written by ``navigate_image_files``, and
  loads it into a normalized SQLite database.  Ingestion is idempotent: one
  row per image, keyed by image name, updated in place when the same image is
  ingested again.
- ``sd_stats_report`` (:mod:`spindoctor.cli.stats.report`, with section
  builders in :mod:`spindoctor.cli.stats.report_sections` and shared helpers
  in :mod:`spindoctor.cli.stats.report_common`) queries the database and
  emits a deterministic report (Markdown text plus PNG charts):
  success/failure counts with failure reasons, a failure taxonomy by scene
  content with per-body failure shares, technique and model usage, offset
  statistics (overall and by instrument/image size), a suspect-offset screen
  against the configured search limits, Cassini BOTSIM NAC/WAC pair
  consistency, cross-technique agreement, confidence-tier calibration
  against observed agreement, and run-time statistics.  Optional drill-down
  flags list example images per category, write per-category filelists, and
  export a one-row-per-image CSV.

Both are cron-friendly single commands and work over any date range and any
instrument.
"""

from spindoctor.cli.stats.ingest import main_ingest
from spindoctor.cli.stats.report import main_report

__all__ = ['main_ingest', 'main_report']
