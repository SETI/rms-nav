"""Navigation statistics system: metadata ingestion and reporting.

Two programs cooperate over a local SQLite database:

- ``sd_stats_ingest`` (:mod:`spindoctor.cli.stats.ingest`) walks one or more
  navigation-results roots (local paths or any URL accepted by ``FCPath``),
  parses every ``*_metadata.json`` written by ``navigate_image_files``, and
  loads it into a normalized SQLite database.  Ingestion is idempotent: one
  row per image, keyed by image name, updated in place when the same image is
  ingested again.
- ``sd_stats_report`` (:mod:`spindoctor.cli.stats.report`) queries the
  database and emits a deterministic report (Markdown text plus PNG charts):
  success/failure counts with failure reasons, technique and model usage,
  per-body and per-ring appearance/usage, offset statistics, cross-technique
  agreement, and confidence-tier calibration against observed agreement.

Both are cron-friendly single commands and work over any date range and any
instrument.
"""

from spindoctor.cli.stats.ingest import main_ingest
from spindoctor.cli.stats.report import main_report

__all__ = ['main_ingest', 'main_report']
