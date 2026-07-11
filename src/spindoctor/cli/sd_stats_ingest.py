#!/usr/bin/env python3
################################################################################
# sd_stats_ingest.py
#
# Ingest per-image navigation metadata JSON files into a local SQLite
# statistics database. See spindoctor.cli.stats for the system overview.
################################################################################

import os
import sys

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.cli.stats.ingest import main_ingest


def main() -> None:
    """Console entry point for ``sd_stats_ingest``."""
    sys.exit(main_ingest())


if __name__ == '__main__':
    main()
