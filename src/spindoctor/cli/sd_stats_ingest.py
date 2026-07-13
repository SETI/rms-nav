#!/usr/bin/env python3
"""Ingest per-image navigation metadata JSON files into a SQLite database.

Dispatch script for the ``sd_stats_ingest`` console entry point.  See
``spindoctor.cli.stats`` for the statistics-system overview.
"""

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
