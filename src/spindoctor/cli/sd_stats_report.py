#!/usr/bin/env python3
"""Generate a deterministic statistics report (Markdown + charts).

Dispatch script for the ``sd_stats_report`` console entry point; reads the
results index written by ``sd_stats_ingest``.  See ``spindoctor.cli.stats``.

This program keeps ``print()`` and takes no logging configuration: its output
*is* terminal text for a person reading a report, and a logger would wrap that
in run-log machinery it has no use for.
"""

import os
import sys

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.cli.stats.report import main_report


def main() -> None:
    """Console entry point for ``sd_stats_report``."""
    sys.exit(main_report())


if __name__ == '__main__':
    main()
