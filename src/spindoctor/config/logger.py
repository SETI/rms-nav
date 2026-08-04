"""The main logger instance.

``MAIN_LOGGER`` carries top-level program events for one run.  It is
configured by :func:`spindoctor.config.logging_config.build_main_logger`, which
a driver reaches through ``build_run_logging`` at startup.

The image logger lives in :mod:`spindoctor.config.log_scope`, which routes
per-image records to whichever image scope is open.
"""

import pdslogger

MAIN_LOGGER = pdslogger.PdsLogger('sd_offset', lognames=False, digits=3)
"""The run's logger, carrying top-level program events for one execution."""
