"""The record of one ingest pass over one root.

A run row is committed before the walk begins and stamped with a finish time
only when the pass is over, so a consumer that looks while a pass is in flight
sees a root whose newest run has not finished, and refuses to read absence of a
row as "this image was never navigated".

A root the walk could not list deliberately never gets its stamp.  A mistyped
or unmounted root must read as one nobody has ingested rather than as one that
holds nothing, and leaving the finish time NULL is what says so.
"""

import datetime
from typing import Any, cast

import sqlalchemy

from spindoctor.cli.stats.ingest.counts import IngestCounts
from spindoctor.results_index import INGEST_RUNS, SCHEMA_VERSION


def _start_run(engine: sqlalchemy.Engine, root_url: str) -> int:
    """Record that an ingest of one root has begun.

    The row is committed before the walk, so a consumer that looks while the
    run is in flight sees a root whose newest run has not finished and refuses
    to read absence from it.

    Parameters:
        engine: The open index.
        root_url: Normalized URL of the root being ingested.

    Returns:
        The run's surrogate identifier.
    """
    with engine.begin() as connection:
        result = connection.execute(
            INGEST_RUNS.insert().values(
                root_url=root_url,
                started_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                finished_utc=None,
                schema_version=SCHEMA_VERSION,
            )
        )
    return int(cast(tuple[Any, ...], result.inserted_primary_key)[0])


def _finish_run(engine: sqlalchemy.Engine, run_id: int, counts: IngestCounts) -> None:
    """Stamp an ingest run as complete and record what it covered.

    Parameters:
        engine: The open index.
        run_id: The run to complete.
        counts: What the run did.
    """
    with engine.begin() as connection:
        connection.execute(
            INGEST_RUNS.update()
            .where(INGEST_RUNS.c.run_id == run_id)
            .values(
                finished_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                files_seen=counts.files_seen,
                files_ingested=counts.files_ingested,
                files_skipped=counts.files_skipped,
                files_failed=counts.files_failed,
                files_removed=counts.files_removed,
                directories_missed=counts.directories_missed,
            )
        )
