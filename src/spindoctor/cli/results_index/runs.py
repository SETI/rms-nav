"""The record of one ingest pass over one root.

A run row is committed before the walk begins and stamped with a finish time
only when the pass is over, so a consumer that looks while a pass is in flight
sees a root whose newest run has not finished, and refuses to read absence of a
row as "this image was never navigated".

A root the walk could not list deliberately never gets its stamp.  A mistyped
or unmounted root must read as one nobody has ingested rather than as one that
holds nothing, and leaving the finish time NULL is what says so.

A pass fanned out over many workers is one run all the same.  What the fan-out
itself did -- how many files it saw, how many rows it removed -- is recorded on
the row as soon as it is known,
because nothing later in the pass can find it out again; the finish time waits
for the workers, so the root stays unreadable until every one of their shares
is accounted for.
"""

import datetime
from dataclasses import dataclass
from typing import Any, cast

import sqlalchemy

from spindoctor.cli.results_index.counts import IngestCounts
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


@dataclass(frozen=True)
class _UnfinishedRun:
    """A run that has begun and has no finish time yet.

    Parameters:
        run_id: The run's surrogate identifier.
        root_url: Normalized URL of the root it covers.
        files_seen: Metadata files the fan-out's walk found, or None when the
            run never recorded a listing at all -- a root nothing could list, or
            a pass that died before it had one.  That is not zero: zero is what
            a root that was listed and holds nothing records, and only zero can
            be accounted for by no shares.
        files_removed: Rows the fan-out deleted, whose documents had left the
            tree.
    """

    run_id: int
    root_url: str
    files_seen: int | None
    files_removed: int | None


def _record_fan_out(engine: sqlalchemy.Engine, run_id: int, counts: IngestCounts) -> None:
    """Record what a fan-out found, without stamping the run as finished.

    The walk happens once, in the program that divides the work up, so what it
    found is written down there: no worker sees the whole root, and the pass
    that completes the run has only the workers' own tallies.  The finish time
    is deliberately left alone, so every consumer keeps treating the root as one
    nobody has ingested until the shares are accounted for.

    Parameters:
        engine: The open index.
        run_id: The run to record against.
        counts: What the fan-out did.
    """
    with engine.begin() as connection:
        connection.execute(
            INGEST_RUNS.update()
            .where(INGEST_RUNS.c.run_id == run_id)
            .values(
                files_seen=counts.files_seen,
                files_ingested=0,
                files_skipped=0,
                files_failed=0,
                files_removed=counts.files_removed,
            )
        )


def _record_shares(engine: sqlalchemy.Engine, run_id: int, counts: IngestCounts) -> None:
    """Record what a pass's shares reported, without stamping the run.

    A pass whose shares do not account for exactly the files its listing found
    stays unfinished, and so does one whose listing was never recorded at all,
    but the shares that did report did real work: their documents are in the
    index.  Writing their tally down is what lets an operator see how far the
    pass got, instead of the row of zeros the fan-out left.

    The row records the log it was last handed rather than the most that has
    ever been reported, so completing again from a shorter log lowers it.  The
    account is a reading of one log, and a run's own numbers are not a place to
    accumulate a history of the attempts to complete it.

    Parameters:
        engine: The open index.
        run_id: The run to record against.
        counts: What the shares reported between them.
    """
    with engine.begin() as connection:
        connection.execute(
            INGEST_RUNS.update()
            .where(INGEST_RUNS.c.run_id == run_id)
            .values(
                files_ingested=counts.files_ingested,
                files_skipped=counts.files_skipped,
                files_failed=counts.files_failed,
            )
        )


def _unfinished_run(connection: sqlalchemy.Connection, root_url: str) -> _UnfinishedRun | None:
    """Return the newest unfinished run of one root.

    Only the newest run is a candidate.  A root whose newest run has finished
    has nothing outstanding, and an older unfinished run under a newer finished
    one is a pass that was abandoned; completing that one would stamp a finish
    time onto a walk nothing came back from.

    Parameters:
        connection: An open connection to the index.
        root_url: The normalized root to ask about.

    Returns:
        The run, or None when the root's newest run has already finished or the
        root has no run at all.
    """
    newest = (
        sqlalchemy.select(
            INGEST_RUNS.c.run_id,
            INGEST_RUNS.c.root_url,
            INGEST_RUNS.c.finished_utc,
            INGEST_RUNS.c.files_seen,
            INGEST_RUNS.c.files_removed,
        )
        .where(INGEST_RUNS.c.root_url == root_url)
        .order_by(INGEST_RUNS.c.run_id.desc())
        .limit(1)
    )
    row = connection.execute(newest).first()
    if row is None or row.finished_utc is not None:
        return None
    return _UnfinishedRun(
        run_id=int(row.run_id),
        root_url=str(row.root_url),
        files_seen=row.files_seen,
        files_removed=row.files_removed,
    )


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
            )
        )
