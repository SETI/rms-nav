"""One ingest pass divided into cloud tasks, and put back together again.

An ingest of an archive-scale root is one listing followed by a great many
independent document reads, which is exactly the shape a task queue serves.
The pass is therefore split in three, and the split is what this module is:

* **Fan-out.**  One program lists each root once, removes the rows whose
  documents have left the tree, records what the walk found on the root's
  ingest run, and divides the files it found into shares.  It is the only
  point in the pass that sees a whole root.
* **A share.**  A worker is handed a list of files with the metrics the
  fan-out's walk reported for them.  It skips the ones the index has already
  read unchanged, reads the rest, writes their rows, and returns what it did.
  It has no run log, so its tally is its return value.
* **Completion.**  The shares' tallies are added up and written to the ingest
  run, which is what finally stamps the root as ingested.

Why a worker never removes a row
--------------------------------

Deleting the rows of documents that have left the tree is licensed by a
complete listing of the root, and a worker holding a share of one has no
evidence about the stubs outside its share: deleting on it would delete its
peers' rows.  Nothing here hands a worker a listing, and the prune refuses
anything that is not a complete one, so the restriction is a property of the
seam rather than a rule a worker has to remember.

Why the fan-out removes them, and completion does not
-----------------------------------------------------

The fan-out is the one moment in the pass when a complete listing exists, and
it is the listing the shares were cut from: every stub a worker writes is one
the listing held, and the prune deletes only stubs it did not hold, so the two
sets cannot intersect however the workers are scheduled.  Removing at
completion instead would mean listing the whole root a second time -- the most
expensive thing an ingest does, and a paid round trip per directory on a cloud
root -- to act on evidence no share ever came from.

What makes a run finishable
---------------------------

The fan-out records how many files it saw.  A run is stamped only when the
shares account for at least that many files between them, because a task that
never reported leaves its documents unread, and a run stamped without them
would tell every consumer that absence of their rows means those images were
never navigated.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import sqlalchemy
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.stats.ingest.chunks import _batched, _ingest_chunk
from spindoctor.cli.stats.ingest.counts import IngestCounts
from spindoctor.cli.stats.ingest.driver import (
    INGEST_COMMIT_CHUNK_SIZE,
    _files_to_read,
    _prune_missing,
)
from spindoctor.cli.stats.ingest.runs import (
    _finish_run,
    _record_fan_out,
    _start_run,
    _unfinished_run,
)
from spindoctor.cli.stats.ingest.store import _recorded_files
from spindoctor.cli.stats.ingest.walk import _ListedFile, _walk_root
from spindoctor.results_index import normalize_root_url

__all__ = [
    'INGEST_TASK_SHARE_SIZE',
    'FanOut',
    'TaskCompletion',
    'TaskResults',
    'complete_ingest_tasks',
    'fan_out_ingest_tasks',
    'ingest_task_share',
    'task_results_from_event_log',
]

INGEST_TASK_SHARE_SIZE = 512
"""How many metadata files one task is handed.

The share bounds what a single task costs and what re-running one costs, and it
is what makes naming every file a share could not read a bounded thing to put in
a task result.
"""

TASK_COMPLETED_EVENT = 'task_completed'
"""Event type under which a worker's return value is written to the event log."""

_TASK_EVENT_PREFIX = 'task_'
"""Prefix of every event type that reports the outcome of one task."""


@dataclass
class _ShareCounts(IngestCounts):
    """A share's tally, which names every file it could not read.

    A pass over a whole root keeps one example file per reason, because a tree
    holding several hundred thousand documents that were never navigation
    results would otherwise be summarized by a list of them.  A share is
    bounded by the fan-out, so naming all of its failures is bounded too -- and
    a worker has no run log to name them in instead.

    Parameters:
        failed_files: Every file this share could not turn into rows.
    """

    failed_files: list[str] = field(default_factory=list)

    def record_failure(self, reason: str, source_file: str) -> None:
        """Count one file that could not be ingested, and name it.

        Parameters:
            reason: What was wrong with it, with nothing file-specific in it.
            source_file: The file, kept both as this reason's example and in
                the share's own list.
        """
        super().record_failure(reason, source_file)
        self.failed_files.append(source_file)


@dataclass(frozen=True)
class _Share:
    """One task's worth of an ingest, as the task data describes it.

    Parameters:
        run_id: The ingest run this share belongs to, echoed back so that
            completion can attribute the tally without guessing.
        root_url: Normalized URL of the root the files live under.
        force: Whether to read every document regardless of what is recorded.
        has_file_metrics: Whether the fan-out's listing reported both a size
            and a modification time for every file of this share.
        files: The files, with the metrics the listing reported for them.
        summary_stubs: Stubs of this share that the walk saw a summary PNG for.
    """

    run_id: int
    root_url: str
    force: bool
    has_file_metrics: bool
    files: list[_ListedFile]
    summary_stubs: set[str]


@dataclass
class FanOut:
    """What dividing an ingest into tasks produced.

    Parameters:
        tasks: The task descriptions, in the shape a cloud-tasks queue loads:
            each a ``task_id`` and the ``data`` one worker is handed.
        counts: What the fan-out itself did -- the files its walks found, the
            rows it removed, the directories it did not list, and the roots it
            could not list at all.
    """

    tasks: list[dict[str, Any]] = field(default_factory=list)
    counts: IngestCounts = field(default_factory=IngestCounts)


@dataclass
class TaskResults:
    """What one worker event log holds.

    Parameters:
        results: The values ``process_task`` returned, newest last.
        lines_unread: Lines of the log that are not JSON objects.  An event log
            is appended to while it is being written, so a partial last line is
            ordinary; a great many of them says the file is not an event log.
        tasks_unfinished: Tasks the log records as having ended without
            returning a value -- an exception, a timeout, a worker that exited.
            Their documents were never read, so the run they belong to cannot be
            stamped.
    """

    results: list[dict[str, Any]] = field(default_factory=list)
    lines_unread: int = 0
    tasks_unfinished: int = 0


@dataclass
class TaskCompletion:
    """What adding the shares up did.

    Parameters:
        counts: The shares' tallies, summed.
        runs_completed: Ingest runs stamped as finished.
        roots_unaccounted: Roots whose shares did not account for every file
            the fan-out saw, each named with the shortfall.  Their runs keep
            their NULL finish times.
        roots_without_a_run: Roots with no unfinished run to complete, each
            named.
        results_unclaimed: Task results naming a run none of the given roots is
            waiting on.
        results_failed: Task results in which a worker reported an error
            instead of a share.
        results_unreadable: Task results that are not the shape a worker
            returns at all.
    """

    counts: IngestCounts = field(default_factory=IngestCounts)
    runs_completed: int = 0
    roots_unaccounted: list[str] = field(default_factory=list)
    roots_without_a_run: list[str] = field(default_factory=list)
    results_unclaimed: int = 0
    results_failed: int = 0
    results_unreadable: int = 0


def _task_files(files: Sequence[_ListedFile], summary_stubs: set[str]) -> list[dict[str, Any]]:
    """Render one share's files as the task data carries them.

    Parameters:
        files: The files of this share.
        summary_stubs: Stubs the whole walk saw a summary PNG for.

    Returns:
        One JSON object per file.  The summary flag travels per file rather
        than as a second list, because it is a property of the file the worker
        is about to write a row for and every consumer of the task data needs
        it beside the stub.
    """
    return [
        {
            'results_path_stub': listed.results_path_stub,
            'mtime_ns': listed.mtime_ns,
            'size_bytes': listed.size_bytes,
            'has_summary_png': listed.results_path_stub in summary_stubs,
        }
        for listed in files
    ]


def fan_out_ingest_tasks(
    engine: sqlalchemy.Engine,
    roots: list[str],
    *,
    force: bool = False,
    share_size: int = INGEST_TASK_SHARE_SIZE,
    logger: PdsLogger,
) -> FanOut:
    """List each root once and divide the documents under it into tasks.

    Each root gets an ingest run, a single walk, and -- when that walk covered
    the whole root -- the removal of the rows whose documents the tree no longer
    holds.  What the walk found is recorded on the run immediately, because
    nothing later in the pass can find it out again; the finish time is left for
    :func:`complete_ingest_tasks`, so until then every consumer treats the root
    as one nobody has ingested.

    A root the walk could not list yields no task and keeps its unfinished run,
    exactly as it does in a pass that reads the documents itself: a mistyped or
    unmounted root is not an empty one.

    Parameters:
        engine: The open index, which must already carry the schema.
        roots: Navigation results roots, each normalized to the form the rows
            record and consumers compare against.
        force: Have the workers read every document, ignoring what is recorded.
        share_size: How many files one task is handed.
        logger: Logger for the per-root scan summary.

    Returns:
        The tasks, and what the fan-out itself did.

    Raises:
        ValueError: If the share size is not at least one file, which would
            divide a root into no tasks at all and lose every document under it.
    """
    if share_size < 1:
        raise ValueError(f'a task share holds at least one file, not {share_size}')
    fan_out = FanOut()
    for root_str in roots:
        root_url = normalize_root_url(root_str)
        root = FCPath(root_url)
        counts = IngestCounts()
        run_id = _start_run(engine, root_url)
        logger.info('Dividing %s into ingest tasks', root_url)
        listing = _walk_root(root, logger=logger)
        counts.files_seen = len(listing.metadata_files)
        counts.directories_missed = listing.directories_missed
        if not listing.root_listed:
            counts.roots_unreadable = 1
            fan_out.counts.add(counts)
            continue
        if listing.covers_whole_root:
            with engine.connect() as connection:
                recorded = _recorded_files(connection, root_url)
            counts.files_removed = _prune_missing(
                engine, root_url, listing, recorded, logger=logger
            )
        _record_fan_out(engine, run_id, counts)
        tasks_of_root = [
            {
                'task_id': f'ingest-{run_id}-{index:06d}',
                'data': {
                    'run_id': run_id,
                    'root_url': root_url,
                    'force': force,
                    'has_file_metrics': listing.has_file_metrics,
                    'files': _task_files(share, listing.summary_stubs),
                },
            }
            for index, share in enumerate(_batched(listing.metadata_files, share_size))
        ]
        fan_out.tasks.extend(tasks_of_root)
        logger.info(
            'Divided %d file(s) under %s into %d task(s)',
            counts.files_seen,
            root_url,
            len(tasks_of_root),
        )
        fan_out.counts.add(counts)
    return fan_out


def _required(task_data: dict[str, Any], key: str, kind: type) -> Any:
    """Read one required value of a task's data, refusing anything else.

    Parameters:
        task_data: The task data.
        key: The value's name.
        kind: The type it must be.

    Returns:
        The value.

    Raises:
        ValueError: If the value is absent or of another type.  A task that does
            not say what to ingest is not something to guess at: every default
            available here would ingest the wrong files or the wrong root.
    """
    if key not in task_data:
        raise ValueError(f'the task carries no "{key}"')
    value = task_data[key]
    if kind is bool:
        acceptable = isinstance(value, bool)
    else:
        # A bool is an int to isinstance, so an identifier declared as a whole
        # number would otherwise accept True and be used as the number one.
        acceptable = isinstance(value, kind) and not isinstance(value, bool)
    if not acceptable:
        raise ValueError(f'the task\'s "{key}" is {type(value).__name__}, not {kind.__name__}')
    return value


def _share_from_task(task_data: dict[str, Any]) -> _Share:
    """Read one task's data into the share it describes.

    Parameters:
        task_data: The task data, as the fan-out wrote it.

    Returns:
        The share.

    Raises:
        ValueError: If the task data is not the shape a fan-out produces,
            naming what was wrong with it.  A worker that guessed at a
            malformed task would write rows for files nobody asked about, under
            a root nobody named, and a consumer cannot tell such a row from a
            correct one.
    """
    run_id = int(_required(task_data, 'run_id', int))
    root_url = str(_required(task_data, 'root_url', str))
    force = bool(_required(task_data, 'force', bool))
    has_file_metrics = bool(_required(task_data, 'has_file_metrics', bool))
    entries = _required(task_data, 'files', list)
    files: list[_ListedFile] = []
    summary_stubs: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f'a "files" entry is {type(entry).__name__}, not an object')
        stub = str(_required(entry, 'results_path_stub', str))
        mtime_ns = entry.get('mtime_ns')
        size_bytes = entry.get('size_bytes')
        if mtime_ns is not None and (isinstance(mtime_ns, bool) or not isinstance(mtime_ns, int)):
            raise ValueError(f'the "mtime_ns" of {stub} is not a whole number')
        if size_bytes is not None and (
            isinstance(size_bytes, bool) or not isinstance(size_bytes, int)
        ):
            raise ValueError(f'the "size_bytes" of {stub} is not a whole number')
        if bool(_required(entry, 'has_summary_png', bool)):
            summary_stubs.add(stub)
        files.append(_ListedFile(results_path_stub=stub, mtime_ns=mtime_ns, size_bytes=size_bytes))
    return _Share(
        run_id=run_id,
        root_url=root_url,
        force=force,
        has_file_metrics=has_file_metrics,
        files=files,
        summary_stubs=summary_stubs,
    )


def ingest_task_share(
    engine: sqlalchemy.Engine, task_data: dict[str, Any], *, logger: PdsLogger
) -> dict[str, Any]:
    """Ingest one share of a root and report what it did.

    The share is skipped, read and written by exactly the rules a pass over a
    whole root uses, over the files the fan-out handed it and no others.  It
    removes no row: nothing here holds a listing, and a share is not evidence
    about the stubs outside it.

    A task re-run over a share it already ingested reads nothing, because every
    one of its files now matches what the index records, which is what makes a
    retried task cheap and harmless.

    Parameters:
        engine: The open index, which must already carry the schema.
        task_data: The task data, as :func:`fan_out_ingest_tasks` wrote it.
        logger: Logger for the per-file failures.  A cloud task has no run log,
            so this discards what it is given and the tally is returned instead.

    Returns:
        The task result: the run and root it belongs to, how many files it
        ingested, skipped and could not read, and the name of every file it
        could not read.

    Raises:
        ValueError: If the task data is not the shape a fan-out produces.
    """
    share = _share_from_task(task_data)
    counts = _ShareCounts()
    root = FCPath(share.root_url)
    stubs = [listed.results_path_stub for listed in share.files]
    with engine.connect() as connection:
        recorded = _recorded_files(connection, share.root_url, stubs=stubs)
    to_read = _files_to_read(
        share.files,
        share.summary_stubs,
        recorded,
        force=share.force,
        has_file_metrics=share.has_file_metrics,
    )
    counts.files_skipped = len(share.files) - len(to_read)
    for chunk in _batched(to_read, INGEST_COMMIT_CHUNK_SIZE):
        _ingest_chunk(
            engine,
            root,
            chunk,
            root_url=share.root_url,
            summary_stubs=share.summary_stubs,
            counts=counts,
            logger=logger,
        )
    return {
        'status': 'ok',
        'run_id': share.run_id,
        'root_url': share.root_url,
        'files_ingested': counts.files_ingested,
        'files_skipped': counts.files_skipped,
        'files_failed': counts.files_failed,
        'failed_files': counts.failed_files,
    }


def task_results_from_event_log(path: FCPath) -> TaskResults:
    """Read the values the workers returned out of a cloud-tasks event log.

    The log is JSON Lines, one event per line, written as each task ends.  Only
    the events that carry a return value are of interest here; the rest are
    counted, because a task that ended without returning one read none of its
    documents.

    Parameters:
        path: The event log.

    Returns:
        The returned values, and what else the log held.
    """
    found = TaskResults()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            found.lines_unread += 1
            continue
        if not isinstance(event, dict):
            found.lines_unread += 1
            continue
        event_type = event.get('event_type')
        if not isinstance(event_type, str) or not event_type.startswith(_TASK_EVENT_PREFIX):
            continue
        if event_type != TASK_COMPLETED_EVENT:
            found.tasks_unfinished += 1
            continue
        result = event.get('result')
        if isinstance(result, dict):
            found.results.append(result)
        else:
            found.tasks_unfinished += 1
    return found


def _share_tally(result: dict[str, Any]) -> tuple[int, IngestCounts] | None:
    """Read one task result as the tally of a share, or refuse it.

    Parameters:
        result: One value a worker returned.

    Returns:
        The run it belongs to and what it did, or None when the value is not a
        share's tally at all.  Its files are then unaccounted for, which is what
        stops the run being stamped.
    """
    run_id = result.get('run_id')
    if isinstance(run_id, bool) or not isinstance(run_id, int):
        return None
    counts = IngestCounts()
    for name in ('files_ingested', 'files_skipped', 'files_failed'):
        value = result.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        setattr(counts, name, value)
    return run_id, counts


def complete_ingest_tasks(
    engine: sqlalchemy.Engine,
    roots: list[str],
    results: Sequence[dict[str, Any]],
    *,
    logger: PdsLogger,
) -> TaskCompletion:
    """Add the shares up and stamp the runs they completed.

    A run is stamped only when its shares account for at least as many files as
    the fan-out's walk saw.  A share that never reported -- a task that failed,
    timed out, or was never run -- leaves its documents unread, and stamping the
    run anyway would tell every consumer that the absence of their rows means
    those images were never navigated.  Such a root keeps its unfinished run and
    is named instead.

    A share counted twice is not a shortfall.  A retried task re-reads nothing,
    because its files already match what the index records, so it reports its
    share as skipped and the total runs past what the walk saw.

    Parameters:
        engine: The open index.
        roots: The navigation results roots whose runs are being completed,
            normalized the way the fan-out normalized them.
        results: The values the workers returned.
        logger: Logger for the per-root outcome.

    Returns:
        What was completed and what was not.
    """
    completion = TaskCompletion()
    by_run: dict[int, IngestCounts] = {}
    results_of_run: dict[int, int] = {}
    for result in results:
        if result.get('status') != 'ok':
            # A worker that could not open the index, or was handed a task it
            # could not read, reports that instead of a tally.  Its files were
            # never read, so the run it belonged to comes up short and is left
            # unfinished; naming the count here is what says why.
            completion.results_failed += 1
            logger.error(
                'A task reported no share: %s', result.get('status_error', '(no reason given)')
            )
            continue
        tally = _share_tally(result)
        if tally is None:
            completion.results_unreadable += 1
            continue
        run_id, counts = tally
        by_run.setdefault(run_id, IngestCounts()).add(counts)
        results_of_run[run_id] = results_of_run.get(run_id, 0) + 1
    claimed: set[int] = set()
    for root_str in roots:
        root_url = normalize_root_url(root_str)
        with engine.connect() as connection:
            run = _unfinished_run(connection, root_url)
        if run is None:
            completion.roots_without_a_run.append(root_url)
            continue
        claimed.add(run.run_id)
        counts = by_run.get(run.run_id, IngestCounts())
        counts.files_seen = run.files_seen or 0
        counts.files_removed = run.files_removed or 0
        counts.directories_missed = run.directories_missed or 0
        accounted = counts.files_ingested + counts.files_skipped + counts.files_failed
        if accounted < counts.files_seen:
            completion.roots_unaccounted.append(
                f'{root_url} ({accounted} of {counts.files_seen} file(s) accounted for)'
            )
            logger.error(
                'The tasks of %s account for %d of the %d file(s) its listing found, so its '
                'ingest run is left unfinished: absence of a row under it is not evidence '
                'that its image was never navigated',
                root_url,
                accounted,
                counts.files_seen,
            )
            completion.counts.add(counts)
            continue
        _finish_run(engine, run.run_id, counts)
        completion.runs_completed += 1
        logger.info(
            'Completed the ingest of %s: %d ingested, %d skipped, %d failed of %d file(s)',
            root_url,
            counts.files_ingested,
            counts.files_skipped,
            counts.files_failed,
            counts.files_seen,
        )
        completion.counts.add(counts)
    completion.results_unclaimed = sum(
        count for run_id, count in results_of_run.items() if run_id not in claimed
    )
    return completion
