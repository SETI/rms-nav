"""Shared tree and index helpers for the statistics tests.

The document factories the trees here are built out of are shared with the rest
of the subtree and live in ``tests/spindoctor/conftest.py``; what is left here
is what only the statistics tests read.

The cloud-task helpers run the same pass in its three separate stages -- divide
a root into shares, ingest a share, add the shares up -- so that a test asserting
on one of them does not have to restate the other two.

:class:`ReplayedFacts` is here for the same reason: a source promises no order,
and the way to measure that a report does not depend on one is to hand it the
same facts in an order the test chose.  Two modules do that -- one over a whole
report, one over a section whose reduction has two candidates to choose
between -- so the stand-in source is written once.
"""

import json
import os
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from tests.spindoctor.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    write_metadata,
)
from tests.spindoctor.results_index.conftest import (
    postgres_decoy_schema,
    postgres_schema,
    postgres_server_url,
    postgres_url,
)

from spindoctor.cli.stats.ingest import (
    TaskCompletion,
    TaskResult,
    complete_ingest_tasks,
    fan_out_ingest_tasks,
    ingest_task_share,
)
from spindoctor.cli.stats.report import build_report
from spindoctor.nav_records import (
    ImageFacts,
    ListedRecord,
    NavRecord,
    RecordSource,
    Selection,
    TreeRecordSource,
    UnreadableFile,
)
from spindoctor.results_index import (
    INGEST_RUNS,
    IndexRecordSource,
    open_index,
)

# The statistics postgres tier runs against a schema of its own, exactly as the
# results-index tier does; re-exporting rather than restating keeps one
# definition of how that schema is created and dropped.
__all__ = ['postgres_decoy_schema', 'postgres_schema', 'postgres_server_url', 'postgres_url']

DATA_DIR = Path(__file__).resolve().parent / 'data'
"""Directory holding the fixture results tree and the frozen report output."""

RESULTS_TREE = DATA_DIR / 'results_tree'
"""Fixture results tree the report regression is measured over."""

GOLDEN_DIR = DATA_DIR / 'golden'
"""Frozen report and CSV output this tree produces, which a change must reproduce."""

GOLDEN_VARIANTS: dict[str, dict[str, Any]] = {
    'full': {'top_n': 5, 'filelists': True, 'csv_export': True},
    'filtered': {
        'instrument': 'coiss',
        'min_image': '1294561202',
        'max_image': '1294563000',
        'top_n': 3,
        'csv_export': True,
    },
}
"""The two report invocations the frozen output under :data:`GOLDEN_DIR` holds.

Between them they cover the report with every drill-down on and with a narrowing
that leaves one instrument, so the parity of the two storages is measured over
the same two invocations the frozen output pins.
"""


class ReplayedFacts:
    """A record source handing back facts a test already read, in an order it chose.

    The seam promises no order, so the way to measure that the report does not
    depend on one is to give it the same facts several times over in several
    orders.  Reading them out of a real source once and replaying them is what
    makes the orders comparable: two storages differ in more than their order,
    and a difference in the output would then say nothing about which.

    Parameters:
        facts: What to yield, in the order to yield it.
    """

    def __init__(self, facts: Sequence[ImageFacts | UnreadableFile]) -> None:
        self._facts = tuple(facts)

    def record(self, stub: str) -> NavRecord:
        """Refuse the per-image lookup, which nothing being measured here asks for.

        Parameters:
            stub: The image's results path stub.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError('replayed facts answer no per-image lookup')

    def records(self, selection: Selection) -> Iterator[NavRecord | UnreadableFile]:
        """Refuse the record stream, which nothing being measured here asks for.

        Parameters:
            selection: Which records were asked for.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError('replayed facts answer no record stream')

    def facts(self, selection: Selection) -> Iterator[ImageFacts | UnreadableFile]:
        """Yield the held facts, in the order this source was built with.

        Parameters:
            selection: Which images were asked for; the facts were already
                narrowed when they were read, so this narrows nothing further.

        Returns:
            The facts, one at a time.
        """
        return iter(self._facts)

    def listing(self, selection: Selection) -> Iterator[ListedRecord]:
        """Refuse the listing, which nothing being measured here asks for.

        Parameters:
            selection: Which files were asked for.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError('replayed facts answer no listing')

    def describe(self) -> str:
        """Say where these records came from, for a run log.

        Returns:
            A phrase naming the replay rather than a storage.
        """
        return 'facts replayed in an order chosen by a test'

    def close(self) -> None:
        """Release what this source holds open, which is nothing."""

    def __enter__(self) -> RecordSource:
        """Enter a run's use of this source.

        Returns:
            The source itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave a run's use of this source.

        Parameters:
            exc_type: The exception's class, when the run is leaving on one.
            exc: The exception, when the run is leaving on one.
            traceback: Its traceback, when the run is leaving on one.
        """
        self.close()


@pytest.fixture
def quiet_logger() -> pdslogger.PdsLogger:
    """Return a logger that keeps ingest chatter out of the test output.

    Returns:
        A logger of its own, so raising its level cannot affect another test.
        The name carries a token that is unique for the life of the process:
        an object's address is not, since the object it belonged to is already
        collected and the next allocation is free to reuse it.
    """
    logger = pdslogger.PdsLogger(f'stats_test_{uuid.uuid4().hex}')
    logger.set_level('ERROR')
    return logger


PINNED_MTIME_NS = 1_700_000_000_000_000_000
"""Modification time given to documents whose metrics must not vary.

An arbitrary instant, in the past, with nanoseconds a filesystem storing whole
seconds would round away to the same value for every file given it.
"""


def write_metadata_in_each(
    roots: Sequence[Path], stub: str, document: dict[str, Any]
) -> list[Path]:
    """Write one document under several roots, indistinguishable but for its root.

    The rows of two roots holding one stub are told apart by the root half of
    the key alone, so a guard against a query that reads the stub alone has to
    hold when the two files match in every other respect -- same bytes, same
    size, same modification time.  Two writes microseconds apart usually do land
    on the same filesystem timestamp and occasionally do not, and a guard that
    depends on which is a guard that passes a root-blind lookup whenever the
    clock ticks between them.  So the time is set here rather than left to the
    clock.

    Parameters:
        roots: The results roots to write under.
        stub: The stub each of them holds.
        document: The document to write into each.

    Returns:
        The paths written.
    """
    written = []
    for root in roots:
        path = write_metadata(root, stub, document)
        os.utime(path, ns=(PINNED_MTIME_NS, PINNED_MTIME_NS))
        written.append(path)
    return written


_NOT_A_NAVIGATION_DOCUMENT = 'not_a_navigation_document'
"""The one key of a document that reads as JSON and holds no navigation result."""


def write_refusal_matching(root: Path, stub: str, document: Path) -> Path:
    """Write a document a pass refuses, matching another file's size and time.

    A refused file is recorded and skipped on the next pass on exactly the
    evidence an ingested one is: the size and the modification time the walk
    reports for it.  So the two halves of a two-root guard on the refusal table
    have to be indistinguishable but for their root, which means the length is
    asked for here rather than left to whatever a document happened to
    serialize to.

    Parameters:
        root: The results root to write under.
        stub: The document's results path stub under that root.
        document: The file whose size and modification time to match.

    Returns:
        The path written.

    Raises:
        ValueError: If the file to match is shorter than the smallest document
            of this shape, which cannot then be padded out to it.
    """
    metrics = document.stat()
    empty = json.dumps({_NOT_A_NAVIGATION_DOCUMENT: ''})
    if metrics.st_size < len(empty):
        raise ValueError(f'{document} holds {metrics.st_size} bytes, fewer than {len(empty)}')
    padding = 'x' * (metrics.st_size - len(empty))
    path = root / f'{stub}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({_NOT_A_NAVIGATION_DOCUMENT: padding}), encoding='utf-8')
    os.utime(path, ns=(metrics.st_mtime_ns, metrics.st_mtime_ns))
    return path


def recorded_lines(
    logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch, *, level: str = 'info'
) -> list[str]:
    """Capture one level of a logger's output, rendered as it would be written.

    ``pdslogger`` writes through its own stream handler, so a test reads what a
    pass told an operator by standing in for the method rather than by capturing
    a stream.

    Parameters:
        logger: The logger to record.
        monkeypatch: Fixture the method is replaced through.
        level: Which method to record.

    Returns:
        The list the lines land in, which fills as the recorded code runs.
    """
    written: list[str] = []

    def recording(message: object, *args: object) -> None:
        written.append(str(message) % args if args else str(message))

    monkeypatch.setattr(logger, level, recording)
    return written


def index_source(url: str, roots: Sequence[Path]) -> IndexRecordSource:
    """Open a record source over the rows an index holds for the given roots.

    Parameters:
        url: The index URL, which must already carry the schema and the rows.
        roots: The results roots whose rows the source answers about.

    Returns:
        The open source, which the caller closes when it is done with it.
    """
    return IndexRecordSource(open_index(url), [root.as_posix() for root in roots], url, ())


@pytest.fixture
def indexed_tree(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> Iterator[RecordSource]:
    """Yield a record source over an index built from the frozen fixture tree.

    Parameters:
        tmp_path: Directory the index file is written into.
        quiet_logger: Logger the ingest reports through.

    Yields:
        The open source, reading rows.
    """
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [RESULTS_TREE], logger=quiet_logger)
    with index_source(url, [RESULTS_TREE]) as source:
        yield source


@pytest.fixture
def walked_tree() -> Iterator[RecordSource]:
    """Yield a record source over the documents of the frozen fixture tree.

    The other half of every parity comparison: the same records, read out of the
    files themselves rather than out of an index ingested from them.

    Yields:
        The open source, reading documents.
    """
    with TreeRecordSource([RESULTS_TREE.as_posix()]) as source:
        yield source


def report_from_the_index(
    url: str, out: Path, *, logger: pdslogger.PdsLogger, **options: Any
) -> Path:
    """Ingest the fixture tree into an index and write one report from its rows.

    One definition of the whole cycle -- ingest, open, build, dispose -- so that
    a change to the report's signature is made once rather than once per backend.

    Parameters:
        url: The index URL to create and ingest into.
        out: Directory receiving the report.
        logger: Logger the ingest reports through.
        options: Report options, passed through to ``build_report``.

    Returns:
        The directory the report was written into.
    """
    ingest_tree(url, [RESULTS_TREE], logger=logger)
    out.mkdir(parents=True, exist_ok=True)
    with index_source(url, [RESULTS_TREE]) as source:
        build_report(source, out, **options)
    return out


def report_from_the_tree(out: Path, **options: Any) -> Path:
    """Write one report over the documents of the fixture tree, opening no index.

    Parameters:
        out: Directory receiving the report.
        options: Report options, passed through to ``build_report``.

    Returns:
        The directory the report was written into.
    """
    out.mkdir(parents=True, exist_ok=True)
    with TreeRecordSource([RESULTS_TREE.as_posix()]) as source:
        build_report(source, out, **options)
    return out


FIRST_STUB = 'VOL/N1454725799_1_CALIB'
"""The stub of the first document every fixture tree below writes."""


def build_tree(root: Path, count: int) -> list[str]:
    """Write a small results tree and return the stubs it holds.

    Parameters:
        root: The results root to write under.
        count: How many documents to write.

    Returns:
        The stubs, in the order the walk will report them.
    """
    stubs = []
    for index in range(count):
        name = f'N{1454725799 + index}_1_CALIB'
        write_metadata(root, f'VOL/{name}', metadata_document(image_name=f'{name}.IMG'))
        stubs.append(f'VOL/{name}')
    return sorted(stubs)


def root_strings(roots: Sequence[Path | str]) -> list[str]:
    """Render results roots as the strings a command line would carry.

    A root reaches a program as text, and two spellings of one root -- with and
    without a trailing separator -- are one root.  A test asking about that has
    to hand the spelling over untouched, which a ``Path`` cannot do: it drops a
    trailing separator the moment it is constructed.

    Parameters:
        roots: The roots, as paths or as the strings an operator typed.

    Returns:
        One string per root.
    """
    return [root.as_posix() if isinstance(root, Path) else root for root in roots]


def fan_out(
    url: str,
    roots: Sequence[Path | str],
    *,
    logger: pdslogger.PdsLogger,
    share_size: int = 2,
    **options: Any,
) -> list[dict[str, Any]]:
    """Create an index and divide the given roots into tasks.

    Parameters:
        url: The index URL to create.
        roots: The results roots to list.
        logger: Logger the fan-out reports through.
        share_size: How many files one task is handed.
        options: Further keyword arguments for the fan-out.

    Returns:
        The task descriptions.
    """
    engine = open_index(url, create=True)
    try:
        return fan_out_ingest_tasks(
            engine,
            root_strings(roots),
            share_size=share_size,
            logger=logger,
            **options,
        ).tasks
    finally:
        engine.dispose()


def run_shares(
    url: str, tasks: Sequence[dict[str, Any]], *, logger: pdslogger.PdsLogger
) -> list[TaskResult]:
    """Ingest every task's share, one after another, as one worker would.

    Parameters:
        url: The index URL, which must already carry the schema.
        tasks: The task descriptions.
        logger: Logger the shares report through.

    Returns:
        What each share returned, under the task that returned it, in task
        order.  A completion tells one task's report from another's by that
        identity, so the helper that runs the shares is where it is attached.
    """
    engine = open_index(url)
    try:
        return [
            TaskResult(
                task_id=str(task['task_id']),
                result=ingest_task_share(engine, task['data'], logger=logger),
            )
            for task in tasks
        ]
    finally:
        engine.dispose()


def reported(task_id: str, result: dict[str, Any]) -> TaskResult:
    """Return one hand-built task result under the task that reported it.

    Parameters:
        task_id: The identity the queue ran the task under.
        result: What that task returned.

    Returns:
        The pair a completion reads.
    """
    return TaskResult(task_id=task_id, result=result)


def complete(
    url: str,
    roots: Sequence[Path | str],
    results: Sequence[TaskResult],
    *,
    logger: pdslogger.PdsLogger,
) -> TaskCompletion:
    """Add up the shares of the given roots and stamp what they completed.

    Parameters:
        url: The index URL.
        roots: The results roots whose runs are being completed.
        results: What the shares returned.
        logger: Logger the completion reports through.

    Returns:
        The completion outcome.
    """
    engine = open_index(url)
    try:
        return complete_ingest_tasks(engine, root_strings(roots), results, logger=logger)
    finally:
        engine.dispose()


def rows_of(url: str, table: sqlalchemy.Table) -> list[tuple[Any, ...]]:
    """Return every row of one table, in a stable order.

    Parameters:
        url: The index URL.
        table: The table to read.

    Returns:
        The rows as tuples, ordered by their text columns so two indexes built
        by different routes compare equal when they hold the same rows.
    """
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            found = [tuple(row) for row in connection.execute(sqlalchemy.select(table))]
    finally:
        engine.dispose()
    return sorted(found, key=repr)


def run_rows(url: str) -> list[Any]:
    """Return every ingest run of an index, oldest first.

    Parameters:
        url: The index URL.

    Returns:
        The rows.
    """
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            return list(
                connection.execute(sqlalchemy.select(INGEST_RUNS).order_by(INGEST_RUNS.c.run_id))
            )
    finally:
        engine.dispose()


def cycle(
    tmp_path: Path, roots: Sequence[Path | str], *, logger: pdslogger.PdsLogger, share_size: int = 2
) -> str:
    """Fan out, ingest every share, and complete, over the given roots.

    Parameters:
        tmp_path: Directory the index is written into.
        roots: The results roots.
        logger: Logger every stage reports through.
        share_size: How many files one task is handed.

    Returns:
        The index URL.
    """
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, roots, logger=logger, share_size=share_size)
    results = run_shares(url, tasks, logger=logger)
    complete(url, roots, results, logger=logger)
    return url
