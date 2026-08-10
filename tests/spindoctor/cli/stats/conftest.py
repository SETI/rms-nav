"""Shared document factories and index helpers for the statistics tests.

The factories build metadata documents in the shape ``navigate_image_files``
writes, so a test that cares about one field does not have to restate the
surrounding document.  The index helpers build a real index over a real tree,
because the ingest guarantees that matter -- what is keyed by what, what is
read a second time -- are properties of the walk and the writer together.

The cloud-task helpers run the same pass in its three separate stages -- divide
a root into shares, ingest a share, add the shares up -- so that a test asserting
on one of them does not have to restate the other two.
"""

import json
import os
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from tests.spindoctor.results_index.conftest import (
    postgres_schema,
    postgres_server_url,
    postgres_url,
)

from spindoctor.cli.stats.ingest import (
    IngestCounts,
    TaskCompletion,
    TaskResult,
    complete_ingest_tasks,
    fan_out_ingest_tasks,
    ingest_metadata_files,
    ingest_task_share,
)
from spindoctor.cli.stats.report import build_report
from spindoctor.results_index import INGEST_RUNS, normalize_root_url, open_index

# The statistics postgres tier runs against a schema of its own, exactly as the
# results-index tier does; re-exporting rather than restating keeps one
# definition of how that schema is created and dropped.
__all__ = ['postgres_schema', 'postgres_server_url', 'postgres_url']

DATA_DIR = Path(__file__).resolve().parent / 'data'
"""Directory holding the fixture results tree and the frozen report output."""

RESULTS_TREE = DATA_DIR / 'results_tree'
"""Fixture results tree the report regression is measured over."""

GOLDEN_DIR = DATA_DIR / 'golden'
"""Report and CSV output this tree produced before the move onto the index."""


def index_url(path: Path) -> str:
    """Return the SQLite URL naming an index file.

    Parameters:
        path: The database file's path.

    Returns:
        The URL.
    """
    return f'sqlite:///{path.as_posix()}'


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


def technique(
    name: str,
    offset: tuple[float, float],
    *,
    confidence: float = 0.7,
    spurious: bool = False,
    at_edge: bool = False,
) -> dict[str, Any]:
    """Build one ``per_technique`` entry.

    Parameters:
        name: Technique class name.
        offset: The technique's ``(dv, du)`` estimate.
        confidence: The technique's calibrated confidence.
        spurious: Whether the technique flagged its own result as spurious.
        at_edge: Whether the fit landed at the edge of its search space.

    Returns:
        The entry.
    """
    return {
        'technique_name': name,
        'feature_ids': [f'{name.lower()}:IAPETUS'],
        'offset_px': list(offset),
        'covariance_px2': [[0.01, 0.0], [0.0, 0.01]],
        'confidence': confidence,
        'spurious': spurious,
        'at_edge': at_edge,
        'diagnostics': {'a': 1},
    }


def metadata_document(
    *,
    image_name: str = 'N1454725799_1_CALIB.IMG',
    instrument: str | None = 'coiss',
    camera: str | None = 'NAC',
    status: str = 'success',
    status_reason: str | None = 'ok',
    status_error: str | None = None,
    offset: list[float] | None = None,
    confidence: float = 0.8,
    confidence_rank: str = 'high',
    per_technique: list[dict[str, Any]] | None = None,
    excluded: list[str] | None = None,
    image_et: float | None = 0.0,
    image_shape: list[int] | None = None,
    elapsed_s: float | None = 3.25,
) -> dict[str, Any]:
    """Build a metadata document in the ``navigate_image_files`` shape.

    Parameters:
        image_name: Recorded ``observation.image_name``.
        instrument: Recorded ``observation.instrument``; None omits the field,
            which models a file that is not a navigation document.
        camera: Recorded ``observation.camera``; None omits the field, as
            happens for an image that never loaded.
        status: Top-level status.
        status_reason: The navigator's explanation; None omits the field.
        status_error: The fatal error; None omits the field.
        offset: The authoritative top-level offset; None omits it, and a
            successful document defaults to one.
        confidence: Top-level confidence.
        confidence_rank: Recorded confidence tier.
        per_technique: Technique entries, from :func:`technique`.
        excluded: Technique names the ensemble excluded.
        image_et: Recorded provenance epoch.
        image_shape: Recorded ``observation.image_shape``; None omits it.
        elapsed_s: Recorded run time; None omits the whole timing section.

    Returns:
        The document.
    """
    if offset is None and status == 'success':
        offset = [1.5, -2.5]
    observation: dict[str, Any] = {
        'image_path': f'/holdings/{image_name}',
        'image_name': image_name,
    }
    if instrument is not None:
        observation['instrument'] = instrument
    if camera is not None:
        observation['camera'] = camera
    if image_shape is not None:
        observation['image_shape'] = image_shape
    navigation_result: dict[str, Any] = {
        'status': status,
        'offset_px': offset,
        'sigma_px': [0.1, 0.2] if offset else None,
        'confidence': confidence,
        'confidence_rank': confidence_rank,
        'covariance_px2': [[0.01, 0.0], [0.0, 0.04]] if offset else None,
        'techniques_used': sorted({t['technique_name'] for t in per_technique or []}),
        'excluded_from_consensus': excluded or [],
        'per_technique': per_technique or [],
        'feature_inventory': [
            {
                'feature_id': 'body_disc:IAPETUS',
                'feature_type': 'BODY_DISC',
                'source_model': 'body:IAPETUS',
                'gated': False,
            },
            {
                'feature_id': 'star:UCAC4:10230452',
                'feature_type': 'STAR',
                'source_model': 'stars',
                'gated': True,
            },
        ],
        'image_classifier': {'class': 'clean', 'noise_sigma': 1.0, 'max_dn': 255.0},
        'provenance': {
            'spindoctor_git_sha': 'abc1234',
            'config_hash': 'deadbeef',
            'image_et': image_et,
            'pipeline_run_iso8601': '2026-07-11T00:00:00Z',
        },
    }
    if status_reason is not None:
        navigation_result['status_reason'] = status_reason
    document: dict[str, Any] = {
        'status': status,
        'observation': observation,
        'navigation_result': navigation_result,
        'confidence': confidence,
    }
    if offset is not None:
        document['offset'] = list(offset)
    if status_error is not None:
        document['status_error'] = status_error
    if elapsed_s is not None:
        document['timing'] = {
            'start_iso8601': '2026-07-11T00:00:00Z',
            'end_iso8601': '2026-07-11T00:00:03.250000Z',
            'elapsed_s': elapsed_s,
        }
    return document


def write_metadata(root: Path, stub: str, document: dict[str, Any]) -> Path:
    """Write one metadata document into a results tree.

    Parameters:
        root: The results root.
        stub: The document's results path stub under that root.
        document: The document to write.

    Returns:
        The path written.
    """
    path = root / f'{stub}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding='utf-8')
    return path


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


def write_summary_png(root: Path, stub: str) -> Path:
    """Write a stand-in summary PNG beside a document.

    A results tree holds one of these beside every navigated image, so it is
    the file a walk of a real root meets most often after the documents
    themselves.  Only its name matters here: nothing opens one.

    Parameters:
        root: The results root.
        stub: The image's results path stub.

    Returns:
        The path written.
    """
    path = root / f'{stub}_summary.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\x89PNG\r\n\x1a\n')
    return path


REFUSED_DOCUMENT = '{"edges": []}'
"""A document that reads as JSON and is not a navigation result of any schema."""


def write_refusal(root: Path, stub: str) -> Path:
    """Write a document under a root that no pass can turn into an image row.

    Parameters:
        root: The results root to write under.
        stub: The document's results path stub under that root.

    Returns:
        The path written.
    """
    path = root / f'{stub}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(REFUSED_DOCUMENT, encoding='utf-8')
    return path


REFUSAL_REPORT_LEAD = 'Documents under '
"""How the standing-refusal line opens, for picking it out of a log."""


def refusal_report(root: Path | str, refused: int) -> str:
    """Return the line a pass writes about the refusals its root's index holds.

    Written once here because three modules read the same line: it is what an
    operator is told about the gap between an error filter answered from the
    index and the same filter answered from the tree, and the number in it is
    the root's own standing total of the refusals that make that gap, rather
    than what any one pass refused.

    Parameters:
        root: The results root the pass covered.
        refused: How many such refusals the index holds under it.

    Returns:
        The line, spelled as the pass writes it.
    """
    return (
        f'{REFUSAL_REPORT_LEAD}{normalize_root_url(root)} an error filter reads from the '
        f'results tree and not from this index: {refused}, whichever pass recorded them. '
        f'Each is a JSON object the ingest refused, so this index records no status for it '
        f'and no error filter answered here selects its image. The count is the whole root, '
        f'so it bounds rather than measures how short a selection over some of its volumes '
        f'comes.'
    )


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


def ingest_tree(
    url: str, roots: list[Path], *, logger: pdslogger.PdsLogger, force: bool = False
) -> IngestCounts:
    """Create an index and ingest one or more results trees into it.

    Parameters:
        url: The index URL to create or add to.
        roots: The results roots to walk.
        logger: Logger the ingest reports through.
        force: Whether to re-read every document.

    Returns:
        What the pass did.
    """
    engine = open_index(url, create=True)
    try:
        return ingest_metadata_files(
            engine, [root.as_posix() for root in roots], force=force, logger=logger
        )
    finally:
        engine.dispose()


@pytest.fixture
def indexed_tree(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> Iterator[sqlalchemy.Connection]:
    """Yield a connection to an index built from the frozen fixture tree.

    Parameters:
        tmp_path: Directory the index file is written into.
        quiet_logger: Logger the ingest reports through.

    Yields:
        An open connection to the index.
    """
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [RESULTS_TREE], logger=quiet_logger)
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            yield connection
    finally:
        engine.dispose()


def report_from_tree(url: str, out: Path, *, logger: pdslogger.PdsLogger, **options: Any) -> Path:
    """Ingest the fixture tree into an index and write one report from it.

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
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            build_report(connection, out, **options)
    finally:
        engine.dispose()
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
