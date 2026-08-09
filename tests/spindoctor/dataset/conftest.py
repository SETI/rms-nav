"""The fixture tree the results-filter tests read, and the helpers that build it.

Every filter has two implementations behind it -- one that walks the results
tree and one that queries a results index -- and the point of the second is that
it answers what the first answers.  So one tree drives every module here: it is
written to disk, ingested into an index, and each filter is then asked of the
tree and of the index and held to one stated answer.  Stating the answer rather
than only comparing the two matters, because two implementations that are wrong
in the same way agree.

The tree covers what the filters distinguish -- a success, a run that finished
without one, three shapes of fatal error, a document that is not valid JSON, a
document that is valid JSON and not an object, and an image with no result files
at all -- and it is ingested alongside a second root holding a fatal SPICE error
and a summary PNG for every one of those stubs.  A query that filtered on the
stub without its root would answer with that second root's rows, and no
single-root fixture can see it happen.

The run rows are stocked the same way, because the run table is keyed by root as
well.  The second root is always passed over last, so its run is the newest in
the index, and it always records a count and a finish time the root under test
does not.

The dataset object the enumeration tests ask their questions of is here too,
since more than one module asks them of the same one.
"""

import uuid
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from filecache import FCPath
from tests.spindoctor.cli.stats.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    write_metadata,
    write_summary_png,
)

from spindoctor.dataset.dataset import ImageFile
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISS
from spindoctor.dataset.results_filter import RESULTS_FILTER_BATCH_SIZE, ResultsFilter
from spindoctor.results_index import (
    INGEST_RUNS,
    SPICE_STATUS_ERROR,
    normalize_root_url,
    open_index,
)

VOLUMES = ['COISS_2001', 'COISS_2002']
"""The volumes the enumeration selected."""

SUCCESS_WITH_PNG = 'COISS_2001/data/a/N1000000001_1_CALIB'
SUCCESS_NO_PNG = 'COISS_2001/data/a/N1000000002_1_CALIB'
FAILURE = 'COISS_2001/data/a/N1000000003_1_CALIB'
SPICE_ERROR = 'COISS_2001/data/b/N1000000004_1_CALIB'
NONSPICE_ERROR = 'COISS_2001/data/b/N1000000005_1_CALIB'
ERROR_WITHOUT_STATUS_ERROR = 'COISS_2001/data/b/N1000000006_1_CALIB'
MALFORMED = 'COISS_2001/data/b/N1000000007_1_CALIB'
NOT_AN_OBJECT = 'COISS_2001/data/b/N1000000008_1_CALIB'
NO_RESULT = 'COISS_2001/data/c/N1000000009_1_CALIB'
OTHER_VOLUME = 'COISS_2002/data/a/N1000000010_1_CALIB'

CANDIDATES = (
    SUCCESS_WITH_PNG,
    SUCCESS_NO_PNG,
    FAILURE,
    SPICE_ERROR,
    NONSPICE_ERROR,
    ERROR_WITHOUT_STATUS_ERROR,
    MALFORMED,
    NOT_AN_OBJECT,
    NO_RESULT,
    OTHER_VOLUME,
)
"""The images offered to the filter, in the order an enumeration yields them."""

WITH_A_DOCUMENT = tuple(stub for stub in CANDIDATES if stub != NO_RESULT)
"""Every candidate whose metadata file exists, however well it reads."""

WITH_A_PNG = (SUCCESS_WITH_PNG, FAILURE, NONSPICE_ERROR, MALFORMED, OTHER_VOLUME)
"""Every candidate a summary PNG was written for, in enumeration order.

One of them is a document the ingest refuses, because a PNG is found beside a
file rather than read out of it: the walk finds ``X_summary.png`` whatever
``X_metadata.json`` turned out to contain.
"""

WITHOUT_A_PNG = tuple(stub for stub in CANDIDATES if stub not in WITH_A_PNG)
"""Every candidate no summary PNG was written for."""

FATAL_ERRORS = (SPICE_ERROR, NONSPICE_ERROR, ERROR_WITHOUT_STATUS_ERROR)
"""Every candidate whose document records a fatal error."""

OTHER_ROOT_NAME = 'other-results'
"""Directory name of the second root every two-root index is built under.

Published rather than repeated where a test stocks that root itself: a stamp
written to a name nothing was ingested under updates no row, and a test whose
assertion is that something is absent then passes for the wrong reason.
"""


@pytest.fixture
def ds() -> DataSetPDS3CassiniISS:
    """Return a Cassini ISS dataset whose holdings root is never read.

    Every test that uses it serves the index rows itself, so the root only has
    to be a path the dataset accepts.

    Returns:
        The dataset.
    """
    return DataSetPDS3CassiniISS('/fake/holdings')


def null_logger() -> pdslogger.PdsLogger:
    """Return a logger that keeps the ingest and the filter quiet.

    Returns:
        A logger discarding everything written to it.
    """
    return pdslogger.NullLogger()


def write_tree(root: Path) -> None:
    """Write the fixture results tree under one root.

    Parameters:
        root: The results root to write into.
    """
    write_metadata(root, SUCCESS_WITH_PNG, metadata_document(image_name='N1000000001_1.IMG'))
    write_summary_png(root, SUCCESS_WITH_PNG)
    write_metadata(root, SUCCESS_NO_PNG, metadata_document(image_name='N1000000002_1.IMG'))
    write_metadata(
        root,
        FAILURE,
        metadata_document(image_name='N1000000003_1.IMG', status='failure', offset=None),
    )
    write_summary_png(root, FAILURE)
    write_metadata(
        root,
        SPICE_ERROR,
        metadata_document(
            image_name='N1000000004_1.IMG',
            status='error',
            status_error=SPICE_STATUS_ERROR,
            offset=None,
        ),
    )
    write_metadata(
        root,
        NONSPICE_ERROR,
        metadata_document(
            image_name='N1000000005_1.IMG',
            status='error',
            status_error='unhandled_exception',
            offset=None,
        ),
    )
    write_summary_png(root, NONSPICE_ERROR)
    write_metadata(
        root,
        ERROR_WITHOUT_STATUS_ERROR,
        metadata_document(image_name='N1000000006_1.IMG', status='error', offset=None),
    )
    write_bytes(root, MALFORMED, b'{"status": "error"')
    # A summary PNG sits beside a document the ingest refuses. This is the
    # ordinary shape of a results root written by an older metadata schema --
    # every image in one has a summary beside a document the ingest will not
    # read -- so both PNG filters have to answer for it as the walk does.
    write_summary_png(root, MALFORMED)
    write_bytes(root, NOT_AN_OBJECT, b'[1, 2, 3]')
    write_metadata(root, OTHER_VOLUME, metadata_document(image_name='N1000000010_1.IMG'))
    write_summary_png(root, OTHER_VOLUME)


def write_bytes(root: Path, stub: str, content: bytes) -> None:
    """Write a metadata file that is not a readable navigation document.

    Parameters:
        root: The results root to write into.
        stub: The image's results path stub.
        content: Exactly what the file holds.
    """
    path = root / f'{stub}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_decoy_tree(root: Path) -> None:
    """Write a second root holding an answer-changing row for every stub.

    Every candidate gets a fatal SPICE error and a summary PNG here, and the one
    candidate the tree under test has no result files for gets a document the
    ingest refuses with a summary PNG beside it, so that a refusal read without
    its root changes both the presence answer and the PNG answer.  Any filter
    that read this root's rows for the other root's stubs therefore answers
    differently, which is what makes the composite key testable at all.

    Parameters:
        root: The second results root to write into.
    """
    for stub in CANDIDATES:
        if stub == NO_RESULT:
            write_bytes(root, stub, b'{"status": "error"')
            write_summary_png(root, stub)
            continue
        write_metadata(
            root,
            stub,
            metadata_document(
                image_name=f'{Path(stub).name}.IMG',
                status='error',
                status_error=SPICE_STATUS_ERROR,
                offset=None,
            ),
        )
        write_summary_png(root, stub)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Write the fixture results tree and return its root.

    Parameters:
        tmp_path: Directory the roots are written under.

    Returns:
        The results root under test.
    """
    root = tmp_path / 'results'
    write_tree(root)
    return root


@pytest.fixture
def indexed(tree: Path, tmp_path: Path) -> str:
    """Ingest the fixture tree and a second root, and return the index URL.

    Parameters:
        tree: The results root under test.
        tmp_path: Directory the index and the second root live under.

    Returns:
        The connection URL of the index.
    """
    decoy = tmp_path / OTHER_ROOT_NAME
    write_decoy_tree(decoy)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [tree, decoy], logger=null_logger())
    return url


def candidate_files(root: Path) -> list[ImageFile]:
    """Build the images an enumeration would offer the filter.

    Parameters:
        root: The results root, only so the stand-in URLs point somewhere.

    Returns:
        One :class:`ImageFile` per candidate, in enumeration order.
    """
    return [
        ImageFile(
            image_file_url=FCPath(root / f'{stub}.IMG'),
            label_file_url=FCPath(root / f'{stub}.LBL'),
            results_path_stub=stub,
        )
        for stub in CANDIDATES
    ]


def select_from(results_filter: ResultsFilter, images: list[ImageFile]) -> list[str]:
    """Run images through the filter exactly as the enumeration does.

    Parameters:
        results_filter: The filter under test.
        images: The candidates, in enumeration order.

    Returns:
        The stubs that passed, in enumeration order.
    """
    kept = [image for image in images if results_filter.passes_presence(image.results_path_stub)]
    selected: list[ImageFile] = []
    for start in range(0, len(kept), RESULTS_FILTER_BATCH_SIZE):
        batch = kept[start : start + RESULTS_FILTER_BATCH_SIZE]
        selected.extend(results_filter.filter_batch(batch))
    return [image.results_path_stub for image in selected]


def selection_of(root: Path, flags: dict[str, bool], *, results_db_url: str | None) -> list[str]:
    """Answer one filter combination over the fixture tree.

    Parameters:
        root: The results root under test.
        flags: The selection flags to apply.
        results_db_url: The index to answer from, or None to read the tree.

    Returns:
        The stubs that passed, in enumeration order.
    """
    results_filter = ResultsFilter(
        VOLUMES,
        str(root),
        logger=null_logger(),
        results_db_url=results_db_url,
        **flags,
    )
    return select_from(results_filter, candidate_files(root))


def one_image_tree(tmp_path: Path) -> tuple[Path, list[ImageFile]]:
    """Write a results root holding one navigated image.

    Parameters:
        tmp_path: Directory the root is written under.

    Returns:
        The root, and the one candidate image ready to filter.
    """
    root = tmp_path / 'results'
    write_metadata(root, SUCCESS_NO_PNG, metadata_document(image_name='N1000000002_1.IMG'))
    return root, [
        ImageFile(
            image_file_url=FCPath(root / 'x.IMG'),
            label_file_url=FCPath(root / 'x.LBL'),
            results_path_stub=SUCCESS_NO_PNG,
        )
    ]


OTHER_ROOT_MISSED = 5
"""How many directories the second root's pass did not list.

The second root is ingested last, so its run is the newest in the index: a count
read from the newest run of the table rather than from the newest run over the
root being enumerated is this one, and reports a gap the enumerated root does
not have -- or, with the two the other way round, reports none where it does.
"""


def stamp_run(url: str, root: Path, **values: Any) -> None:
    """Record something about the pass over one root, and about no other root.

    A run row is one root's, exactly as an image row is, so an update without a
    root names every pass in the index and makes the two roots indistinguishable
    in the column it writes.

    Parameters:
        url: The index to write into.
        root: The results root whose newest pass is being described.
        values: Column values to record on it.
    """
    engine = open_index(url)
    try:
        with engine.begin() as connection:
            connection.execute(
                INGEST_RUNS.update()
                .where(INGEST_RUNS.c.root_url == normalize_root_url(root))
                .values(**values)
            )
    finally:
        engine.dispose()


def index_of_two_roots(tmp_path: Path, root: Path, *, missed: int) -> str:
    """Ingest a tree, and a second tree whose pass missed directories after it.

    The count is written rather than provoked, so that the test is about what a
    consumer does with a count and not about what makes a walk record one.  The
    second root carries a count of its own for the same reason the fixture tree
    is ingested beside a decoy: a query answering from the wrong root's run row
    passes every single-root assertion.

    Parameters:
        tmp_path: Directory the index file is written into.
        root: The results root to ingest and describe.
        missed: How many directories the pass over that root is recorded as
            having missed.

    Returns:
        The connection URL of the index.
    """
    decoy = tmp_path / OTHER_ROOT_NAME
    write_metadata(decoy, SUCCESS_NO_PNG, metadata_document(image_name='N1000000002_1.IMG'))
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root, decoy], logger=null_logger())
    stamp_run(url, root, directories_missed=missed)
    stamp_run(url, decoy, directories_missed=OTHER_ROOT_MISSED)
    return url


def reporting_logger() -> pdslogger.PdsLogger:
    """Return a logger whose output a test reads back.

    Returns:
        A logger of its own, so raising its level cannot affect another test.
    """
    return pdslogger.PdsLogger(f'results_filter_test_{uuid.uuid4().hex}')


UNLISTABLE = 'COISS_2001/data/c'
"""A directory under the root that one pass cannot list."""


def refusing_to_list(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    """Make one directory refuse to be listed, as a permission or a share can.

    Provoked rather than written into the run row, because what is under test is
    what the prune does when the walk comes back incomplete, and that is decided
    by the walk and not by the count it records.

    Parameters:
        monkeypatch: Fixture the refusal is installed through.
        directory: The directory that will refuse.
    """
    listing = FCPath.iterdir_metadata
    refused = directory.as_posix()

    def refuse(self: FCPath) -> Any:
        if self.as_posix() == refused:
            raise OSError('this directory may not be listed')
        return listing(self)

    monkeypatch.setattr(FCPath, 'iterdir_metadata', refuse)


def index_without_a_table(tmp_path: Path, root: Path, table: str) -> str:
    """Ingest a tree into an index and then take one of its tables away.

    This is the shape of an index whose account was granted the rows it reports
    on and not the bookkeeping beside them, and of one restored from a partial
    dump.  A connection lost between the open and the query fails the same way
    and cannot be provoked as cheaply.

    Parameters:
        tmp_path: Directory the index file is written into.
        root: The results root to ingest.
        table: Name of the table to drop.

    Returns:
        The connection URL of the index.
    """
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    engine = open_index(url)
    try:
        with engine.begin() as connection:
            connection.execute(sqlalchemy.text(f'DROP TABLE {table}'))
    finally:
        engine.dispose()
    return url


def reported_line(out: str) -> str:
    """Return the line reporting what the index holds.

    Parameters:
        out: Everything the filter wrote.

    Returns:
        The one line naming the counts and the age of the answer.
    """
    return next(line for line in out.splitlines() if 'Results index holds' in line)
