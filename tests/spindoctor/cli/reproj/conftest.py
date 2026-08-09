"""The one fixture tree both pointing sources are measured against.

The point of these tests is equivalence, so the documents are written once and
both sources are pointed at the same ones: the file-backed source reads the
tree, and the index-backed source reads an index built from that same tree by
the real ingest.  A factory that built two trees, or an index written by hand,
could agree with itself while the two paths a program takes disagreed.

The recorded attitudes are real: they are what the pointing computation produced
for a Cassini NAC frame at its own navigated offset.  A hand-written rotation of
zeros and ones round-trips through JSON whatever the storage does with it, and
the reader's flip gate holds the recovered rotation to 1e-9, so only a value
carrying a full float64 mantissa can show that the storage kept one.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pdslogger
import pytest
from filecache import FCPath
from sqlalchemy.engine import Engine
from tests.spindoctor.cli.stats.conftest import metadata_document, write_metadata
from tests.spindoctor.results_index.conftest import (
    postgres_schema,
    postgres_server_url,
    postgres_url,
)

from spindoctor.cli.reproj.pointing_source import (
    FilePointingSource,
    IndexPointingSource,
    PointingSource,
)
from spindoctor.cli.stats.ingest import ingest_metadata_files
from spindoctor.dataset.dataset import ImageFile
from spindoctor.results_index import normalize_root_url, open_index

# The pointing postgres tier runs against a schema of its own, exactly as the
# results-index tier does; re-exporting rather than restating keeps one
# definition of how that schema is created and dropped.
__all__ = ['postgres_schema', 'postgres_server_url', 'postgres_url']

CMATRIX_ORIGINAL = [
    0.963676611721357,
    0.24393954782302113,
    0.1087238935521779,
    -0.23632299640079324,
    0.5892234094236003,
    0.772636534962837,
    0.1244139437257574,
    -0.7702657144097279,
    0.6254693436224319,
]
"""The as-flown attitude recorded for a real Cassini NAC frame, row-major."""

CMATRIX = [
    0.9636758075215185,
    0.24394452671199518,
    0.10871985046444949,
    -0.23632717093513517,
    0.5892492544213016,
    0.7726155476313796,
    0.1244122432702933,
    -0.7702443664521031,
    0.6254959709488565,
]
"""The corrected attitude recorded for that frame at its navigated offset."""

MIDTIME_ET = 136576860.1724845
"""That frame's exposure midtime, which the reader's gate holds to 1e-6 s."""

TIMES: dict[str, Any] = {
    'start_et': 136576860.0424845,
    'stop_et': 136576860.30248448,
    'midtime_et': MIDTIME_ET,
    'exposure_s': 0.26,
    'sclk_start': '1/1461997416.044',
    'sclk_midtime': '1/1461997416.078',
    'sclk_stop': '1/1461997416.111',
}
"""The exposure epochs recorded beside those attitudes."""

POINTING: dict[str, Any] = {
    'cmatrix': CMATRIX,
    'cmatrix_original': CMATRIX_ORIGINAL,
    'camera_frame': 'CASSINI_ISS_NAC',
    'camera_frame_id': -82360,
    'ck_frame_id': -82000,
}
"""A complete recorded pointing block, in the shape its producer writes."""

FITTED_ROTATION_POINTING: dict[str, Any] = {
    'cmatrix_original': CMATRIX_ORIGINAL,
    'camera_frame': 'CASSINI_ISS_NAC',
    'camera_frame_id': -82360,
    'ck_frame_id': -82000,
}
"""What a result that fitted a camera rotation records: a baseline, no cmatrix."""

OFFSET = [5.6005, 1.0788]
"""The navigated offset those attitudes were computed from."""

VOLUME = 'COISS_2001/data/1461994336_1462054659'
"""Volume and range directory every stub of the fixture tree sits under."""

CMATRIX_STUB = f'{VOLUME}/N100_1_CALIB'
FITTED_STUB = f'{VOLUME}/N101_1_CALIB'
NO_POINTING_STUB = f'{VOLUME}/N102_1_CALIB'
FAILED_STUB = f'{VOLUME}/N103_1_CALIB'
NULL_OFFSET_STUB = f'{VOLUME}/N104_1_CALIB'
NO_MIDTIME_STUB = f'{VOLUME}/N105_1_CALIB'
MALFORMED_OFFSET_STUB = f'{VOLUME}/N106_1_CALIB'
NO_OFFSET_KEY_STUB = f'{VOLUME}/N107_1_CALIB'
NON_FINITE_OFFSET_STUB = f'{VOLUME}/N108_1_CALIB'
BOOLEAN_OFFSET_STUB = f'{VOLUME}/N109_1_CALIB'
NOT_A_ROTATION_STUB = f'{VOLUME}/N110_1_CALIB'
UNNAVIGATED_STUB = f'{VOLUME}/N999_1_CALIB'
"""Stubs of the fixture tree; the last is deliberately never written."""


def image_file(stub: str) -> ImageFile:
    """Build an image file naming the stub its record is looked up under.

    Parameters:
        stub: The results path stub.

    Returns:
        The image file.
    """
    name = stub.rsplit('/', 1)[-1]
    return ImageFile(
        image_file_url=FCPath(f'/holdings/{name}.IMG'),
        label_file_url=FCPath(f'/holdings/{name}.LBL'),
        results_path_stub=stub,
        index_file_row={},
    )


def document(stub: str, **overrides: Any) -> dict[str, Any]:
    """Build one navigation document, named for the image it records.

    Parameters:
        stub: The results path stub, whose basename names the image.
        overrides: Fields passed through to the document factory.

    Returns:
        The document.
    """
    name = stub.rsplit('/', 1)[-1]
    return metadata_document(image_name=f'{name}.IMG', **overrides)


def _reason_tree() -> dict[str, dict[str, Any]]:
    """Build one document per classification the two sources must agree about.

    Returns:
        Results path stub mapped to the document recorded there.
    """
    tree: dict[str, dict[str, Any]] = {
        CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET, times=TIMES, pointing=POINTING),
        FITTED_STUB: document(
            FITTED_STUB, offset=OFFSET, times=TIMES, pointing=FITTED_ROTATION_POINTING
        ),
        NO_POINTING_STUB: document(NO_POINTING_STUB, offset=OFFSET),
        FAILED_STUB: document(
            FAILED_STUB, status='error', status_error='missing_spice_data', offset=None
        ),
        NULL_OFFSET_STUB: document(NULL_OFFSET_STUB),
        # A pointing block whose exposure epochs carry no midtime: the gates
        # cannot run without one, so the record is malformed rather than bare.
        NO_MIDTIME_STUB: document(
            NO_MIDTIME_STUB,
            offset=OFFSET,
            times={key: value for key, value in TIMES.items() if key != 'midtime_et'},
            pointing=POINTING,
        ),
        MALFORMED_OFFSET_STUB: document(MALFORMED_OFFSET_STUB, offset=[1.0]),
        NO_OFFSET_KEY_STUB: document(NO_OFFSET_KEY_STUB),
        NON_FINITE_OFFSET_STUB: document(NON_FINITE_OFFSET_STUB, offset=[float('nan'), 2.0]),
        BOOLEAN_OFFSET_STUB: document(BOOLEAN_OFFSET_STUB, offset=[True, False]),
        # Nine finite numbers that are not a rotation: ingest stores them, and
        # the reader's validator is what refuses them, in both paths alike.
        NOT_A_ROTATION_STUB: document(
            NOT_A_ROTATION_STUB,
            offset=OFFSET,
            times=TIMES,
            pointing={**POINTING, 'cmatrix': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]},
        ),
    }
    # The two shapes of "no usable offset on a successful record" are made here
    # rather than by the factory, which writes the key only when it has a value
    # and so cannot express a key holding null.
    tree[NULL_OFFSET_STUB]['offset'] = None
    del tree[NO_OFFSET_KEY_STUB]['offset']
    return tree


REASON_TREE = _reason_tree()
"""The fixture tree, one document per classification under test."""


@pytest.fixture
def quiet_ingest_logger() -> pdslogger.PdsLogger:
    """Return a logger that keeps ingest chatter out of the test output.

    Returns:
        A logger of its own, named uniquely for the life of the process, so
        raising its level cannot affect another test.
    """
    logger = pdslogger.PdsLogger(f'pointing_source_test_{uuid.uuid4().hex}')
    logger.set_level('ERROR')
    return logger


def build_tree(root: Path, documents: dict[str, dict[str, Any]]) -> None:
    """Write a results tree of navigation documents.

    Parameters:
        root: The results root to write under.
        documents: Results path stub mapped to the document recorded there.
    """
    for stub, content in documents.items():
        write_metadata(root, stub, content)


def index_for(roots: list[Path], database: Path, *, logger: pdslogger.PdsLogger) -> Engine:
    """Ingest one or more results trees and return the open index.

    Parameters:
        roots: The results roots to walk.
        database: Path of the SQLite file to create.
        logger: Logger the ingest reports through.

    Returns:
        The open index, which the caller disposes of.
    """
    engine = open_index(f'sqlite:///{database.as_posix()}', create=True)
    ingest_metadata_files(engine, [root.as_posix() for root in roots], logger=logger)
    return engine


@pytest.fixture
def sources(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> Iterator[dict[str, PointingSource]]:
    """Yield both sources over one results tree covering every reachable reason.

    Parameters:
        tmp_path: Directory the tree and the index are written under.
        quiet_ingest_logger: Logger the ingest reports through.

    Yields:
        The file-backed source under ``'file'`` and the index-backed one under
        ``'index'``, both answering for the same root.
    """
    root = tmp_path / 'nav'
    build_tree(root, REASON_TREE)
    engine = index_for([root], tmp_path / 'index.sqlite3', logger=quiet_ingest_logger)
    try:
        yield {
            'file': FilePointingSource(FCPath(root)),
            'index': IndexPointingSource(engine, normalize_root_url(root)),
        }
    finally:
        engine.dispose()
