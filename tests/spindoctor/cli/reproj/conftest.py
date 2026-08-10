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

CAMERA_FRAME_ONLY_POINTING: dict[str, Any] = {'camera_frame': 'CASSINI_ISS_NAC'}
"""A pointing block holding only the one field the index has no column for.

No navigation writes it -- a block always carries the baseline and both frame
identities -- so it is here to pin what the two paths do with a block that
leaves no trace in a row.
"""

FLOAT_FRAME_ID_POINTING: dict[str, Any] = {
    'camera_frame_id': -82360.0,
    'ck_frame_id': -82000.0,
}
"""A block whose only columned fields are identities written as floats.

The columns are integer ones, so neither value survives ingest and the row
records no block, exactly as for a block holding nothing columned at all.
"""


def nested(flat: list[float]) -> list[list[float]]:
    """Rewrite nine row-major values as the 3x3 nesting of them.

    Parameters:
        flat: The nine values, row-major.

    Returns:
        Three rows of three.
    """
    return [flat[0:3], flat[3:6], flat[6:9]]


def one_element_rows(flat: list[float]) -> list[list[float]]:
    """Rewrite nine row-major values as nine rows of one.

    A shape no producer writes and every reader that assembles an array
    accepts, because nine rows of one reshape into 3x3 exactly as nine scalars
    do.  It is here because a store that judged the entries instead of
    assembling them held nothing for it.

    Parameters:
        flat: The nine values, row-major.

    Returns:
        Nine rows of one.
    """
    return [[value] for value in flat]


TOO_BIG_FOR_A_FLOAT = 10**400
"""A JSON integer literal no float can hold.

JSON bounds no integer, so a document can carry one; ``float()`` of it raises
rather than overflowing to an infinity, which is a reader that raises where its
contract says it answers.
"""


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
NAN_MIDTIME_STUB = f'{VOLUME}/N111_1_CALIB'
NESTED_CMATRIX_STUB = f'{VOLUME}/N112_1_CALIB'
NO_TOP_LEVEL_STATUS_STUB = f'{VOLUME}/N113_1_CALIB'
CAMERA_FRAME_ONLY_STUB = f'{VOLUME}/N114_1_CALIB'
NO_STATUS_ERROR_STUB = f'{VOLUME}/N115_1_CALIB'
SUCCESS_NO_OFFSET_KEY_STUB = f'{VOLUME}/N116_1_CALIB'
OVER_LONG_OFFSET_STUB = f'{VOLUME}/N117_1_CALIB'
NUMERIC_STRING_OFFSET_STUB = f'{VOLUME}/N118_1_CALIB'
NESTED_ORIGINAL_STUB = f'{VOLUME}/N119_1_CALIB'
NESTED_NOT_A_ROTATION_STUB = f'{VOLUME}/N120_1_CALIB'
RAGGED_CMATRIX_STUB = f'{VOLUME}/N121_1_CALIB'
UNSTORABLE_CMATRIX_ALONE_STUB = f'{VOLUME}/N122_1_CALIB'
FLOAT_FRAME_ID_STUB = f'{VOLUME}/N123_1_CALIB'
LITERAL_UNKNOWN_STATUS_STUB = f'{VOLUME}/N124_1_CALIB'
NULL_STATUS_ERROR_STUB = f'{VOLUME}/N125_1_CALIB'
ONE_ELEMENT_ROWS_CMATRIX_STUB = f'{VOLUME}/N126_1_CALIB'
ONE_ELEMENT_ROWS_ORIGINAL_STUB = f'{VOLUME}/N127_1_CALIB'
RAGGED_NINE_CMATRIX_STUB = f'{VOLUME}/N128_1_CALIB'
HUGE_INT_IN_CMATRIX_STUB = f'{VOLUME}/N129_1_CALIB'
HUGE_INT_MIDTIME_STUB = f'{VOLUME}/N130_1_CALIB'
HUGE_INT_OFFSET_STUB = f'{VOLUME}/N131_1_CALIB'
LITERAL_UNKNOWN_ERROR_STUB = f'{VOLUME}/N132_1_CALIB'
REFUSED_DOCUMENT_STUB = f'{VOLUME}/N133_1_CALIB'
ZERO_OFFSET_STUB = f'{VOLUME}/N134_1_CALIB'
ZERO_EPOCH_STUB = f'{VOLUME}/N135_1_CALIB'
UNNAVIGATED_STUB = f'{VOLUME}/N999_1_CALIB'
"""Stubs of the fixture tree; the last is deliberately never written."""

ZERO_OFFSET = [0.0, 0.0]
"""A navigated offset of exactly no pixels.

A recorded value that is present and false at once, which is what separates
asking whether a column holds a value from asking whether the value is true.
The pair is applied like any other, so a rebuild that read it as no pair at all
would build an offset-corrected product through the document and an uncorrected
one through the row.
"""

ZERO_TIMES: dict[str, Any] = {
    'start_et': 0.0,
    'stop_et': 0.0,
    'midtime_et': 0.0,
    'exposure_s': 0.0,
    'sclk_start': '1/0000000000.000',
    'sclk_midtime': '1/0000000000.000',
    'sclk_stop': '1/0000000000.000',
}
"""Exposure epochs at the J2000 epoch itself, every number among them a zero.

``midtime_et`` is the one the pointing ladder cannot run without, so a rebuild
that dropped a present zero would report a clean corrected attitude as a
malformed pointing block.
"""

ZERO_FRAME_ID_POINTING: dict[str, Any] = {**POINTING, 'camera_frame_id': 0, 'ck_frame_id': 0}
"""A complete pointing block whose two frame identities are recorded zeros."""

REFUSED_DOCUMENT_REASON = 'no observation.instrument'
"""Why the ingest refuses the one document in the tree it cannot read.

Named here because the refusal a consumer reports has to carry it: an operator
told only that the index cannot answer for an image has nothing to fix.
"""


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


def reason_tree() -> dict[str, dict[str, Any]]:
    """Build one document per classification the two sources must agree about.

    A fresh tree of fresh documents per call, rather than one shared mapping:
    every document here is mutable, and a test that edited a nested one would
    otherwise change what every later test in the run was given.

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
        # A midtime that is a number and is not one: NaN defeats every
        # comparison, so a guard written as one does not hold it, and the gate
        # it would reach ties a recorded attitude to its observation.
        NAN_MIDTIME_STUB: document(
            NAN_MIDTIME_STUB,
            offset=OFFSET,
            times={**TIMES, 'midtime_et': float('nan')},
            pointing=POINTING,
        ),
        # A rotation written as a 3x3 nesting rather than as the nine row-major
        # floats its producer writes.  Both shapes are read by the one function
        # ingest stores rotations through, so the nesting reaches the index as
        # the nine values it denotes and both paths apply the same attitude.
        NESTED_CMATRIX_STUB: document(
            NESTED_CMATRIX_STUB,
            offset=OFFSET,
            times=TIMES,
            pointing={**POINTING, 'cmatrix': nested(CMATRIX)},
        ),
        # The baseline written the same way.  It is gated against the
        # observation exactly as the corrected attitude is, so a store that
        # read one shape for one matrix and both for the other would refuse
        # through an index what it applies through a document.
        NESTED_ORIGINAL_STUB: document(
            NESTED_ORIGINAL_STUB,
            offset=OFFSET,
            times=TIMES,
            pointing={**POINTING, 'cmatrix_original': nested(CMATRIX_ORIGINAL)},
        ),
        # A nesting of nine finite numbers that is not a rotation: stored like
        # any other nine finite numbers, and refused by the validator both
        # paths apply to what was stored.
        NESTED_NOT_A_ROTATION_STUB: document(
            NESTED_NOT_A_ROTATION_STUB,
            offset=OFFSET,
            times=TIMES,
            pointing={**POINTING, 'cmatrix': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]},
        ),
        # Three rows that are not three rows of three.  Read as an array this
        # is not a matrix at all, and the shape reader refuses it before numpy
        # is asked to make one of it.
        RAGGED_CMATRIX_STUB: document(
            RAGGED_CMATRIX_STUB,
            offset=OFFSET,
            times=TIMES,
            pointing={**POINTING, 'cmatrix': [[1.0, 2.0, 3.0], [4.0, 5.0], [6.0, 7.0, 8.0]]},
        ),
        # A corrected attitude and a baseline neither storage can hold, in a
        # block with no frame identities either, so the row records no block.
        UNSTORABLE_CMATRIX_ALONE_STUB: document(
            UNSTORABLE_CMATRIX_ALONE_STUB,
            offset=OFFSET,
            times=TIMES,
            pointing={'cmatrix': 'not a matrix', 'cmatrix_original': 'not a matrix'},
        ),
        FLOAT_FRAME_ID_STUB: document(
            FLOAT_FRAME_ID_STUB, offset=OFFSET, times=TIMES, pointing=FLOAT_FRAME_ID_POINTING
        ),
        # An offset of three numbers where the reader wants two.  Neither
        # storage may take the first two of them: a pointing built from part of
        # a recorded value is a pointing nobody recorded.
        OVER_LONG_OFFSET_STUB: document(OVER_LONG_OFFSET_STUB, offset=[*OFFSET, 9.9]),
        # An offset written as two numeric strings, which the reader converts
        # and applies, so the index has to store what it converts them to.
        NUMERIC_STRING_OFFSET_STUB: document(NUMERIC_STRING_OFFSET_STUB, offset=OFFSET),
        # A document naming, as its own outcome, the word a document naming
        # none is recorded as.  The two are one row, so both are reported as
        # that word rather than one of them as nothing.
        LITERAL_UNKNOWN_STATUS_STUB: document(
            LITERAL_UNKNOWN_STATUS_STUB, status='unknown', offset=None
        ),
        # An unsuccessful outcome whose error field is present and null, which
        # is a field naming no error just as an absent one is.
        NULL_STATUS_ERROR_STUB: document(NULL_STATUS_ERROR_STUB, status='failed', offset=None),
        # A document naming no outcome of its own, beside a nested copy that
        # names one.  The ladder's first question is the top-level field, so a
        # column standing the nested one in for it would apply a corrected
        # attitude to a record the same document supplies no pointing for.
        NO_TOP_LEVEL_STATUS_STUB: document(
            NO_TOP_LEVEL_STATUS_STUB, offset=OFFSET, times=TIMES, pointing=POINTING
        ),
        CAMERA_FRAME_ONLY_STUB: document(
            CAMERA_FRAME_ONLY_STUB,
            offset=OFFSET,
            times=TIMES,
            pointing=CAMERA_FRAME_ONLY_POINTING,
        ),
        # An unsuccessful outcome naming no error, which is every failed and
        # conflicted navigation: the field is written only by a document
        # recording an image that would not load.
        NO_STATUS_ERROR_STUB: document(NO_STATUS_ERROR_STUB, status='failed', offset=None),
        # A successful outcome carrying a usable attitude and no offset field.
        # No navigation writes it, since a result with no offset is never a
        # success.  Both paths build the product the rest of the record
        # supplies; only the name they give the offset shortfall differs.
        SUCCESS_NO_OFFSET_KEY_STUB: document(
            SUCCESS_NO_OFFSET_KEY_STUB, offset=OFFSET, times=TIMES, pointing=POINTING
        ),
        # A rotation written as nine rows of one.  Every reader that assembles
        # an array reshapes it into the same 3x3 the flat nine denote, so the
        # store has to hold it: a store that judged the entries one at a time
        # instead would leave the corrected attitude applied through a document
        # and an OffsetFOV applied through a row.
        ONE_ELEMENT_ROWS_CMATRIX_STUB: document(
            ONE_ELEMENT_ROWS_CMATRIX_STUB,
            offset=OFFSET,
            times=TIMES,
            pointing={**POINTING, 'cmatrix': one_element_rows(CMATRIX)},
        ),
        # The baseline written the same way, which is gated against the
        # observation exactly as the corrected attitude is.
        ONE_ELEMENT_ROWS_ORIGINAL_STUB: document(
            ONE_ELEMENT_ROWS_ORIGINAL_STUB,
            offset=OFFSET,
            times=TIMES,
            pointing={**POINTING, 'cmatrix_original': one_element_rows(CMATRIX_ORIGINAL)},
        ),
        # Nine entries that pass the count and are of shapes no single array
        # can hold.  Assembling them raises, and that refusal has to be the
        # malformed record it is rather than an exception through a classifier.
        RAGGED_NINE_CMATRIX_STUB: document(
            RAGGED_NINE_CMATRIX_STUB,
            offset=OFFSET,
            times=TIMES,
            pointing={
                **POINTING,
                'cmatrix': [[1.0, 2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]],
            },
        ),
        # An integer no float can hold, in each of the three places a reader
        # converts one.  Each is a malformed value to both storages; a reader
        # that raised on it would cost the image in one path and the whole
        # document in the other.
        HUGE_INT_IN_CMATRIX_STUB: document(
            HUGE_INT_IN_CMATRIX_STUB,
            offset=OFFSET,
            times=TIMES,
            pointing={**POINTING, 'cmatrix': [TOO_BIG_FOR_A_FLOAT, *CMATRIX[1:]]},
        ),
        HUGE_INT_MIDTIME_STUB: document(
            HUGE_INT_MIDTIME_STUB,
            offset=OFFSET,
            times={**TIMES, 'midtime_et': TOO_BIG_FOR_A_FLOAT},
            pointing=POINTING,
        ),
        HUGE_INT_OFFSET_STUB: document(HUGE_INT_OFFSET_STUB, offset=OFFSET),
        # An unsuccessful outcome naming, as its own error, the word a record
        # naming none is reported under.  Stored as the record naming none it
        # reads as, so the column cannot come to hold one thing and the reader
        # report another.
        LITERAL_UNKNOWN_ERROR_STUB: document(
            LITERAL_UNKNOWN_ERROR_STUB, status='failed', offset=None, status_error='unknown'
        ),
        # A document the ingest refuses whole, carrying a usable corrected
        # attitude all the same.  Read as a file it supplies that attitude;
        # the index holds no record of it and must say so rather than report
        # the image as one nothing navigated.
        REFUSED_DOCUMENT_STUB: document(
            REFUSED_DOCUMENT_STUB,
            instrument=None,
            offset=OFFSET,
            times=TIMES,
            pointing=POINTING,
        ),
        # An offset of two zeros: a navigation that found the pointing already
        # right records one, and it is a pair like any other.  Every column of
        # the row it becomes holds a value that is false when it is read for
        # its truth rather than for its presence.
        ZERO_OFFSET_STUB: document(ZERO_OFFSET_STUB, offset=ZERO_OFFSET),
        # And the same on the other half of the rebuild: exposure epochs at the
        # J2000 epoch and frame identities of zero, beside a corrected attitude
        # the ladder does apply.  A rebuild reading presence as truth loses the
        # midtime and reports this record as a malformed pointing block.
        ZERO_EPOCH_STUB: document(
            ZERO_EPOCH_STUB, offset=OFFSET, times=ZERO_TIMES, pointing=ZERO_FRAME_ID_POINTING
        ),
    }
    # The two shapes of "no usable offset on a successful record" are made here
    # rather than by the factory, which writes the key only when it has a value
    # and so cannot express a key holding null.
    tree[NULL_OFFSET_STUB]['offset'] = None
    del tree[NO_OFFSET_KEY_STUB]['offset']
    del tree[SUCCESS_NO_OFFSET_KEY_STUB]['offset']
    # Likewise made here: the factory writes the top-level status always, and
    # the nested copy of it is what the shape under test is about.
    del tree[NO_TOP_LEVEL_STATUS_STUB]['status']
    # And likewise the values the factory's own types cannot express: an offset
    # of two numeric strings, and an error field present and holding null.
    tree[NUMERIC_STRING_OFFSET_STUB]['offset'] = [str(OFFSET[0]), str(OFFSET[1])]
    tree[NULL_STATUS_ERROR_STUB]['status_error'] = None
    tree[HUGE_INT_OFFSET_STUB]['offset'] = [TOO_BIG_FOR_A_FLOAT, 1.0]
    return tree


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
    build_tree(root, reason_tree())
    engine = index_for([root], tmp_path / 'index.sqlite3', logger=quiet_ingest_logger)
    try:
        yield {
            'file': FilePointingSource(FCPath(root)),
            'index': IndexPointingSource(engine, normalize_root_url(root)),
        }
    finally:
        engine.dispose()
