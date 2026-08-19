"""The per-image facts, from the index and from the documents, compared.

The acceptance test of the fourth question on the seam: over one results tree,
the facts a source reading the documents yields and the facts a source reading
an index ingested from that same tree yields are the same facts -- every field of
the image, every technique row, every feature-source row, and the same refusal
for every file neither can read.  A consumer cannot see which storage answered,
so anything the two disagree about is a report that changes when an operator
points it at a database.

Two things are built into every test here because both are this codebase's own
defect classes.

**Two roots, and the second differs in the value under test.**  The key is
``(root_url, results_path_stub)``, one index serves several roots, and a query
that filters on the stub alone passes every single-root test there is.  So both
roots hold the same stubs with different values, and the comparisons are made
against each of them in turn: a query that dropped the root answers out of
whichever row the server reached first, and one of the two directions catches it.
A source over *one* root cannot show everything the key does, though: the merge
pairs the wrong rows only where one stream spans two roots, so the last section
reads two roots in one stream as well.

**Neither backend's order is trusted.**  Both streams are sorted in Python
before anything is compared, on the whole key, and so are the child rows within
one image, because the index stores no ordinal for a technique and a server
returns rows of one key in whatever order it likes.
"""

import importlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from tests.spindoctor.cli.stats.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    write_metadata,
)
from tests.spindoctor.results_index.conftest import (
    feature_source_row,
    image_row,
    technique_row,
)

from spindoctor.nav_records import (
    METADATA_SUFFIX,
    NOT_A_NAVIGATION_DOCUMENT,
    NOT_VALID_JSON,
    RETRIEVE_BATCH_SIZE,
    ImageFacts,
    Selection,
    TreeRecordSource,
    UnreadableFile,
    normalize_root_url,
)
from spindoctor.results_index import (
    FEATURE_SOURCES,
    IMAGES,
    TECHNIQUES,
    IndexRecordSource,
    open_index,
    open_record_source,
)

facts_stream_module = importlib.import_module('spindoctor.results_index.facts_stream')
"""The module the merge lives in, which one test replaces a statement inside.

Bound by importing the submodule rather than by reading the name off the
package: the package is free to re-export the function of the same name, and a
patch aimed at the name would then be aimed at a function object instead of at
the module whose attribute the merge reads.
"""

MISSION = 'coiss'
"""The instrument identity the mission-filtered reads below keep."""

OTHER_MISSION = 'vgiss'
"""An instrument identity of another mission's documents in the same tree."""

COLUMNS = (IMAGES.c.status, IMAGES.c.offset_dv, IMAGES.c.offset_du)
"""A consumer's columns, which narrow a record and must not narrow the facts."""

OTHER_MISSION_STUB = 'VOL0/C1454725_CALIB'
"""An image of another mission, sorting ahead of every image kept below.

Ahead deliberately: a mission-filtered stream drops it, and its technique rows
are then the first rows a child stream would meet.  A merge that did not restrict
the children to the images it yields would stall on them and hand every later
image no techniques at all.
"""

SUCCESS_STUB = 'VOL1/N1454725799_1_CALIB'
"""A navigated image, carrying everything a fully populated document carries."""

FAILURE_STUB = 'VOL1/N1454725800_1_CALIB'
"""An image the navigation reached and did not solve."""

ERROR_STUB = 'VOL2/N1454725801_1_CALIB'
"""An image whose run ended in a fatal error, naming it."""

UNLOADED_STUB = 'VOL2/N1454725802_1_CALIB'
"""An image that never loaded: no navigation result, and no outcome named.

Its epoch is under ``observation`` rather than under the provenance a run that
loaded the image would have written, which is the pair of columns that says
which of the two a document carried.
"""

REFUSED_STUB = 'VOL1/junk'
"""A JSON object of some other tool's shape, which no pass reads as a document.

The second root holds a navigable document at this same stub, so a refusal
excluded by the presence of a record under the *other* root would go unreported
here.
"""

TORN_STUB = 'VOL2/torn'
"""A file no JSON value comes out of at all."""

EXTRA_STUB = 'VOL2/N9999999999_1_CALIB'
"""An image only the second root holds, so a root-blind stream yields too much."""

OTHER_ROOT_REFUSED_STUB = 'VOL2/junk_second'
"""A file only the second root refused, at a stub the first root has nothing at.

Its own stub rather than a shared one: a read of the first root naming it has to
come back empty, and a stub both roots held would come back full whether or not
the refusal query knew about the root.
"""

REFUSED_DOCUMENT = '{"edges": []}'
"""What sits at the refused stub."""

TORN_DOCUMENT = '{"observation":'
"""What sits at the torn stub."""

NESTED_STUB = 'VOL2/nested'
"""A file whose nesting a decoder may give up on rather than fail to parse.

Its own class of refusal: how deep a decoder will follow nesting, and whether it
reports giving up as a decoding error or as something else entirely, is that
decoder's business.  Whatever it does, no value came out of the file, and the
two storages have to say that in the same words -- one of them naming the
exception it met and the other not is a difference an operator reads as the two
disagreeing about the file.
"""

NESTED_DOCUMENT = '{"a":' * 20000
"""What sits at the nested stub: twenty thousand opening braces and nothing else."""


@dataclass(frozen=True)
class RootValues:
    """What one root's documents record where the other root's record something else.

    Every value a test below reads is in here, so a query that dropped the root
    half of the key reads the wrong one of these rather than reading nothing.

    Parameters:
        offset: The authoritative offset every image of this root records.
        twist_covariance: The 3x3 covariance the navigated image records, whose
            rotation row and column no per-axis sigma states.
        technique_covariance: The 2x2 covariance each of its techniques records.
        excluded: The technique names the ensemble excluded, in an order that
            is not their sorted order, so a storage that sorted them is caught.
        camera_frame_id: The frame identifier the navigated image records, which
            the two roots record differently.
        observation_et: The epoch the image that never loaded records under its
            observation.
        midtime: The exposure midtime the navigated image records.
    """

    offset: list[float]
    twist_covariance: list[list[float]]
    technique_covariance: list[list[float]]
    excluded: list[str]
    camera_frame_id: int
    observation_et: float
    midtime: float


FIRST_VALUES = RootValues(
    offset=[1.5, -2.5],
    twist_covariance=[
        [0.010, 0.002, 0.0003],
        [0.002, 0.040, 0.0005],
        [0.0003, 0.0005, 0.000001],
    ],
    technique_covariance=[[0.11, 0.012], [0.012, 0.13]],
    excluded=['StarRefineNav', 'BodyLimbNav', 'RingEdgeNav'],
    camera_frame_id=-82360,
    observation_et=5000.0,
    midtime=100.0,
)
"""What the first root's documents record."""

SECOND_VALUES = RootValues(
    offset=[9.5, -8.5],
    twist_covariance=[
        [0.090, 0.008, 0.0007],
        [0.008, 0.070, 0.0009],
        [0.0007, 0.0009, 0.000004],
    ],
    technique_covariance=[[0.21, 0.022], [0.022, 0.23]],
    excluded=['RingEdgeNav', 'StarRefineNav', 'BodyLimbNav'],
    camera_frame_id=-84001,
    observation_et=6000.0,
    midtime=300.0,
)
"""What the second root's record instead, so the two are told apart by value."""

IDENTITY_CMATRIX = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
"""A recorded rotation, of no astronomical significance."""


def _technique(name: str, offset: list[float], values: RootValues) -> dict[str, Any]:
    """Build one ``per_technique`` entry.

    Parameters:
        name: The technique's class name, which is the identity of its row.
        offset: Its own estimate.
        values: The root's differing values, for the covariance it records.

    Returns:
        The entry.
    """
    return {
        'technique_name': name,
        'feature_ids': [f'{name.lower()}:IAPETUS'],
        'offset_px': offset,
        'covariance_px2': values.technique_covariance,
        'confidence': 0.7,
        'spurious': False,
        'at_edge': False,
        'diagnostics': {'iterations': 4},
    }


def _success_document(values: RootValues) -> dict[str, Any]:
    """Build the navigated image's document.

    Parameters:
        values: The root's differing values.

    Returns:
        The document.
    """
    document = metadata_document(
        image_name='N1454725799_1.IMG',
        instrument=MISSION,
        offset=values.offset,
        excluded=values.excluded,
        image_shape=[1024, 1024],
        per_technique=[
            _technique('StarFieldFromCatalogNav', [1.0, 2.0], values),
            _technique('BodyLimbNav', [1.1, 2.1], values),
        ],
        times={
            'midtime_et': values.midtime,
            'start_et': values.midtime - 1.0,
            'stop_et': values.midtime + 1.0,
            'exposure_s': 2.0,
        },
        pointing={
            'camera_frame': 'CASSINI_ISS_NAC',
            'camera_frame_id': values.camera_frame_id,
            'ck_frame_id': -82000,
            'cmatrix': IDENTITY_CMATRIX,
            'cmatrix_original': IDENTITY_CMATRIX,
        },
    )
    # A twist-fitted result, whose rotation row and column carry the terms a
    # per-axis sigma cannot state.  The builder writes a 2x2, so it is replaced
    # here rather than parameterized into a helper nothing else needs.
    document['navigation_result']['covariance_px2'] = values.twist_covariance
    document['navigation_result']['rotation_deg'] = 0.25
    document['navigation_result']['sigma_rotation_deg'] = 0.01
    return document


def _failure_document(values: RootValues) -> dict[str, Any]:
    """Build the document of an image the navigation did not solve.

    Parameters:
        values: The root's differing values, for the epoch it records.

    Returns:
        The document.
    """
    return metadata_document(
        image_name='N1454725800_1.IMG',
        instrument=MISSION,
        status='failed',
        status_reason='no technique produced a usable offset',
        offset=None,
        times={'midtime_et': values.midtime + 10.0},
    )


def _error_document(values: RootValues) -> dict[str, Any]:
    """Build the document of a run that ended in a fatal error.

    Parameters:
        values: The root's differing values, for the epoch it records.

    Returns:
        The document.
    """
    return metadata_document(
        image_name='N1454725801_1.IMG',
        instrument=MISSION,
        status='error',
        status_error='SPICE(SPKINSUFFDATA)',
        status_reason=None,
        offset=None,
        times={'midtime_et': values.midtime + 20.0},
    )


def _unloaded_document(values: RootValues) -> dict[str, Any]:
    """Build the document of an image that never loaded.

    Parameters:
        values: The root's differing values, for the observation epoch.

    Returns:
        The document, which names no outcome and carries no navigation result:
        its epoch is the one the navigator read out of the dataset index, and
        that is recorded under the observation rather than as provenance.
    """
    return {
        'observation': {
            'image_name': 'N1454725802_1.IMG',
            'instrument': MISSION,
            'image_et': values.observation_et,
            'image_path': '/holdings/N1454725802_1.IMG',
        }
    }


def _other_mission_document(values: RootValues) -> dict[str, Any]:
    """Build a document of the mission a filtered read drops.

    Parameters:
        values: The root's differing values.

    Returns:
        The document, which carries technique rows of its own so that dropping
        the image has to drop them too.
    """
    return metadata_document(
        image_name='C1454725.IMG',
        instrument=OTHER_MISSION,
        offset=values.offset,
        per_technique=[_technique('StarUniqueMatchNav', [3.0, 4.0], values)],
        times={'midtime_et': values.midtime},
    )


def _write_text(root: Path, stub: str, text: str) -> None:
    """Write one file where a navigation document belongs, whatever it holds.

    Parameters:
        root: The results root.
        stub: The document's results path stub under it.
        text: What to write.
    """
    path = root / f'{stub}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def _write_root(tmp_path: Path, name: str, values: RootValues, *, extra: bool) -> Path:
    """Write one results root holding every kind of file the tests read.

    Parameters:
        tmp_path: Directory the root is written under.
        name: The root's directory name.
        values: The values this root's documents record.
        extra: Whether this root also holds an image the other does not, and a
            navigable document where the other holds a refusal.

    Returns:
        The results root.
    """
    root = tmp_path / name
    write_metadata(root, OTHER_MISSION_STUB, _other_mission_document(values))
    write_metadata(root, SUCCESS_STUB, _success_document(values))
    write_metadata(root, FAILURE_STUB, _failure_document(values))
    write_metadata(root, ERROR_STUB, _error_document(values))
    write_metadata(root, UNLOADED_STUB, _unloaded_document(values))
    _write_text(root, TORN_STUB, TORN_DOCUMENT)
    _write_text(root, NESTED_STUB, NESTED_DOCUMENT)
    if extra:
        write_metadata(root, EXTRA_STUB, _failure_document(values))
        write_metadata(root, REFUSED_STUB, _failure_document(values))
        _write_text(root, OTHER_ROOT_REFUSED_STUB, REFUSED_DOCUMENT)
    else:
        _write_text(root, REFUSED_STUB, REFUSED_DOCUMENT)
    return root


class TwoRoots:
    """Two results roots differing in every value read, and the index of both.

    Parameters:
        first: The root whose documents record the first set of values.
        second: The root whose documents record the second set, holding one
            image the first does not and a navigable document where the first
            holds a refusal.
        url: The index both were ingested into.
    """

    def __init__(self, first: Path, second: Path, url: str) -> None:
        self.first = first
        self.second = second
        self.url = url

    def root(self, which: str) -> Path:
        """Return one of the two roots by name.

        Parameters:
            which: ``'first'`` or ``'second'``.

        Returns:
            That root.
        """
        return self.first if which == 'first' else self.second


@pytest.fixture
def quiet_logger() -> pdslogger.PdsLogger:
    """Return a logger that writes nowhere.

    Returns:
        The logger, so an ingest driven by a test says nothing.
    """
    return pdslogger.NullLogger()


@pytest.fixture
def two_roots(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> TwoRoots:
    """Write two results roots and ingest both into one index.

    Parameters:
        tmp_path: Directory the roots and the index are written under.
        quiet_logger: Logger the ingest reports through.

    Returns:
        The two roots and the index holding both.
    """
    first = _write_root(tmp_path, 'results-first', FIRST_VALUES, extra=False)
    second = _write_root(tmp_path, 'results-second', SECOND_VALUES, extra=True)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first, second], logger=quiet_logger)
    return TwoRoots(first, second, url)


BOTH_ROOTS = [pytest.param('first', id='first-root'), pytest.param('second', id='second-root')]
"""Read every comparison against each root, so a root-blind query fails one."""


def _stub_of(found: ImageFacts | UnreadableFile) -> str:
    """Return the stub of one thing a facts stream yielded.

    Parameters:
        found: The facts, or the file no facts came out of.

    Returns:
        The image's results path stub, which both shapes carry.
    """
    if isinstance(found, UnreadableFile):
        return found.stub
    return str(found.image['results_path_stub'])


def _technique_key(row: dict[str, Any]) -> str:
    """Return what identifies one technique row within its image.

    Parameters:
        row: The row.

    Returns:
        The technique's name.
    """
    return str(row['technique_name'])


def _feature_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return what identifies one feature-source row within its image.

    Parameters:
        row: The row.

    Returns:
        The feature type and the source that offered it.
    """
    return str(row['feature_type']), str(row['source_model']), str(row['source_name'])


def _root_of(found: ImageFacts | UnreadableFile) -> str:
    """Return the root one thing a facts stream yielded is held under.

    Parameters:
        found: The facts, or the file no facts came out of.

    Returns:
        The root URL.  An image carries it as a column of its own; a file no
        facts came out of carries only where its document is, which is that
        root joined to the stub it is a key under.
    """
    if isinstance(found, UnreadableFile):
        return str(found.path)[: -len(f'/{found.stub}{METADATA_SUFFIX}')]
    return str(found.image['root_url'])


def _key_of(found: ImageFacts | UnreadableFile) -> tuple[str, str]:
    """Return the whole key of one thing a facts stream yielded.

    Parameters:
        found: The facts, or the file no facts came out of.

    Returns:
        The root URL and the results path stub.
    """
    return _root_of(found), _stub_of(found)


def _in_order(found: Any) -> list[ImageFacts | UnreadableFile]:
    """Return what a stream yielded, sorted by the whole key in Python.

    Neither storage promises an order, and a server sorts text under its own
    collation, so the comparison is made in one order that belongs to neither.
    On the whole key rather than on the stub: a stream over two roots yields one
    stub twice, and a sort on half a key leaves those two in whichever order
    they arrived, which is a property of the backend rather than of the answer.

    Parameters:
        found: What the stream yielded.

    Returns:
        The same things, in key order.
    """
    return sorted(found, key=_key_of)


def _images(found: list[ImageFacts | UnreadableFile]) -> list[dict[str, Any]]:
    """Return each image's own values, in the order the stream was sorted into.

    Parameters:
        found: What a stream yielded, sorted.

    Returns:
        One mapping per image.
    """
    return [one.image for one in found if isinstance(one, ImageFacts)]


def _techniques(found: list[ImageFacts | UnreadableFile]) -> list[list[dict[str, Any]]]:
    """Return each image's technique rows, sorted within the image.

    Parameters:
        found: What a stream yielded, sorted.

    Returns:
        One list per image, each in technique-name order.
    """
    return [
        sorted(one.techniques, key=_technique_key) for one in found if isinstance(one, ImageFacts)
    ]


def _feature_sources(found: list[ImageFacts | UnreadableFile]) -> list[list[dict[str, Any]]]:
    """Return each image's feature-source rows, sorted within the image.

    Parameters:
        found: What a stream yielded, sorted.

    Returns:
        One list per image, each in feature-source order.
    """
    return [
        sorted(one.feature_sources, key=_feature_key)
        for one in found
        if isinstance(one, ImageFacts)
    ]


def _refusals(found: list[ImageFacts | UnreadableFile]) -> list[tuple[str, str]]:
    """Return every file no facts came out of, and why.

    Parameters:
        found: What a stream yielded, sorted.

    Returns:
        The stub and reason of each.
    """
    return [(one.stub, one.reason) for one in found if isinstance(one, UnreadableFile)]


def _from_tree(
    held: TwoRoots, which: str, selection: Selection
) -> list[ImageFacts | UnreadableFile]:
    """Read the facts of one root out of its documents.

    Parameters:
        held: The two roots and their index.
        which: Which root to read.
        selection: What to read.

    Returns:
        What the stream yielded, in stub order.
    """
    return _in_order(TreeRecordSource([held.root(which)]).facts(selection))


def _from_index(
    held: TwoRoots, which: str, selection: Selection
) -> list[ImageFacts | UnreadableFile]:
    """Read the facts of one root out of the index both roots were ingested into.

    Parameters:
        held: The two roots and their index.
        which: Which root to read.
        selection: What to read.

    Returns:
        What the stream yielded, in stub order.
    """
    with open_record_source([held.root(which)], results_db_url=held.url, columns=COLUMNS) as source:
        return _in_order(source.facts(selection))


def _facts_of(found: list[ImageFacts | UnreadableFile], stub: str) -> ImageFacts:
    """Return one image's facts out of what a stream yielded.

    Parameters:
        found: What the stream yielded.
        stub: The image to pick out.

    Returns:
        Its facts.
    """
    picked = [one for one in found if isinstance(one, ImageFacts) and _stub_of(one) == stub]
    assert len(picked) == 1
    return picked[0]


# ---------------------------------------------------------------------------
# The acceptance test: one tree, two storages, one answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_the_two_storages_cover_the_same_files(two_roots: TwoRoots, which: str) -> None:
    """Every file the selection covers, from either storage.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    from_tree = [_stub_of(one) for one in _from_tree(two_roots, which, Selection())]
    from_index = [_stub_of(one) for one in _from_index(two_roots, which, Selection())]
    assert from_index == from_tree


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_the_two_storages_agree_on_every_image_field(two_roots: TwoRoots, which: str) -> None:
    """Field by field, which is the whole of what a consumer reads.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    from_tree = _images(_from_tree(two_roots, which, Selection()))
    from_index = _images(_from_index(two_roots, which, Selection()))
    assert from_index == from_tree


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_the_two_storages_agree_on_every_technique_row(two_roots: TwoRoots, which: str) -> None:
    """The child rows a reader of documents gets for nothing.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    from_tree = _techniques(_from_tree(two_roots, which, Selection()))
    from_index = _techniques(_from_index(two_roots, which, Selection()))
    assert from_index == from_tree


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_the_two_storages_agree_on_every_feature_source_row(
    two_roots: TwoRoots, which: str
) -> None:
    """The aggregated inventory, which is the other table the merge reads.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    from_tree = _feature_sources(_from_tree(two_roots, which, Selection()))
    from_index = _feature_sources(_from_index(two_roots, which, Selection()))
    assert from_index == from_tree


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_the_two_storages_refuse_the_same_files_for_the_same_reason(
    two_roots: TwoRoots, which: str
) -> None:
    """A file that is no navigation document, reported alike by both.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    from_tree = _refusals(_from_tree(two_roots, which, Selection()))
    from_index = _refusals(_from_index(two_roots, which, Selection()))
    assert from_index == from_tree


def test_the_comparison_covers_a_refused_file(two_roots: TwoRoots) -> None:
    """Without which the agreement above would be an agreement about nothing.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    reasons = dict(_refusals(_from_tree(two_roots, 'first', Selection())))
    assert reasons[REFUSED_STUB].startswith(NOT_A_NAVIGATION_DOCUMENT)


def test_the_comparison_covers_a_file_that_is_not_json(two_roots: TwoRoots) -> None:
    """The other family of reason, which says the parse produced nothing.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    reasons = dict(_refusals(_from_tree(two_roots, 'first', Selection())))
    assert reasons[TORN_STUB] == NOT_VALID_JSON


def test_the_comparison_covers_a_file_the_decoder_gave_up_on(two_roots: TwoRoots) -> None:
    """Whichever way the decoder gave up, both storages state the one reason for it.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    reasons = dict(_refusals(_from_tree(two_roots, 'first', Selection())))
    assert reasons[NESTED_STUB] == NOT_VALID_JSON


def test_the_index_gives_that_file_the_reason_the_documents_give_it(
    two_roots: TwoRoots,
) -> None:
    """Named on its own as well, because it is the case the two once differed on.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    reasons = dict(_refusals(_from_index(two_roots, 'first', Selection())))
    assert reasons[NESTED_STUB] == NOT_VALID_JSON


def test_a_refusal_is_reported_although_the_other_root_holds_a_record_there(
    two_roots: TwoRoots,
) -> None:
    """A record under one root is no evidence about another root's refusal.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    stubs = [stub for stub, _reason in _refusals(_from_index(two_roots, 'first', Selection()))]
    assert REFUSED_STUB in stubs


def test_the_other_roots_extra_image_is_not_this_roots(two_roots: TwoRoots) -> None:
    """A stream that dropped the root would yield an image this root never held.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    stubs = [_stub_of(one) for one in _from_index(two_roots, 'first', Selection())]
    assert EXTRA_STUB not in stubs


# ---------------------------------------------------------------------------
# The values the comparison is made of
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_a_twist_covariance_survives_whole(two_roots: TwoRoots, which: str) -> None:
    """Its rotation row and column carry terms no per-axis sigma states.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    values = FIRST_VALUES if which == 'first' else SECOND_VALUES
    found = _facts_of(_from_index(two_roots, which, Selection()), SUCCESS_STUB)
    assert found.image['covariance_px2'] == values.twist_covariance


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_a_per_technique_covariance_survives_whole(two_roots: TwoRoots, which: str) -> None:
    """Stored as a matrix, so a reader derives the sigmas rather than the reverse.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    values = FIRST_VALUES if which == 'first' else SECOND_VALUES
    found = _facts_of(_from_index(two_roots, which, Selection()), SUCCESS_STUB)
    assert [row['covariance_px2'] for row in found.techniques] == [
        values.technique_covariance,
        values.technique_covariance,
    ]


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_an_exclusion_list_keeps_the_order_it_was_written_in(
    two_roots: TwoRoots, which: str
) -> None:
    """Which is not sorted order, so a storage that sorted the list is caught.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    values = FIRST_VALUES if which == 'first' else SECOND_VALUES
    found = _facts_of(_from_index(two_roots, which, Selection()), SUCCESS_STUB)
    assert found.image['excluded_from_consensus'] == values.excluded


def test_the_exclusion_list_under_test_is_not_in_sorted_order() -> None:
    """Without which the test above would hold whether or not order survived."""
    assert FIRST_VALUES.excluded != sorted(FIRST_VALUES.excluded)


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_each_roots_frame_id_comes_back_from_that_root(two_roots: TwoRoots, which: str) -> None:
    """The two roots name two frames, so a read of one may not answer with the other.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    values = FIRST_VALUES if which == 'first' else SECOND_VALUES
    found = _facts_of(_from_index(two_roots, which, Selection()), SUCCESS_STUB)
    assert found.image['camera_frame_id'] == values.camera_frame_id


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_an_image_that_never_loaded_records_its_epoch_under_the_observation(
    two_roots: TwoRoots, which: str
) -> None:
    """The column that says which of the two epochs a document carried.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    values = FIRST_VALUES if which == 'first' else SECOND_VALUES
    found = _facts_of(_from_index(two_roots, which, Selection()), UNLOADED_STUB)
    assert found.image['observation_image_et'] == values.observation_et


def test_an_image_that_never_loaded_records_no_provenance_epoch(two_roots: TwoRoots) -> None:
    """The other half of the pair, which is what makes the two columns worth having.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = _facts_of(_from_index(two_roots, 'first', Selection()), UNLOADED_STUB)
    assert found.image['provenance_image_et'] is None


def test_a_navigated_image_records_its_epoch_as_provenance(two_roots: TwoRoots) -> None:
    """Which is the case the pair of columns has to tell apart from the other.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = _facts_of(_from_index(two_roots, 'first', Selection()), SUCCESS_STUB)
    assert found.image['provenance_image_et'] == 0.0


def test_an_image_naming_no_outcome_carries_no_error(two_roots: TwoRoots) -> None:
    """An absent outcome and a recorded error are different facts.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = _facts_of(_from_index(two_roots, 'first', Selection()), UNLOADED_STUB)
    assert found.image['status_error'] is None


def test_a_run_that_ended_in_an_error_names_it(two_roots: TwoRoots) -> None:
    """The vocabulary an error filter matches verbatim.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = _facts_of(_from_index(two_roots, 'first', Selection()), ERROR_STUB)
    assert found.image['status_error'] == 'SPICE(SPKINSUFFDATA)'


# ---------------------------------------------------------------------------
# What the index alone owes
# ---------------------------------------------------------------------------


def test_the_facts_carry_every_column_whatever_the_consumer_selected(
    two_roots: TwoRoots,
) -> None:
    """The columns narrow a record; the facts are the whole row by definition.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = _facts_of(_from_index(two_roots, 'first', Selection()), SUCCESS_STUB)
    assert set(found.image) == {column.name for column in IMAGES.columns}


def test_the_consumers_columns_are_fewer_than_the_whole_row() -> None:
    """Without which the test above would hold whatever the statement selected."""
    assert len(COLUMNS) < len(IMAGES.columns)


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_a_mission_filter_keeps_one_missions_images(two_roots: TwoRoots, which: str) -> None:
    """The restriction both storages honour the same way.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    selection = Selection(instrument=MISSION)
    from_tree = [_stub_of(one) for one in _from_tree(two_roots, which, selection)]
    from_index = [_stub_of(one) for one in _from_index(two_roots, which, selection)]
    assert from_index == from_tree


def test_the_children_of_a_dropped_image_are_not_merged_onto_a_kept_one(
    two_roots: TwoRoots,
) -> None:
    """The dropped image sorts first, so its rows are the first the merge meets.

    A merge reading every child row under the root would hold that image's
    techniques against a key it never yields, and every image after it would
    come back with none.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = _facts_of(_from_index(two_roots, 'first', Selection(instrument=MISSION)), SUCCESS_STUB)
    assert [_technique_key(row) for row in sorted(found.techniques, key=_technique_key)] == [
        'BodyLimbNav',
        'StarFieldFromCatalogNav',
    ]


def test_the_dropped_image_really_holds_child_rows(two_roots: TwoRoots) -> None:
    """Without which the test above would prove nothing about the merge.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = _facts_of(_from_index(two_roots, 'first', Selection()), OTHER_MISSION_STUB)
    assert [_technique_key(row) for row in found.techniques] == ['StarUniqueMatchNav']


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_a_subtree_filter_keeps_one_subtrees_images(two_roots: TwoRoots, which: str) -> None:
    """Answered from a column on one storage and from a walk on the other.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    selection = Selection(subtrees=('VOL1',))
    from_tree = [_stub_of(one) for one in _from_tree(two_roots, which, selection)]
    from_index = [_stub_of(one) for one in _from_index(two_roots, which, selection)]
    assert from_index == from_tree


def test_a_selection_naming_stubs_is_answered_in_the_order_it_names_them(
    two_roots: TwoRoots,
) -> None:
    """Naming an image is not a narrowing, so the answer lines up with the ask.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    named = Selection(stubs=(ERROR_STUB, SUCCESS_STUB, REFUSED_STUB))
    with open_record_source(
        [two_roots.first], results_db_url=two_roots.url, columns=COLUMNS
    ) as source:
        found = list(source.facts(named))
    assert [_stub_of(one) for one in found] == [ERROR_STUB, SUCCESS_STUB, REFUSED_STUB]


def test_a_selection_naming_stubs_reads_the_selected_roots_values(two_roots: TwoRoots) -> None:
    """The other root holds the same stub, recording something else.

    Read against the *first* root, whose rows were written first.  A batch read
    builds what it found with a dictionary update over a stream ordered by the
    key, so a query that dropped the root half would be answered by whichever
    row came back last -- which is the root whose URL sorts last, and asking for
    that one would be answered correctly by the defect.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    named = Selection(stubs=(SUCCESS_STUB,))
    with open_record_source(
        [two_roots.first], results_db_url=two_roots.url, columns=COLUMNS
    ) as source:
        found = list(source.facts(named))
    assert _facts_of(found, SUCCESS_STUB).image['covariance_px2'] == FIRST_VALUES.twist_covariance


def test_the_root_read_first_is_not_the_one_a_root_blind_read_would_answer_with(
    two_roots: TwoRoots,
) -> None:
    """Without which the direction of the test above would be the wrong way round.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    assert normalize_root_url(two_roots.first) < normalize_root_url(two_roots.second)


def test_a_selection_naming_stubs_carries_the_child_rows(two_roots: TwoRoots) -> None:
    """The merge runs per batch, so a named read has to reach it too.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    named = Selection(stubs=(SUCCESS_STUB,))
    with open_record_source(
        [two_roots.first], results_db_url=two_roots.url, columns=COLUMNS
    ) as source:
        found = list(source.facts(named))
    rows = sorted(_facts_of(found, SUCCESS_STUB).techniques, key=_technique_key)
    assert [_technique_key(row) for row in rows] == ['BodyLimbNav', 'StarFieldFromCatalogNav']


def test_a_selection_naming_more_stubs_than_one_batch_answers_every_one(
    two_roots: TwoRoots,
) -> None:
    """A caller is free to name a mission's worth, and the read is cut into batches.

    Asked for in more names than one batch binds, so a read that answered the
    first batch and dropped the rest would hand a queue task back short.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    named = tuple([ERROR_STUB, SUCCESS_STUB] * 40)
    with open_record_source(
        [two_roots.first], results_db_url=two_roots.url, columns=COLUMNS
    ) as source:
        found = [_stub_of(one) for one in source.facts(Selection(stubs=named))]
    assert found == list(named)


def test_the_named_stubs_really_do_cross_a_batch_boundary() -> None:
    """Without which the test above would hold whatever the batching did."""
    assert len([ERROR_STUB, SUCCESS_STUB] * 40) > RETRIEVE_BATCH_SIZE


def test_named_stubs_still_honour_the_mission(two_roots: TwoRoots) -> None:
    """A selection is a narrowing whatever else it names.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    named = Selection(stubs=(SUCCESS_STUB, OTHER_MISSION_STUB), instrument=MISSION)
    with open_record_source(
        [two_roots.first], results_db_url=two_roots.url, columns=COLUMNS
    ) as source:
        found = [_stub_of(one) for one in source.facts(named)]
    assert found == [SUCCESS_STUB]


def test_named_stubs_still_honour_a_time_bound(two_roots: TwoRoots) -> None:
    """The other half of what a selection restricts a named read by.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    named = Selection(stubs=(SUCCESS_STUB, ERROR_STUB), start_et=FIRST_VALUES.midtime + 5.0)
    with open_record_source(
        [two_roots.first], results_db_url=two_roots.url, columns=COLUMNS
    ) as source:
        found = [_stub_of(one) for one in source.facts(named)]
    assert found == [ERROR_STUB]


def test_a_named_stub_only_the_other_root_holds_yields_nothing(two_roots: TwoRoots) -> None:
    """Naming a key does not stop it being a key under one root.

    The stub is one only the other root holds, so a query that bound the keys
    and dropped the root would hand this run an image nobody asked for -- and
    do it whichever order the server returned its rows in, which naming a stub
    both roots hold cannot show.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with open_record_source(
        [two_roots.first], results_db_url=two_roots.url, columns=COLUMNS
    ) as source:
        found = list(source.facts(Selection(stubs=(EXTRA_STUB,))))
    assert found == []


def test_a_named_stub_only_the_other_root_refused_yields_nothing(two_roots: TwoRoots) -> None:
    """The refusal half of a named read carries its own root term.

    The other root refused a file at a stub this root holds nothing at, so a
    refusal query blind to the root would report this run a shortfall that is
    not its own.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with open_record_source(
        [two_roots.first], results_db_url=two_roots.url, columns=COLUMNS
    ) as source:
        found = list(source.facts(Selection(stubs=(OTHER_ROOT_REFUSED_STUB,))))
    assert found == []


def test_the_other_roots_refused_file_really_is_refused_there(two_roots: TwoRoots) -> None:
    """Without which the test above would hold over a file nothing recorded.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with open_record_source(
        [two_roots.second], results_db_url=two_roots.url, columns=COLUMNS
    ) as source:
        found = list(source.facts(Selection(stubs=(OTHER_ROOT_REFUSED_STUB,))))
    assert [type(one) for one in found] == [UnreadableFile]


def test_a_selection_naming_a_root_the_source_does_not_hold_is_refused(
    two_roots: TwoRoots, tmp_path: Path
) -> None:
    """Refused where a caller asked, rather than partway through its loop.

    Parameters:
        two_roots: The two ingested roots and their index.
        tmp_path: Directory the unheld root would be under.
    """
    with (
        open_record_source(
            [two_roots.first], results_db_url=two_roots.url, columns=COLUMNS
        ) as source,
        pytest.raises(ValueError, match='does not hold'),
    ):
        source.facts(Selection(roots=(str(tmp_path / 'elsewhere'),)))


def test_a_stream_of_facts_gives_its_connection_back(two_roots: TwoRoots) -> None:
    """Three cursors on one connection, all of them released when it is done.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    events: list[str] = []
    engine = open_index(two_roots.url)
    sqlalchemy.event.listen(engine, 'checkout', lambda *_args: events.append('out'))
    sqlalchemy.event.listen(engine, 'checkin', lambda *_args: events.append('in'))
    with IndexRecordSource(engine, [two_roots.first], two_roots.url, COLUMNS) as source:
        list(source.facts(Selection()))
    assert events.count('out') == events.count('in')


def test_a_stream_of_facts_takes_a_connection_out_of_the_pool(two_roots: TwoRoots) -> None:
    """Without which the balance above would hold over nothing at all.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    events: list[str] = []
    engine = open_index(two_roots.url)
    sqlalchemy.event.listen(engine, 'checkout', lambda *_args: events.append('out'))
    with IndexRecordSource(engine, [two_roots.first], two_roots.url, COLUMNS) as source:
        list(source.facts(Selection()))
    assert events.count('out') == 1


# ---------------------------------------------------------------------------
# The merge, against the order the backend would otherwise hand back
# ---------------------------------------------------------------------------

REVERSED_STUBS = ('VOL/B_CALIB', 'VOL/A_CALIB')
"""Two images, written in the order a sorted read does not return them in."""


class ReversedIndex:
    """An index whose child rows are written in the opposite order to its images.

    What this shape shows is that the merge pairs a child row with its image by
    the key it carries: written back to front, insertion order pairs every image
    with the wrong one, so a merge reading the two streams off against each
    other by position hands each image another image's children.

    What it does not show is that the statements ask for an order.  The child
    read joins to ``images`` on the key, and a backend is free to answer that
    join off the child table's own unique index and hand the rows back in key
    order whether or not the statement said to sort them, which is what SQLite
    does here.  That the statements ask is pinned by reading the statements
    themselves, and what it costs when they do not is pinned on the tier that
    runs against a planner free to choose otherwise.

    Parameters:
        first: The root the tests read.
        second: A second root holding the same stubs with other children.
        url: The index holding both.
    """

    def __init__(self, first: Path, second: Path, url: str) -> None:
        self.first = first
        self.second = second
        self.url = url


def _child_name(root_url: str, stub: str) -> str:
    """Return a technique name naming the image it belongs to.

    Parameters:
        root_url: The root half of the image's key.
        stub: The stub half.

    Returns:
        A name no other image's row carries, so a mis-paired row is visible.
    """
    return f'{Path(root_url).name}:{stub}'


@pytest.fixture
def reversed_index(tmp_path: Path) -> ReversedIndex:
    """Write an index whose rows defeat every order but the one the merge asks for.

    Parameters:
        tmp_path: Directory the index is written under.

    Returns:
        The two roots and the index holding both.
    """
    first = tmp_path / 'reversed-first'
    second = tmp_path / 'reversed-second'
    url = index_url(tmp_path / 'reversed.sqlite3')
    keys = [(normalize_root_url(root), stub) for root in (second, first) for stub in REVERSED_STUBS]
    engine = open_index(url, create=True)
    try:
        with engine.begin() as connection:
            connection.execute(
                IMAGES.insert(),
                [
                    image_row(root_url=root_url, results_path_stub=stub, subtree='VOL')
                    for root_url, stub in keys
                ],
            )
            # The children go in back to front, so insertion order pairs every
            # image with the wrong one and only the sort puts them right.
            connection.execute(
                TECHNIQUES.insert(),
                [
                    technique_row(
                        root_url=root_url,
                        results_path_stub=stub,
                        technique_name=_child_name(root_url, stub),
                    )
                    for root_url, stub in reversed(keys)
                ],
            )
            connection.execute(
                FEATURE_SOURCES.insert(),
                [
                    feature_source_row(
                        root_url=root_url,
                        results_path_stub=stub,
                        source_name=_child_name(root_url, stub),
                    )
                    for root_url, stub in reversed(keys)
                ],
            )
    finally:
        engine.dispose()
    return ReversedIndex(first, second, url)


def _merged_children(held: ReversedIndex, root: Path, pick: str) -> dict[str, list[str]]:
    """Return the child names the merge gave each image of one root.

    Parameters:
        held: The index whose rows are written back to front.
        root: The root to read.
        pick: ``'techniques'`` or ``'feature_sources'``.

    Returns:
        The names, by stub.
    """
    engine = open_index(held.url)
    with IndexRecordSource(engine, [root], held.url, COLUMNS) as source:
        found = [one for one in source.facts(Selection()) if isinstance(one, ImageFacts)]
    if pick == 'techniques':
        return {
            _stub_of(one): [str(row['technique_name']) for row in one.techniques] for one in found
        }
    return {
        _stub_of(one): [str(row['source_name']) for row in one.feature_sources] for one in found
    }


def test_the_merge_gives_each_image_its_own_technique_rows(
    reversed_index: ReversedIndex,
) -> None:
    """Written back to front, so a positional merge pairs them all wrongly.

    Parameters:
        reversed_index: The index whose child rows are written in reverse.
    """
    root_url = normalize_root_url(reversed_index.first)
    assert _merged_children(reversed_index, reversed_index.first, 'techniques') == {
        stub: [_child_name(root_url, stub)] for stub in REVERSED_STUBS
    }


def test_the_merge_gives_each_image_its_own_feature_source_rows(
    reversed_index: ReversedIndex,
) -> None:
    """The other child table, merged by the same rule.

    Parameters:
        reversed_index: The index whose child rows are written in reverse.
    """
    root_url = normalize_root_url(reversed_index.first)
    assert _merged_children(reversed_index, reversed_index.first, 'feature_sources') == {
        stub: [_child_name(root_url, stub)] for stub in REVERSED_STUBS
    }


def test_the_merge_reads_the_selected_roots_children(reversed_index: ReversedIndex) -> None:
    """The other root holds the same stubs, with children named for itself.

    Parameters:
        reversed_index: The index whose child rows are written in reverse.
    """
    root_url = normalize_root_url(reversed_index.second)
    assert _merged_children(reversed_index, reversed_index.second, 'techniques') == {
        stub: [_child_name(root_url, stub)] for stub in REVERSED_STUBS
    }


def _unordered_keys(url: str, table: sqlalchemy.Table) -> list[tuple[Any, Any]]:
    """Return the keys of one table in the order the server hands them back unasked.

    Every column is selected, because that is what the merge selects and a
    narrower statement is answered off an index whose order is the key's rather
    than the table's own.

    Parameters:
        url: The index to read.
        table: The table to read.

    Returns:
        One key per row, in the order they arrived.
    """
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            return [
                (row.root_url, row.results_path_stub)
                for row in connection.execute(sqlalchemy.select(table))
            ]
    finally:
        engine.dispose()


def test_the_rows_really_do_arrive_in_two_different_orders(
    reversed_index: ReversedIndex,
) -> None:
    """Without which the three tests above hold whatever the merge does.

    Parameters:
        reversed_index: The index whose child rows are written in reverse.
    """
    images = _unordered_keys(reversed_index.url, IMAGES)
    techniques = _unordered_keys(reversed_index.url, TECHNIQUES)
    assert techniques != images


def _issued_by(url: str, root: Path, selection: Selection) -> list[str]:
    """Return the statements a stream of facts sent to the server.

    Parameters:
        url: The index to read.
        root: The root to read.
        selection: What to read.

    Returns:
        The SQL of each statement, in the order it was issued.
    """
    issued: list[str] = []
    engine = open_index(url)
    sqlalchemy.event.listen(
        engine,
        'before_cursor_execute',
        lambda conn, cursor, statement, *rest: issued.append(statement),
    )
    with IndexRecordSource(engine, [root], url, COLUMNS) as source:
        list(source.facts(selection))
    return issued


def _ordering_of(statement: str) -> str:
    """Return what one statement sorts on.

    Parameters:
        statement: The SQL of a statement carrying an ``ORDER BY``.

    Returns:
        The sort terms, with their whitespace flattened.
    """
    return ' '.join(statement.split('ORDER BY')[1].split())


def test_every_statement_of_the_merge_orders_on_the_whole_key(two_roots: TwoRoots) -> None:
    """Adjacent rows in one order are the whole of what lets three streams merge.

    Read off the statements the source issued rather than off the source code.
    The merge never compares two keys for their order, so it can only be right
    while the three streams arrive in one order, and the sort is on the whole
    key because one index serves several roots: an image stream sorted on the
    stub alone interleaves two roots where a child stream sorted on the key does
    not, and every image whose rows are met out of turn is handed none of them.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    issued = _issued_by(two_roots.url, two_roots.first, Selection())
    assert sorted(_ordering_of(one) for one in issued if 'ORDER BY' in one) == [
        'feature_sources.root_url, feature_sources.results_path_stub',
        'images.root_url, images.results_path_stub',
        'techniques.root_url, techniques.results_path_stub',
    ]


def _child_statement_reading_every_row(
    table: sqlalchemy.Table, conditions: Sequence[sqlalchemy.ColumnElement[bool]]
) -> sqlalchemy.Select[Any]:
    """Return a child statement restricted to nothing at all.

    Stands in for any way a child stream could come to carry a key the image
    stream does not yield -- a write landing between the statements, a join
    dropped, a condition stated over one statement and not the others.

    Parameters:
        table: The child table to read.
        conditions: What the merge asked to restrict the images by, ignored.

    Returns:
        The statement, ordered by the key as the real one is.
    """
    del conditions
    return sqlalchemy.select(table).order_by(table.c.root_url, table.c.results_path_stub)


def test_a_child_row_belonging_to_no_yielded_image_fails_the_read(
    two_roots: TwoRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A merge that waits gives every later image nothing, and says nothing.

    A row the image stream never yields is one the merge waits for to the end of
    the pass, handing every image after it an empty list that is
    indistinguishable from an image with no rows of its own.  So the pass has to
    end by asking whether anything is still being waited for.

    Parameters:
        two_roots: The two ingested roots and their index.
        monkeypatch: Fixture the unrestricted child read is installed through.
    """
    monkeypatch.setattr(facts_stream_module, '_child_statement', _child_statement_reading_every_row)
    with (
        open_record_source(
            [two_roots.first], results_db_url=two_roots.url, columns=COLUMNS
        ) as source,
        pytest.raises(ValueError, match='did not answer from one state'),
    ):
        list(source.facts(Selection(instrument=MISSION)))


def test_a_read_whose_child_rows_are_all_claimed_does_not_fail(two_roots: TwoRoots) -> None:
    """Without which the guard above would be free to refuse every pass.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = _from_index(two_roots, 'first', Selection(instrument=MISSION))
    assert [_stub_of(one) for one in found if isinstance(one, ImageFacts)] != []


def test_the_fixture_documents_are_what_the_ingest_read(two_roots: TwoRoots) -> None:
    """The tree half of every comparison is a tree an ingest could read.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    written = json.loads(
        (two_roots.first / f'{SUCCESS_STUB}_metadata.json').read_text(encoding='utf-8')
    )
    assert written['navigation_result']['covariance_px2'] == FIRST_VALUES.twist_covariance


# ---------------------------------------------------------------------------
# Two roots read as one stream
# ---------------------------------------------------------------------------

INTERLEAVED_FIRST = ('VOL/S1_CALIB', 'VOL/S3_CALIB')
"""What the first of the interleaved roots holds."""

INTERLEAVED_SECOND = ('VOL/S2_CALIB',)
"""What the second holds: one stub that sorts between the first root's two.

Key order and stub order therefore disagree, which is the shape a sort that lost
its root component hands back interleaved: the merge meets one image's rows
before the image it is assembling, waits, and gives that image none of its own.
"""

BOUNDARY_FIRST = ('VOL/S1_CALIB', 'VOL/S5_CALIB')
"""What the first of the boundary roots holds."""

BOUNDARY_SECOND = ('VOL/S5_CALIB', 'VOL/S6_CALIB')
"""What the second holds, beginning at the stub the first one ends on.

Two adjacent image groups sharing a stub is the shape a merge key that lost its
root half mis-pairs under: the first root's image takes the second root's rows
as well as its own, and the second root's image comes back with none.
"""


class TwoRootPairs:
    """Four results roots in one index, in two pairs, each read as one stream.

    Parameters:
        interleaved: The pair whose stubs interleave between the two roots.
        boundary: The pair where one root's last stub is the other's first.
        url: The index holding all four.
    """

    def __init__(
        self, interleaved: tuple[Path, Path], boundary: tuple[Path, Path], url: str
    ) -> None:
        self.interleaved = interleaved
        self.boundary = boundary
        self.url = url


@pytest.fixture
def two_root_pairs(tmp_path: Path) -> TwoRootPairs:
    """Write one index whose four roots make up the two shapes read below.

    Written as rows rather than ingested from trees, because what each shape
    turns on is which stubs each root holds and nothing about the documents.
    The child rows go in back to front, so insertion order pairs every image
    with the wrong one.

    Parameters:
        tmp_path: Directory the index is written under.

    Returns:
        The two pairs and the index holding all four roots.
    """
    held = {
        'boundary-a': BOUNDARY_FIRST,
        'boundary-b': BOUNDARY_SECOND,
        'interleaved-a': INTERLEAVED_FIRST,
        'interleaved-b': INTERLEAVED_SECOND,
    }
    roots = {name: tmp_path / name for name in held}
    keys = [
        (normalize_root_url(roots[name]), stub) for name, stubs in held.items() for stub in stubs
    ]
    url = index_url(tmp_path / 'pairs.sqlite3')
    engine = open_index(url, create=True)
    try:
        with engine.begin() as connection:
            connection.execute(
                IMAGES.insert(),
                [
                    image_row(root_url=root_url, results_path_stub=stub, subtree='VOL')
                    for root_url, stub in keys
                ],
            )
            connection.execute(
                TECHNIQUES.insert(),
                [
                    technique_row(
                        root_url=root_url,
                        results_path_stub=stub,
                        technique_name=_child_name(root_url, stub),
                    )
                    for root_url, stub in reversed(keys)
                ],
            )
            connection.execute(
                FEATURE_SOURCES.insert(),
                [
                    feature_source_row(
                        root_url=root_url,
                        results_path_stub=stub,
                        source_name=_child_name(root_url, stub),
                    )
                    for root_url, stub in reversed(keys)
                ],
            )
    finally:
        engine.dispose()
    return TwoRootPairs(
        (roots['interleaved-a'], roots['interleaved-b']),
        (roots['boundary-a'], roots['boundary-b']),
        url,
    )


def _children_of(url: str, roots: Sequence[Path], pick: str) -> dict[tuple[str, str], list[str]]:
    """Return the child names the merge gave each image of a stream over two roots.

    Parameters:
        url: The index to read.
        roots: The roots the source is opened over, read as one stream.
        pick: ``'techniques'`` or ``'feature_sources'``.

    Returns:
        The names, by the whole key of the image they were merged onto.
    """
    engine = open_index(url)
    with IndexRecordSource(engine, list(roots), url, COLUMNS) as source:
        found = [one for one in source.facts(Selection()) if isinstance(one, ImageFacts)]
    if pick == 'techniques':
        return {
            _key_of(one): [str(row['technique_name']) for row in one.techniques] for one in found
        }
    return {_key_of(one): [str(row['source_name']) for row in one.feature_sources] for one in found}


def _each_images_own_child(
    roots: Sequence[Path], held: Sequence[Sequence[str]]
) -> dict[tuple[str, str], list[str]]:
    """Return the one child row each image of these roots was written with.

    Parameters:
        roots: The roots, in the order their stubs are given.
        held: What each of them holds.

    Returns:
        The child name of each image, by the whole key.
    """
    return {
        (normalize_root_url(root), stub): [_child_name(normalize_root_url(root), stub)]
        for root, stubs in zip(roots, held, strict=True)
        for stub in stubs
    }


def test_two_roots_in_one_stream_give_each_image_its_own_technique_rows(
    two_root_pairs: TwoRootPairs,
) -> None:
    """The stubs interleave, so key order and stub order are two different orders.

    A sort that named the stub alone would hand the image stream and the child
    streams back interleaved differently, and an image whose rows are met out of
    turn is given none of them.

    Parameters:
        two_root_pairs: The four roots and the index holding them.
    """
    assert _children_of(
        two_root_pairs.url, two_root_pairs.interleaved, 'techniques'
    ) == _each_images_own_child(two_root_pairs.interleaved, (INTERLEAVED_FIRST, INTERLEAVED_SECOND))


def test_two_roots_in_one_stream_give_each_image_its_own_feature_source_rows(
    two_root_pairs: TwoRootPairs,
) -> None:
    """The other child table, merged onto the same stream by the same rule.

    Parameters:
        two_root_pairs: The four roots and the index holding them.
    """
    assert _children_of(
        two_root_pairs.url, two_root_pairs.interleaved, 'feature_sources'
    ) == _each_images_own_child(two_root_pairs.interleaved, (INTERLEAVED_FIRST, INTERLEAVED_SECOND))


def test_the_interleaved_roots_really_do_disagree_with_stub_order() -> None:
    """Without which the two tests above would hold whatever the sort named."""
    assert [*INTERLEAVED_FIRST, *INTERLEAVED_SECOND] != sorted(
        [*INTERLEAVED_FIRST, *INTERLEAVED_SECOND]
    )


def test_a_stub_the_next_root_begins_with_keeps_the_first_roots_rows_off_it(
    two_root_pairs: TwoRootPairs,
) -> None:
    """One root's last stub is the next root's first, so the two groups adjoin.

    A merge comparing stubs alone takes both groups of child rows for the first
    of the two images and leaves the second with none, and there is no simpler
    shape it goes wrong under.

    Parameters:
        two_root_pairs: The four roots and the index holding them.
    """
    assert _children_of(
        two_root_pairs.url, two_root_pairs.boundary, 'techniques'
    ) == _each_images_own_child(two_root_pairs.boundary, (BOUNDARY_FIRST, BOUNDARY_SECOND))


def test_the_boundary_roots_really_do_share_a_stub_where_they_meet() -> None:
    """Without which the test above would hold whatever the merge compared."""
    assert BOUNDARY_FIRST[-1] == BOUNDARY_SECOND[0]


def test_a_selection_naming_one_root_narrows_a_stream_over_two(
    two_root_pairs: TwoRootPairs,
) -> None:
    """A source is free to hold two roots and be asked about one of them.

    Parameters:
        two_root_pairs: The four roots and the index holding them.
    """
    first, _second = two_root_pairs.interleaved
    engine = open_index(two_root_pairs.url)
    held = list(two_root_pairs.interleaved)
    with IndexRecordSource(engine, held, two_root_pairs.url, COLUMNS) as source:
        found = sorted(_key_of(one) for one in source.facts(Selection(roots=(str(first),))))
    assert found == sorted((normalize_root_url(first), stub) for stub in INTERLEAVED_FIRST)


def test_named_stubs_over_a_source_holding_two_roots_are_refused(
    two_root_pairs: TwoRootPairs,
) -> None:
    """A stub is a key under a root, so two roots would answer one name twice.

    Parameters:
        two_root_pairs: The four roots and the index holding them.
    """
    engine = open_index(two_root_pairs.url)
    with (
        IndexRecordSource(
            engine, list(two_root_pairs.interleaved), two_root_pairs.url, COLUMNS
        ) as source,
        pytest.raises(ValueError, match='selection of keys'),
    ):
        source.facts(Selection(stubs=(INTERLEAVED_FIRST[0],)))
