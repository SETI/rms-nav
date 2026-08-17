"""Reading the document a navigation record is written to.

Every reader of a results tree resolves a stub to a path, reads a document and
decides what an unreadable one means through this module, whichever program it
belongs to and whether or not that program can read a results index.  So the
rules are tested once, here, rather than once per reader: which paths a root may
be read at, what makes a file a document, and which unreadable file is reported
against a mission rather than passed over as another mission's.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from spindoctor.support.nav_document import (
    ABSOLUTE_PATH_FRAGMENT,
    METADATA_SUFFIX,
    NULL_BYTE_IN_PATH,
    PATH_OUTSIDE_ROOT,
    read_document,
    read_documents,
    resolved_document_path,
    stub_for_document,
)


def test_a_metadata_file_that_is_not_json_is_counted(tmp_path: Path) -> None:
    """It names no image, so it is reported rather than given a report row."""
    (tmp_path / f'broken{METADATA_SUFFIX}').write_text('{not json')
    records, unreadable = read_documents(FCPath(str(tmp_path)), 'coiss')
    assert records == []
    assert len(unreadable) == 1


def test_a_metadata_file_holding_a_json_array_is_counted(tmp_path: Path) -> None:
    """Valid JSON that is not a document is unreadable for the same reason."""
    (tmp_path / f'listy{METADATA_SUFFIX}').write_text('[1, 2]')
    _records, unreadable = read_documents(FCPath(str(tmp_path)), 'coiss')
    assert len(unreadable) == 1


def test_a_document_from_another_mission_is_not_considered(tmp_path: Path) -> None:
    """A run is per mission, and another mission's images are not its business."""
    (tmp_path / f'other{METADATA_SUFFIX}').write_text(
        json.dumps({'status': 'success', 'observation': {'instrument': 'vgiss'}})
    )
    records, unreadable = read_documents(FCPath(str(tmp_path)), 'coiss')
    assert records == []
    assert len(unreadable) == 0


@pytest.mark.parametrize(
    'document',
    [
        {'status': 'error'},
        {'status': 'error', 'observation': 'later'},
        {'status': 'error', 'observation': {'image_name': 'A_CALIB'}},
        {'status': 'error', 'observation': {'instrument': None}},
    ],
    ids=['no-observation', 'observation-not-a-block', 'no-instrument', 'instrument-null'],
)
def test_a_document_naming_no_instrument_is_counted_as_unreadable(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    """Only a document that names a mission can be another mission's.

    One with no readable instrument is unreadable, not foreign: skipping it
    silently would let a truncated document vanish from every mission's run
    without a trace.

    Parameters:
        document: A JSON object whose observation names no instrument.
    """
    (tmp_path / f'mute{METADATA_SUFFIX}').write_text(json.dumps(document))
    records, unreadable = read_documents(FCPath(str(tmp_path)), 'coiss')
    assert records == []
    assert len(unreadable) == 1
    assert unreadable[0][1] == 'names no instrument to attribute it to a mission'


def test_a_stub_resolves_to_its_document_under_the_root(tmp_path: Path) -> None:
    """The ordinary case: a stub is a path under the root with the suffix back on."""
    resolved = resolved_document_path(tmp_path, 'COISS_2001/N1454725799')
    assert resolved.path is not None
    assert resolved.path.as_posix().endswith(f'COISS_2001/N1454725799{METADATA_SUFFIX}')


def test_a_resolved_document_names_no_refusal(tmp_path: Path) -> None:
    """A stub the root may be read at breaks no rule, and says so."""
    assert resolved_document_path(tmp_path, 'COISS_2001/N1454725799').refusal is None


def test_a_stub_carrying_a_null_byte_is_refused(tmp_path: Path) -> None:
    """It would resolve perfectly well and then fail at the filesystem."""
    assert resolved_document_path(tmp_path, 'COISS_2001/N145\x004725799').refusal == (
        NULL_BYTE_IN_PATH
    )


def test_an_absolute_stub_is_refused(tmp_path: Path) -> None:
    """An absolute fragment names a file under no root at all."""
    assert resolved_document_path(tmp_path, '/etc/passwd').refusal == ABSOLUTE_PATH_FRAGMENT


def test_a_stub_that_escapes_its_root_is_refused(tmp_path: Path) -> None:
    """A key is not a path, and a key holding ``..`` is a file outside the root."""
    assert resolved_document_path(tmp_path, '../../elsewhere/N1454725799').refusal == (
        PATH_OUTSIDE_ROOT
    )


def test_an_escaping_stub_reports_what_it_resolved_to(tmp_path: Path) -> None:
    """The report has to show the path, so the resolution travels with the refusal."""
    resolved = resolved_document_path(tmp_path, '../../elsewhere/N1454725799')
    assert resolved.resolved is not None
    assert resolved.resolved.as_posix().endswith(f'elsewhere/N1454725799{METADATA_SUFFIX}')


def test_a_refused_stub_carries_no_path(tmp_path: Path) -> None:
    """A caller reads the path, so a refusal must not hand back a usable one."""
    assert resolved_document_path(tmp_path, '/etc/passwd').path is None


def test_a_documents_stub_is_its_path_under_the_root(tmp_path: Path) -> None:
    """The stub and the path are two spellings of one identity, and must agree.

    This is the inverse of the resolution above: the walk turns a path into a
    stub, an index row records that stub, and a reader turns it back into a path.
    A disagreement between the two would make an ingested image unfindable by the
    key it was ingested under.
    """
    root = FCPath(str(tmp_path))
    document = root / 'COISS_2001' / f'N1454725799{METADATA_SUFFIX}'
    stub = stub_for_document(root, document)
    assert stub == 'COISS_2001/N1454725799'


def test_a_stub_round_trips_through_the_path_and_back(tmp_path: Path) -> None:
    """Resolving a stub and taking the stub of the result returns the stub."""
    root = FCPath(str(tmp_path))
    resolved = resolved_document_path(root, 'COISS_2001/N1454725799')
    assert resolved.path is not None
    assert stub_for_document(root.expanduser().resolve(), resolved.path) == (
        'COISS_2001/N1454725799'
    )


def test_a_document_is_read_as_the_object_it_holds(tmp_path: Path) -> None:
    """The ordinary case, which the refusals below are the exceptions to."""
    path = tmp_path / f'N1454725799{METADATA_SUFFIX}'
    path.write_text(json.dumps({'status': 'success'}))
    assert read_document(FCPath(str(path))) == {'status': 'success'}


def test_a_file_that_is_not_there_raises_the_error_a_caller_distinguishes(
    tmp_path: Path,
) -> None:
    """An unnavigated image and an unreadable document are reported differently."""
    with pytest.raises(FileNotFoundError):
        read_document(FCPath(str(tmp_path / f'absent{METADATA_SUFFIX}')))
