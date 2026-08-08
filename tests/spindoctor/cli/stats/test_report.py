"""Tests for the statistics report over the results index.

The report is the one program that requires an index, and its output is a
contract: the same rows and the same options always produce byte-identical
Markdown.  These pin what each section says; the frozen comparison in
``test_report_regression`` pins that the whole of it still says what it said
before the queries moved onto the index.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pdslogger
import pytest
from filecache import FCPath
from sqlalchemy import Connection

from spindoctor.cli.stats.classify import (
    date_from_image_et,
    datetime_from_image_et,
    image_number_from_name,
)
from spindoctor.cli.stats.report import build_report, main_report
from spindoctor.cli.stats.report_common import count_pct, image_name_from_filename
from spindoctor.cli.stats.report_sections import IMAGE_COLUMNS, resolve_offset_limit
from spindoctor.dataset import DataSetPDS3CassiniISS, DataSetPDS3VoyagerISS
from spindoctor.results_index import open_index

from .conftest import index_url, ingest_tree, metadata_document, technique, write_metadata


def _indexed(
    tmp_path: Path, documents: dict[str, dict[str, Any]], logger: pdslogger.PdsLogger
) -> Iterator[Connection]:
    """Write documents into a tree, ingest them, and yield a connection.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        documents: Results path stub to the document written at it.
        logger: Logger the ingest reports through.

    Yields:
        An open connection to the index.
    """
    root = tmp_path / 'results'
    for stub, document in documents.items():
        write_metadata(root, stub, document)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=logger)
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            yield connection
    finally:
        engine.dispose()


def _standard_documents() -> dict[str, dict[str, Any]]:
    """Return three images over two volumes.

    Two Cassini successes with per-technique results, one of them with an
    ensemble exclusion, and one Voyager failure.

    Returns:
        Results path stub to document.
    """
    return {
        'COISS_2001/N1000000001_1_CALIB': metadata_document(
            image_name='N1000000001_1_CALIB.IMG',
            image_shape=[1024, 1024],
            per_technique=[
                technique('BodyDiscCorrelateNav', (1.5, -2.5)),
                technique('BodyLimbNav', (1.7, -2.3)),
            ],
        ),
        'COISS_2001/N1000000002_1_CALIB': metadata_document(
            image_name='N1000000002_1_CALIB.IMG',
            image_shape=[1024, 1024],
            confidence=0.3,
            confidence_rank='medium',
            per_technique=[technique('StarUniqueMatchNav', (0.1, 0.2))],
            excluded=['BodyBlobNav'],
        ),
        'VGISS_5101/C3250013_GEOMED': metadata_document(
            image_name='C3250013_GEOMED.IMG',
            instrument='vgiss',
            status='failed',
            status_reason='no_features_extracted',
            offset=None,
            confidence=0.0,
            confidence_rank='failed',
            image_et=-6.0e8,
            image_shape=[1000, 1000],
        ),
    }


@pytest.fixture
def standard(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> Iterator[Connection]:
    """Yield a connection to an index holding the three standard documents.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.

    Yields:
        An open connection.
    """
    yield from _indexed(tmp_path, _standard_documents(), quiet_logger)


# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------


def test_date_from_image_et_j2000() -> None:
    """The epoch itself, as the date filters compare it."""
    assert date_from_image_et(0.0) == '2000-01-01'


def test_date_from_image_et_none() -> None:
    """An image with no epoch gets no date rather than a wrong one."""
    assert date_from_image_et(None) is None


def test_datetime_from_image_et_keeps_the_time() -> None:
    """The selection table shows a time; a bare date collapses a whole day."""
    assert datetime_from_image_et(0.0) == '2000-01-01T11:58:56'


@pytest.mark.parametrize(
    ('image_name', 'expected'),
    [
        ('N1454725799_1_CALIB.IMG', 1454725799),
        ('/some/dir/W1728613298_8.IMG', 1728613298),
        ('lor_0003103486_0x630_sci.fit', 3103486),
        ('1454725799', 1454725799),
        ('no-digits-here', None),
        (None, None),
    ],
)
def test_image_number_from_name(image_name: str | None, expected: int | None) -> None:
    """The value ingest stores in the column the range filter compares.

    Parameters:
        image_name: The name to read.
        expected: The number it holds.
    """
    assert image_number_from_name(image_name) == expected


@pytest.mark.parametrize(
    ('instrument', 'filename', 'expected'),
    [
        ('coiss', 'N1454725799_1_CALIB.IMG', 'N1454725799'),
        ('coiss', '/holdings/data/W1728613298_8.IMG', 'W1728613298'),
        ('vgiss', 'C3250013_GEOMED.IMG', 'C3250013'),
        ('gossi', 'C0349632000R.IMG', 'C0349632000R'),
        ('nhlorri', 'lor_0003103486_0x630_sci.fit', 'lor_0003103486'),
        # An unregistered instrument only loses its extension.
        ('mystery', 'X9999999.IMG', 'X9999999'),
    ],
)
def test_image_name_from_filename(instrument: str, filename: str, expected: str) -> None:
    """Every printed name is the token --image-filelist selects on.

    Parameters:
        instrument: The instrument whose naming rule applies.
        filename: The recorded image name.
        expected: The dataset-level image name.
    """
    assert image_name_from_filename(instrument, filename) == expected


def test_image_name_from_filename_is_idempotent() -> None:
    """Re-deriving a name that is already an image name changes nothing."""
    assert image_name_from_filename('coiss', 'N1454725799') == 'N1454725799'


def test_count_pct_formats_share() -> None:
    """Every count in the report carries its percentage."""
    assert count_pct(5, 158) == '5 (3.2%)'


def test_count_pct_zero_total() -> None:
    """An empty denominator renders 0.0% rather than dividing by zero."""
    assert count_pct(0, 0) == '0 (0.0%)'


# ---------------------------------------------------------------------------
# Offset-limit resolution
# ---------------------------------------------------------------------------


def test_resolve_offset_limit_coiss_nac_by_size() -> None:
    """Cassini NAC CALIB limits come from the per-size margin table."""
    assert resolve_offset_limit('coiss', 'N1454725799_1_CALIB.IMG', 1024) == (50.0, 140.0)


def test_resolve_offset_limit_coiss_wac() -> None:
    """Cassini WAC limits use the wac detector block."""
    assert resolve_offset_limit('coiss', 'W1454725799_1_CALIB.IMG', 512) == (5.0, 10.0)


def test_resolve_offset_limit_requires_shape_for_size_tables() -> None:
    """A size-keyed margin table cannot resolve without a recorded shape."""
    result = resolve_offset_limit('coiss', 'N1454725799_1_CALIB.IMG', None)
    assert result == 'image shape not recorded in the database'


def test_resolve_offset_limit_unknown_instrument() -> None:
    """An unregistered instrument has no configured limit to screen against."""
    result = resolve_offset_limit('mystery', 'X123.IMG', 1024)
    assert 'no configured search limit' in str(result)


def test_resolve_offset_limit_missing_size_entry() -> None:
    """A size with no margin entry reports the failure instead of guessing."""
    result = resolve_offset_limit('vgiss', 'C3250013_GEOMED.IMG', 1024)
    assert 'no extfov_margin_vu entry for image size 1024' in str(result)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_build_report_writes_the_status_table(standard: Connection, tmp_path: Path) -> None:
    """One column per instrument, then a total, every count with a percentage."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| status | coiss | vgiss | total |' in text


def test_build_report_counts_successes(standard: Connection, tmp_path: Path) -> None:
    """An instrument column's percentage is of that instrument's images."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| success | 2 (100.0%) | 0 (0.0%) | 2 (66.7%) |' in text


def test_build_report_counts_failures(standard: Connection, tmp_path: Path) -> None:
    """A failed image is counted under its own instrument."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| failed | 0 (0.0%) | 1 (100.0%) | 1 (33.3%) |' in text


def test_build_report_names_the_failure_reason(standard: Connection, tmp_path: Path) -> None:
    """The navigator's own explanation reaches the failure-reason table."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert 'no_features_extracted' in text


def test_a_status_error_reaches_the_reason_table(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A fatal error has no status_reason, so the reason column falls through.

    The two vocabularies live in two columns, and a report of failure reasons
    wants whichever of them the document carried.
    """
    documents = {
        'COISS_2001/N1000000009_1_CALIB': metadata_document(
            image_name='N1000000009_1_CALIB.IMG',
            status='error',
            status_reason=None,
            status_error='missing_spice_data',
            offset=None,
        )
    }
    for connection in _indexed(tmp_path, documents, quiet_logger):
        text = build_report(connection, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| error | missing_spice_data | 1 (100.0%) | 1 (100.0%) |' in text


def test_a_status_reason_wins_over_a_status_error(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A document carrying both is described by the navigator's explanation."""
    documents = {
        'COISS_2001/N1000000009_1_CALIB': metadata_document(
            image_name='N1000000009_1_CALIB.IMG',
            status='failed',
            status_reason='no_features_extracted',
            status_error='missing_spice_data',
            offset=None,
        )
    }
    for connection in _indexed(tmp_path, documents, quiet_logger):
        text = build_report(connection, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| failed | no_features_extracted | 1 (100.0%) | 1 (100.0%) |' in text


def test_build_report_writes_charts(standard: Connection, tmp_path: Path) -> None:
    """Each categorical section gets a stacked bar chart beside its table."""
    out = tmp_path / 'report'
    build_report(standard, out)
    assert (out / 'status_counts.png').exists()


def test_build_report_writes_one_histogram_per_camera(standard: Connection, tmp_path: Path) -> None:
    """Offsets are never pooled across cameras, and neither are their charts."""
    out = tmp_path / 'report'
    build_report(standard, out)
    assert (out / 'offsets_hist_coiss_NAC.png').exists()


def test_build_report_selection_section(standard: Connection, tmp_path: Path) -> None:
    """The report opens with per-instrument counts and image/date bounds."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert (
        '| coiss | 2 (66.7%) | N1000000001 | N1000000002 '
        '| 2000-01-01T11:58:56 | 2000-01-01T11:58:56 |' in text
    )


def test_build_report_selection_covers_every_instrument(
    standard: Connection, tmp_path: Path
) -> None:
    """Each instrument gets a row, with its own image and date bounds."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert (
        '| vgiss | 1 (33.3%) | C3250013 | C3250013 '
        '| 1980-12-27T01:19:09 | 1980-12-27T01:19:09 |' in text
    )


def test_build_report_reports_the_total(standard: Connection, tmp_path: Path) -> None:
    """The selection section closes with what the filters actually selected."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert 'Total images: 3' in text


def test_dates_ignore_a_dateless_extreme_image(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A dateless image at the number range's edge does not hide the time span."""
    documents = _standard_documents()
    documents['COISS_2001/N0000000001_1_CALIB'] = metadata_document(
        image_name='N0000000001_1_CALIB.IMG',
        status='error',
        status_reason=None,
        status_error='missing_spice_data',
        offset=None,
        image_et=None,
    )
    for connection in _indexed(tmp_path, documents, quiet_logger):
        text = build_report(connection, tmp_path / 'report').read_text(encoding='utf-8')
    assert (
        '| coiss | 3 (75.0%) | N0000000001 | N1000000002 '
        '| 2000-01-01T11:58:56 | 2000-01-01T11:58:56 |' in text
    )


def test_offsets_separate_nac_from_wac(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """Two cameras of one instrument get their own rows."""
    documents = _standard_documents()
    documents['COISS_2001/W1000000004_1_CALIB'] = metadata_document(
        image_name='W1000000004_1_CALIB.IMG',
        camera='WAC',
        image_shape=[512, 512],
        offset=[0.4, 0.6],
    )
    for connection in _indexed(tmp_path, documents, quiet_logger):
        text = build_report(connection, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| coiss | WAC | dV | 1 (33.3%) |' in text


def test_each_camera_gets_its_own_histogram(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One Cassini WAC pixel is ten NAC pixels; a pooled chart describes neither."""
    documents = _standard_documents()
    documents['COISS_2001/W1000000004_1_CALIB'] = metadata_document(
        image_name='W1000000004_1_CALIB.IMG',
        camera='WAC',
        image_shape=[512, 512],
        offset=[0.4, 0.6],
    )
    out = tmp_path / 'report'
    for connection in _indexed(tmp_path, documents, quiet_logger):
        build_report(connection, out)
    assert (out / 'offsets_hist_coiss_WAC.png').exists()


def test_confidence_tiers_always_listed(standard: Connection, tmp_path: Path) -> None:
    """A tier with no images reads as an explicit zero, not a missing row."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    section = text.split('## Confidence calibration')[1]
    assert '| low | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |' in section


def test_confidence_tiers_are_in_tier_order(standard: Connection, tmp_path: Path) -> None:
    """Descending confidence, so the table reads as a calibration curve."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    section = text.split('## Confidence calibration')[1]
    tiers = [line.split('|')[1].strip() for line in section.splitlines() if line.startswith('| ')]
    assert tiers[:6] == ['tier', 'high', 'medium', 'low', 'failed', 'conflicted']


def test_offset_statistics_are_headed_per_camera(standard: Connection, tmp_path: Path) -> None:
    """Pointing error belongs to the camera, so the table is keyed by one."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| instrument | camera | axis | images | mean | median | stdev | min | max |' in text


def test_a_failed_instrument_contributes_no_offset_group(
    standard: Connection, tmp_path: Path
) -> None:
    """Only successful images have an offset to describe."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    section = text.split('## Offset statistics')[1].split('##')[0]
    assert '| vgiss |' not in section


def test_build_report_accepts_fcpath_output_dir(standard: Connection, tmp_path: Path) -> None:
    """Every artifact writes through FCPath, so the output may be a cloud URL."""
    out = FCPath(str(tmp_path / 'report'))
    report_path = build_report(standard, out, top_n=2, filelists=True, csv_export=True)
    assert 'Total images: 3' in report_path.read_text(encoding='utf-8')


def test_build_report_writes_the_csv_through_fcpath(standard: Connection, tmp_path: Path) -> None:
    """The CSV goes to the same place by the same route."""
    build_report(
        standard, FCPath(str(tmp_path / 'report')), top_n=2, filelists=True, csv_export=True
    )
    assert (tmp_path / 'report' / 'images.csv').exists()


def test_instrument_filter_excludes_other_instruments(standard: Connection, tmp_path: Path) -> None:
    """A filtered report has no column for an instrument it filtered out."""
    text = build_report(standard, tmp_path / 'report', instrument='vgiss').read_text(
        encoding='utf-8'
    )
    assert 'coiss' not in text


def test_instrument_filter_reports_only_that_instrument(
    standard: Connection, tmp_path: Path
) -> None:
    """And no row for an outcome only the excluded instrument had."""
    text = build_report(standard, tmp_path / 'report', instrument='vgiss').read_text(
        encoding='utf-8'
    )
    status_table = text.split('## Success / failure')[1].split('![status]')[0]
    assert '| success |' not in status_table


def test_date_filter_excludes_out_of_range(standard: Connection, tmp_path: Path) -> None:
    """The Voyager frame is two decades outside this range."""
    text = build_report(
        standard, tmp_path / 'report', start_date='1999-01-01', end_date='2001-01-01'
    ).read_text(encoding='utf-8')
    assert 'Total images: 2' in text


def test_min_image_filter(standard: Connection, tmp_path: Path) -> None:
    """The range filter compares the ingested column, on any backend."""
    text = build_report(standard, tmp_path / 'report', min_image='N1000000002').read_text(
        encoding='utf-8'
    )
    assert 'Total images: 1' in text


def test_min_image_filter_is_reported(standard: Connection, tmp_path: Path) -> None:
    """A report says what it was filtered by, so a stale one is recognizable."""
    text = build_report(standard, tmp_path / 'report', min_image='N1000000002').read_text(
        encoding='utf-8'
    )
    assert 'image number >= 1000000002' in text


def test_max_image_filter(standard: Connection, tmp_path: Path) -> None:
    """Only the Voyager frame falls at or below this number."""
    text = build_report(standard, tmp_path / 'report', max_image='5000000').read_text(
        encoding='utf-8'
    )
    assert 'Total images: 1' in text


def test_root_filter_selects_one_root(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A report may span roots, so it can also be asked for one."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_metadata(first, 'VOL/N1000000001_1_CALIB', metadata_document())
    write_metadata(
        second,
        'VOL/N1000000002_1_CALIB',
        metadata_document(image_name='N1000000002_1_CALIB.IMG'),
    )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first, second], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        text = build_report(connection, tmp_path / 'report', roots=[first.as_posix()]).read_text(
            encoding='utf-8'
        )
    engine.dispose()
    assert 'Total images: 1' in text


def test_a_report_over_every_root_sees_every_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Naming no root reports over the whole index."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_metadata(first, 'VOL/N1000000001_1_CALIB', metadata_document())
    write_metadata(
        second,
        'VOL/N1000000002_1_CALIB',
        metadata_document(image_name='N1000000002_1_CALIB.IMG'),
    )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first, second], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        text = build_report(connection, tmp_path / 'report').read_text(encoding='utf-8')
    engine.dispose()
    assert 'Total images: 2' in text


def test_a_named_root_is_reported_as_a_filter(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A root-restricted report says which root it covered."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1000000001_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        text = build_report(connection, tmp_path / 'report', roots=[root.as_posix()]).read_text(
            encoding='utf-8'
        )
    engine.dispose()
    assert f'root in {root.as_posix()}' in text


def test_build_report_rejects_digitless_image_bound(standard: Connection, tmp_path: Path) -> None:
    """A bound with no digits names nothing and is refused rather than ignored."""
    with pytest.raises(ValueError, match='contains no digits'):
        build_report(standard, tmp_path / 'report', min_image='nodigits')


def test_failure_taxonomy_classifies_by_content(standard: Connection, tmp_path: Path) -> None:
    """The failed Voyager frame recorded one body, so it is single-body."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| single-body | 0 (0.0%) | 1 (100.0%) | 1 (33.3%) |' in text


def test_failure_taxonomy_breaks_down_by_reason(standard: Connection, tmp_path: Path) -> None:
    """Each content category carries the reasons its images failed for."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| single-body | no_features_extracted | 0 (0.0%) | 1 (100.0%) | 1 (33.3%) |' in text


def test_per_body_failure_share_counts_failures(standard: Connection, tmp_path: Path) -> None:
    """A body with a high failure share points at a modeling problem."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| IAPETUS | vgiss | 1 (100.0%) | 0 (0.0%) | 1.000 |' in text


def test_per_body_failure_share_counts_successes(standard: Connection, tmp_path: Path) -> None:
    """The same body on the instrument where it worked."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| IAPETUS | coiss | 0 (0.0%) | 2 (100.0%) | 0.000 |' in text


def test_offset_by_group_section(standard: Connection, tmp_path: Path) -> None:
    """Offsets additionally break down by camera and image size."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| coiss | NAC | 1024x1024 | 2 (100.0%) ' in text


def test_technique_usage_counts_images(standard: Connection, tmp_path: Path) -> None:
    """Each technique is credited with the images it ran on."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| BodyDiscCorrelateNav | coiss | 1 (50.0%) | 1 (100.0%) | 0.700 |' in text


def test_technique_usage_counts_non_spurious_runs(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A spurious result is counted as run but not as good.

    The share is computed from a boolean column, which a backend with a native
    boolean type will not do arithmetic on.
    """
    documents = {
        'COISS_2001/N1000000001_1_CALIB': metadata_document(
            image_name='N1000000001_1_CALIB.IMG',
            per_technique=[technique('BodyLimbNav', (1.0, 1.0), spurious=True)],
        ),
        'COISS_2001/N1000000002_1_CALIB': metadata_document(
            image_name='N1000000002_1_CALIB.IMG',
            per_technique=[technique('BodyLimbNav', (1.0, 1.0))],
        ),
    }
    for connection in _indexed(tmp_path, documents, quiet_logger):
        text = build_report(connection, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| BodyLimbNav | coiss | 2 (100.0%) | 1 (50.0%) | 0.700 |' in text


def test_a_spurious_technique_is_left_out_of_the_agreement(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Cross-technique agreement compares results the techniques stood behind."""
    documents = {
        'COISS_2001/N1000000001_1_CALIB': metadata_document(
            image_name='N1000000001_1_CALIB.IMG',
            per_technique=[
                technique('BodyLimbNav', (1.0, 1.0)),
                technique('StarUniqueMatchNav', (40.0, 40.0), spurious=True),
            ],
        )
    }
    for connection in _indexed(tmp_path, documents, quiet_logger):
        text = build_report(connection, tmp_path / 'report').read_text(encoding='utf-8')
    assert 'BodyLimbNav vs StarUniqueMatchNav' not in text


def test_source_usage_counts_images_once_per_source(standard: Connection, tmp_path: Path) -> None:
    """An image contributing several feature types of one source counts once."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| body:IAPETUS | IAPETUS | 2 (100.0%) | 1 (100.0%) | 3 (100.0%) |' in text


def test_suspect_offset_section_flags_an_offset_near_the_limit(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An offset pinned near the search boundary may be a correlation artifact."""
    documents = _standard_documents()
    # The NAC 1024 CALIB margin is (50, 140); |dV| = 49 is 0.98 of the limit.
    documents['COISS_2001/N1000000003_1_CALIB'] = metadata_document(
        image_name='N1000000003_1_CALIB.IMG', image_shape=[1024, 1024], offset=[49.0, 10.0]
    )
    for connection in _indexed(tmp_path, documents, quiet_logger):
        text = build_report(connection, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| N1000000003 | coiss | 49.000 | 10.000 |' in text


def test_suspect_offset_section_counts_what_it_screened(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A share is only meaningful beside how many images it could screen."""
    documents = _standard_documents()
    documents['COISS_2001/N1000000003_1_CALIB'] = metadata_document(
        image_name='N1000000003_1_CALIB.IMG', image_shape=[1024, 1024], offset=[49.0, 10.0]
    )
    for connection in _indexed(tmp_path, documents, quiet_logger):
        text = build_report(connection, tmp_path / 'report').read_text(encoding='utf-8')
    assert 'Suspect images: 1 (25.0%) of 3 screened.' in text


def test_suspect_offset_section_reports_unresolved_limits(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An image whose limit cannot be resolved is called out, not dropped."""
    documents = {'X9999999': metadata_document(image_name='X9999999.IMG', instrument='mystery')}
    for connection in _indexed(tmp_path, documents, quiet_logger):
        text = build_report(connection, tmp_path / 'report').read_text(encoding='utf-8')
    assert "mystery: no configured search limit for instrument 'mystery' (1 image(s))" in text


def _botsim_documents() -> dict[str, dict[str, Any]]:
    """Return three BOTSIM pairs: consistent, inconsistent, and half-navigated.

    Returns:
        Results path stub to document.
    """
    documents: dict[str, dict[str, Any]] = {}
    for name, offset in (
        ('N1454725799_1_CALIB', [10.0, -20.0]),
        ('W1454725799_1_CALIB', [1.0, -2.0]),
        ('N1454725900_1_CALIB', [12.0, -20.0]),
        ('W1454725900_1_CALIB', [1.0, -2.0]),
        ('N1454726000_1_CALIB', [1.0, 1.0]),
    ):
        documents[f'COISS_2001/{name}'] = metadata_document(
            image_name=f'{name}.IMG', image_shape=[1024, 1024], offset=offset
        )
    documents['COISS_2001/W1454726000_1_CALIB'] = metadata_document(
        image_name='W1454726000_1_CALIB.IMG',
        status='failed',
        status_reason='no_features_extracted',
        offset=None,
        confidence=0.0,
        confidence_rank='failed',
    )
    return documents


def test_botsim_section_identifies_pairs(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A pair is two frames sharing one spacecraft-clock count."""
    for connection in _indexed(tmp_path, _botsim_documents(), quiet_logger):
        text = build_report(connection, tmp_path / 'report', top_n=5).read_text(encoding='utf-8')
    assert '| pairs identified | 3 |' in text


def test_botsim_section_compares_only_navigated_pairs(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A pair whose WAC frame failed is identified but not compared."""
    for connection in _indexed(tmp_path, _botsim_documents(), quiet_logger):
        text = build_report(connection, tmp_path / 'report', top_n=5).read_text(encoding='utf-8')
    assert '| pairs with both navigated | 2 |' in text


def test_botsim_section_reports_the_residual(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Residuals are 0.0 and 2.0, so the median is 1.0."""
    for connection in _indexed(tmp_path, _botsim_documents(), quiet_logger):
        text = build_report(connection, tmp_path / 'report', top_n=5).read_text(encoding='utf-8')
    assert '| median residual (px) | 1.000 |' in text


def test_botsim_section_names_the_worst_pair(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The worst-pairs table leads with the inconsistent pair, by image name."""
    for connection in _indexed(tmp_path, _botsim_documents(), quiet_logger):
        text = build_report(connection, tmp_path / 'report', top_n=5).read_text(encoding='utf-8')
    assert '| 1454725900 | N1454725900 | W1454725900 | 2.000 | 0.000 | 2.000 |' in text


def test_runtime_section_per_instrument(standard: Connection, tmp_path: Path) -> None:
    """Per-instrument run times, as a share of that instrument's images."""
    text = build_report(standard, tmp_path / 'report', top_n=2).read_text(encoding='utf-8')
    assert '| coiss | 2 (100.0%) | 6.500 | 3.250 | 3.250 | 3.250 | 3.250 | 0.000 |' in text


def test_runtime_section_pools_when_more_than_one_instrument(
    standard: Connection, tmp_path: Path
) -> None:
    """The pooled row says something new only once two instruments fed it."""
    text = build_report(standard, tmp_path / 'report', top_n=2).read_text(encoding='utf-8')
    assert '| (all) | 3 (100.0%) | 9.750 | 3.250 | 3.250 | 3.250 | 3.250 | 0.000 |' in text


def test_runtime_section_lists_the_slowest(standard: Connection, tmp_path: Path) -> None:
    """With a drill-down count, the slowest images are named."""
    text = build_report(standard, tmp_path / 'report', top_n=2).read_text(encoding='utf-8')
    assert 'Slowest 2 image(s):' in text


def test_runtime_section_skipped_without_timing(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A tree whose documents carry no timing gets no run-time section."""
    documents = {'COISS_2001/N1454725799_1_CALIB': metadata_document(elapsed_s=None)}
    for connection in _indexed(tmp_path, documents, quiet_logger):
        text = build_report(connection, tmp_path / 'report').read_text(encoding='utf-8')
    assert '## Run-time statistics' not in text


def test_top_n_lists_examples(standard: Connection, tmp_path: Path) -> None:
    """Examples are grouped by instrument and printed as image names."""
    text = build_report(standard, tmp_path / 'report', top_n=5).read_text(encoding='utf-8')
    assert '- no_features_extracted / vgiss: C3250013' in text


def test_top_n_lists_exclusion_examples(standard: Connection, tmp_path: Path) -> None:
    """The ensemble-exclusion section drills down the same way."""
    text = build_report(standard, tmp_path / 'report', top_n=5).read_text(encoding='utf-8')
    assert '- BodyBlobNav / coiss: N1000000002' in text


def test_the_exclusion_section_counts_images(standard: Connection, tmp_path: Path) -> None:
    """The exclusion set is a JSON column, and the empty set is not a category."""
    text = build_report(standard, tmp_path / 'report').read_text(encoding='utf-8')
    assert '| BodyBlobNav | 1 (50.0%) | 0 (0.0%) | 1 (33.3%) |' in text


def test_no_exclusions_means_no_exclusion_section(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An index where the ensemble excluded nothing has nothing to report."""
    documents = {'COISS_2001/N1454725799_1_CALIB': metadata_document()}
    for connection in _indexed(tmp_path, documents, quiet_logger):
        text = build_report(connection, tmp_path / 'report').read_text(encoding='utf-8')
    assert '## Ensemble outlier exclusions' not in text


def test_filelists_written(standard: Connection, tmp_path: Path) -> None:
    """One full image-name list per category and instrument."""
    out = tmp_path / 'report'
    build_report(standard, out, filelists=True)
    written = out / 'filelists' / 'failure_reason_no_features_extracted_vgiss.txt'
    assert written.read_text(encoding='utf-8') == (
        '# failure_reason_no_features_extracted_vgiss (1 image(s))\nC3250013\n'
    )


def test_filelists_cover_the_exclusion_section(standard: Connection, tmp_path: Path) -> None:
    """Every drill-down section writes its lists, not only the first."""
    out = tmp_path / 'report'
    build_report(standard, out, filelists=True)
    written = out / 'filelists' / 'excluded_BodyBlobNav_coiss.txt'
    assert written.read_text(encoding='utf-8') == (
        '# excluded_BodyBlobNav_coiss (1 image(s))\nN1000000002\n'
    )


def test_filelists_are_referenced_from_the_report(standard: Connection, tmp_path: Path) -> None:
    """A list nobody is pointed at is a list nobody reads."""
    text = build_report(standard, tmp_path / 'report', filelists=True).read_text(encoding='utf-8')
    assert 'filelists/failure_reason_no_features_extracted_vgiss.txt' in text


def test_filelists_are_image_filelist_readable(standard: Connection, tmp_path: Path) -> None:
    """Every filelist line is a comment or a name the dataset layer accepts."""
    out = tmp_path / 'report'
    build_report(standard, out, filelists=True)
    validators = {
        'coiss': DataSetPDS3CassiniISS._img_name_valid,
        'vgiss': DataSetPDS3VoyagerISS._img_name_valid,
    }
    invalid = [
        f'{path.name}: {line!r}'
        for path in sorted((out / 'filelists').glob('*.txt'))
        for line in path.read_text(encoding='utf-8').splitlines()
        if not line.startswith('#')
        if not validators[path.stem.rsplit('_', 1)[-1]](line)
    ]
    assert invalid == []


def test_csv_export_is_announced(standard: Connection, tmp_path: Path) -> None:
    """The report says the CSV exists and how many rows it holds."""
    text = build_report(standard, tmp_path / 'report', csv_export=True).read_text(encoding='utf-8')
    assert 'images.csv (3 row(s))' in text


def test_csv_export_writes_one_row_per_image(standard: Connection, tmp_path: Path) -> None:
    """Three images, plus the header."""
    out = tmp_path / 'report'
    build_report(standard, out, csv_export=True)
    assert len((out / 'images.csv').read_text(encoding='utf-8').splitlines()) == 4


def test_csv_export_carries_every_column_in_schema_order(
    standard: Connection, tmp_path: Path
) -> None:
    """The export is the whole row, so a question the report skips is answerable."""
    out = tmp_path / 'report'
    build_report(standard, out, csv_export=True)
    header = (out / 'images.csv').read_text(encoding='utf-8').splitlines()[0].split(',')
    assert header[: len(IMAGE_COLUMNS)] == list(IMAGE_COLUMNS)


def test_csv_export_starts_at_the_key_columns(standard: Connection, tmp_path: Path) -> None:
    """A row that does not say which image it is cannot be joined to anything."""
    assert IMAGE_COLUMNS[:2] == ('root_url', 'results_path_stub')


def test_csv_export_ends_with_the_aggregates(standard: Connection, tmp_path: Path) -> None:
    """The per-image counts the row itself does not carry."""
    out = tmp_path / 'report'
    build_report(standard, out, csv_export=True)
    header = (out / 'images.csv').read_text(encoding='utf-8').splitlines()[0].split(',')
    assert header[-4:] == ['n_technique_rows', 'n_feature_sources', 'n_features', 'n_gated']


def test_csv_export_counts_features_as_zero_when_there_are_none(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An image with no inventory exports a zero rather than an empty cell."""
    document = metadata_document()
    document['navigation_result']['feature_inventory'] = []
    for connection in _indexed(tmp_path, {'VOL/N1454725799_1_CALIB': document}, quiet_logger):
        build_report(connection, tmp_path / 'report', csv_export=True)
    row = (tmp_path / 'report' / 'images.csv').read_text(encoding='utf-8').splitlines()[1]
    assert row.endswith(',0,0,0,0')


def test_report_is_deterministic(standard: Connection, tmp_path: Path) -> None:
    """The same index and options twice produce byte-identical Markdown."""
    first = build_report(
        standard, tmp_path / 'a', top_n=3, filelists=True, csv_export=True
    ).read_text(encoding='utf-8')
    second = build_report(
        standard, tmp_path / 'b', top_n=3, filelists=True, csv_export=True
    ).read_text(encoding='utf-8')
    assert first == second


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_main_report_writes_a_report(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver opens the index it was named and writes the report."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    out = tmp_path / 'report'
    exit_code = main_report(['--results-db', url, '--output-dir', str(out)])
    assert exit_code == 0


def test_main_report_accepts_the_drill_down_flags(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The range, suspect, and CSV flags parse and take effect."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    out = tmp_path / 'report'
    main_report(
        [
            '--results-db',
            url,
            '--output-dir',
            str(out),
            '--top-n',
            '3',
            '--filelists',
            '--csv',
            '--suspect-fraction',
            '0.8',
            '--min-image',
            '1',
        ]
    )
    text = (out / 'report.md').read_text(encoding='utf-8')
    assert 'at least 0.80 of the per-axis maximum expected pointing' in text


def test_main_report_accepts_a_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The root is normalized the way ingest normalized it, so it matches."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    out = tmp_path / 'report'
    exit_code = main_report(
        ['--results-db', url, '--root', f'{root.as_posix()}/', '--output-dir', str(out)]
    )
    assert exit_code == 0


def test_main_report_refuses_a_root_nobody_ingested(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absence of rows under a root is not evidence that nothing was navigated."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    with pytest.raises(SystemExit):
        main_report(
            [
                '--results-db',
                url,
                '--root',
                str(tmp_path / 'never-ingested'),
                '--output-dir',
                str(tmp_path / 'report'),
            ]
        )


def test_main_report_names_the_roots_it_does_hold(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """The message has to be actionable, so it says what the index does cover."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    with pytest.raises(SystemExit):
        main_report(
            [
                '--results-db',
                url,
                '--root',
                str(tmp_path / 'never-ingested'),
                '--output-dir',
                str(tmp_path / 'report'),
            ]
        )
    assert root.as_posix() in capsys.readouterr().err


def test_main_report_without_an_index_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """This program has no file-reading mode, and the message says which flag."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    exit_code = main_report(['--output-dir', str(tmp_path / 'report')])
    assert exit_code == 1


def test_main_report_without_an_index_names_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """A refusal that does not say what to type is a refusal nobody can act on."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    main_report(['--output-dir', str(tmp_path / 'report')])
    assert '--results-db' in capsys.readouterr().err


def test_main_report_refuses_an_index_that_is_not_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A consumer never creates an index; it reports that there is none."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    missing = tmp_path / 'absent.sqlite3'
    exit_code = main_report(
        ['--results-db', index_url(missing), '--output-dir', str(tmp_path / 'report')]
    )
    assert exit_code == 1


def test_main_report_leaves_no_database_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty database would answer every question with "not navigated"."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    missing = tmp_path / 'absent.sqlite3'
    main_report(['--results-db', index_url(missing), '--output-dir', str(tmp_path / 'report')])
    assert not missing.exists()


def test_main_report_honors_the_none_sentinel(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exported index URL can be overridden on the command line."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    monkeypatch.setenv('NAV_RESULTS_DB', url)
    exit_code = main_report(['--results-db', 'none', '--output-dir', str(tmp_path / 'report')])
    assert exit_code == 1


def test_main_report_reads_the_environment_variable(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine with one index need not name it on every invocation."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    monkeypatch.setenv('NAV_RESULTS_DB', url)
    exit_code = main_report(['--output-dir', str(tmp_path / 'report')])
    assert exit_code == 0
