"""Hermetic tests for ``spindoctor.cli.ck.report``.

These pin which metadata field each report column comes from, that an image
appears in the report exactly once with either a source file or an omission
reason, and that a malformed value is refused rather than rendered.  The only
kernel any of them needs is the leapseconds kernel the UTC column is converted
with.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from filecache import FCPath
from tests.spindoctor.cli.ck.conftest import (
    CASSINI_CAMERA_FRAME,
    CASSINI_CK_FRAME_ID,
    ET0,
    KernelPool,
    axis_rotation,
    image_metadata,
)

from spindoctor.cli.ck.images import OmissionReason
from spindoctor.cli.ck.report import (
    REPORT_COLUMNS,
    ImageFacts,
    ReportRow,
    read_image_facts,
    report_text,
    utc_for_et,
    write_report,
)

_CORRECTED = axis_rotation(np.array([0.1, -0.7, 0.3]), 1.1)
_UNCORRECTED = axis_rotation(np.array([0.1, -0.7, 0.3]), 1.2)

# Values chosen so that no two columns hold the same number: a row that read
# sigma where it meant offset, or du where it meant dv, would still be wrong.
_OFFSET = (-3.25, 1.125)
_SIGMA = (0.0625, 0.03125)
_CONFIDENCE = 0.8125
_SCLK = '1/1484573295.118'


def _metadata(**overrides: Any) -> dict[str, Any]:
    """Build a fully-populated navigated image's metadata, with fields replaced.

    Parameters:
        overrides: Keyword arguments passed through to the metadata builder.

    Returns:
        The metadata dict.
    """
    defaults: dict[str, Any] = {
        'image_name': 'N1484573295_1.IMG',
        'cmatrix': _CORRECTED,
        'cmatrix_original': _UNCORRECTED,
        'camera_frame': CASSINI_CAMERA_FRAME,
        'ck_frame_id': CASSINI_CK_FRAME_ID,
        'start_et': ET0,
        'stop_et': ET0 + 2.0,
        'sclk_midtime': _SCLK,
        'offset': _OFFSET,
        'sigma_px': _SIGMA,
        'confidence': _CONFIDENCE,
        'confidence_rank': 'high',
        'status_reason': 'ensemble_agreement',
    }
    defaults.update(overrides)
    return image_metadata(**defaults)


def _facts(pool: KernelPool, **overrides: Any) -> ImageFacts:
    """Read the report facts of an image's metadata, with fields replaced.

    Parameters:
        pool: The furnished kernel pool, needed for the UTC conversion.
        overrides: Keyword arguments passed through to the metadata builder.

    Returns:
        The facts.
    """
    assert pool is not None
    return read_image_facts(_metadata(**overrides))


def _row(facts: ImageFacts) -> dict[str, str]:
    """Return one row's cells keyed by column name.

    Parameters:
        facts: The image's facts.

    Returns:
        The cells of a row that names a source file.
    """
    row = ReportRow(facts=facts, source_bc='03236_04002ra_nav.bc', omission_reason=None)
    return dict(zip(REPORT_COLUMNS, row.values(), strict=True))


# ---------------------------------------------------------------------------
# Where each column comes from
# ---------------------------------------------------------------------------


def test_the_offset_columns_come_from_the_top_level_offset(pool: KernelPool) -> None:
    """The offset pair is the unrounded top-level ``[dv, du]``."""
    facts = _facts(pool)
    assert facts.offset_dv == pytest.approx(_OFFSET[0])
    assert facts.offset_du == pytest.approx(_OFFSET[1])


def test_the_sigma_columns_come_from_the_navigation_result(pool: KernelPool) -> None:
    """The sigma pair is the ``navigation_result`` one, reported as recorded."""
    facts = _facts(pool)
    assert facts.sigma_dv == pytest.approx(_SIGMA[0])
    assert facts.sigma_du == pytest.approx(_SIGMA[1])


def test_the_confidence_column_comes_from_the_top_level(pool: KernelPool) -> None:
    """Confidence is the top-level value, not a per-technique one."""
    assert _facts(pool).confidence == pytest.approx(_CONFIDENCE)


def test_the_rank_and_status_reason_come_from_the_navigation_result(pool: KernelPool) -> None:
    """Both live under ``navigation_result``, unlike the status beside them."""
    facts = _facts(pool)
    assert facts.confidence_rank == 'high'
    assert facts.status_reason == 'ensemble_agreement'


def test_the_status_column_comes_from_the_top_level(pool: KernelPool) -> None:
    """The status is the top-level one."""
    assert _facts(pool, status='conflicted').status == 'conflicted'


def test_the_clock_column_is_the_recorded_midtime_string(pool: KernelPool) -> None:
    """The clock column is ``times.sclk_midtime``, not one computed here."""
    assert _facts(pool).sclk == _SCLK


def test_the_epoch_column_is_the_exposure_midtime(pool: KernelPool) -> None:
    """The epoch column is the midtime, so all three time columns agree."""
    assert _facts(pool).et == pytest.approx(ET0 + 1.0)


def test_the_utc_column_is_that_same_epoch(pool: KernelPool) -> None:
    """The UTC column is the midtime converted, not the start."""
    assert _facts(pool).utc == utc_for_et(ET0 + 1.0)


def test_the_image_name_comes_from_the_observation(pool: KernelPool) -> None:
    """The name is the observation's, which is what the segment is named."""
    assert _facts(pool).image_name == 'N1484573295_1.IMG'


# ---------------------------------------------------------------------------
# Images that measured less than everything
# ---------------------------------------------------------------------------


def test_a_load_error_document_reports_a_name_and_a_status(pool: KernelPool) -> None:
    """An image with no navigation result still gets its row."""
    assert pool is not None
    facts = read_image_facts(
        {'status': 'failed', 'observation': {'image_name': 'N1484573295_1.IMG'}}
    )
    assert facts.image_name == 'N1484573295_1.IMG'
    assert facts.status == 'failed'


def test_a_load_error_document_reports_no_measurement(pool: KernelPool) -> None:
    """Nothing is invented for an image that measured nothing."""
    assert pool is not None
    facts = read_image_facts(
        {'status': 'failed', 'observation': {'image_name': 'N1484573295_1.IMG'}}
    )
    assert facts.et is None
    assert facts.offset_dv is None
    assert facts.confidence is None
    assert facts.status_reason is None


def test_a_result_with_no_offset_reports_none(pool: KernelPool) -> None:
    """A failed navigation records no offset, and none is reported."""
    facts = _facts(pool, offset=None)
    assert facts.offset_dv is None
    assert facts.offset_du is None


def test_a_null_sigma_is_read_as_not_recorded(pool: KernelPool) -> None:
    """The pipeline writes ``null`` for a sigma it has none of."""
    metadata = _metadata()
    metadata['navigation_result']['sigma_px'] = None
    facts = read_image_facts(metadata)
    assert facts.sigma_dv is None
    assert facts.sigma_du is None


def test_an_unrecorded_value_is_an_empty_cell(pool: KernelPool) -> None:
    """An empty cell means not measured, which zero would not."""
    cells = _row(_facts(pool, offset=None))
    assert cells['offset_dv'] == ''
    assert cells['offset_du'] == ''


def test_a_measured_zero_is_not_an_empty_cell(pool: KernelPool) -> None:
    """A navigated offset of zero is a measurement and reads as one."""
    cells = _row(_facts(pool, offset=(0.0, 0.0)))
    assert cells['offset_dv'] == '0.0'


# ---------------------------------------------------------------------------
# Malformed values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'value', [float('nan'), float('inf'), float('-inf')], ids=['nan', 'inf', 'minus-inf']
)
def test_a_non_finite_confidence_is_refused(pool: KernelPool, value: float) -> None:
    """A NaN confidence would be reported as a number no reader can attribute."""
    assert pool is not None
    with pytest.raises(ValueError, match='not finite'):
        read_image_facts(_metadata(confidence=value))


@pytest.mark.parametrize('value', [float('nan'), float('inf')], ids=['nan', 'inf'])
def test_a_non_finite_offset_element_is_refused(pool: KernelPool, value: float) -> None:
    """Nor is a non-finite offset rendered into the report."""
    assert pool is not None
    with pytest.raises(ValueError, match='non-finite'):
        read_image_facts(_metadata(offset=(value, 1.0)))


def test_a_non_finite_midtime_is_refused(pool: KernelPool) -> None:
    """A non-finite midtime would reach the UTC conversion."""
    assert pool is not None
    with pytest.raises(ValueError, match='not finite'):
        read_image_facts(_metadata(midtime_et=float('nan')))


def test_an_offset_of_one_element_is_refused(pool: KernelPool) -> None:
    """An offset is a pair; one number leaves an axis unaccounted for."""
    assert pool is not None
    with pytest.raises(ValueError, match='not two'):
        read_image_facts(_metadata(offset=(1.0,)))


def test_an_offset_of_three_elements_is_refused(pool: KernelPool) -> None:
    """So does a third number, which no column would carry."""
    assert pool is not None
    with pytest.raises(ValueError, match='not two'):
        read_image_facts(_metadata(offset=(1.0, 2.0, 3.0)))


def test_an_offset_that_is_not_a_sequence_is_refused(pool: KernelPool) -> None:
    """A bare number where a pair belongs is a malformed document."""
    assert pool is not None
    metadata = _metadata()
    metadata['offset'] = 1.0
    with pytest.raises(TypeError, match='not a pair'):
        read_image_facts(metadata)


def test_an_offset_of_text_is_refused(pool: KernelPool) -> None:
    """Text that parses as a number is still not a number."""
    assert pool is not None
    metadata = _metadata()
    metadata['offset'] = ['1.0', '2.0']
    with pytest.raises(TypeError, match='not a number'):
        read_image_facts(metadata)


def test_an_offset_of_booleans_is_refused(pool: KernelPool) -> None:
    """JSON ``true`` counts as an int in Python and is not an offset."""
    assert pool is not None
    metadata = _metadata()
    metadata['offset'] = [True, False]
    with pytest.raises(TypeError, match='not a number'):
        read_image_facts(metadata)


def test_a_null_status_is_refused(pool: KernelPool) -> None:
    """``str(None)`` is ``'None'``, which reads in the report as a status."""
    assert pool is not None
    metadata = _metadata()
    metadata['status'] = None
    with pytest.raises(TypeError, match='not a string'):
        read_image_facts(metadata)


def test_a_null_rank_is_refused(pool: KernelPool) -> None:
    """Nothing the pipeline writes as text is ever null, so a null is malformed."""
    assert pool is not None
    metadata = _metadata()
    metadata['navigation_result']['confidence_rank'] = None
    with pytest.raises(TypeError, match='not a string'):
        read_image_facts(metadata)


def test_an_absent_rank_is_read_as_not_recorded(pool: KernelPool) -> None:
    """A document that never carried one reports none."""
    assert pool is not None
    metadata = _metadata()
    del metadata['navigation_result']['confidence_rank']
    assert read_image_facts(metadata).confidence_rank is None


def test_a_null_confidence_is_refused(pool: KernelPool) -> None:
    """The same rule holds for a number the pipeline always writes."""
    assert pool is not None
    metadata = _metadata()
    metadata['confidence'] = None
    with pytest.raises(TypeError, match='not a number'):
        read_image_facts(metadata)


def test_a_numeric_rank_is_refused(pool: KernelPool) -> None:
    """The rank is a name; a number where one belongs is a malformed record."""
    assert pool is not None
    with pytest.raises(TypeError, match='not a string'):
        read_image_facts(_metadata(confidence_rank=3))


def test_an_empty_image_name_is_refused() -> None:
    """A row no image can be attributed to is worse than no row."""
    with pytest.raises(ValueError, match='image_name is empty'):
        ImageFacts(
            image_name='',
            utc=None,
            et=None,
            sclk=None,
            offset_dv=None,
            offset_du=None,
            sigma_dv=None,
            sigma_du=None,
            confidence=None,
            confidence_rank=None,
            status='failed',
            status_reason=None,
        )


def test_a_times_block_that_is_not_a_block_is_refused(pool: KernelPool) -> None:
    """A ``times`` field holding text is malformed, not an image without times."""
    assert pool is not None
    metadata = _metadata()
    metadata['navigation_result']['times'] = 'later'
    with pytest.raises(ValueError, match='not a section'):
        read_image_facts(metadata)


# ---------------------------------------------------------------------------
# A row says either where the segment went or why there is none
# ---------------------------------------------------------------------------


def test_a_row_names_a_source_or_a_reason(pool: KernelPool) -> None:
    """A row with both would be counted twice by every consumer."""
    with pytest.raises(ValueError, match='not both and not neither'):
        ReportRow(
            facts=_facts(pool),
            source_bc='03236_04002ra_nav.bc',
            omission_reason=OmissionReason.BOTSIM_LOSER,
        )


def test_a_row_with_neither_is_refused(pool: KernelPool) -> None:
    """And a row with neither says nothing about what became of the image."""
    with pytest.raises(ValueError, match='not both and not neither'):
        ReportRow(facts=_facts(pool), source_bc=None, omission_reason=None)


def test_an_empty_source_file_name_is_refused(pool: KernelPool) -> None:
    """An empty name reads in the report exactly like an omitted image."""
    with pytest.raises(ValueError, match='empty source file name'):
        ReportRow(facts=_facts(pool), source_bc='', omission_reason=None)


def test_an_omitted_image_names_its_reason(pool: KernelPool) -> None:
    """The reason is written as the value the closed set declares."""
    row = ReportRow(
        facts=_facts(pool),
        source_bc=None,
        omission_reason=OmissionReason.NO_REPRODUCING_BASELINE,
    )
    cells = dict(zip(REPORT_COLUMNS, row.values(), strict=True))
    assert cells['omission_reason'] == 'no_reproducing_baseline'
    assert cells['source_bc'] == ''


def test_an_assigned_image_names_its_file(pool: KernelPool) -> None:
    """An image with a segment names the file carrying it and no reason."""
    cells = _row(_facts(pool))
    assert cells['source_bc'] == '03236_04002ra_nav.bc'
    assert cells['omission_reason'] == ''


# ---------------------------------------------------------------------------
# The report as a whole
# ---------------------------------------------------------------------------


def test_the_report_starts_with_its_column_names(pool: KernelPool) -> None:
    """A consumer keys on names, because the column set is expected to grow."""
    text = report_text([ReportRow(facts=_facts(pool), source_bc='a_nav.bc', omission_reason=None)])
    assert text.splitlines()[0] == ','.join(REPORT_COLUMNS)


def test_every_image_appears_exactly_once(pool: KernelPool) -> None:
    """One row per image, in the order given."""
    rows = [
        ReportRow(
            facts=_facts(pool, image_name='A.IMG'), source_bc='a_nav.bc', omission_reason=None
        ),
        ReportRow(
            facts=_facts(pool, image_name='B.IMG'),
            source_bc=None,
            omission_reason=OmissionReason.BOTSIM_LOSER,
        ),
    ]
    lines = report_text(rows).splitlines()
    assert len(lines) == 3
    assert lines[1].startswith('A.IMG,')
    assert lines[2].startswith('B.IMG,')


def test_an_image_reported_twice_is_refused(pool: KernelPool) -> None:
    """A report holding one image twice would be counted twice."""
    row = ReportRow(facts=_facts(pool), source_bc='a_nav.bc', omission_reason=None)
    with pytest.raises(ValueError, match='more than once'):
        report_text([row, row])


def test_a_field_holding_a_comma_is_quoted(pool: KernelPool) -> None:
    """A comma inside a cell must not become a column boundary."""
    text = report_text(
        [
            ReportRow(
                facts=_facts(pool, status_reason='a,b'),
                source_bc='a_nav.bc',
                omission_reason=None,
            )
        ]
    )
    assert '"a,b"' in text


def test_write_report_writes_what_report_text_renders(pool: KernelPool, tmp_path: Path) -> None:
    """The file on disk is the rendered text and nothing else."""
    rows = [ReportRow(facts=_facts(pool), source_bc='a_nav.bc', omission_reason=None)]
    path = FCPath(str(tmp_path / 'report.csv'))
    write_report(path, rows)
    assert path.read_text() == report_text(rows)
