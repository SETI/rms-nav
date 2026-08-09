"""Tests for the pure helpers the report is assembled from.

None of these reads a database. They cover the values the report derives from an
image name or an epoch, the search limit it resolves from configuration, and the
filter fragment every query splices in -- which carries named binds and its
values beside it, because the fragment is spliced into statements that carry
binds of their own and an inlined literal is a quoting bug waiting for a value
that carries a quote.
"""

import pytest

from spindoctor.cli.stats.classify import (
    date_from_image_et,
    datetime_from_image_et,
    image_number_from_name,
)
from spindoctor.cli.stats.report_common import count_pct, image_name_from_filename, where_clause
from spindoctor.cli.stats.report_sections import resolve_offset_limit

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
# The filter fragment
# ---------------------------------------------------------------------------


def test_a_filter_is_a_named_bind_carrying_its_value_beside_it() -> None:
    """An inlined literal is a quoting bug waiting for a value that carries a quote.

    The fragment is only half of it; the value travels beside it as a parameter.
    """
    where, params = where_clause(instrument='coiss', start_date=None, end_date=None)
    assert where == ' WHERE instrument = :instrument'
    assert params == {'instrument': 'coiss'}
