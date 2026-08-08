"""Hermetic tests for the round trip's pure-logic helpers.

The round trip proper needs local binary kernels and the holdings, so its
module skips wherever those are absent -- which would leave the input-domain
checks of its helpers unexercised everywhere but a fully provisioned machine.
The helpers tested here read nothing but their arguments, the repo-local image
library and ``tmp_path``, so this module carries no environment guard and no
``integration`` marker and runs in the plain unit tier.  The two
``pixel_scales`` tests stay with the round trip: measuring a real FOV
genuinely needs oops resources.
"""

import json
from pathlib import Path

import pytest

from tests.integration.ck_round_trip import (
    angle_to_pixels,
    read_json,
    record_index_for_tick,
    sidecar_for,
)

# A round-trip cohort frame, used to show the sidecar lookup finds a real
# library entry; the library lives in the repo, so no holdings are needed.
_CASSINI_NAC = 'N1461997416_1_CALIB'


@pytest.mark.parametrize(
    'angle_rad', [float('nan'), float('inf'), float('-inf')], ids=['nan', 'inf', '-inf']
)
def test_angle_to_pixels_refuses_a_non_finite_angle(angle_rad: float) -> None:
    """A non-finite angle divides into a non-finite number of pixels.

    Which then compares False against every tolerance there is, and so reads as
    a pass.

    Parameters:
        angle_rad: The angle that must be refused.
    """
    with pytest.raises(ValueError, match='not a finite number of radians'):
        angle_to_pixels(angle_rad, 6.0e-6)


@pytest.mark.parametrize(
    'scale_rad_px',
    [0.0, -6.0e-6, float('nan'), float('inf')],
    ids=['zero', 'negative', 'nan', 'inf'],
)
def test_angle_to_pixels_refuses_an_unusable_scale(scale_rad_px: float) -> None:
    """A scale that is not a positive angle converts nothing.

    Parameters:
        scale_rad_px: The scale that must be refused.
    """
    with pytest.raises(ValueError, match='not a positive angle'):
        angle_to_pixels(1.0e-6, scale_rad_px)


def test_angle_to_pixels_converts_at_the_measured_scale() -> None:
    """One pixel of angle is one pixel."""
    assert angle_to_pixels(1.2e-5, 6.0e-6) == pytest.approx(2.0)


def test_record_index_for_tick_finds_the_record() -> None:
    """A time tag a segment holds names the record holding it."""
    assert record_index_for_tick([10.0, 20.0, 30.0], 20.0) == 1


def test_record_index_for_tick_refuses_an_epoch_with_no_record() -> None:
    """An epoch between two records is an interpolation, not a record.

    Comparing a readback against the nearest record would turn an assertion
    about what was written into an assertion about what SPICE interpolated.
    """
    with pytest.raises(ValueError, match='no record sits at encoded clock time'):
        record_index_for_tick([10.0, 20.0, 30.0], 25.0)


def test_record_index_for_tick_refuses_an_empty_segment() -> None:
    """A segment with no records holds no epoch at all."""
    with pytest.raises(ValueError, match='holds no records'):
        record_index_for_tick([], 20.0)


@pytest.mark.parametrize('tick', [float('nan'), float('inf')], ids=['nan', 'inf'])
def test_record_index_for_tick_refuses_a_non_finite_epoch(tick: float) -> None:
    """A non-finite time tag matches nothing and reports nothing.

    Parameters:
        tick: The encoded clock time that must be refused.
    """
    with pytest.raises(ValueError, match='not a finite tick'):
        record_index_for_tick([10.0, 20.0], tick)


@pytest.mark.parametrize(
    'ticks',
    [[10.0, float('nan'), 30.0], [10.0, 30.0, float('nan')]],
    ids=['before-the-match', 'after-the-match'],
)
def test_record_index_for_tick_refuses_a_non_finite_tag(ticks: list[float]) -> None:
    """A segment whose tags are not finite cannot be searched by comparison.

    Every tag is validated, including one sitting after the record that would
    have matched: a search that stopped at the match would report an index from
    a tag list it never finished examining.

    Parameters:
        ticks: The tag list holding a non-finite value.
    """
    with pytest.raises(ValueError, match='non-finite time tag'):
        record_index_for_tick(ticks, 30.0)


def test_sidecar_for_refuses_an_unknown_image() -> None:
    """An image id the library does not hold is named rather than guessed at."""
    with pytest.raises(ValueError, match='no image library sidecar is named'):
        sidecar_for('N0000000000_0_CALIB')


def test_sidecar_for_finds_a_cohort_frame() -> None:
    """Each cohort frame is a real library entry, found by its own id."""
    assert sidecar_for(_CASSINI_NAC).image_id == _CASSINI_NAC


def test_read_json_refuses_a_missing_file(tmp_path: Path) -> None:
    """A step's findings that are not there mean the step did not finish."""
    with pytest.raises(ValueError, match='did not complete'):
        read_json(tmp_path / 'absent.json')


def test_read_json_refuses_a_document_that_is_not_an_object(tmp_path: Path) -> None:
    """A JSON list where a step's findings belong is a malformed file."""
    path = tmp_path / 'list.json'
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match='not a JSON object'):
        read_json(path)
