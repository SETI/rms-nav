import pytest
from tests.config import URL_VOYAGER_ISS_IO_01

import nav.obs.obs_inst_voyager_iss as obstvgiss
from nav.obs.obs_inst_voyager_iss import _voyager_if_factor, _voyager_spacecraft_digit

# Real Voyager VICAR LAB02 records (fixed-format strings; index 4 is the
# spacecraft digit).  Captured from a holdings GEOMED image.
_LAB02_V1 = 'VGR-1   FDS 20621.33   PICNO 1326J2-002   SCET 79.189 16:08:47         C'
_LAB02_V2 = 'VGR-2   FDS 20621.33   PICNO 1326J2-002   SCET 79.189 16:08:47         C'
_LABEL3 = 'FOR (I/F)*10000., MULTIPLY DN VALUE BY               1.00000'


def test_voyager_iss_basic() -> None:
    obs = obstvgiss.ObsVoyagerISS.from_file(URL_VOYAGER_ISS_IO_01)
    assert obs.midtime == -646429822.8760977


def test_voyager_iss_metadata_spacecraft_lid() -> None:
    """Public metadata derives the instrument-host LID from LAB02."""
    obs = obstvgiss.ObsVoyagerISS.from_file(URL_VOYAGER_ISS_IO_01)
    meta = obs.get_public_metadata()
    # The IO test image is a Voyager 2 (VGR-2) frame.
    assert meta['instrument_host_lid'].endswith('spacecraft.vg2')


def test_spacecraft_digit_voyager1() -> None:
    """A VGR-1 LAB02 record yields spacecraft digit '1'."""
    assert _voyager_spacecraft_digit(_LAB02_V1) == '1'


def test_spacecraft_digit_voyager2() -> None:
    """A VGR-2 LAB02 record yields spacecraft digit '2'."""
    assert _voyager_spacecraft_digit(_LAB02_V2) == '2'


def test_spacecraft_digit_too_short_raises() -> None:
    """A LAB02 string shorter than 5 chars raises a clear ValueError."""
    with pytest.raises(ValueError, match='Unexpected Voyager LAB02 format'):
        _voyager_spacecraft_digit('VGR')


def test_spacecraft_digit_non_string_raises() -> None:
    """A non-string LAB02 value raises a clear ValueError."""
    with pytest.raises(ValueError, match='Unexpected Voyager LAB02 format'):
        _voyager_spacecraft_digit(None)


def test_spacecraft_digit_unknown_id_raises() -> None:
    """A LAB02 with an out-of-range spacecraft id raises ValueError."""
    bad = 'VGR-3   FDS 20621.33'
    with pytest.raises(ValueError, match='expected 1 or 2'):
        _voyager_spacecraft_digit(bad)


def test_if_factor_parses_value() -> None:
    """A well-formed LABEL3 yields its trailing numeric factor."""
    assert _voyager_if_factor(_LABEL3) == 1.0


def test_if_factor_missing_phrase_raises() -> None:
    """A LABEL3 lacking the fixed phrase raises a clear ValueError."""
    with pytest.raises(ValueError, match='Unexpected Voyager LABEL3 format'):
        _voyager_if_factor('SOME OTHER LABEL 1.0')


def test_if_factor_non_numeric_remainder_raises() -> None:
    """A LABEL3 whose remainder is not numeric raises a clear ValueError."""
    bad = 'FOR (I/F)*10000., MULTIPLY DN VALUE BY    NOT_A_NUMBER'
    with pytest.raises(ValueError, match='Unexpected Voyager LABEL3 format'):
        _voyager_if_factor(bad)


def test_if_factor_non_string_raises() -> None:
    """A non-string LABEL3 value raises a clear ValueError."""
    with pytest.raises(ValueError, match='Unexpected Voyager LABEL3 format'):
        _voyager_if_factor(None)
