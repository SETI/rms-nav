import oops
import pytest

from nav.support.misc import (dec_rad_to_dms,
                              ra_rad_to_hms,
                              safe_lstrip_zero)


def test_ra_rad_to_hms() -> None:
    assert ra_rad_to_hms(0) == '00h00m00.000s'
    assert ra_rad_to_hms(1 * oops.RPD) == '00h04m00.000s'
    assert ra_rad_to_hms(14 * oops.RPD) == '00h56m00.000s'
    assert ra_rad_to_hms(15 * oops.RPD) == '01h00m00.000s'
    assert ra_rad_to_hms(359 * oops.RPD) == '23h56m00.000s'
    assert ra_rad_to_hms(360 * oops.RPD) == '00h00m00.000s'
    assert ra_rad_to_hms(.0006 / 60 / 60 / 24 * 360 * oops.RPD) == '00h00m00.001s'
    assert ra_rad_to_hms(59.999 / 60 / 60 / 24 * 360 * oops.RPD) == '00h00m59.999s'
    with pytest.raises(ValueError):
        ra_rad_to_hms(-1)


def test_dec_rad_to_dms() -> None:
    assert dec_rad_to_dms(0) == '+000d00m00.000s'
    assert dec_rad_to_dms(1 * oops.RPD) == '+001d00m00.000s'
    assert dec_rad_to_dms(14 * oops.RPD) == '+014d00m00.000s'
    assert dec_rad_to_dms(15 * oops.RPD) == '+015d00m00.000s'
    assert dec_rad_to_dms(179 * oops.RPD) == '+179d00m00.000s'
    assert dec_rad_to_dms(180 * oops.RPD) == '-180d00m00.000s'
    assert dec_rad_to_dms(.0006 / 60 / 60 * oops.RPD) == '+000d00m00.001s'
    assert dec_rad_to_dms(59.9994 / 60 / 60 * oops.RPD) == '+000d00m59.999s'

    assert dec_rad_to_dms(-1 * oops.RPD) == '-001d00m00.000s'
    assert dec_rad_to_dms(-14 * oops.RPD) == '-014d00m00.000s'
    assert dec_rad_to_dms(-15 * oops.RPD) == '-015d00m00.000s'
    assert dec_rad_to_dms(-179 * oops.RPD) == '-179d00m00.000s'
    assert dec_rad_to_dms(-180 * oops.RPD) == '+180d00m00.000s'
    assert dec_rad_to_dms(-.0006 / 60 / 60 * oops.RPD) == '-000d00m00.001s'
    assert dec_rad_to_dms(-59.9994 / 60 / 60 * oops.RPD) == '-000d00m59.999s'


def test_safe_lstrip_zero() -> None:
    assert safe_lstrip_zero('0000000000') == '0'
    assert safe_lstrip_zero('0000000001') == '1'
    assert safe_lstrip_zero('0000000009') == '9'
    assert safe_lstrip_zero('0000000010') == '10'
    assert safe_lstrip_zero('000000001f') == '1f'
    assert safe_lstrip_zero('0000000020') == '20'
    assert safe_lstrip_zero('00000000ff') == 'ff'
    assert safe_lstrip_zero('00000000FF') == 'FF'
    assert safe_lstrip_zero('00000000100') == '100'
