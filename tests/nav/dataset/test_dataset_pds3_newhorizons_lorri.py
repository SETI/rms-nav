import pytest

import nav.dataset.dataset_pds3_newhorizons_lorri as dsnhlor


@pytest.fixture
def ds_newhorizons_lorri():
    """New Horizons LORRI dataset fixture for testing."""
    return dsnhlor.DataSetPDS3NewHorizonsLORRI()


def test_newhorizons_lorri_yield_basic(ds_newhorizons_lorri) -> None:
    ret = list(ds_newhorizons_lorri.yield_image_files_index(max_filenames=2))
    assert len(ret) == 2
    assert ret[0].image_files[0].label_file_url.as_posix().endswith(
        'volumes/NHxxLO_xxxx/NHLALO_2001/data/20060224_000310/lor_0003103486_0x630_sci.lbl')
    assert ret[1].image_files[0].label_file_url.as_posix().endswith(
        'volumes/NHxxLO_xxxx/NHLALO_2001/data/20060224_000310/lor_0003103486_0x631_sci.lbl')


def test_newhorizons_lorri_yield_vol_start(ds_newhorizons_lorri) -> None:
    ret = list(ds_newhorizons_lorri.yield_image_files_index(max_filenames=1,
                                                            vol_start='NHLALO_2001'))
    assert len(ret) == 1
    assert ret[0].image_files[0].label_file_url.as_posix().endswith(
        'volumes/NHxxLO_xxxx/NHLALO_2001/data/20060224_000310/lor_0003103486_0x630_sci.lbl')


def test_newhorizons_lorri_yield_vol_end(ds_newhorizons_lorri) -> None:
    ret = list(ds_newhorizons_lorri.yield_image_files_index(vol_end='NHJULO_2001'))
    assert len(ret) == 2364
    assert ret[-1].image_files[0].label_file_url.as_posix().endswith(
        'volumes/NHxxLO_xxxx/NHJULO_2001/data/20070611_004390/lor_0043906321_0x636_sci.lbl')


def test_newhorizons_lorri_yield_img_start_num(ds_newhorizons_lorri) -> None:
    ret = list(ds_newhorizons_lorri.yield_image_files_index(max_filenames=2,
                                                            img_start_num=19683707))
    assert len(ret) == 2
    assert ret[0].image_files[0].label_file_url.as_posix().endswith('lor_0019683707_0x630_sci.lbl')
    assert ret[1].image_files[0].label_file_url.as_posix().endswith('lor_0019683711_0x630_sci.lbl')


def test_newhorizons_lorri_yield_img_end_num(ds_newhorizons_lorri) -> None:
    ret = list(ds_newhorizons_lorri.yield_image_files_index(img_end_num=19683707))
    assert len(ret) == 995
    assert ret[-2].image_files[0].label_file_url.as_posix().endswith(
        'volumes/NHxxLO_xxxx/NHLALO_2001/data/20060904_001968/lor_0019683693_0x630_sci.lbl')
    assert ret[-1].image_files[0].label_file_url.as_posix().endswith(
        'volumes/NHxxLO_xxxx/NHLALO_2001/data/20060904_001968/lor_0019683707_0x630_sci.lbl')


def test_newhorizons_lorri_yield_volumes(ds_newhorizons_lorri) -> None:
    ret = list(ds_newhorizons_lorri.yield_image_files_index(volumes=['NHLALO_2001', 'NHJULO_2001']))
    assert len(ret) == 2364
    ret2 = [x.image_files[0].label_file_url.as_posix() for x in ret]
    ret3 = [x for x in ret2 if 'NHLALO_2001' not in x and 'NHJULO_2001' not in x]
    assert len(ret3) == 0
