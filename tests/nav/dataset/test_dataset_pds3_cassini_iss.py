import argparse

import pytest

import nav.dataset.dataset_pds3_cassini_iss as dscoiss


@pytest.fixture
def ds_cassini_iss() -> dscoiss.DataSetPDS3CassiniISS:
    """Cassini ISS dataset fixture for testing."""
    return dscoiss.DataSetPDS3CassiniISS()


def test_cassini_iss_yield_basic(ds_cassini_iss: dscoiss.DataSetPDS3CassiniISS) -> None:
    ret = list(ds_cassini_iss.yield_image_files_index(max_filenames=2))
    assert len(ret) == 2
    assert (
        ret[0]
        .image_files[0]
        .label_file_url.as_posix()
        .endswith(
            'calibrated/COISS_1xxx/COISS_1001/data/1294561143_1295221348/W1294561143_1_CALIB.LBL'
        )
    )
    assert (
        ret[1]
        .image_files[0]
        .label_file_url.as_posix()
        .endswith(
            'calibrated/COISS_1xxx/COISS_1001/data/1294561143_1295221348/W1294561202_1_CALIB.LBL'
        )
    )


def test_cassini_iss_yield_vol_start(
    ds_cassini_iss: dscoiss.DataSetPDS3CassiniISS,
) -> None:
    ret = list(ds_cassini_iss.yield_image_files_index(max_filenames=1, vol_start='COISS_2009'))
    assert len(ret) == 1
    assert (
        ret[0]
        .image_files[0]
        .label_file_url.as_posix()
        .endswith(
            'calibrated/COISS_2xxx/COISS_2009/data/1484573295_1484664788/N1484573295_1_CALIB.LBL'
        )
    )


def test_cassini_iss_yield_vol_end(
    ds_cassini_iss: dscoiss.DataSetPDS3CassiniISS,
) -> None:
    ret = list(ds_cassini_iss.yield_image_files_index(vol_end='COISS_1002'))
    assert len(ret) == 8868
    assert (
        ret[-1]
        .image_files[0]
        .label_file_url.as_posix()
        .endswith(
            'calibrated/COISS_1xxx/COISS_1002/data/1353707153_1353756211/W1353756211_1_CALIB.LBL'
        )
    )


def test_cassini_iss_yield_img_start_num(
    ds_cassini_iss: dscoiss.DataSetPDS3CassiniISS,
) -> None:
    ret = list(ds_cassini_iss.yield_image_files_index(max_filenames=2, img_start_num=1353634555))
    assert len(ret) == 2
    assert ret[0].image_files[0].label_file_url.as_posix().endswith('N1353634555_1_CALIB.LBL')
    assert ret[1].image_files[0].label_file_url.as_posix().endswith('W1353634555_1_CALIB.LBL')


def test_cassini_iss_yield_img_end_num(
    ds_cassini_iss: dscoiss.DataSetPDS3CassiniISS,
) -> None:
    ret = list(ds_cassini_iss.yield_image_files_index(img_end_num=1294561202))
    assert len(ret) == 2
    assert (
        ret[0]
        .image_files[0]
        .label_file_url.as_posix()
        .endswith(
            'calibrated/COISS_1xxx/COISS_1001/data/1294561143_1295221348/W1294561143_1_CALIB.LBL'
        )
    )
    assert (
        ret[1]
        .image_files[0]
        .label_file_url.as_posix()
        .endswith(
            'calibrated/COISS_1xxx/COISS_1001/data/1294561143_1295221348/W1294561202_1_CALIB.LBL'
        )
    )


def test_cassini_iss_yield_volumes(
    ds_cassini_iss: dscoiss.DataSetPDS3CassiniISS,
) -> None:
    ret = list(ds_cassini_iss.yield_image_files_index(volumes=['COISS_1001', 'COISS_2009']))
    assert len(ret) == 8421
    ret2 = [x.image_files[0].label_file_url.as_posix() for x in ret]
    ret3 = [x for x in ret2 if 'COISS_1001' not in x and 'COISS_2009' not in x]
    assert len(ret3) == 0


def test_cassini_iss_camera(ds_cassini_iss: dscoiss.DataSetPDS3CassiniISS) -> None:
    arguments = argparse.Namespace(camera='WAC')
    ret = list(
        ds_cassini_iss.yield_image_files_index(
            max_filenames=1, volumes=['COISS_1001'], arguments=arguments
        )
    )
    assert len(ret) == 1
    assert ret[0].image_files[0].label_file_url.as_posix().endswith('W1294561143_1_CALIB.LBL')
    arguments = argparse.Namespace(camera='NAC')
    ret = list(
        ds_cassini_iss.yield_image_files_index(
            max_filenames=1, volumes=['COISS_1001'], arguments=arguments
        )
    )
    assert len(ret) == 1
    assert ret[0].image_files[0].label_file_url.as_posix().endswith('N1294562651_1_CALIB.LBL')


def test_cassini_iss_camera_invalid(
    ds_cassini_iss: dscoiss.DataSetPDS3CassiniISS,
) -> None:
    arguments = argparse.Namespace(camera='foo')
    with pytest.raises(ValueError):
        next(
            ds_cassini_iss.yield_image_files_index(
                max_filenames=1, volumes=['COISS_1001'], arguments=arguments
            )
        )


def test_cassini_iss_group_botsim(
    ds_cassini_iss: dscoiss.DataSetPDS3CassiniISS,
) -> None:
    ret = list(
        ds_cassini_iss.yield_image_files_index(
            group='botsim', img_start_num=1294562768, img_end_num=1294562949
        )
    )
    assert len(ret) == 3
    assert len(ret[0].image_files) == 1
    assert len(ret[1].image_files) == 2
    assert len(ret[2].image_files) == 1
    assert ret[0].image_files[0].label_file_url.as_posix().endswith('N1294562768_1_CALIB.LBL')
    assert ret[1].image_files[0].label_file_url.as_posix().endswith('N1294562836_1_CALIB.LBL')
    assert ret[1].image_files[1].label_file_url.as_posix().endswith('W1294562835_1_CALIB.LBL')
    assert ret[2].image_files[0].label_file_url.as_posix().endswith('W1294562949_1_CALIB.LBL')
