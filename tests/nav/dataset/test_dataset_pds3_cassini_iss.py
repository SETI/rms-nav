import argparse

import nav.dataset.dataset_pds3_cassini_iss as dscoiss

# Create this once so we can take advantage of PdsTable caching
_DS = dscoiss.DataSetPDS3CassiniISS()


def test_cassini_iss_yield_basic() -> None:
    ret = _DS.yield_filenames_index(max_filenames=2, retrieve_files=False)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 2
    assert ret2[0].endswith(
        'calibrated/COISS_1xxx/COISS_1001/data/1294561143_1295221348/W1294561143_1_CALIB.LBL')
    assert ret2[1].endswith(
        'calibrated/COISS_1xxx/COISS_1001/data/1294561143_1295221348/W1294561202_1_CALIB.LBL')


def test_cassini_iss_yield_vol_start() -> None:
    ret = _DS.yield_filenames_index(max_filenames=1, vol_start='COISS_2009',
                                    retrieve_files=False)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 1
    assert ret2[0].endswith(
        'calibrated/COISS_2xxx/COISS_2009/data/1484573295_1484664788/N1484573295_1_CALIB.LBL')


def test_cassini_iss_yield_vol_end() -> None:
    ret = _DS.yield_filenames_index(vol_end='COISS_1002', retrieve_files=False)
    ret2 = [x[0] for x in ret]
    assert len(ret2) == 8868
    assert ret2[-1].as_posix().endswith(
        'calibrated/COISS_1xxx/COISS_1002/data/1353707153_1353756211/W1353756211_1_CALIB.LBL')


def test_cassini_iss_yield_img_start_num() -> None:
    ret = _DS.yield_filenames_index(max_filenames=2, img_start_num=1353634555,
                                    retrieve_files=False)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 2
    assert ret2[0].endswith('N1353634555_1_CALIB.LBL')
    assert ret2[1].endswith('W1353634555_1_CALIB.LBL')


def test_cassini_iss_yield_img_end_num() -> None:
    ret = _DS.yield_filenames_index(img_end_num=1294561202, retrieve_files=False)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 2
    assert ret2[0].endswith(
        'calibrated/COISS_1xxx/COISS_1001/data/1294561143_1295221348/W1294561143_1_CALIB.LBL')
    assert ret2[1].endswith(
        'calibrated/COISS_1xxx/COISS_1001/data/1294561143_1295221348/W1294561202_1_CALIB.LBL')


def test_cassini_iss_yield_volumes() -> None:
    ret = _DS.yield_filenames_index(volumes=['COISS_1001', 'COISS_2009'],
                                    retrieve_files=False)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 8421
    ret3 = [x for x in ret2 if 'COISS_1001' not in x and 'COISS_2009' not in x]
    assert len(ret3) == 0


def test_cassini_iss_camera() -> None:
    arguments = argparse.Namespace(camera='WAC')
    ret = _DS.yield_filenames_index(max_filenames=1, volumes=['COISS_1001'],
                                    retrieve_files=False, arguments=arguments)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 1
    assert ret2[0].endswith('W1294561143_1_CALIB.LBL')
    arguments = argparse.Namespace(camera='NAC')
    ret = _DS.yield_filenames_index(max_filenames=1, volumes=['COISS_1001'],
                                    retrieve_files=False, arguments=arguments)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 1
    assert ret2[0].endswith('N1294562651_1_CALIB.LBL')
