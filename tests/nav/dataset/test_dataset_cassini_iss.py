import nav.dataset.dataset_cassini_iss as dscoiss

# Create this once so we can take advantage of PdsTable caching
_DS = dscoiss.DataSetCassiniISS()


def test_cassini_iss_yield_basic() -> None:
    ret = _DS.yield_image_filenames_index(max_filenames=2)
    ret2 = [x.as_posix() for x in ret]
    assert len(ret2) == 2
    assert ret2[0].endswith(
        'volumes/COISS_1xxx/COISS_1001/data/1294561143_1295221348/W1294561143_1.LBL')
    assert ret2[1].endswith(
        'volumes/COISS_1xxx/COISS_1001/data/1294561143_1295221348/W1294561202_1.LBL')


def test_cassini_iss_yield_vol_start() -> None:
    ret = _DS.yield_image_filenames_index(max_filenames=1, vol_start='COISS_2009')
    ret2 = [x.as_posix() for x in ret]
    assert len(ret2) == 1
    assert ret2[0].endswith(
        'volumes/COISS_2xxx/COISS_2009/data/1484573295_1484664788/N1484573295_1.LBL')


def test_cassini_iss_yield_vol_end() -> None:
    ret = _DS.yield_image_filenames_index(vol_end='COISS_1002')
    ret2 = list(ret)
    assert len(ret2) == 8868
    assert ret2[-1].as_posix().endswith(
        'volumes/COISS_1xxx/COISS_1002/data/1353707153_1353756211/W1353756211_1.LBL')


def test_cassini_iss_yield_img_start_num() -> None:
    ret = _DS.yield_image_filenames_index(max_filenames=2, img_start_num=1353634555)
    ret2 = [x.as_posix() for x in ret]
    assert len(ret2) == 2
    assert ret2[0].endswith('N1353634555_1.LBL')
    assert ret2[1].endswith('W1353634555_1.LBL')


def test_cassini_iss_yield_img_end_num() -> None:
    ret = _DS.yield_image_filenames_index(img_end_num=1294561202)
    ret2 = [x.as_posix() for x in ret]
    assert len(ret2) == 2
    assert ret2[0].endswith(
        'volumes/COISS_1xxx/COISS_1001/data/1294561143_1295221348/W1294561143_1.LBL')
    assert ret2[1].endswith(
        'volumes/COISS_1xxx/COISS_1001/data/1294561143_1295221348/W1294561202_1.LBL')


def test_cassini_iss_yield_volumes() -> None:
    ret = _DS.yield_image_filenames_index(volumes=['COISS_1001', 'COISS_2009'])
    ret2 = [x.as_posix() for x in ret]
    assert len(ret2) == 8421
    ret3 = [x for x in ret2 if 'COISS_1001' not in x and 'COISS_2009' not in x]
    assert len(ret3) == 0


def test_cassini_iss_camera() -> None:
    ret = _DS.yield_image_filenames_index(max_filenames=1, volumes=['COISS_1001'],
                                          camera='W')
    ret2 = [x.as_posix() for x in ret]
    assert len(ret2) == 1
    assert ret2[0].endswith('W1294561143_1.LBL')
    ret = _DS.yield_image_filenames_index(max_filenames=1, volumes=['COISS_1001'],
                                          camera='N')
    ret2 = [x.as_posix() for x in ret]
    assert len(ret2) == 1
    assert ret2[0].endswith('N1294562651_1.LBL')
