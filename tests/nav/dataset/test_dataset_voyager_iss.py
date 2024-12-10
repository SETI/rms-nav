import nav.dataset.dataset_voyager_iss as dsvgiss

# Create this once so we can take advantage of PdsTable caching
_DS = dsvgiss.DataSetVoyagerISS()


def test_voyager_iss_yield_basic():
    ret = _DS.yield_image_filenames_index(max_filenames=2)
    ret = [x.as_posix() for x in ret]
    assert len(ret) == 2
    assert ret[0].endswith(
        'volumes/VGISS_5xxx/VGISS_5101/DATA/C13854XX/C1385455_GEOMED.LBL')
    assert ret[1].endswith(
        'volumes/VGISS_5xxx/VGISS_5101/DATA/C13894XX/C1389407_GEOMED.LBL')


def test_voyager_iss_yield_vol_start():
    ret = _DS.yield_image_filenames_index(max_filenames=1, vol_start='VGISS_8201')
    ret = [x.as_posix() for x in ret]
    assert len(ret) == 1
    assert ret[0].endswith(
        'volumes/VGISS_8xxx/VGISS_8201/DATA/C08966XX/C0896631_GEOMED.LBL')


def test_voyager_iss_yield_vol_end():
    ret = _DS.yield_image_filenames_index(vol_end='VGISS_5102')
    ret = list(ret)
    assert len(ret) == 928
    assert ret[-1].as_posix().endswith(
        'volumes/VGISS_5xxx/VGISS_5102/DATA/C15199XX/C1519931_GEOMED.LBL')


def test_voyager_iss_yield_img_start_num():
    ret = _DS.yield_image_filenames_index(max_filenames=2, img_start_num=1469548)
    ret = [x.as_posix() for x in ret]
    assert len(ret) == 2
    assert ret[0].endswith('C1469548_GEOMED.LBL')
    assert ret[1].endswith('C1469550_GEOMED.LBL')


def test_voyager_iss_yield_img_end_num():
    ret = _DS.yield_image_filenames_index(img_end_num=1469550)
    ret = [x.as_posix() for x in ret]
    assert len(ret) == 117
    assert ret[-2].endswith(
        'volumes/VGISS_5xxx/VGISS_5101/DATA/C14695XX/C1469548_GEOMED.LBL')
    assert ret[-1].endswith(
        'volumes/VGISS_5xxx/VGISS_5101/DATA/C14695XX/C1469550_GEOMED.LBL')


def test_voyager_iss_yield_volumes():
    ret = _DS.yield_image_filenames_index(volumes=['VGISS_5101', 'VGISS_8201'])
    ret = [x.as_posix() for x in ret]
    assert len(ret) == 1403
    ret = [x for x in ret if 'VGISS_5101' not in x and 'VGISS_8201' not in x]
    assert len(ret) == 0

# def test_voyager_iss_camera():
#     ret = _DS.yield_image_filenames_index(max_filenames=1, volumes=['COISS_1001'],
#                                           camera='W')
#     ret = [x.as_posix() for x in ret]
#     assert len(ret) == 1
#     assert ret[0].endswith('W1294561143_1.LBL')
#     ret = _DS.yield_image_filenames_index(max_filenames=1, volumes=['COISS_1001'],
#                                           camera='N')
#     ret = [x.as_posix() for x in ret]
#     assert len(ret) == 1
#     assert ret[0].endswith('N1294562651_1.LBL')
