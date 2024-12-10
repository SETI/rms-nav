import nav.dataset.dataset_galileo_ssi as dsgossi

# Create this once so we can take advantage of PdsTable caching
_DS = dsgossi.DataSetGalileoSSI()


def test_galileo_ssi_yield_basic():
    ret = _DS.yield_image_filenames_index(max_filenames=2)
    ret = [x.as_posix() for x in ret]
    assert len(ret) == 2
    assert ret[0].endswith(
        'volumes/GO_0xxx/GO_0002/RAW_CAL/C0003061100R.LBL')
    assert ret[1].endswith(
        'volumes/GO_0xxx/GO_0002/RAW_CAL/C0003061200R.LBL')

def test_galileo_ssi_yield_vol_start():
    ret = _DS.yield_image_filenames_index(max_filenames=1, vol_start='GO_0020')
    ret = [x.as_posix() for x in ret]
    assert len(ret) == 1
    assert ret[0].endswith(
        'volumes/GO_0xxx/GO_0020/E11/IO/C0420361523R.LBL')


def test_galileo_ssi_yield_vol_end():
    ret = _DS.yield_image_filenames_index(vol_end='GO_0003')
    ret = list(ret)
    assert len(ret) == 1600
    assert ret[-1].as_posix().endswith(
        'volumes/GO_0xxx/GO_0003/MOON/C0061060800R.LBL')


def test_galileo_ssi_yield_img_start_num():
    ret = _DS.yield_image_filenames_index(max_filenames=2, img_start_num=59468500)
    ret = [x.as_posix() for x in ret]
    assert len(ret) == 2
    assert ret[0].endswith('C0059468500R.LBL')
    assert ret[1].endswith('C0059468545R.LBL')


def test_galileo_ssi_yield_img_end_num():
    ret = _DS.yield_image_filenames_index(img_end_num=59468500)
    ret = [x.as_posix() for x in ret]
    assert len(ret) == 181
    assert ret[-2].endswith(
        'volumes/GO_0xxx/GO_0002/RAW_CAL/C0059468445R.LBL')
    assert ret[-1].endswith(
        'volumes/GO_0xxx/GO_0002/RAW_CAL/C0059468500R.LBL')


def test_galileo_ssi_yield_volumes():
    ret = _DS.yield_image_filenames_index(volumes=['GO_0003', 'GO_0020'])
    ret = [x.as_posix() for x in ret]
    assert len(ret) == 1504
    ret = [x for x in ret if 'GO_0003' not in x and 'GO_0020' not in x]
    assert len(ret) == 0

# def test_galileo_ssi_camera():
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
