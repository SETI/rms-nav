import nav.dataset.dataset_newhorizons_lorri as dsnhlor

# Create this once so we can take advantage of PdsTable caching
_DS = dsnhlor.DataSetNewHorizonsLORRI()


def test_newhorizons_lorri_yield_basic():
    ret = _DS.yield_image_filenames_index(max_filenames=2)
    ret = [x.as_posix() for x in ret]
    assert len(ret) == 2
    assert ret[0].endswith(
        'volumes/NHxxLO_xxxx/NHLALO_2001/data/20060224_000310/lor_0003103486_0x630_sci.lbl')
    assert ret[1].endswith(
        'volumes/NHxxLO_xxxx/NHLALO_2001/data/20060224_000310/lor_0003103486_0x631_sci.lbl')

def test_newhorizons_lorri_yield_vol_start():
    ret = _DS.yield_image_filenames_index(max_filenames=1, vol_start='NHLALO_2001')
    ret = [x.as_posix() for x in ret]
    assert len(ret) == 1
    assert ret[0].endswith(
        'volumes/NHxxLO_xxxx/NHLALO_2001/data/20060224_000310/lor_0003103486_0x630_sci.lbl')


def test_newhorizons_lorri_yield_vol_end():
    ret = _DS.yield_image_filenames_index(vol_end='NHJULO_2001')
    ret = list(ret)
    assert len(ret) == 2364
    assert ret[-1].as_posix().endswith(
        'volumes/NHxxLO_xxxx/NHJULO_2001/data/20070611_004390/lor_0043906321_0x636_sci.lbl')


def test_newhorizons_lorri_yield_img_start_num():
    ret = _DS.yield_image_filenames_index(max_filenames=2, img_start_num=19683707)
    ret = [x.as_posix() for x in ret]
    assert len(ret) == 2
    assert ret[0].endswith('lor_0019683707_0x630_sci.lbl')
    assert ret[1].endswith('lor_0019683711_0x630_sci.lbl')


def test_newhorizons_lorri_yield_img_end_num():
    ret = _DS.yield_image_filenames_index(img_end_num=19683707)
    ret = [x.as_posix() for x in ret]
    assert len(ret) == 995
    assert ret[-2].endswith(
        'volumes/NHxxLO_xxxx/NHLALO_2001/data/20060904_001968/lor_0019683693_0x630_sci.lbl')
    assert ret[-1].endswith(
        'volumes/NHxxLO_xxxx/NHLALO_2001/data/20060904_001968/lor_0019683707_0x630_sci.lbl')


def test_newhorizons_lorri_yield_volumes():
    ret = _DS.yield_image_filenames_index(volumes=['NHLALO_2001', 'NHJULO_2001'])
    ret = [x.as_posix() for x in ret]
    assert len(ret) == 2364
    ret = [x for x in ret if 'NHLALO_2001' not in x and 'NHJULO_2001' not in x]
    assert len(ret) == 0

# def test_newhorizons_lorri_camera():
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
