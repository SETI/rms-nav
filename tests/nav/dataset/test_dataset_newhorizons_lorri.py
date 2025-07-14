import nav.dataset.dataset_newhorizons_lorri as dsnhlor

# Create this once so we can take advantage of PdsTable caching
_DS = dsnhlor.DataSetNewHorizonsLORRI()


def test_newhorizons_lorri_yield_basic() -> None:
    ret = _DS.yield_filenames_index(max_filenames=2, retrieve_files=False)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 2
    assert ret2[0].endswith(
        'volumes/NHxxLO_xxxx/NHLALO_2001/data/20060224_000310/lor_0003103486_0x630_sci.lbl')
    assert ret2[1].endswith(
        'volumes/NHxxLO_xxxx/NHLALO_2001/data/20060224_000310/lor_0003103486_0x631_sci.lbl')


def test_newhorizons_lorri_yield_vol_start() -> None:
    ret = _DS.yield_filenames_index(max_filenames=1, vol_start='NHLALO_2001',
                                          retrieve_files=False)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 1
    assert ret2[0].endswith(
        'volumes/NHxxLO_xxxx/NHLALO_2001/data/20060224_000310/lor_0003103486_0x630_sci.lbl')


def test_newhorizons_lorri_yield_vol_end() -> None:
    ret = _DS.yield_filenames_index(vol_end='NHJULO_2001', retrieve_files=False)
    ret2 = [x[0] for x in ret]
    assert len(ret2) == 2364
    assert ret2[-1].as_posix().endswith(
        'volumes/NHxxLO_xxxx/NHJULO_2001/data/20070611_004390/lor_0043906321_0x636_sci.lbl')


def test_newhorizons_lorri_yield_img_start_num() -> None:
    ret = _DS.yield_filenames_index(max_filenames=2, img_start_num=19683707,
                                          retrieve_files=False)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 2
    assert ret2[0].endswith('lor_0019683707_0x630_sci.lbl')
    assert ret2[1].endswith('lor_0019683711_0x630_sci.lbl')


def test_newhorizons_lorri_yield_img_end_num() -> None:
    ret = _DS.yield_filenames_index(img_end_num=19683707, retrieve_files=False)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 995
    assert ret2[-2].endswith(
        'volumes/NHxxLO_xxxx/NHLALO_2001/data/20060904_001968/lor_0019683693_0x630_sci.lbl')
    assert ret2[-1].endswith(
        'volumes/NHxxLO_xxxx/NHLALO_2001/data/20060904_001968/lor_0019683707_0x630_sci.lbl')


def test_newhorizons_lorri_yield_volumes() -> None:
    ret = _DS.yield_filenames_index(volumes=['NHLALO_2001', 'NHJULO_2001'],
                                          retrieve_files=False)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 2364
    ret3 = [x for x in ret2 if 'NHLALO_2001' not in x and 'NHJULO_2001' not in x]
    assert len(ret3) == 0
