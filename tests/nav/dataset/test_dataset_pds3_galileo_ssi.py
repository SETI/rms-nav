import pytest

import nav.dataset.dataset_pds3_galileo_ssi as dsgossi


@pytest.fixture
def ds_galileo_ssi():
    """Galileo SSI dataset fixture for testing."""
    return dsgossi.DataSetPDS3GalileoSSI()


def test_galileo_ssi_yield_basic(ds_galileo_ssi) -> None:
    ret = ds_galileo_ssi.yield_filenames_index(max_filenames=2, retrieve_files=False)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 2
    assert ret2[0].endswith(
        'volumes/GO_0xxx/GO_0002/RAW_CAL/C0003061100R.LBL')
    assert ret2[1].endswith(
        'volumes/GO_0xxx/GO_0002/RAW_CAL/C0003061200R.LBL')


def test_galileo_ssi_yield_vol_start(ds_galileo_ssi) -> None:
    ret = ds_galileo_ssi.yield_filenames_index(max_filenames=1, vol_start='GO_0020',
                                               retrieve_files=False)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 1
    assert ret2[0].endswith(
        'volumes/GO_0xxx/GO_0020/E11/IO/C0420361523R.LBL')


def test_galileo_ssi_yield_vol_end(ds_galileo_ssi) -> None:
    ret = ds_galileo_ssi.yield_filenames_index(vol_end='GO_0003', retrieve_files=False)
    ret2 = [x[0] for x in ret]
    assert len(ret2) == 1600
    assert ret2[-1].as_posix().endswith(
        'volumes/GO_0xxx/GO_0003/MOON/C0061060800R.LBL')


def test_galileo_ssi_yield_img_start_num(ds_galileo_ssi) -> None:
    ret = ds_galileo_ssi.yield_filenames_index(max_filenames=2, img_start_num=59468500,
                                               retrieve_files=False)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 2
    assert ret2[0].endswith('C0059468500R.LBL')
    assert ret2[1].endswith('C0059468545R.LBL')


def test_galileo_ssi_yield_img_end_num(ds_galileo_ssi) -> None:
    ret = ds_galileo_ssi.yield_filenames_index(img_end_num=59468500, retrieve_files=False)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 181
    assert ret2[-2].endswith(
        'volumes/GO_0xxx/GO_0002/RAW_CAL/C0059468445R.LBL')
    assert ret2[-1].endswith(
        'volumes/GO_0xxx/GO_0002/RAW_CAL/C0059468500R.LBL')


def test_galileo_ssi_yield_volumes(ds_galileo_ssi) -> None:
    ret = ds_galileo_ssi.yield_filenames_index(volumes=['GO_0003', 'GO_0020'],
                                               retrieve_files=False)
    ret2 = [x[0].as_posix() for x in ret]
    assert len(ret2) == 1504
    ret3 = [x for x in ret2 if 'GO_0003' not in x and 'GO_0020' not in x]
    assert len(ret3) == 0
