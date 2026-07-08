import argparse
from typing import Any

import pytest
from filecache import FCPath
from tests.config import REQUIRES_EXTERNAL_DATA

import spindoctor.dataset.dataset_pds3_cassini_iss as dscoiss
from spindoctor.dataset.dataset import ImageFile

pytestmark = REQUIRES_EXTERNAL_DATA


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


def test_cassini_iss_choose_random_count(
    ds_cassini_iss: dscoiss.DataSetPDS3CassiniISS,
) -> None:
    ret = list(
        ds_cassini_iss.yield_image_files_index(choose_random_images=3, volumes=['COISS_1001'])
    )
    assert len(ret) == 3
    for group in ret:
        assert len(group.image_files) == 1
        assert 'COISS_1001' in group.image_files[0].label_file_url.as_posix()


def test_cassini_iss_choose_random_restrictive_filter_terminates(
    ds_cassini_iss: dscoiss.DataSetPDS3CassiniISS,
) -> None:
    # A restrictive image-number range that matches only a handful of rows must
    # terminate and return only matching images (never livelock), even though we ask
    # for more random images than exist in range.
    ret = list(
        ds_cassini_iss.yield_image_files_index(
            choose_random_images=100,
            volumes=['COISS_1001'],
            img_start_num=1294561143,
            img_end_num=1294561202,
        )
    )
    # Only two images fall in this range; the sampler returns exactly those and stops.
    assert len(ret) == 2
    names = sorted(g.image_files[0].image_file_name for g in ret)
    assert names[0].startswith('W1294561143')
    assert names[1].startswith('W1294561202')


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


def _make_botsim_imagefile(
    name: str, *, shutter: str, image_time: str, observation_id: str
) -> ImageFile:
    """Build a synthetic ImageFile carrying the index columns the grouper reads."""
    url = FCPath(f'calibrated/COISS_2xxx/COISS_2001/data/0000000000_0000000000/{name}_1_CALIB.IMG')
    row: dict[str, Any] = {
        'SHUTTER_MODE_ID': shutter,
        'IMAGE_NUMBER': name[1:],
        'OBSERVATION_ID': observation_id,
        'IMAGE_TIME': image_time,
    }
    return ImageFile(
        image_file_url=url,
        label_file_url=url.with_suffix('.LBL'),
        results_path_stub=name,
        index_file_row=row,
    )


def test_cassini_iss_group_botsim_lone_frame_not_dropped(
    ds_cassini_iss: dscoiss.DataSetPDS3CassiniISS,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Sequence: a true pair, a lone BOTSIM frame (its partner camera missing), then
    # another true pair. The lone frame must be yielded on its own, never dropped, and
    # only genuine N/W partners are paired.
    sequence = [
        _make_botsim_imagefile(
            'W2000000001',
            shutter='BOTSIM',
            image_time='2004-001T00:00:00.000',
            observation_id='OBS_A',
        ),
        _make_botsim_imagefile(
            'N2000000002',
            shutter='BOTSIM',
            image_time='2004-001T00:00:00.900',
            observation_id='OBS_A',
        ),
        _make_botsim_imagefile(
            'N2000000050',
            shutter='BOTSIM',
            image_time='2004-001T00:05:00.000',
            observation_id='OBS_B',
        ),
        _make_botsim_imagefile(
            'W2000000100',
            shutter='BOTSIM',
            image_time='2004-001T00:10:00.000',
            observation_id='OBS_C',
        ),
        _make_botsim_imagefile(
            'N2000000101',
            shutter='BOTSIM',
            image_time='2004-001T00:10:00.900',
            observation_id='OBS_C',
        ),
    ]

    def fake_yield(**_kwargs: Any) -> Any:
        yield from sequence

    monkeypatch.setattr(ds_cassini_iss, '_yield_image_files_index', fake_yield)

    ret = list(ds_cassini_iss.yield_image_files_index(group='botsim'))

    # Three groups: pair, lone, pair. No frame dropped (5 frames total).
    assert len(ret) == 3
    assert len(ret[0].image_files) == 2
    assert len(ret[1].image_files) == 1
    assert len(ret[2].image_files) == 2
    total_frames = sum(len(g.image_files) for g in ret)
    assert total_frames == len(sequence)
    # First pair: NAC listed first.
    assert ret[0].image_files[0].image_file_name.startswith('N2000000002')
    assert ret[0].image_files[1].image_file_name.startswith('W2000000001')
    # Lone frame is the orphan BOTSIM with no partner.
    assert ret[1].image_files[0].image_file_name.startswith('N2000000050')
    # Second pair: NAC listed first.
    assert ret[2].image_files[0].image_file_name.startswith('N2000000101')
    assert ret[2].image_files[1].image_file_name.startswith('W2000000100')


def test_cassini_iss_group_botsim_no_pair_when_observation_differs(
    ds_cassini_iss: dscoiss.DataSetPDS3CassiniISS,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Opposite cameras, both BOTSIM, close in time, but DIFFERENT observation IDs must
    # NOT be paired; both are yielded individually.
    sequence = [
        _make_botsim_imagefile(
            'W2000000001',
            shutter='BOTSIM',
            image_time='2004-001T00:00:00.000',
            observation_id='OBS_A',
        ),
        _make_botsim_imagefile(
            'N2000000002',
            shutter='BOTSIM',
            image_time='2004-001T00:00:00.900',
            observation_id='OBS_B',
        ),
    ]

    def fake_yield(**_kwargs: Any) -> Any:
        yield from sequence

    monkeypatch.setattr(ds_cassini_iss, '_yield_image_files_index', fake_yield)

    ret = list(ds_cassini_iss.yield_image_files_index(group='botsim'))
    assert len(ret) == 2
    assert len(ret[0].image_files) == 1
    assert len(ret[1].image_files) == 1
