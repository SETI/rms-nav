"""Spec-first tests for the per-image backplane driver and the backplane config.

Contract under test (docs/dev_guide/dev_guide_backplanes.rst "Pipeline overview" and
"Restrictions and assumptions"): the driver reads the per-image navigation
``_metadata.json`` from the nav results root, refuses to proceed unless
``status == 'success'``, builds the snapshot with a zero extended-FOV margin, wraps
its FOV in an ``OffsetFOV`` carrying the navigated ``(dv, du)`` offset (``(0, 0)``
with a warning when the offset is None), and hands the per-source results to the
merge and the writer.  The configured backplane list in
``config_900_backplanes.yaml`` must name real ``oops.Backplane`` methods and declare
angle units in radians.
"""

import json
from pathlib import Path
from typing import Any, ClassVar, cast

import cspyce
import numpy as np
import oops
import pdslogger
import pytest
from astropy.io import fits
from filecache import FCPath
from oops.backplane import Backplane

from spindoctor.cli.backplanes import backplanes as backplanes_mod
from spindoctor.cli.backplanes.backplanes import generate_backplanes_image_files
from spindoctor.config import (
    DEFAULT_CONFIG,
    MAIN_LOGGER,
    Config,
    LogLevels,
    LogSinks,
    build_main_logger,
    set_log_levels,
)
from spindoctor.config.program_names import SD_BACKPLANES
from spindoctor.dataset.dataset import ImageFile, ImageFiles
from spindoctor.obs import Obs, ObsSnapshotInst
from spindoctor.support.types import PathLike

from .conftest import HermeticObs, inventory_entry, make_snapshot

SHAPE_VU = (10, 12)


# ---------------------------------------------------------------------------
# Configuration contract (config_900_backplanes.yaml)
# ---------------------------------------------------------------------------


def _config_entries(kind: str) -> list[dict[str, Any]]:
    """Return the shipping backplane entries of the given kind.

    Parameters:
        kind: Either 'bodies' or 'rings'.
    """
    return cast('list[dict[str, Any]]', getattr(DEFAULT_CONFIG.backplanes, kind))


@pytest.mark.parametrize('kind', ['bodies', 'rings'])
def test_default_config_entries_have_required_fields(kind: str) -> None:
    """Every shipping backplane entry declares name, method, and units.

    Parameters:
        kind: The config list under test ('bodies' or 'rings').
    """
    entries = _config_entries(kind)
    assert len(entries) > 0
    for entry in entries:
        assert entry.get('name'), f'{kind} entry missing name: {entry}'
        assert entry.get('method'), f'{kind} entry missing method: {entry}'
        assert entry.get('units'), f'{kind} entry missing units: {entry}'


@pytest.mark.parametrize('kind', ['bodies', 'rings'])
def test_default_config_methods_exist_on_oops_backplane(kind: str) -> None:
    """Every configured method resolves to a callable on oops.Backplane.

    The dev guide warns there is no compile-time check on the YAML; this test is
    that check.

    Parameters:
        kind: The config list under test ('bodies' or 'rings').
    """
    for entry in _config_entries(kind):
        method = entry['method']
        assert callable(getattr(Backplane, method, None)), (
            f'{kind} backplane {entry["name"]} names unknown oops method {method}'
        )


def test_default_config_angle_backplanes_declare_radians() -> None:
    """Angle-valued backplanes (angles, longitudes, latitudes) are declared in rad.

    The viewer converts BUNIT=rad HDUs to degrees for display; a mislabeled angle
    plane would silently ship wrong units.
    """
    angle_names = {'body_longitude', 'body_latitude', 'ring_longitude'}
    for kind in ('bodies', 'rings'):
        for entry in _config_entries(kind):
            name = entry['name']
            if name.endswith('_angle') or name in angle_names:
                assert entry['units'] == 'rad', f'{name} must declare rad units'


# ---------------------------------------------------------------------------
# Driver plumbing helpers
# ---------------------------------------------------------------------------


def _image_files(images_root: FCPath, *stubs: str) -> ImageFiles:
    """Build an ImageFiles batch with one local image file per stub.

    Parameters:
        images_root: Directory that receives the placeholder image files.
        *stubs: Results path stubs, one per image.
    """
    files = []
    for stub in stubs or ('IMG1',):
        img = images_root / f'{stub}.IMG'
        img.write_bytes(b'')
        files.append(
            ImageFile(
                image_file_url=img,
                label_file_url=img,
                results_path_stub=stub,
            )
        )
    return ImageFiles(image_files=files)


def _write_nav_metadata(nav_root: FCPath, stub: str, doc: dict[str, Any]) -> None:
    """Write a navigation metadata JSON for the given stub.

    Parameters:
        nav_root: The nav results root directory.
        stub: The results path stub.
        doc: The metadata document to serialize.
    """
    (nav_root / f'{stub}_metadata.json').write_text(json.dumps(doc))


def _roots(root: FCPath) -> tuple[FCPath, FCPath]:
    """Create and return (nav_results_root, backplane_results_root) directories.

    Parameters:
        root: Temporary directory the roots are created under.
    """
    nav_root = root / 'nav'
    bp_root = root / 'bp'
    nav_root.mkdir()
    bp_root.mkdir()
    return nav_root, bp_root


def _obs_class_for(obs: Any) -> tuple[type[ObsSnapshotInst], list[dict[str, Any]]]:
    """Build an obs class whose from_file returns a fixed observation.

    Parameters:
        obs: The observation (or arbitrary object) from_file should return.

    Returns:
        The ObsSnapshotInst subclass and the list that records the keyword
        arguments of every from_file call.
    """
    from_file_calls: list[dict[str, Any]] = []

    class _DriverObs(HermeticObs):
        """HermeticObs whose from_file records its arguments and returns ``obs``."""

        calls: ClassVar[list[dict[str, Any]]] = from_file_calls

        @staticmethod
        def from_file(
            path: PathLike,
            *,
            config: Config | None = None,
            extfov_margin_vu: tuple[int, int] | None = None,
            **kwargs: Any,
        ) -> Obs:
            """Record the call and return the fixed observation.

            Parameters:
                path: Image file path requested by the driver (recorded).
                config: Ignored.
                extfov_margin_vu: Extended-FOV margin requested (recorded).
                **kwargs: Ignored.
            """
            from_file_calls.append({'path': path, 'extfov_margin_vu': extfov_margin_vu})
            return cast(Obs, obs)

    return _DriverObs, from_file_calls


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Replace the per-source, merge, and writer stages with recorders.

    Parameters:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        Dict of recorded calls per stage ('bodies', 'rings', 'merge', 'write').
    """
    calls: dict[str, list[Any]] = {'bodies': [], 'rings': [], 'merge': [], 'write': []}
    sentinel_master: dict[str, Any] = {'sentinel_plane': np.ones(SHAPE_VU, dtype=np.float32)}

    def fake_bodies(snapshot: Any, config: Any, *, logger: Any) -> dict[str, Any]:
        """Record the snapshot and return a one-body sentinel bodies_result.

        Parameters:
            snapshot: The observation handed to the body stage (recorded).
            config: Ignored.
            logger: Ignored.
        """
        calls['bodies'].append(snapshot)
        return {'FAKEBODY': {}}

    def fake_rings(snapshot: Any, config: Any, *, logger: Any) -> None:
        """Record the snapshot and report no ring backplanes.

        Parameters:
            snapshot: The observation handed to the ring stage (recorded).
            config: Ignored.
            logger: Ignored.
        """
        calls['rings'].append(snapshot)
        return None

    def fake_merge(
        snapshot: Any, *, bodies_result: Any, rings_result: Any
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Record the per-source results and return the sentinel master arrays.

        Parameters:
            snapshot: The observation handed to the merge (sizes the ID map).
            bodies_result: Body-stage result (recorded).
            rings_result: Ring-stage result (recorded).
        """
        calls['merge'].append({'bodies_result': bodies_result, 'rings_result': rings_result})
        return sentinel_master, np.zeros(snapshot.data.shape, dtype=np.int32)

    def fake_write(**kwargs: Any) -> None:
        """Record the writer keyword arguments without writing anything.

        Parameters:
            **kwargs: The write_fits keyword arguments (recorded verbatim).
        """
        calls['write'].append(kwargs)

    monkeypatch.setattr(backplanes_mod, 'create_body_backplanes', fake_bodies)
    monkeypatch.setattr(backplanes_mod, 'create_ring_backplanes', fake_rings)
    monkeypatch.setattr(backplanes_mod, 'merge_sources_into_master', fake_merge)
    monkeypatch.setattr(backplanes_mod, 'write_fits', fake_write)
    return calls


def _run(
    tmp_path: Path,
    *,
    metadata: dict[str, Any] | None,
    obs: Any | None = None,
    write_output_files: bool = True,
) -> tuple[Any, list[dict[str, Any]], FCPath, FCPath]:
    """Prepare roots and metadata, run the driver, and return the pieces.

    Parameters:
        tmp_path: pytest-provided temporary directory, wrapped once into FCPath.
        metadata: Nav metadata document to write; None writes nothing.
        obs: Observation returned by from_file; defaults to a fresh simulated
            HermeticObs.
        write_output_files: Forwarded to the driver.

    Returns:
        The observation, the recorded from_file calls, the nav root, and the
        backplane root.
    """
    root = FCPath(tmp_path)
    nav_root, bp_root = _roots(root)
    if metadata is not None:
        _write_nav_metadata(nav_root, 'IMG1', metadata)
    snapshot = obs if obs is not None else make_snapshot(shape_vu=SHAPE_VU, simulated=True)
    obs_class, from_file_calls = _obs_class_for(snapshot)
    generate_backplanes_image_files(
        obs_class,
        _image_files(root, 'IMG1'),
        nav_results_root=nav_root,
        backplane_results_root=bp_root,
        write_output_files=write_output_files,
    )
    return snapshot, from_file_calls, nav_root, bp_root


# ---------------------------------------------------------------------------
# Driver behavior
# ---------------------------------------------------------------------------


def test_driver_rejects_multi_image_batches(tmp_path: Path) -> None:
    """A batch with more than one image is rejected.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    root = FCPath(tmp_path)
    nav_root, bp_root = _roots(root)
    obs_class, _ = _obs_class_for(make_snapshot(shape_vu=SHAPE_VU, simulated=True))
    with pytest.raises(ValueError, match='exactly one image per batch'):
        generate_backplanes_image_files(
            obs_class,
            _image_files(root, 'IMG1', 'IMG2'),
            nav_results_root=nav_root,
            backplane_results_root=bp_root,
        )


def test_driver_missing_metadata_raises(tmp_path: Path) -> None:
    """A missing navigation metadata file is a hard error.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    with pytest.raises(FileNotFoundError):
        _run(tmp_path, metadata=None)


def test_driver_invalid_metadata_json_raises(tmp_path: Path) -> None:
    """Unparseable navigation metadata is a hard error.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    root = FCPath(tmp_path)
    nav_root, bp_root = _roots(root)
    (nav_root / 'IMG1_metadata.json').write_text('this is not json')
    obs_class, _ = _obs_class_for(make_snapshot(shape_vu=SHAPE_VU, simulated=True))
    with pytest.raises(json.JSONDecodeError):
        generate_backplanes_image_files(
            obs_class,
            _image_files(root, 'IMG1'),
            nav_results_root=nav_root,
            backplane_results_root=bp_root,
        )


def test_driver_skips_image_with_failed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An image whose navigation did not succeed is skipped without output.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    calls = _stub_pipeline(monkeypatch)
    _, from_file_calls, _, bp_root = _run(
        tmp_path, metadata={'status': 'error', 'offset': [0.0, 0.0]}
    )
    assert from_file_calls == []
    assert calls['write'] == []
    assert list(bp_root.iterdir()) == []


def test_driver_skips_image_with_missing_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata without a status field is treated as a failed navigation.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    calls = _stub_pipeline(monkeypatch)
    _, from_file_calls, _, _ = _run(tmp_path, metadata={'offset': [0.0, 0.0]})
    assert from_file_calls == []
    assert calls['write'] == []


def test_driver_requires_offset_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful metadata without an offset field is a hard error.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    _stub_pipeline(monkeypatch)
    with pytest.raises(ValueError, match='"offset" field not found'):
        _run(tmp_path, metadata={'status': 'success'})


def test_driver_defaults_none_offset_to_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A None offset falls back to (0, 0) and the pipeline still runs.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    calls = _stub_pipeline(monkeypatch)
    snapshot, _, _, _ = _run(tmp_path, metadata={'status': 'success', 'offset': None})
    assert isinstance(snapshot.fov, oops.fov.OffsetFOV)
    assert snapshot.fov.uv_offset[0] == 0.0
    assert snapshot.fov.uv_offset[1] == 0.0
    assert len(calls['merge']) == 1


def test_driver_applies_offset_as_du_dv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The metadata (dv, du) offset becomes an OffsetFOV uv_offset of (du, dv).

    The nav metadata stores offsets in (v, u) order while oops FOVs use (u, v);
    swapping them is the classic axis bug this test pins down.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    _stub_pipeline(monkeypatch)
    snapshot, _, _, _ = _run(tmp_path, metadata={'status': 'success', 'offset': [2.5, -1.25]})
    assert isinstance(snapshot.fov, oops.fov.OffsetFOV)
    assert snapshot.fov.uv_offset[0] == -1.25
    assert snapshot.fov.uv_offset[1] == 2.5


def test_driver_reads_image_with_zero_extfov_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The snapshot is built on the sensor only (extfov margin (0, 0)).

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    _stub_pipeline(monkeypatch)
    _, from_file_calls, _, _ = _run(tmp_path, metadata={'status': 'success', 'offset': [0.0, 0.0]})
    assert len(from_file_calls) == 1
    assert from_file_calls[0]['extfov_margin_vu'] == (0, 0)


def test_driver_rejects_non_snapshot_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An observation that is not an ObsSnapshot is rejected.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    _stub_pipeline(monkeypatch)
    with pytest.raises(TypeError, match='Expected ObsSnapshot'):
        _run(tmp_path, metadata={'status': 'success', 'offset': [0.0, 0.0]}, obs=object())


def test_driver_skips_writing_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """write_output_files=False runs the pipeline but writes nothing.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    calls = _stub_pipeline(monkeypatch)
    _run(
        tmp_path,
        metadata={'status': 'success', 'offset': [0.0, 0.0]},
        write_output_files=False,
    )
    assert len(calls['merge']) == 1
    assert calls['write'] == []


def test_driver_writes_fits_under_backplane_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The FITS output path is <backplane_root>/<stub>_backplanes.fits.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    calls = _stub_pipeline(monkeypatch)
    _, _, _, bp_root = _run(tmp_path, metadata={'status': 'success', 'offset': [0.0, 0.0]})
    assert len(calls['write']) == 1
    fits_file_path = calls['write'][0]['fits_file_path']
    assert fits_file_path.as_posix() == (bp_root / 'IMG1_backplanes.fits').as_posix()


def test_driver_passes_merge_outputs_to_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer receives exactly the master arrays and ID map the merge produced.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    calls = _stub_pipeline(monkeypatch)
    _run(tmp_path, metadata={'status': 'success', 'offset': [0.0, 0.0]})
    write_kwargs = calls['write'][0]
    assert 'sentinel_plane' in write_kwargs['master_by_type']
    assert calls['merge'][0]['bodies_result'] == {'FAKEBODY': {}}
    assert write_kwargs['rings_result'] is None


# ---------------------------------------------------------------------------
# Hermetic end-to-end run on a simulated observation
# ---------------------------------------------------------------------------

EXPECTED_BODY_HDUS = {
    'BODY_LONGITUDE',
    'BODY_LATITUDE',
    'BODY_INCIDENCE_ANGLE',
    'BODY_EMISSION_ANGLE',
    'BODY_PHASE_ANGLE',
    'BODY_FINEST_RESOLUTION',
    'BODY_COARSEST_RESOLUTION',
}


def _simulated_mimas_snapshot() -> tuple[HermeticObs, np.ndarray]:
    """Build a simulated snapshot with one masked body for the end-to-end run.

    Returns:
        The observation and the full-frame boolean MIMAS mask.
    """
    mask = np.zeros(SHAPE_VU, dtype=bool)
    mask[2:6, 3:7] = True
    inventory = {
        'MIMAS': inventory_entry(
            u_min=3,
            u_max=6,
            v_min=2,
            v_max=5,
            body_range=500000.0,
            center_uv=(4.5, 3.5),
            u_pixel_size=4.0,
            v_pixel_size=4.0,
        )
    }
    snap = make_snapshot(
        shape_vu=SHAPE_VU,
        simulated=True,
        sim_inventory=inventory,
        sim_body_mask_map={'MIMAS': mask},
    )
    return snap, mask


def test_end_to_end_simulated_image_writes_expected_fits(tmp_path: Path) -> None:
    """A full driver run on a simulated image produces the documented FITS layout.

    Uses the shipping config_900_backplanes.yaml body list, the real per-source,
    merge, and writer stages, and no SPICE beyond the built-in body-name table.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    snap, mask = _simulated_mimas_snapshot()
    _, _, _, bp_root = _run(
        tmp_path, metadata={'status': 'success', 'offset': [1.0, -2.0]}, obs=snap
    )
    fits_path = bp_root / 'IMG1_backplanes.fits'
    assert fits_path.exists()
    with fits.open(fits_path.get_local_path()) as hdul:
        names = [hdu.name for hdu in hdul]
        assert names[1] == 'BODY_ID_MAP'
        assert set(names[1:]) == {'BODY_ID_MAP'} | EXPECTED_BODY_HDUS
        assert hdul['BODY_LATITUDE'].header['BUNIT'] == 'rad'
        assert hdul['BODY_FINEST_RESOLUTION'].header['BUNIT'] == 'km/pixel'
        id_map = hdul['BODY_ID_MAP'].data
        assert np.all(id_map[mask] == int(cspyce.bodn2c('MIMAS')))
        assert np.all(id_map[~mask] == 0)
        assert np.all(hdul['BODY_LATITUDE'].data[mask] > 0.0)


def test_end_to_end_simulated_image_applies_offset(tmp_path: Path) -> None:
    """The end-to-end run wraps the FOV with the navigated (dv, du) offset.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    snap, _ = _simulated_mimas_snapshot()
    _run(tmp_path, metadata={'status': 'success', 'offset': [1.0, -2.0]}, obs=snap)
    assert isinstance(snap.fov, oops.fov.OffsetFOV)
    assert snap.fov.uv_offset[0] == -2.0
    assert snap.fov.uv_offset[1] == 1.0


def test_end_to_end_simulated_image_writes_sidecar(tmp_path: Path) -> None:
    """The end-to-end run writes the sidecar with per-body stats and inventory.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    snap, _ = _simulated_mimas_snapshot()
    _, _, _, bp_root = _run(
        tmp_path, metadata={'status': 'success', 'offset': [1.0, -2.0]}, obs=snap
    )
    sidecar = bp_root / 'IMG1_backplane_metadata.json'
    assert sidecar.exists()
    metadata = json.loads(sidecar.read_text())
    body = metadata['bodies']['MIMAS']
    assert set(body['backplanes']) == {e['name'] for e in _config_entries('bodies')}
    assert body['center_uv'] == [3.5, 4.5]
    assert body['center_range'] == 500000.0


# ---------------------------------------------------------------------------
# Backplanes computed on uncorrected pointing
# ---------------------------------------------------------------------------


def _run_with_null_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], str]:
    """Generate backplanes for an image whose navigation recorded no offset.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        The driver's result and the text written to the run's log.
    """
    _stub_pipeline(monkeypatch)
    root = FCPath(tmp_path)
    nav_root, bp_root = _roots(root)
    _write_nav_metadata(nav_root, 'IMG1', {'status': 'success', 'offset': None})
    obs_class, _ = _obs_class_for(make_snapshot(shape_vu=SHAPE_VU, simulated=True))

    levels = LogLevels()
    set_log_levels(levels)
    log_path = build_main_logger(
        MAIN_LOGGER,
        SD_BACKPLANES,
        LogSinks(log_root=root / 'runlog', main_console=False),
        levels,
        timestamp='2026-07-29T12-00-00',
    )
    try:
        result = generate_backplanes_image_files(
            obs_class,
            _image_files(root, 'IMG1'),
            nav_results_root=nav_root,
            backplane_results_root=bp_root,
            write_output_files=False,
        )
    finally:
        for handler in list(MAIN_LOGGER.handlers):
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
    assert log_path is not None
    with log_path.open('r') as stream:
        return result, str(stream.read())


def test_uncorrected_pointing_reaches_the_run_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backplanes built on uncorrected pointing say so where a run is watched.

    The product carries no sign of it, so someone following a batch would
    otherwise have to open every image's log to find out.
    """
    _, log_text = _run_with_null_offset(tmp_path, monkeypatch)
    assert 'uncorrected pointing' in log_text


def test_uncorrected_pointing_names_the_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And says which image, since a batch has many."""
    _, log_text = _run_with_null_offset(tmp_path, monkeypatch)
    assert 'IMG1' in log_text


def test_uncorrected_pointing_is_returned_to_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cloud task has no run log, so the result is how the fact leaves it."""
    result, _ = _run_with_null_offset(tmp_path, monkeypatch)
    assert result['uncorrected_pointing'] is True


def test_a_navigated_image_is_not_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An image with a real offset carries no such flag."""
    _stub_pipeline(monkeypatch)
    root = FCPath(tmp_path)
    nav_root, bp_root = _roots(root)
    _write_nav_metadata(nav_root, 'IMG1', {'status': 'success', 'offset': [1.5, -2.5]})
    obs_class, _ = _obs_class_for(make_snapshot(shape_vu=SHAPE_VU, simulated=True))
    result = generate_backplanes_image_files(
        obs_class,
        _image_files(root, 'IMG1'),
        nav_results_root=nav_root,
        backplane_results_root=bp_root,
        write_output_files=False,
    )
    assert 'uncorrected_pointing' not in result
