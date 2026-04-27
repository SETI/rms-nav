"""Tests for ``nav.nav_orchestrator.provenance.Provenance``."""

from nav.nav_orchestrator.provenance import Provenance


def test_provenance_minimal_construction() -> None:
    """The minimal required fields are version + ET + ISO timestamp."""
    prov = Provenance(
        rms_nav_version='0.5.2',
        image_et=414504000.0,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
    )
    assert prov.rms_nav_version == '0.5.2'
    assert prov.image_et == 414504000.0
    assert prov.spice_kernels == ()
    assert prov.spice_kernel_count == 0
    assert prov.static_data_hashes == {}


def test_provenance_carries_kernels_and_hashes() -> None:
    """Optional kernel list and hash dict are stored verbatim."""
    prov = Provenance(
        rms_nav_version='0.5.2',
        image_et=1.0,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
        spice_kernels=('a.bsp', 'b.tpc'),
        spice_kernel_count=2,
        static_data_hashes={'config_220_body_shape.yaml': 'deadbeef'},
    )
    assert prov.spice_kernels == ('a.bsp', 'b.tpc')
    assert prov.static_data_hashes['config_220_body_shape.yaml'] == 'deadbeef'
