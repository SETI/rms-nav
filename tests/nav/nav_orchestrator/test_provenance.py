"""Tests for ``nav.nav_orchestrator.provenance.Provenance``."""

from nav.nav_orchestrator.provenance import (
    Provenance,
    ProvenanceMetadata,
    collect_provenance_metadata,
)


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
        static_data_hashes={'config_220_body_shape.yaml': 'deadbeef'},
    )
    assert prov.spice_kernels == ('a.bsp', 'b.tpc')
    assert prov.spice_kernel_count == 2
    assert prov.static_data_hashes['config_220_body_shape.yaml'] == 'deadbeef'


def test_provenance_static_data_hashes_is_immutable() -> None:
    """``static_data_hashes`` is wrapped as ``MappingProxyType``."""
    import pytest

    prov = Provenance(
        rms_nav_version='0.5.2',
        image_et=1.0,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
        static_data_hashes={'a.yaml': 'aa'},
    )
    with pytest.raises(TypeError) as exc_info:
        prov.static_data_hashes['b.yaml'] = 'bb'  # type: ignore[index]
    # ``MappingProxyType`` raises ``TypeError`` with "does not support
    # item assignment" when assignment is attempted.
    assert 'item assignment' in str(exc_info.value)


def test_collect_provenance_metadata_returns_dataclass() -> None:
    """``collect_provenance_metadata`` returns a populated dataclass."""
    from collections.abc import Mapping
    from types import MappingProxyType

    meta = collect_provenance_metadata()
    assert isinstance(meta, ProvenanceMetadata)
    # The git SHA may be ``None`` if the working tree isn't a repo, but
    # the field must exist.
    assert isinstance(meta.git_sha, str | type(None))
    assert isinstance(meta.spice_kernels, tuple)
    # ``static_data_hashes`` is exposed as ``Mapping[str, str]`` and the
    # implementation returns a ``MappingProxyType``; pin both shape and
    # contents so a future refactor that swaps the wrapping type still
    # has to satisfy the read-only-mapping contract.
    assert isinstance(meta.static_data_hashes, Mapping)
    assert isinstance(meta.static_data_hashes, MappingProxyType)
    for key, value in meta.static_data_hashes.items():
        assert isinstance(key, str)
        assert isinstance(value, str)


def test_collect_provenance_metadata_hashes_static_data_yamls() -> None:
    """The hash dict covers config_220_body_shape.yaml and the inst configs."""
    meta = collect_provenance_metadata()
    names = set(meta.static_data_hashes.keys())
    # Every shipped 4N0 instrument block plus the body-shape catalogue
    # must appear; ring catalogues (3N0) are also static data.
    assert 'config_220_body_shape.yaml' in names
    assert 'config_400_inst_coiss.yaml' in names
    assert 'config_310_saturn_rings.yaml' in names
    # SHA-256 hex digest is 64 chars.
    for digest in meta.static_data_hashes.values():
        assert len(digest) == 64
        assert all(c in '0123456789abcdef' for c in digest)


def test_collect_provenance_metadata_is_byte_identical_across_calls() -> None:
    """Two consecutive calls produce identical hashes (modulo wall-clock)."""
    a = collect_provenance_metadata()
    b = collect_provenance_metadata()
    assert dict(a.static_data_hashes) == dict(b.static_data_hashes)
    assert a.spice_kernels == b.spice_kernels
