"""Spec-first tests for PDS4 collection inventories and global index files (phase 2).

Contract under test (docs/user_guide/user_guide_pds4_bundle.rst "Summary Pass" /
"Summary Pass Outputs" and docs/dev_guide/dev_guide_pds4.rst "Pipeline overview"):
``generate_collection_files`` scans the bundle's ``data/`` tree for
``*_backplanes.lblx`` labels, sorts them by image name, and writes the
``collection_data.tab`` / ``collection_browse.tab`` inventories (``Member
Status`` + ``LIDVID_LID`` columns, one ``P`` row per product, LIDVIDs from the
dataset's ``pds4_image_name_to_*_lidvid`` builders) plus the matching
``.lblx`` labels when the templates exist.  ``generate_global_index_files``
scans ``data/`` for ``*_supplemental.txt`` files and writes
``document/supplemental/global_index_bodies.tab`` (one row per image/body) and
``global_index_rings.tab`` (one row per image with ring backplanes), with
min/max columns for each configured backplane type formatted to 5 decimal
places, plus their labels when templates exist.

Known bug pinned here as strict xfail: #139 (the global-index LID is hand-built
without the ``urn:nasa:pds:`` prefix and without the dataset's image-name
transformation, so it cannot cross-reference the product LIDVIDs in the
collection inventories).
"""

from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from spindoctor.cli.pds4.collections import (
    generate_collection_files,
    generate_global_index_files,
)
from spindoctor.config import MAIN_LOGGER
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISSSaturn

from .conftest import (
    COLLECTION_BROWSE_TEMPLATE,
    COLLECTION_DATA_TEMPLATE,
    GLOBAL_INDEX_TEMPLATE,
    BundleEnv,
    make_bundle_env,
    read_tab,
    touch_label,
    write_supplemental,
    write_templates,
)

BODY_STATS = {'MIMAS': {'backplanes': {'latitude': {'min': 1.234567891, 'max': 2}}}}
RING_STATS = {'backplanes': {'radius': {'min': 74500.0, 'max': 136800.987654}}}


def _run_collections(env: BundleEnv) -> None:
    """Run generate_collection_files against the environment's bundle root.

    Parameters:
        env: The hermetic bundle environment to process.
    """
    generate_collection_files(
        FCPath(env.bundle_results_root), env.dataset.as_dataset(), MAIN_LOGGER
    )


def _run_global_index(env: BundleEnv) -> None:
    """Run generate_global_index_files against the environment's bundle root.

    Parameters:
        env: The hermetic bundle environment to process.
    """
    generate_global_index_files(
        FCPath(env.bundle_results_root), env.dataset.as_dataset(), MAIN_LOGGER
    )


def _index_env(tmp_path: Path) -> BundleEnv:
    """Build an environment with body/ring backplane types configured.

    Parameters:
        tmp_path: Base temporary directory.

    Returns:
        A :class:`BundleEnv` whose config lists 'latitude'/'resolution' body
        backplanes and a 'radius' ring backplane.
    """
    return make_bundle_env(
        tmp_path,
        bodies=[{'name': 'latitude'}, {'name': 'resolution'}],
        rings=[{'name': 'radius'}],
    )


# ---------------------------------------------------------------------------
# generate_collection_files: inventories
# ---------------------------------------------------------------------------


def test_missing_data_dir_raises(tmp_path: Path) -> None:
    """A bundle without a data directory is rejected with FileNotFoundError."""
    env = make_bundle_env(tmp_path)
    with pytest.raises(FileNotFoundError, match='Data directory does not exist'):
        _run_collections(env)


def test_empty_data_dir_writes_header_only_inventories(tmp_path: Path) -> None:
    """An empty data tree yields inventories with only the header row."""
    env = make_bundle_env(tmp_path)
    (env.bundle_dir / 'data').mkdir(parents=True)
    _run_collections(env)
    data_rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    assert data_rows == [['Member Status', 'LIDVID_LID']]
    browse_rows = read_tab(env.bundle_dir / 'browse' / 'collection_browse.tab')
    assert browse_rows == [['Member Status', 'LIDVID_LID']]


def test_data_inventory_row_per_label_with_primary_status(tmp_path: Path) -> None:
    """Each *_backplanes.lblx yields one P row with the dataset's data LIDVID."""
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    _run_collections(env)
    rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    assert len(rows) == 2
    assert rows[1] == ['P', 'urn:nasa:pds:fake_bundle:data:1234567890w::1.0']


def test_browse_inventory_uses_browse_lidvids(tmp_path: Path) -> None:
    """The browse inventory lists the same images with browse LIDVIDs."""
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    _run_collections(env)
    rows = read_tab(env.bundle_dir / 'browse' / 'collection_browse.tab')
    assert rows[1] == ['P', 'urn:nasa:pds:fake_bundle:browse:1234567890w::1.0']


def test_inventory_rows_sorted_by_image_name_not_path(tmp_path: Path) -> None:
    """Inventory rows sort by extracted image name, ignoring shard directories."""
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'zz9/1111111111n')
    touch_label(env.bundle_dir / 'data', 'aa0/2222222222w')
    _run_collections(env)
    rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    assert rows[1][1] == 'urn:nasa:pds:fake_bundle:data:1111111111n::1.0'
    assert rows[2][1] == 'urn:nasa:pds:fake_bundle:data:2222222222w::1.0'


def test_duplicate_image_names_produce_duplicate_rows(tmp_path: Path) -> None:
    """Characterization: the same image name in two shards is listed twice."""
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    touch_label(env.bundle_dir / 'data', 'shard1/1234567890w')
    _run_collections(env)
    rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    assert len(rows) == 3
    assert rows[1] == rows[2]


def test_non_backplane_label_files_ignored(tmp_path: Path) -> None:
    """Only *_backplanes.lblx files are inventoried from the data tree."""
    env = make_bundle_env(tmp_path)
    other = env.bundle_dir / 'data' / 'shard0' / '1234567890w_other.lblx'
    other.parent.mkdir(parents=True)
    other.write_text('<x/>\n', encoding='utf-8')
    _run_collections(env)
    rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    assert rows == [['Member Status', 'LIDVID_LID']]


# ---------------------------------------------------------------------------
# generate_collection_files: labels
# ---------------------------------------------------------------------------


def test_collection_labels_rendered_when_templates_exist(tmp_path: Path) -> None:
    """Collection labels render with the CSV path and (empty) date-range variables."""
    env = make_bundle_env(
        tmp_path,
        template_contents={
            'collection_data.lblx': COLLECTION_DATA_TEMPLATE,
            'collection_browse.lblx': COLLECTION_BROWSE_TEMPLATE,
        },
    )
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    _run_collections(env)
    data_label = env.bundle_dir / 'data' / 'collection_data.lblx'
    text = data_label.read_text(encoding='utf-8')
    assert str(FCPath(env.bundle_dir) / 'data' / 'collection_data.tab') in text
    assert '<start></start>' in text
    assert '<stop></stop>' in text
    browse_label = env.bundle_dir / 'browse' / 'collection_browse.lblx'
    browse_text = browse_label.read_text(encoding='utf-8')
    assert str(FCPath(env.bundle_dir) / 'browse' / 'collection_browse.tab') in browse_text


def test_collection_labels_skipped_when_templates_missing(tmp_path: Path) -> None:
    """Missing collection templates skip the labels but still write the inventories."""
    env = make_bundle_env(tmp_path, template_contents={})
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    _run_collections(env)
    assert (env.bundle_dir / 'data' / 'collection_data.tab').is_file()
    assert not (env.bundle_dir / 'data' / 'collection_data.lblx').exists()
    assert not (env.bundle_dir / 'browse' / 'collection_browse.lblx').exists()


# ---------------------------------------------------------------------------
# generate_global_index_files: tables
# ---------------------------------------------------------------------------


def test_bodies_index_header_from_configured_backplane_types(tmp_path: Path) -> None:
    """The bodies index header lists min/max columns per configured body backplane."""
    env = _index_env(tmp_path)
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    _run_global_index(env)
    rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert rows[0] == [
        'LID',
        'body_name',
        'path_to_image_file',
        'latitude_min',
        'latitude_max',
        'resolution_min',
        'resolution_max',
    ]


def test_bodies_index_one_row_per_image_body(tmp_path: Path) -> None:
    """The bodies index has one row per (image, body) pair."""
    env = _index_env(tmp_path)
    two_bodies: dict[str, Any] = {
        'MIMAS': {'backplanes': {'latitude': {'min': 1.0, 'max': 2.0}}},
        'ENCELADUS': {'backplanes': {'latitude': {'min': 3.0, 'max': 4.0}}},
    }
    write_supplemental(env.bundle_dir / 'data', 'shard0/1111111111n', bodies=two_bodies)
    write_supplemental(env.bundle_dir / 'data', 'shard0/2222222222w', bodies=BODY_STATS)
    _run_global_index(env)
    rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert len(rows) == 4
    body_names = [row[1] for row in rows[1:]]
    assert body_names == ['MIMAS', 'ENCELADUS', 'MIMAS']


def test_bodies_index_numeric_values_formatted_to_five_decimals(tmp_path: Path) -> None:
    """Numeric min/max values are written with exactly five decimal places."""
    env = _index_env(tmp_path)
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    _run_global_index(env)
    rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert rows[1][3] == '1.23457'
    assert rows[1][4] == '2.00000'


def test_bodies_index_missing_backplane_values_blank(tmp_path: Path) -> None:
    """Backplane types absent from a body's stats produce empty columns."""
    env = _index_env(tmp_path)
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    _run_global_index(env)
    rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert rows[1][5] == ''
    assert rows[1][6] == ''


def test_index_path_to_image_file_is_data_relative(tmp_path: Path) -> None:
    """The path_to_image_file column points at data/<stub>_backplanes.lblx."""
    env = _index_env(tmp_path)
    write_supplemental(
        env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS, rings=RING_STATS
    )
    _run_global_index(env)
    bodies_rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert bodies_rows[1][2] == 'data/shard0/1234567890w_backplanes.lblx'
    rings_rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_rings.tab')
    assert rings_rows[1][1] == 'data/shard0/1234567890w_backplanes.lblx'


def test_rings_index_row_only_for_images_with_ring_backplanes(tmp_path: Path) -> None:
    """Images without ring backplanes are omitted from the rings index."""
    env = _index_env(tmp_path)
    write_supplemental(env.bundle_dir / 'data', 'shard0/1111111111n', bodies=BODY_STATS)
    write_supplemental(env.bundle_dir / 'data', 'shard0/2222222222w', rings=RING_STATS)
    _run_global_index(env)
    rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_rings.tab')
    assert rows[0] == ['LID', 'path_to_image_file', 'radius_min', 'radius_max']
    assert len(rows) == 2
    assert rows[1][1] == 'data/shard0/2222222222w_backplanes.lblx'
    assert rows[1][2] == '74500.00000'
    assert rows[1][3] == '136800.98765'


def test_no_supplemental_files_writes_header_only_indexes(tmp_path: Path) -> None:
    """With no supplemental files (even no data dir), header-only tables are written."""
    env = _index_env(tmp_path)
    _run_global_index(env)
    bodies_rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert len(bodies_rows) == 1
    rings_rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_rings.tab')
    assert len(rings_rows) == 1


def test_unreadable_supplemental_skipped_with_logged_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed supplemental file is skipped; other images are still indexed."""
    env = _index_env(tmp_path)
    write_supplemental(env.bundle_dir / 'data', 'shard0/1111111111n', raw_text='not json')
    write_supplemental(env.bundle_dir / 'data', 'shard0/2222222222w', bodies=BODY_STATS)
    _run_global_index(env)
    rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert len(rows) == 2
    assert '2222222222w' in rows[1][0]
    assert 'Error reading supplemental file' in capsys.readouterr().out


def test_global_index_labels_rendered_with_file_records(tmp_path: Path) -> None:
    """Global index labels render with FILE_RECORDS set to the row counts."""
    env = _index_env(tmp_path)
    write_templates(
        Path(env.dataset.pds4_bundle_template_dir()),
        {
            'global_index_bodies.lblx': GLOBAL_INDEX_TEMPLATE,
            'global_index_rings.lblx': GLOBAL_INDEX_TEMPLATE,
        },
    )
    two_bodies: dict[str, Any] = {
        'MIMAS': {'backplanes': {'latitude': {'min': 1.0, 'max': 2.0}}},
        'ENCELADUS': {'backplanes': {'latitude': {'min': 3.0, 'max': 4.0}}},
    }
    write_supplemental(
        env.bundle_dir / 'data', 'shard0/1234567890w', bodies=two_bodies, rings=RING_STATS
    )
    _run_global_index(env)
    supplemental_dir = env.bundle_dir / 'document' / 'supplemental'
    bodies_text = (supplemental_dir / 'global_index_bodies.lblx').read_text(encoding='utf-8')
    assert '<records>2</records>' in bodies_text
    rings_text = (supplemental_dir / 'global_index_rings.lblx').read_text(encoding='utf-8')
    assert '<records>1</records>' in rings_text


def test_global_index_labels_skipped_when_templates_missing(tmp_path: Path) -> None:
    """Missing global index templates skip the labels but still write the tables."""
    env = _index_env(tmp_path)
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    _run_global_index(env)
    supplemental_dir = env.bundle_dir / 'document' / 'supplemental'
    assert (supplemental_dir / 'global_index_bodies.tab').is_file()
    assert not (supplemental_dir / 'global_index_bodies.lblx').exists()
    assert not (supplemental_dir / 'global_index_rings.lblx').exists()


# ---------------------------------------------------------------------------
# LID cross-referencing (known bug #139 and LID round trips)
# ---------------------------------------------------------------------------


def _cross_reference_env(tmp_path: Path) -> BundleEnv:
    """Build a one-image bundle with a label and a supplemental file in place.

    Parameters:
        tmp_path: Base temporary directory.

    Returns:
        The environment, ready for both phase-2 generators.
    """
    env = _index_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    write_supplemental(
        env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS, rings=RING_STATS
    )
    return env


@pytest.mark.xfail(
    strict=True,
    reason='#139: the global-index LID is hand-built as <bundle>:data:<image> without '
    'the urn:nasa:pds: prefix and without the dataset LID builder, so it does not '
    'match the product LIDVIDs in collection_data.tab',
)
def test_global_index_bodies_lid_matches_collection_inventory(tmp_path: Path) -> None:
    """#139 round trip: the bodies-index LID equals the collection inventory LID."""
    env = _cross_reference_env(tmp_path)
    _run_collections(env)
    _run_global_index(env)
    inventory_rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    inventory_lid = inventory_rows[1][1].split('::')[0]
    bodies_rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert bodies_rows[1][0] == inventory_lid


@pytest.mark.xfail(
    strict=True,
    reason='#139: the global-index LID is hand-built as <bundle>:data:<image> without '
    'the urn:nasa:pds: prefix and without the dataset LID builder, so it does not '
    'match the product LIDVIDs in collection_data.tab',
)
def test_global_index_rings_lid_matches_collection_inventory(tmp_path: Path) -> None:
    """#139 round trip: the rings-index LID equals the collection inventory LID."""
    env = _cross_reference_env(tmp_path)
    _run_collections(env)
    _run_global_index(env)
    inventory_rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    inventory_lid = inventory_rows[1][1].split('::')[0]
    rings_rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_rings.tab')
    assert rings_rows[1][0] == inventory_lid


def test_inventory_lidvid_round_trips_with_canonical_builders(tmp_path: Path) -> None:
    """With canonical LID builders, the inventory LIDVID is the data LID plus ::1.0."""
    env = _cross_reference_env(tmp_path)
    _run_collections(env)
    rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    expected = env.dataset.pds4_image_name_to_data_lid('1234567890w') + '::1.0'
    assert rows[1][1] == expected


# ---------------------------------------------------------------------------
# Real Cassini dataset: LID/LIDVID construction and template tree
# ---------------------------------------------------------------------------


def _cassini_dataset(tmp_path: Path) -> DataSetPDS3CassiniISSSaturn:
    """Construct the reference Cassini Saturn dataset on a local holdings root.

    Parameters:
        tmp_path: Base temporary directory used as the (empty) holdings root.
    """
    return DataSetPDS3CassiniISSSaturn(tmp_path / 'holdings')


def test_cassini_data_lid_canonical_form(tmp_path: Path) -> None:
    """The Cassini data LID rotates the camera letter to a lowercase suffix."""
    dataset = _cassini_dataset(tmp_path)
    lid = dataset.pds4_image_name_to_data_lid('N1454725799')
    assert lid == 'urn:nasa:pds:cassini_iss_saturn_backplanes_rsfrench2027:data:1454725799n'


def test_cassini_lidvid_version_field(tmp_path: Path) -> None:
    """LIDVIDs append a ::1.0 version to the corresponding LID."""
    dataset = _cassini_dataset(tmp_path)
    lid = dataset.pds4_image_name_to_data_lid('N1454725799')
    lidvid = dataset.pds4_image_name_to_data_lidvid('N1454725799')
    assert lidvid == f'{lid}::1.0'


def test_cassini_browse_lid_uses_browse_collection(tmp_path: Path) -> None:
    """Browse LIDs differ from data LIDs only in the collection segment."""
    dataset = _cassini_dataset(tmp_path)
    lid = dataset.pds4_image_name_to_browse_lid('N1454725799')
    assert lid == 'urn:nasa:pds:cassini_iss_saturn_backplanes_rsfrench2027:browse:1454725799n'


def test_cassini_lid_strips_version_suffix_and_extension(tmp_path: Path) -> None:
    """Image-name version suffixes and extensions do not leak into the LID."""
    dataset = _cassini_dataset(tmp_path)
    plain = dataset.pds4_image_name_to_data_lid('N1454725799')
    suffixed = dataset.pds4_image_name_to_data_lid('N1454725799_1.IMG')
    assert suffixed == plain


@pytest.mark.xfail(
    strict=True,
    reason='#256: generate_collection_files feeds the '
    'on-disk image name (already LID-part form, e.g. 1454725799n) back into '
    'pds4_image_name_to_data_lidvid, which re-applies the N1454725799 -> 1454725799n '
    'rotation, yielding 454725799n1 - the inventory LIDVID cannot match the DATA_LID '
    'rendered inside the product label',
)
def test_cassini_inventory_lidvid_matches_label_lid(tmp_path: Path) -> None:
    """The Cassini collection inventory LIDVID matches the label's DATA_LID."""
    dataset = _cassini_dataset(tmp_path)
    bundle_results_root = tmp_path / 'bundle'
    bundle_dir = bundle_results_root / dataset.pds4_bundle_name()
    touch_label(bundle_dir / 'data', '1454xxxxxx/145472xxxx/1454725799n')
    generate_collection_files(FCPath(bundle_results_root), dataset, MAIN_LOGGER)
    rows = read_tab(bundle_dir / 'data' / 'collection_data.tab')
    inventory_lid = rows[1][1].split('::')[0]
    label_lid = dataset.pds4_image_name_to_data_lid('N1454725799')
    assert inventory_lid == label_lid


@pytest.mark.parametrize(
    'template_name',
    [
        'bundle.lblx',
        'readme.txt',
        'data.lblx',
        'browse.lblx',
        'collection_data.lblx',
        'collection_browse.lblx',
        'collection_context.lblx',
        'collection_context.csv',
        'collection_document.lblx',
        'collection_document.csv',
        'collection_xml_schema.lblx',
        'collection_xml_schema.csv',
        'global_index_bodies.lblx',
        'global_index_rings.lblx',
        'cassini-iss-saturn-backplanes-user-guide.lblx',
    ],
)
def test_cassini_template_tree_ships_documented_files(tmp_path: Path, template_name: str) -> None:
    """Every file in the dev guide's reference template tree ships as package data."""
    dataset = _cassini_dataset(tmp_path)
    template_dir = Path(dataset.pds4_bundle_template_dir())
    assert (template_dir / template_name).is_file()
