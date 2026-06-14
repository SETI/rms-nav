"""Tests for nested merging behavior of :meth:`Config.update_config`.

A user override of one nested key must combine key-by-key with the bundled
defaults so sibling keys under the same sub-block survive (CODE-CFG-1).
"""

from __future__ import annotations

from pathlib import Path

from nav.config import Config
from nav.config.config import _deep_merge


def test_update_config_preserves_sibling_defaults(tmp_path: Path) -> None:
    """A user override of one leaf keeps unrelated sibling defaults."""
    config = Config()
    config._config_dict = {'bodies': {'MIMAS': {'radii_km': [1], 'albedo': 2}}}
    config._update_attrdicts()

    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text('bodies:\n  MIMAS:\n    albedo: 9\n', encoding='utf-8')
    config.update_config(user_path, read_default=False)

    assert config.bodies['MIMAS']['radii_km'] == [1]
    assert config.bodies['MIMAS']['albedo'] == 9


def test_deep_merge_recurses_into_nested_mappings() -> None:
    """Nested mappings combine key-by-key with overlay leaves winning."""
    base = {'MIMAS': {'radii_km': [1], 'albedo': 2}}
    overlay = {'MIMAS': {'albedo': 9}}
    assert _deep_merge(base, overlay) == {'MIMAS': {'radii_km': [1], 'albedo': 9}}


def test_deep_merge_overlay_replaces_non_mapping() -> None:
    """When either side is not a mapping, the overlay value replaces it."""
    base = {'k': [1, 2]}
    overlay = {'k': [3]}
    assert _deep_merge(base, overlay) == {'k': [3]}


def test_deep_merge_does_not_mutate_inputs() -> None:
    """The inputs are left unchanged after merging."""
    base = {'a': {'x': 1}}
    overlay = {'a': {'y': 2}}
    _deep_merge(base, overlay)
    assert base == {'a': {'x': 1}}


def test_category_returns_cached_instance() -> None:
    """CODE-CFG-3: category() returns the same cached AttrDict across calls."""
    config = Config()
    config._config_dict = {'bodies': {'MIMAS': {'albedo': 2}}}
    config._update_attrdicts()
    first = config.category('bodies')
    second = config.category('bodies')
    assert first is second


def test_category_cache_rebuilt_on_reload() -> None:
    """CODE-CFG-3: the category cache is dropped when the config is reloaded."""
    config = Config()
    config._config_dict = {'bodies': {'MIMAS': {'albedo': 2}}}
    config._update_attrdicts()
    before = config.category('bodies')
    config._config_dict = {'bodies': {'MIMAS': {'albedo': 9}}}
    config._update_attrdicts()
    after = config.category('bodies')
    assert after is not before
    assert after['MIMAS']['albedo'] == 9


def test_reread_no_path_drops_removed_keys() -> None:
    """CODE-CFG-2: a no-path reread starts from an empty dict (no stale keys)."""
    config = Config()
    # Simulate a prior load that contains a key the bundled YAML does not.
    config._config_dict = {'general': {'stale_only_key': 123}}
    config._update_attrdicts()
    # Reread from the bundled config_files glob.
    config.read_config(reread=True)
    assert 'stale_only_key' not in config._config_dict.get('general', {})
