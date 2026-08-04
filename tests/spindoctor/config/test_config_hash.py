"""Tests for what the provenance configuration digest covers.

The digest answers one question -- was this result produced by the same
configuration as that one -- so it has to cover everything that could change a
result and nothing that could not. A setting that cannot move an offset but
still changes the digest makes every result of a run compare as differently
configured against the archive it belongs to.
"""

from pathlib import Path

import pytest

from spindoctor.config.config import HASH_EXCLUDED_SECTIONS, Config


def _hash(tmp_path: Path, body: str = '') -> str:
    """Return the resolved digest of the shipped configuration plus an override.

    Parameters:
        tmp_path: Directory the override file is written into.
        body: YAML text applied as an override; empty for the defaults alone.

    Returns:
        The digest.
    """
    config = Config()
    config.read_config()
    if body:
        override = tmp_path / 'override.yaml'
        override.write_text(body)
        config.update_config(str(override))
    return str(config.resolved_config_hash())


@pytest.mark.parametrize(
    'body',
    [
        'logging:\n  image: DEBUG\n',
        'logging:\n  main: WARNING\n',
        'logging:\n  techniques:\n    titan_haze: DEBUG\n',
        'logging:\n  models:\n    rings: NONE\n',
        'logging:\n  other:\n    annotate: DEBUG\n',
        'logging:\n  programs:\n    sd_mosaic:\n      main: DEBUG\n',
        'logging:\n  strict_scope: true\n',
    ],
)
def test_a_logging_setting_does_not_change_the_digest(tmp_path: Path, body: str) -> None:
    """Looking harder at a run does not make its results differently configured.

    Raising one component's level is an everyday thing to do while
    investigating, and it would otherwise re-stamp every result of that run.
    """
    assert _hash(tmp_path, body) == _hash(tmp_path)


def test_a_setting_that_can_change_a_result_does_change_the_digest(tmp_path: Path) -> None:
    """The exclusion is narrow: anything the pipeline reads still counts."""
    assert _hash(tmp_path, 'offset:\n  correlation_fft_upsample_factor: 256\n') != _hash(tmp_path)


def test_the_digest_is_stable_across_repeated_resolution(tmp_path: Path) -> None:
    """The same configuration resolves to the same digest every time."""
    assert _hash(tmp_path) == _hash(tmp_path)


def test_only_logging_is_excluded() -> None:
    """The excluded set is exactly what was argued for, not a growing list.

    Every section added to it stops being able to distinguish two results, so
    growth wants the same argument made again rather than a quiet append.
    """
    assert sorted(HASH_EXCLUDED_SECTIONS) == ['logging']


def test_an_excluded_section_is_still_present_in_the_configuration() -> None:
    """Excluding a section from the digest does not remove it from the config."""
    config = Config()
    config.read_config()
    assert config.logging is not None
