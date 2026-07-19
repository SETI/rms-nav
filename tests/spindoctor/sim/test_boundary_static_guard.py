"""Static guard: navigator-side code never references a truth attribute name.

The information boundary's filtered channel (``obs.nav_params``) is structural,
but the full scene truth still rides the observation object as plain
attributes (``obs.sim_params``, the planted ``sim_offset_v`` / ``sim_offset_u``,
and the renderer's output records) -- one attribute access away from any
navigator-side model's ``self.obs``.  This module is the mechanical consumer
guard: it walks the AST of every module under ``spindoctor.nav_model``,
``spindoctor.nav_technique``, and ``spindoctor.nav_orchestrator`` and fails if
any of them references a truth-bearing attribute name, either as an attribute
access (``obs.sim_params``) or as a bare string literal (``getattr(obs,
'sim_params')``, ``vars(obs)['sim_params']``).

The deny-list is every ``sim_``-prefixed attribute ``ObsSim.from_file`` stores
on the snapshot, plus the historical ``sim_star_list`` name so it cannot
return; a completeness test parses ``obs_inst_sim.py`` so a new truth
attribute added there without a deny-list extension fails here, not silently.

Legitimate exceptions, should one ever exist, go in ``_ALLOWED_REFERENCES``:
a mapping from ``(module path relative to the spindoctor package, name)`` to a
one-line justification.  The allowlist is currently empty -- no navigator-side
module has a sanctioned reason to name a truth attribute.
"""

import ast
from pathlib import Path

import pytest

import spindoctor

# Truth-bearing attribute names stored on the simulated observation by
# ObsSim.from_file (src/spindoctor/obs/obs_inst_sim.py), plus the historical
# sim_star_list attribute (removed from the observation; denied so a future
# change cannot quietly reintroduce a consumer).
TRUTH_ATTRIBUTE_DENYLIST: frozenset[str] = frozenset(
    {
        'sim_params',
        'sim_offset_v',
        'sim_offset_u',
        'sim_time',
        'sim_body_models',
        'sim_inventory',
        'sim_body_order_near_to_far',
        'sim_body_index_map',
        'sim_body_mask_map',
        'sim_star_list',
    }
)

# Navigator-side packages the guard scans, relative to the spindoctor package.
_NAVIGATOR_PACKAGES: tuple[str, ...] = ('nav_model', 'nav_technique', 'nav_orchestrator')

# Sanctioned references: (module path relative to the spindoctor package,
# denied name) -> one-line justification.  Empty by design; add an entry only
# with a justification a reviewer can check.
_ALLOWED_REFERENCES: dict[tuple[str, str], str] = {}

_PACKAGE_ROOT = Path(spindoctor.__file__).parent


def _truth_references(source: str, *, filename: str) -> list[tuple[str, int, str]]:
    """Return every truth-attribute reference in ``source``.

    A reference is an attribute access whose attribute name is on the
    deny-list, or a string literal exactly equal to a denied name (the
    ``getattr`` / ``vars()`` evasion).  Bare identifiers (parameter names,
    local variables) are deliberately not flagged: the leak surface is the
    observation object's attribute namespace, not the word itself.

    Parameters:
        source: Python source text to scan.
        filename: Label used in the returned reference records.

    Returns:
        List of ``(filename, line number, denied name)`` tuples.
    """
    references: list[tuple[str, int, str]] = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Attribute) and node.attr in TRUTH_ATTRIBUTE_DENYLIST:
            references.append((filename, node.lineno, node.attr))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in TRUTH_ATTRIBUTE_DENYLIST
        ):
            references.append((filename, node.lineno, node.value))
    return references


def _package_violations(package: str) -> list[str]:
    """Return formatted, non-allowlisted truth references in ``package``."""
    violations: list[str] = []
    for path in sorted((_PACKAGE_ROOT / package).rglob('*.py')):
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        for filename, lineno, name in _truth_references(
            path.read_text(encoding='utf-8'), filename=rel
        ):
            if (rel, name) in _ALLOWED_REFERENCES:
                continue
            violations.append(f'{filename}:{lineno}: references truth attribute {name!r}')
    return violations


def test_denylist_covers_every_obs_sim_truth_attribute() -> None:
    """Every sim_* attribute ObsSim stores on the snapshot is on the deny-list.

    Parses ``obs_inst_sim.py`` and collects the attribute names assigned onto
    the snapshot with the ``sim_`` prefix, so adding a truth attribute there
    without extending the deny-list fails here.
    """
    source = (_PACKAGE_ROOT / 'obs' / 'obs_inst_sim.py').read_text(encoding='utf-8')
    stored: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr.startswith('sim_'):
                stored.add(target.attr)
    assert stored
    missing = stored - TRUTH_ATTRIBUTE_DENYLIST
    assert not missing, (
        f'ObsSim stores truth attribute(s) {sorted(missing)} not covered by the '
        f'static guard deny-list; extend TRUTH_ATTRIBUTE_DENYLIST in the same change'
    )


@pytest.mark.parametrize('package', _NAVIGATOR_PACKAGES)
def test_navigator_package_references_no_truth_attribute(package: str) -> None:
    """No module in the navigator-side package references a truth attribute."""
    violations = _package_violations(package)
    assert not violations, (
        'navigator-side code references simulated-scene truth attributes '
        '(read obs.nav_params instead, or add an _ALLOWED_REFERENCES entry '
        'with a justification):\n' + '\n'.join(violations)
    )


def test_guard_flags_an_attribute_access() -> None:
    """The scanner detects a direct truth-attribute access."""
    references = _truth_references('def f(obs):\n    return obs.sim_params\n', filename='probe.py')
    assert references == [('probe.py', 2, 'sim_params')]


def test_guard_flags_a_string_literal_evasion() -> None:
    """The scanner detects the getattr string-literal form."""
    references = _truth_references(
        "def f(obs):\n    return getattr(obs, 'sim_offset_v')\n", filename='probe.py'
    )
    assert references == [('probe.py', 2, 'sim_offset_v')]


def test_allowlist_entries_name_existing_modules() -> None:
    """Every allowlist entry points at a module that still exists."""
    for rel, _name in _ALLOWED_REFERENCES:
        assert (_PACKAGE_ROOT / rel).is_file(), f'stale allowlist entry: {rel}'
