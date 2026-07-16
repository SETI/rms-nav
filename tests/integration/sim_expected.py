"""Expected-outcome assertion machinery for sim scenes (the ``expected`` block).

A sim scene may carry a scene-level ``expected`` block declaring the outcome the
navigator should produce: a ``status`` (``success`` / ``failed`` / ``conflicted``),
an optional ``confidence_tier`` (one of the five navigation ranks, or null to
assert the status only), and an optional ``status_reason``.  This module loads
that block and asserts a :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`
against it with clear failure messages.

It is the sim analog of the image-library sidecar's expected-outcome regression,
modeled on that taxonomy (status / confidence_tier / status_reason and its
cross-field rules) but implemented independently: a sim scene is not a sidecar,
so nothing here imports ``tests.integration.sidecar``.  The ``expected`` block is
a test-only scene key -- read here, fed to neither the renderer nor the
navigator, and stripped from ``nav_params`` by the information boundary.

The expected-fail scenes are the reason this exists: when a scene renders every
star in the wrong place, or an overwhelming confounder field, the CORRECT
navigation outcome is a failed / low-confidence result -- never a confident wrong
offset -- and this machinery is what turns that requirement into a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import iter_scene_paths, load_sim_scene


@dataclass(frozen=True)
class ExpectedOutcome:
    """A scene's declared expected navigation outcome.

    Parameters:
        status: The expected fused status (``success`` / ``failed`` /
            ``conflicted``).
        confidence_tier: The expected confidence rank, or ``None`` to leave the
            tier unasserted (assert the status only).
        status_reason: The expected ``status_reason`` value, or ``None`` to
            leave it unasserted.
    """

    status: str
    confidence_tier: str | None
    status_reason: str | None


def expected_from_scene(sim_params: dict[str, Any]) -> ExpectedOutcome | None:
    """Return the scene's :class:`ExpectedOutcome`, or None when it has no block.

    Parameters:
        sim_params: A validated scene mapping (the ``expected`` block, if
            present, was validated by the sim schema).

    Returns:
        The parsed expected outcome, or ``None`` for a scene with no ``expected``
        block.
    """
    block = sim_params.get('expected')
    if not isinstance(block, dict):
        return None
    return ExpectedOutcome(
        status=str(block['status']),
        confidence_tier=block.get('confidence_tier'),
        status_reason=block.get('status_reason'),
    )


def iter_expected_scene_paths(root: Path) -> list[Path]:
    """Return every catalog scene under ``root`` that carries an ``expected`` block."""
    return [path for path in iter_scene_paths(root) if load_sim_scene(path).get('expected')]


def navigate_scene(sim_params: dict[str, Any]) -> NavResult:
    """Render and navigate a scene in-process, returning its NavResult.

    Parameters:
        sim_params: A validated scene mapping.

    Returns:
        The orchestrator's fused result for the rendered frame.
    """
    obs = ObsSim.from_file('/tmp/sim_expected.yaml', sim_params=sim_params)
    orchestrator = NavOrchestrator(build_models_for_obs(obs), only_models='*', only_techniques='*')
    return orchestrator.navigate(obs)


def assert_result_matches_expected(
    *, scene_name: str, expected: ExpectedOutcome, result: NavResult
) -> None:
    """Assert a NavResult matches a scene's expected outcome, with clear messages.

    The status is always checked; the confidence tier and the status_reason are
    checked only when the scene asserts them (a null tier or an omitted reason
    leaves that field unconstrained).

    Parameters:
        scene_name: The scene name, used in the failure messages.
        expected: The scene's declared expected outcome.
        result: The navigator's fused result for the scene.

    Raises:
        AssertionError: On any mismatch, naming the scene and the fields.
    """
    assert result.status == expected.status, (
        f'{scene_name}: expected status {expected.status!r}, got {result.status!r} '
        f'(status_reason {result.status_reason.value!r})'
    )
    if expected.confidence_tier is not None:
        assert result.confidence_rank == expected.confidence_tier, (
            f'{scene_name}: expected confidence tier {expected.confidence_tier!r}, '
            f'got {result.confidence_rank!r}'
        )
    if expected.status_reason is not None:
        assert result.status_reason.value == expected.status_reason, (
            f'{scene_name}: expected status_reason {expected.status_reason!r}, '
            f'got {result.status_reason.value!r}'
        )
