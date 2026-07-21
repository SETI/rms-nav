"""Tests for the survivorship (selection-effect) model.

Pin the three facts the reliability-gate finding leans on: a gate with no
dependence on the scene keeps everyone; separate per-technique gates on
independent errors do not manufacture cross-covariance (they only attenuate
each marginal variance); and a shared-latent gate attenuates the shared
cross-covariance toward zero, more so as the survival fraction falls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selection import (
    SelectionTrial,
    StratumStats,
    stratum_stats,
    synthetic_selection_trial,
)


def _trial(
    *,
    common_gain: float,
    common_gate: float,
    self_gate: float,
    keep_frac: float,
    n: int = 40000,
) -> SelectionTrial:
    """Run one trial with a fixed generator and a default size."""
    rng = np.random.default_rng(20260721)
    return synthetic_selection_trial(
        rng,
        n,
        common_gain=common_gain,
        common_gate=common_gate,
        self_gate=self_gate,
        keep_frac=keep_frac,
    )


def test_stratum_stats_below_two_samples_is_zero() -> None:
    """A one-sample stratum reports zero moments rather than raising."""
    stats = stratum_stats(np.array([1.0]), np.array([2.0]))
    assert stats == StratumStats(n=1, var_a=0.0, var_b=0.0, cov_ab=0.0, corr_ab=0.0)


def test_stratum_stats_rejects_misaligned() -> None:
    """Mismatched array lengths raise ValueError."""
    with pytest.raises(ValueError, match='must align'):
        stratum_stats(np.zeros(3), np.zeros(4))


def test_stratum_stats_rejects_non_1d() -> None:
    """A 2-D error array raises ValueError before the shape-match check."""
    with pytest.raises(ValueError, match='must be 1-D'):
        stratum_stats(np.zeros((3, 2)), np.zeros((3, 2)))


def test_trial_rejects_bad_n() -> None:
    """A trial with fewer than two scenes raises."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match='n must be at least 2'):
        synthetic_selection_trial(
            rng, 1, common_gain=0.0, common_gate=0.0, self_gate=1.0, keep_frac=0.5
        )


def test_trial_rejects_bad_keep_frac() -> None:
    """A survival fraction outside (0, 1) raises."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match='keep_frac'):
        synthetic_selection_trial(
            rng, 100, common_gain=0.0, common_gate=0.0, self_gate=1.0, keep_frac=1.0
        )


def test_determinism() -> None:
    """Two trials with the same seed produce identical survivor covariance."""
    a = _trial(common_gain=1.0, common_gate=1.0, self_gate=0.0, keep_frac=0.5)
    b = _trial(common_gain=1.0, common_gate=1.0, self_gate=0.0, keep_frac=0.5)
    assert a.survivor.cov_ab == b.survivor.cov_ab


def test_inert_gate_keeps_everyone() -> None:
    """A gate that depends on neither the latent nor the error selects nobody out."""
    trial = _trial(common_gain=1.0, common_gate=0.0, self_gate=0.0, keep_frac=0.5)
    assert trial.keep_frac_actual == 1.0


def test_inert_gate_survivor_equals_full() -> None:
    """With nobody dropped the survivor covariance equals the full covariance."""
    trial = _trial(common_gain=1.0, common_gate=0.0, self_gate=0.0, keep_frac=0.5)
    assert trial.survivor.cov_ab == trial.full.cov_ab


def test_separate_gate_does_not_manufacture_covariance() -> None:
    """Independent errors, self-gated separately: survivor cross-cov stays ~0."""
    trial = _trial(common_gain=0.0, common_gate=0.0, self_gate=1.0, keep_frac=0.5)
    assert abs(trial.survivor.cov_ab) < 0.05


def test_separate_gate_attenuates_marginal_variance() -> None:
    """Self-gating still shrinks each marginal variance (survivors are easy frames)."""
    trial = _trial(common_gain=0.0, common_gate=0.0, self_gate=1.0, keep_frac=0.5)
    assert trial.survivor.var_a < 0.6 * trial.full.var_a


def test_shared_latent_gate_attenuates_cross_covariance() -> None:
    """A shared-latent gate drives the survivor cross-cov below the full value."""
    trial = _trial(common_gain=1.0, common_gate=1.0, self_gate=0.0, keep_frac=0.5)
    assert trial.survivor.cov_ab < 0.5 * trial.full.cov_ab


def test_shared_latent_gate_keeps_cross_covariance_positive() -> None:
    """The attenuated shared covariance stays on the same (positive) side of zero."""
    trial = _trial(common_gain=1.0, common_gate=1.0, self_gate=0.0, keep_frac=0.5)
    assert trial.survivor.cov_ab > 0.0


def test_shared_latent_attenuation_grows_with_dropout() -> None:
    """Heavier dropout attenuates the shared cross-covariance further."""
    light = _trial(common_gain=1.0, common_gate=1.0, self_gate=0.0, keep_frac=0.9)
    heavy = _trial(common_gain=1.0, common_gate=1.0, self_gate=0.0, keep_frac=0.5)
    assert heavy.survivor.cov_ab < light.survivor.cov_ab


def test_marginal_attenuation_grows_with_dropout() -> None:
    """Heavier dropout shrinks the marginal variance further."""
    light = _trial(common_gain=0.0, common_gate=0.0, self_gate=1.0, keep_frac=0.9)
    heavy = _trial(common_gain=0.0, common_gate=0.0, self_gate=1.0, keep_frac=0.5)
    assert heavy.survivor.var_a < light.survivor.var_a
