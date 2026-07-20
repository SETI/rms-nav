"""Unit tests for the covariance-components estimator on synthetic errors.

Every test draws technique errors from known covariances, feeds only the
resulting offsets to the solve, and checks recovery (where the composition
identifies the parameters) or the explicit degeneracy report (where it
cannot).  No spindoctor imports: the estimator is a standalone module.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from estimator import (  # noqa: E402
    EstimatorSpec,
    FrameSample,
    solve_covariance_components,
)


def _rot(theta: float) -> np.ndarray:
    """Rotation matrix whose columns are the basis vectors at ``theta``."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _draw(rng: np.random.Generator, cov: np.ndarray) -> np.ndarray:
    """One 2-D zero-mean Gaussian draw with covariance ``cov``."""
    return rng.multivariate_normal([0.0, 0.0], cov)


C_LIMB = np.array([[0.04, 0.01], [0.01, 0.36]])
C_DISC = np.array([[0.09, -0.02], [-0.02, 0.16]])
C_BLOB = np.array([[0.25, 0.0], [0.0, 0.25]])
S2_RING = 0.0025


def _make_frames(
    rng: np.random.Generator,
    n: int,
    *,
    with_ring: bool = False,
    ring_axis: str = 'diverse',
    limb_rotating: bool = False,
    with_blob: bool = False,
    common_limb_ring: float = 0.0,
) -> list[FrameSample]:
    """Build synthetic frames for the standard limb/disc/ring/blob setup.

    Parameters:
        rng: Random generator.
        n: Number of frames.
        with_ring: Include the rank1 ring instance.
        ring_axis: ``'diverse'`` for a uniform per-frame axis angle,
            ``'fixed'`` for a constant axis.
        limb_rotating: Draw the limb error in a per-frame rotating basis.
        with_blob: Include an isotropic full-rank blob instance.
        common_limb_ring: Standard deviation of an isotropic common-mode
            error added to both limb and ring (the injected shared bias).

    Returns:
        The frame list.
    """
    truth = np.zeros(2)
    frames: list[FrameSample] = []
    for _ in range(n):
        theta = float(rng.uniform(0.0, 2.0 * math.pi)) if limb_rotating else 0.0
        r_mat = _rot(theta)
        e_limb = r_mat @ _draw(rng, C_LIMB)
        e_disc = _draw(rng, C_DISC)
        offsets = {
            'limb': (truth[0] + e_limb[0], truth[1] + e_limb[1]),
            'disc': (truth[0] + e_disc[0], truth[1] + e_disc[1]),
        }
        basis = {'limb': theta}
        axis: dict[str, float] = {}
        common = np.zeros(2)
        if common_limb_ring > 0.0:
            common = rng.normal(0.0, common_limb_ring, size=2)
            offsets['limb'] = (offsets['limb'][0] + common[0], offsets['limb'][1] + common[1])
        if with_ring:
            alpha = float(rng.uniform(0.0, 2.0 * math.pi)) if ring_axis == 'diverse' else 0.7
            a_vec = np.array([math.cos(alpha), math.sin(alpha)])
            s_err = float(rng.normal(0.0, math.sqrt(S2_RING))) + float(a_vec @ common)
            # The rank1 instance reports a 2-D offset whose tangential part
            # is garbage; the solve must only ever use the axis projection.
            garbage = float(rng.normal(0.0, 30.0))
            t_vec = np.array([-a_vec[1], a_vec[0]])
            o_ring = truth + s_err * a_vec + garbage * t_vec
            offsets['ring'] = (float(o_ring[0]), float(o_ring[1]))
            axis['ring'] = alpha
        if with_blob:
            e_blob = _draw(rng, C_BLOB)
            offsets['blob'] = (truth[0] + e_blob[0], truth[1] + e_blob[1])
        frames.append(FrameSample(offsets=offsets, basis_angle_rad=basis, axis_angle_rad=axis))
    return frames


def test_three_full_rank_recovery() -> None:
    """Three co-occurring full-rank techniques recover all three matrices."""
    rng = np.random.default_rng(11)
    frames = _make_frames(rng, 6000, with_blob=True)
    specs = [
        EstimatorSpec('limb', 'full'),
        EstimatorSpec('disc', 'full'),
        EstimatorSpec('blob', 'full'),
    ]
    result = solve_covariance_components(frames, specs)
    assert result.null_space.shape[0] == 0
    np.testing.assert_allclose(result.covariances['limb'], C_LIMB, atol=0.03)
    np.testing.assert_allclose(result.covariances['disc'], C_DISC, atol=0.03)
    np.testing.assert_allclose(result.covariances['blob'], C_BLOB, atol=0.03)


def test_three_full_rank_identifiability_scores() -> None:
    """A fully identifiable system scores ~1 on every parameter."""
    rng = np.random.default_rng(12)
    frames = _make_frames(rng, 500, with_blob=True)
    specs = [
        EstimatorSpec('limb', 'full'),
        EstimatorSpec('disc', 'full'),
        EstimatorSpec('blob', 'full'),
    ]
    result = solve_covariance_components(frames, specs)
    for name, score in result.identifiability.items():
        assert score > 0.99, name


def test_limb_disc_alone_not_separable() -> None:
    """One matrix equation, two unknown matrices: 3-dim null space."""
    rng = np.random.default_rng(13)
    frames = _make_frames(rng, 3000)
    specs = [EstimatorSpec('limb', 'full'), EstimatorSpec('disc', 'full')]
    result = solve_covariance_components(frames, specs)
    assert result.null_space.shape[0] == 3
    assert result.condition_number == math.inf


def test_limb_disc_alone_recovers_combined_sum() -> None:
    """The combined covariance survives even though the split does not."""
    rng = np.random.default_rng(14)
    frames = _make_frames(rng, 6000)
    specs = [EstimatorSpec('limb', 'full'), EstimatorSpec('disc', 'full')]
    result = solve_covariance_components(frames, specs)
    combined = result.covariances['limb'] + result.covariances['disc']
    np.testing.assert_allclose(combined, C_LIMB + C_DISC, atol=0.04)


def test_limb_disc_alone_solution_not_unique() -> None:
    """Adding a null-space vector leaves the residual unchanged."""
    rng = np.random.default_rng(15)
    frames = _make_frames(rng, 200)
    specs = [EstimatorSpec('limb', 'full'), EstimatorSpec('disc', 'full')]
    result = solve_covariance_components(frames, specs)
    assert result.null_space.shape[0] > 0
    # Rebuild the residual for the reported params and for a shifted copy.
    shifted = result.params + 5.0 * result.null_space[0]
    diffs = []
    for frame in frames:
        o_l = np.asarray(frame.offsets['limb'])
        o_d = np.asarray(frame.offsets['disc'])
        diffs.append(o_l - o_d)
    d_arr = np.asarray(diffs) - np.mean(diffs, axis=0)
    targets = np.stack(
        [d_arr[:, 0] ** 2, d_arr[:, 0] * d_arr[:, 1], d_arr[:, 1] ** 2], axis=1
    ).ravel()
    design = np.tile(np.hstack([np.eye(3), np.eye(3)]), (len(frames), 1))
    res_a = design @ result.params - targets
    res_b = design @ shifted - targets
    np.testing.assert_allclose(res_a, res_b, atol=1e-9)


def test_ring_diverse_axis_makes_system_identifiable() -> None:
    """limb+disc+ring with a rotating radial axis recovers everything."""
    rng = np.random.default_rng(16)
    frames = _make_frames(rng, 8000, with_ring=True, ring_axis='diverse')
    specs = [
        EstimatorSpec('limb', 'full'),
        EstimatorSpec('disc', 'full'),
        EstimatorSpec('ring', 'rank1'),
    ]
    result = solve_covariance_components(frames, specs)
    assert result.null_space.shape[0] == 0
    np.testing.assert_allclose(result.covariances['limb'], C_LIMB, atol=0.04)
    np.testing.assert_allclose(result.covariances['disc'], C_DISC, atol=0.04)
    assert result.rank1_variances['ring'] == pytest.approx(S2_RING, abs=0.02)


def test_ring_fixed_axis_degenerates() -> None:
    """A frozen radial axis leaves a 2-dim null space (radial-only split).

    With the axis fixed, the ring leg constrains each body technique's
    covariance only along that axis: the ring variance and the per-technique
    radial-radial projections stay identifiable (the full limb+disc sum
    closes them), but the split of the remaining elements does not.
    """
    rng = np.random.default_rng(17)
    frames = _make_frames(rng, 6000, with_ring=True, ring_axis='fixed')
    specs = [
        EstimatorSpec('limb', 'full'),
        EstimatorSpec('disc', 'full'),
        EstimatorSpec('ring', 'rank1'),
    ]
    result = solve_covariance_components(frames, specs)
    assert result.null_space.shape[0] == 2
    assert result.identifiability['ring:s2'] > 0.99
    assert result.identifiability['limb:c11'] < 0.99
    # The radial-radial projection is the identifiable per-technique combo:
    # q(alpha) . vech(C) survives the degeneracy and matches truth.
    alpha = 0.7
    q = np.array(
        [
            math.cos(alpha) ** 2,
            2.0 * math.cos(alpha) * math.sin(alpha),
            math.sin(alpha) ** 2,
        ]
    )
    c_hat = result.covariances['limb']
    radial_hat = float(q @ [c_hat[0, 0], c_hat[0, 1], c_hat[1, 1]])
    radial_true = float(q @ [C_LIMB[0, 0], C_LIMB[0, 1], C_LIMB[1, 1]])
    assert radial_hat == pytest.approx(radial_true, abs=0.04)


def test_rotating_limb_basis_recovery() -> None:
    """A limb covariance stationary only in its rotating frame is recovered."""
    rng = np.random.default_rng(18)
    frames = _make_frames(
        rng, 8000, with_ring=True, ring_axis='diverse', limb_rotating=True, with_blob=True
    )
    specs = [
        EstimatorSpec('limb', 'full', basis='rotating'),
        EstimatorSpec('disc', 'full'),
        EstimatorSpec('ring', 'rank1'),
        EstimatorSpec('blob', 'full'),
    ]
    result = solve_covariance_components(frames, specs)
    assert result.null_space.shape[0] == 0
    np.testing.assert_allclose(result.covariances['limb'], C_LIMB, atol=0.05)


def test_rotating_limb_without_rotation_flag_is_wrong() -> None:
    """Ignoring the rotation misreads the limb anisotropy (the plan's trap)."""
    rng = np.random.default_rng(19)
    frames = _make_frames(rng, 6000, with_blob=True, limb_rotating=True)
    specs = [
        EstimatorSpec('limb', 'full'),  # wrong: treats basis as image-fixed
        EstimatorSpec('disc', 'full'),
        EstimatorSpec('blob', 'full'),
    ]
    result = solve_covariance_components(frames, specs)
    # The rotating anisotropic limb looks isotropic when averaged in image
    # coordinates: the recovered off-diagonal collapses and the diagonal
    # elements average, hiding the 9x anisotropy.
    c = result.covariances['limb']
    mean_diag = (C_LIMB[0, 0] + C_LIMB[1, 1]) / 2.0
    assert c[0, 0] == pytest.approx(mean_diag, abs=0.05)
    assert c[1, 1] == pytest.approx(mean_diag, abs=0.05)


def test_declared_pair_covariance_recovers_common_mode() -> None:
    """A shared limb+ring error shows up in the declared pair covariance."""
    rng = np.random.default_rng(20)
    sigma_common = 0.5
    frames = _make_frames(
        rng,
        8000,
        with_ring=True,
        ring_axis='diverse',
        with_blob=True,
        common_limb_ring=sigma_common,
    )
    specs = [
        EstimatorSpec('limb', 'full'),
        EstimatorSpec('disc', 'full'),
        EstimatorSpec('ring', 'rank1'),
        EstimatorSpec('blob', 'full'),
    ]
    result = solve_covariance_components(
        frames, specs, pair_covariances=[('limb', 'ring')]
    )
    gamma = result.pair_covariances[('limb', 'ring')]
    assert isinstance(gamma, float)
    assert gamma == pytest.approx(sigma_common**2, abs=0.05)
    # The limb covariance must absorb the common mode (it is a real part of
    # the limb error), inflated by sigma_common^2 on the diagonal.
    np.testing.assert_allclose(
        result.covariances['limb'],
        C_LIMB + np.eye(2) * sigma_common**2,
        atol=0.06,
    )


def test_undeclared_common_mode_misattributes() -> None:
    """Without the declared pair, the solve misattributes the common mode.

    The limb-ring common error cancels in the limb-ring difference, so the
    three-cornered-hat shifts it onto the techniques outside the coupled
    pair -- the classic agreement-masks-shared-bias failure the campaign
    must demonstrate.
    """
    rng = np.random.default_rng(21)
    sigma_common = 0.5
    frames = _make_frames(
        rng,
        8000,
        with_ring=True,
        ring_axis='diverse',
        with_blob=True,
        common_limb_ring=sigma_common,
    )
    specs = [
        EstimatorSpec('limb', 'full'),
        EstimatorSpec('disc', 'full'),
        EstimatorSpec('ring', 'rank1'),
        EstimatorSpec('blob', 'full'),
    ]
    result = solve_covariance_components(frames, specs)
    # The recovered limb covariance understates the true limb error (which
    # includes the common mode): the shared component cancels in the
    # limb-ring difference and the least-squares fit spreads the resulting
    # inconsistency, so a material fraction of the common-mode variance
    # vanishes from the limb estimate.
    true_limb = C_LIMB + np.eye(2) * sigma_common**2
    understated = float(np.trace(true_limb - result.covariances['limb'])) / 2.0
    assert understated > 0.3 * sigma_common**2


def test_shared_group_across_bodies() -> None:
    """Two limb instances sharing a group separate limb from disc."""
    rng = np.random.default_rng(22)
    frames = []
    for _ in range(6000):
        e_l_a = _draw(rng, C_LIMB)
        e_l_b = _draw(rng, C_LIMB)
        e_d_a = _draw(rng, C_DISC)
        e_d_b = _draw(rng, C_DISC)
        frames.append(
            FrameSample(
                offsets={
                    'limb@A': (e_l_a[0], e_l_a[1]),
                    'limb@B': (e_l_b[0], e_l_b[1]),
                    'disc@A': (e_d_a[0], e_d_a[1]),
                    'disc@B': (e_d_b[0], e_d_b[1]),
                }
            )
        )
    specs = [
        EstimatorSpec('limb@A', 'full', group='limb'),
        EstimatorSpec('limb@B', 'full', group='limb'),
        EstimatorSpec('disc@A', 'full', group='disc'),
        EstimatorSpec('disc@B', 'full', group='disc'),
    ]
    result = solve_covariance_components(frames, specs)
    assert result.null_space.shape[0] == 0
    np.testing.assert_allclose(result.covariances['limb'], C_LIMB, atol=0.03)
    np.testing.assert_allclose(result.covariances['disc'], C_DISC, atol=0.03)


def test_pair_mean_diff_reports_bias() -> None:
    """A deterministic offset between techniques lands in the mean channel."""
    rng = np.random.default_rng(23)
    frames = []
    for _ in range(2000):
        e_l = _draw(rng, C_LIMB)
        e_d = _draw(rng, C_DISC)
        frames.append(
            FrameSample(
                offsets={
                    'limb': (0.4 + e_l[0], -0.2 + e_l[1]),
                    'disc': (e_d[0], e_d[1]),
                }
            )
        )
    specs = [EstimatorSpec('limb', 'full'), EstimatorSpec('disc', 'full')]
    result = solve_covariance_components(frames, specs)
    mean = result.pair_mean_diff[('limb', 'disc')]
    assert isinstance(mean, np.ndarray)
    np.testing.assert_allclose(mean, [0.4, -0.2], atol=0.03)
    # The bias must NOT inflate the recovered combined covariance.
    combined = result.covariances['limb'] + result.covariances['disc']
    np.testing.assert_allclose(combined, C_LIMB + C_DISC, atol=0.05)


def test_bootstrap_intervals_cover_truth() -> None:
    """Bootstrap CIs on an identifiable system cover the true diagonal."""
    rng = np.random.default_rng(24)
    frames = _make_frames(rng, 1500, with_blob=True)
    specs = [
        EstimatorSpec('limb', 'full'),
        EstimatorSpec('disc', 'full'),
        EstimatorSpec('blob', 'full'),
    ]
    result = solve_covariance_components(frames, specs, n_bootstrap=100, bootstrap_seed=7)
    lo, hi = result.bootstrap_ci['limb:c22']
    assert lo < C_LIMB[1, 1]
    assert hi > C_LIMB[1, 1]


def test_duplicate_names_rejected() -> None:
    """Duplicate instance names raise."""
    frames = [FrameSample(offsets={'a': (0.0, 0.0), 'b': (0.0, 0.0)})]
    specs = [EstimatorSpec('a', 'full'), EstimatorSpec('a', 'full')]
    with pytest.raises(ValueError, match='unique'):
        solve_covariance_components(frames, specs)


def test_empty_cohort_rejected() -> None:
    """An empty frame list raises."""
    specs = [EstimatorSpec('a', 'full'), EstimatorSpec('b', 'full')]
    with pytest.raises(ValueError, match='non-empty'):
        solve_covariance_components([], specs)


def test_missing_axis_angle_rejected() -> None:
    """A rank1 instance without its per-frame axis raises."""
    frames = [FrameSample(offsets={'a': (0.0, 0.0), 'r': (0.0, 0.0)})]
    specs = [EstimatorSpec('a', 'full'), EstimatorSpec('r', 'rank1')]
    with pytest.raises(ValueError, match='axis_angle_rad'):
        solve_covariance_components(frames, specs)


def test_rank1_pair_skipped_when_axes_perpendicular() -> None:
    """Two rank1 instances with orthogonal axes share no measured component."""
    frames = [
        FrameSample(
            offsets={'r1': (0.1, 0.0), 'r2': (0.0, 0.1)},
            axis_angle_rad={'r1': 0.0, 'r2': math.pi / 2.0},
        )
        for _ in range(10)
    ]
    specs = [EstimatorSpec('r1', 'rank1'), EstimatorSpec('r2', 'rank1')]
    with pytest.raises(ValueError, match='co-occurs'):
        solve_covariance_components(frames, specs)
