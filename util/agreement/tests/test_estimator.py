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

from estimator import (
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
    result = solve_covariance_components(frames, specs, pair_covariances=[('limb', 'ring')])
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


def _rotating_bias_frames(
    rng: np.random.Generator,
    n: int,
    mu_limb: tuple[float, float],
    mu_disc: tuple[float, float] | None,
) -> list[FrameSample]:
    """Frames with geometry-locked (rotating-frame) technique biases.

    The limb error is drawn in a per-frame rotating basis with a constant
    bias ``mu_limb`` in that basis; when ``mu_disc`` is given the disc
    error also carries a rotating bias with the same per-frame angle (the
    partial-body situation where both techniques are pulled toward the
    body center).  A rank1 ring with a diverse axis completes the system.

    Parameters:
        rng: Random generator.
        n: Number of frames.
        mu_limb: Limb bias in the rotating frame.
        mu_disc: Optional disc bias in the same rotating frame.

    Returns:
        The frame list.
    """
    c_limb = np.array([[0.3, 0.0], [0.0, 0.5]])
    c_disc = np.array([[0.05, 0.0], [0.0, 0.05]])
    s2_ring = 0.01
    frames: list[FrameSample] = []
    for _ in range(n):
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        r = _rot(theta)
        e_l = r @ (_draw(rng, c_limb) + np.asarray(mu_limb))
        e_d = _draw(rng, c_disc)
        if mu_disc is not None:
            e_d = e_d + r @ np.asarray(mu_disc)
        alpha = float(rng.uniform(0.0, 2.0 * math.pi))
        a = np.array([math.cos(alpha), math.sin(alpha)])
        t = np.array([-a[1], a[0]])
        e_r = float(rng.normal(0.0, math.sqrt(s2_ring))) * a + float(rng.normal(0.0, 30.0)) * t
        frames.append(
            FrameSample(
                offsets={
                    'limb': (float(e_l[0]), float(e_l[1])),
                    'disc': (float(e_d[0]), float(e_d[1])),
                    'ring': (float(e_r[0]), float(e_r[1])),
                },
                basis_angle_rad={'limb': theta, 'disc': theta},
                axis_angle_rad={'ring': alpha},
            )
        )
    return frames


def test_rotating_bias_aliases_under_constant_centering() -> None:
    """A rotating-frame bias is invisible to constant centering and aliases.

    The image-frame mean of a rotating bias is ~0 over a diverse cohort,
    so the constant channel subtracts nothing and mu mu^T lands in the
    recovered covariance (C + mu mu^T) with the system fully identifiable
    and no negative-variance symptom anywhere.
    """
    rng = np.random.default_rng(30)
    frames = _rotating_bias_frames(rng, 5000, (-2.0, 0.0), None)
    specs = [
        EstimatorSpec('limb', 'full', basis='rotating'),
        EstimatorSpec('disc', 'full'),
        EstimatorSpec('ring', 'rank1'),
    ]
    result = solve_covariance_components(frames, specs, mean_model='constant')
    mean = result.pair_mean_diff[('limb', 'disc')]
    assert isinstance(mean, np.ndarray)
    # The constant bias channel sees nothing...
    assert float(np.abs(mean).max()) < 0.1
    # ...and the limb variance is inflated to ~ C + mu^2 = 0.3 + 4.0.
    assert result.covariances['limb'][0, 0] == pytest.approx(4.3, abs=0.3)
    assert result.null_space.shape[0] == 0


def test_rotating_mean_model_absorbs_geometry_locked_bias() -> None:
    """Experiment A: the rotating mean columns restore the true covariance."""
    rng = np.random.default_rng(31)
    frames = _rotating_bias_frames(rng, 6000, (-2.0, 0.0), None)
    specs = [
        EstimatorSpec('limb', 'full', basis='rotating'),
        EstimatorSpec('disc', 'full'),
        EstimatorSpec('ring', 'rank1'),
    ]
    result = solve_covariance_components(frames, specs)
    assert result.covariances['limb'][0, 0] == pytest.approx(0.3, abs=0.06)
    assert result.covariances['limb'][1, 1] == pytest.approx(0.5, abs=0.08)
    assert result.rank1_variances['ring'] == pytest.approx(0.01, abs=0.05)
    # The fitted mean model exposes the bias it absorbed.
    model = result.pair_mean_model[('limb', 'ring')]
    assert model['limb:mu1'] == pytest.approx(-2.0, abs=0.1)


def test_shared_rotating_bias_repaired_when_both_declared() -> None:
    """Experiment B: shared rotating biases stop corrupting the solve.

    With limb and disc both biased in the same rotating frame and both
    declared rotating, the mean model absorbs the shared component: the
    disc covariance no longer goes negative and every value returns to
    truth.  (With disc left undeclared the corruption persists -- the
    absorption follows the declaration.)
    """
    rng = np.random.default_rng(32)
    frames = _rotating_bias_frames(rng, 6000, (-2.0, 0.0), (-1.5, 0.0))
    specs_declared = [
        EstimatorSpec('limb', 'full', basis='rotating'),
        EstimatorSpec('disc', 'full', basis='rotating'),
        EstimatorSpec('ring', 'rank1'),
    ]
    result = solve_covariance_components(frames, specs_declared)
    assert result.covariances['limb'][0, 0] == pytest.approx(0.3, abs=0.06)
    assert result.covariances['disc'][0, 0] == pytest.approx(0.05, abs=0.05)
    assert result.covariances['disc'][0, 0] > 0.0
    assert result.rank1_variances['ring'] == pytest.approx(0.01, abs=0.05)
    # Undeclared disc: the corruption persists (documented limitation).
    specs_naive = [
        EstimatorSpec('limb', 'full', basis='rotating'),
        EstimatorSpec('disc', 'full'),
        EstimatorSpec('ring', 'rank1'),
    ]
    naive = solve_covariance_components(frames, specs_naive, mean_model='constant')
    assert naive.covariances['disc'][0, 0] < 0.0
    # And auto mode does NOT rescue an undeclared member: the limb's mean
    # columns absorb only the limb-disc difference of the shared bias, so
    # the disc's own rotating bias still aliases (c11 inflated an order of
    # magnitude above the 0.05 truth, though no longer negative).
    auto_naive = solve_covariance_components(frames, specs_naive)
    assert auto_naive.covariances['disc'][0, 0] > 0.3


def test_bootstrap_coverage_rate() -> None:
    """Bootstrap CIs cover the truth for nearly all identifiable parameters.

    Not a single-parameter spot check: all nine parameters of a fully
    identifiable three-technique system are checked at once.  At 95%
    nominal coverage the expectation is ~8.55 covered; requiring at least
    7 catches gross miscoverage while tolerating the ~7% chance of two
    marginal misses (percentile bootstrap on variance parameters is
    slightly anti-conservative even with per-replicate re-centering).
    """
    rng = np.random.default_rng(33)
    frames = _make_frames(rng, 1500, with_blob=True)
    specs = [
        EstimatorSpec('limb', 'full'),
        EstimatorSpec('disc', 'full'),
        EstimatorSpec('blob', 'full'),
    ]
    result = solve_covariance_components(frames, specs, n_bootstrap=150, bootstrap_seed=9)
    truth = {
        'limb': C_LIMB,
        'disc': C_DISC,
        'blob': C_BLOB,
    }
    covered = 0
    total = 0
    for group, cov in truth.items():
        for name, value in (
            (f'{group}:c11', cov[0, 0]),
            (f'{group}:c12', cov[0, 1]),
            (f'{group}:c22', cov[1, 1]),
        ):
            lo, hi = result.bootstrap_ci[name]
            total += 1
            if lo <= value <= hi:
                covered += 1
    assert total == 9
    assert covered >= 7
