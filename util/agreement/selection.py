"""Survivorship (selection-effect) model for a filtering shared layer.

The reliability gate differs in kind from the gradient / edge-distance-transform
and noise-sigma channels the bias-independence stage injects through: those
*shift* a surviving technique's offset, so their coupling is an offset
regression.  The gate only *admits or drops* a feature -- a surviving
technique's offset is unchanged -- so its common-mode effect is a **selection
effect**, not a bias.  A cohort assembled downstream of the gate (every real
agreement cohort is) is the set of scenes whose techniques survived, and the
covariances measured on it describe that survivor population rather than the
whole population.

This module is the truth-based instrument for that effect.  It works on planted
error arrays (never on real frames), so every number it returns is an exact
statement about how survivorship distorts a covariance -- the axis on which the
simulator's word is final.  Two facts it makes precise:

- **Separate per-technique gates do not manufacture cross-covariance.**  When
  two techniques have independent errors and each is admitted by its own gate,
  the joint-survival event factorizes across the two, so the survivor
  cross-covariance stays at its (zero) population value.  What separate gating
  *does* do is attenuate each technique's *marginal* covariance when survival
  tracks that technique's own error -- survivors are the low-error frames, so a
  per-technique sigma measured on survivors understates the population sigma.
- **A shared scene latent is the only channel that distorts the pairwise
  covariance.**  When a common scene-quality factor drives both techniques'
  errors *and* their admission, conditioning on survival truncates that
  latent's variance and so attenuates the shared (cross) covariance toward zero.

The model plants ``e_i = common_gain * q + eps_i`` with a shared scene latent
``q`` and independent per-technique noise ``eps_i``, then admits a scene when
both techniques' "badness" (a weighted sum of the shared-latent and own-error
magnitudes) is small enough to hit a target survival fraction.  Sweeping the
gate weights and the survival fraction bounds the distortion: at a given real
per-technique dropout fraction and a given strength of the reliability-vs-error
coupling, the induced change in any recovered covariance element is at most what
this model reports.  What the model cannot supply -- the actual strength of that
coupling on real frames -- is a property of the real reliability scores, outside
the simulator's envelope.

All statistics are per-axis scalars (variance and covariance along one image
axis): the agreement solve reduces every rank-1 (ring) pair to a per-axis
projection, and the per-axis form is where the bound is cleanest.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'SelectionTrial',
    'StratumStats',
    'stratum_stats',
    'synthetic_selection_trial',
]


@dataclass(frozen=True)
class StratumStats:
    """Per-axis second moments of one stratum's two error series.

    Parameters:
        n: Number of samples in the stratum.
        var_a: Variance of instance A's error.
        var_b: Variance of instance B's error.
        cov_ab: Cross-covariance of the two errors.
        corr_ab: Cross-correlation (``0`` when either variance is zero).
    """

    n: int
    var_a: float
    var_b: float
    cov_ab: float
    corr_ab: float


@dataclass(frozen=True)
class SelectionTrial:
    """Full-population vs survivor statistics for one synthetic trial.

    Parameters:
        keep_frac_target: Requested scene survival fraction.
        keep_frac_actual: Achieved survival fraction (both techniques kept).
        full: Statistics over every planted scene.
        survivor: Statistics over the survivor stratum only.
    """

    keep_frac_target: float
    keep_frac_actual: float
    full: StratumStats
    survivor: StratumStats


def stratum_stats(err_a: NDArray[np.float64], err_b: NDArray[np.float64]) -> StratumStats:
    """Second moments of a paired error stratum.

    Parameters:
        err_a: Instance A errors (1-D).
        err_b: Instance B errors (1-D), aligned with ``err_a``.

    Returns:
        The stratum statistics.  Variances and covariance use ``ddof=1``;
        below two samples they are reported as zero.

    Raises:
        ValueError: if the two arrays differ in length.
    """
    if err_a.shape != err_b.shape:
        raise ValueError(f'err_a and err_b must align; got {err_a.shape} and {err_b.shape}')
    n = int(err_a.shape[0])
    if n < 2:
        return StratumStats(n=n, var_a=0.0, var_b=0.0, cov_ab=0.0, corr_ab=0.0)
    var_a = float(np.var(err_a, ddof=1))
    var_b = float(np.var(err_b, ddof=1))
    cov_ab = float(np.cov(err_a, err_b, ddof=1)[0, 1])
    denom = math.sqrt(var_a * var_b) if var_a > 0.0 and var_b > 0.0 else 0.0
    corr_ab = cov_ab / denom if denom > 0.0 else 0.0
    return StratumStats(n=n, var_a=var_a, var_b=var_b, cov_ab=cov_ab, corr_ab=corr_ab)


def _threshold_for_keep(
    badness_a: NDArray[np.float64], badness_b: NDArray[np.float64], keep_frac: float
) -> float:
    """Bisection for the badness cutoff that admits ``keep_frac`` of scenes.

    A scene is admitted when both techniques' badness is at or below the
    returned cutoff; badness is admitted-low, so the kept fraction rises
    monotonically with the cutoff.

    Parameters:
        badness_a: Instance A per-scene badness (admitted when low).
        badness_b: Instance B per-scene badness.
        keep_frac: Target joint survival fraction in (0, 1).

    Returns:
        The cutoff value.
    """
    lo = 0.0
    hi = float(max(badness_a.max(), badness_b.max()))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        kept = float(np.mean((badness_a <= mid) & (badness_b <= mid)))
        if kept < keep_frac:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def synthetic_selection_trial(
    rng: np.random.Generator,
    n: int,
    *,
    common_gain: float,
    common_gate: float,
    self_gate: float,
    keep_frac: float,
) -> SelectionTrial:
    """Plant paired errors, admit a survivor stratum, return both strata's moments.

    The two techniques share a scene latent ``q ~ N(0, 1)`` scaled by
    ``common_gain`` (the only source of population cross-covariance) plus
    independent unit noise.  A scene's per-technique badness is
    ``common_gate * q**2 + self_gate * eps_i**2`` -- large when the shared
    latent is unfavorable (common-mode admission pressure) or when the
    technique's own error is large (self-selection).  A scene survives when
    both techniques' badness clears the cutoff chosen to hit ``keep_frac``.

    Parameters:
        rng: Seeded generator (determinism is the caller's).
        n: Number of planted scenes.
        common_gain: Shared-latent gain into each error (sets population
            cross-covariance ``common_gain**2``).
        common_gate: Weight of the shared latent in the admission badness
            (the channel that attenuates the shared covariance).
        self_gate: Weight of each technique's own error in its admission
            badness (the channel that attenuates the marginal variance).
        keep_frac: Target scene survival fraction in (0, 1).

    Returns:
        The trial's full-population and survivor statistics.

    Raises:
        ValueError: if ``n`` is below 2 or ``keep_frac`` is not in (0, 1).
    """
    if n < 2:
        raise ValueError(f'n must be at least 2; got {n}')
    if not 0.0 < keep_frac < 1.0:
        raise ValueError(f'keep_frac must lie in (0, 1); got {keep_frac}')
    q = rng.standard_normal(n)
    eps_a = rng.standard_normal(n)
    eps_b = rng.standard_normal(n)
    err_a = common_gain * q + eps_a
    err_b = common_gain * q + eps_b
    badness_a = common_gate * q**2 + self_gate * eps_a**2
    badness_b = common_gate * q**2 + self_gate * eps_b**2
    cutoff = _threshold_for_keep(badness_a, badness_b, keep_frac)
    survive = (badness_a <= cutoff) & (badness_b <= cutoff)
    full = stratum_stats(err_a, err_b)
    survivor = stratum_stats(err_a[survive], err_b[survive])
    return SelectionTrial(
        keep_frac_target=keep_frac,
        keep_frac_actual=float(np.mean(survive)),
        full=full,
        survivor=survivor,
    )


def _fmt(stats: StratumStats) -> str:
    """One-line rendering of a stratum's moments."""
    return (
        f'n={stats.n:5d} var_a={stats.var_a:6.3f} var_b={stats.var_b:6.3f} '
        f'cov={stats.cov_ab:+.3f} corr={stats.corr_ab:+.3f}'
    )


def main(argv: list[str] | None = None) -> int:
    """Print the selection-effect bound grid for a range of gate strengths.

    Sweeps the survival fraction and the two gate channels and reports, for
    each cell, the survivor-vs-full change in the marginal variance and the
    cross-covariance -- the bound the campaign record quotes.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--n', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=20260721)
    parser.add_argument(
        '--common-gain',
        type=float,
        default=1.0,
        help='shared-latent gain (population cross-covariance is its square)',
    )
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    print(f'# Selection-effect bound grid (n={args.n}, common_gain={args.common_gain})')
    print('# marginal attenuation = survivor var_a / full var_a;')
    print('# cross attenuation     = survivor cov / full cov.')
    print('# Read each regime at the right common_gain: the no-manufacture result')
    print('# (independent errors -> survivor cov stays 0) needs --common-gain 0.0;')
    print('# at common_gain > 0 the separate self-gate row instead shows an existing')
    print('# shared covariance left ~unchanged (cov ratio ~1.0), not attenuated.')
    for label, common_gate, self_gate in (
        ('separate self-gate (independent errors)', 0.0, 1.0),
        ('shared-latent gate', 1.0, 0.0),
        ('mixed gate', 1.0, 1.0),
    ):
        print(f'\n## {label}: common_gate={common_gate} self_gate={self_gate}')
        for keep in (0.9, 0.75, 0.5, 0.25):
            trial = synthetic_selection_trial(
                rng,
                args.n,
                common_gain=args.common_gain,
                common_gate=common_gate,
                self_gate=self_gate,
                keep_frac=keep,
            )
            var_ratio = trial.survivor.var_a / trial.full.var_a if trial.full.var_a else 0.0
            cov_ratio = trial.survivor.cov_ab / trial.full.cov_ab if trial.full.cov_ab else 0.0
            print(
                f'keep={keep:.2f} (actual {trial.keep_frac_actual:.2f}): '
                f'var_a x{var_ratio:.3f}  cov x{cov_ratio:.3f}  | '
                f'full[{_fmt(trial.full)}] surv[{_fmt(trial.survivor)}]'
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
