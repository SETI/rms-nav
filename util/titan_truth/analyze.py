"""Report planted-truth accuracy and solve the reported-sigma scales.

Reads the JSONL rows ``collect.py`` writes and answers the three questions the
haze navigator's accuracy claim rests on:

1. **How accurate is it?**  Per-axis recovery-error percentiles, on the clean
   family (where the estimator's assumptions hold, which is what the published
   bound is stated on) and on every stress family beside it.

2. **Is the reported uncertainty honest?**  The z-score ``error / sigma`` per
   axis.  A standard deviation of 1 means the technique's own covariance
   describes its errors; the campaign solves for the ``cross_sigma_scale`` /
   ``along_sigma_scale`` that make it so.  The solve inverts the covariance
   model exactly: the scale multiplies the fit sigma BEFORE the isotropic
   model-error floor is added in quadrature, so
   ``sigma(s) = hypot(s / s_now * sigma_fit, floor)`` with ``sigma_fit``
   recovered from each row's reported sigma and the floor recorded in the
   run's manifest.

   The solve runs on the UNCENSORED rows only.  Each axis sigma is clamped to
   a configured per-axis floor before the model-error floor is applied, and a
   row sitting on that clamp reports the floor no matter what multiplier
   produced it -- so including clamped rows would measure the floor and
   attribute it to the scale.  Both the all-row z-score (what the ensemble
   actually consumes) and the uncensored one (what the scale controls) are
   reported, along with how many rows are clamped; when the clamped fraction
   is large, the floor, not the scale, is what sets the reported uncertainty.

3. **Does it ever lock confidently wrong?**  The error distribution restricted
   to results the technique called confident, which is the bound that matters
   operationally: a wrong answer the pipeline believes is worse than a
   refusal.

Run::

    venv/bin/python util/titan_truth/analyze.py _work/titan_truth/rows.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# Acceptance bounds the campaign is graded against (plan Section 6, Phase D).
CLEAN_CROSS_P95_PX: float = 1.0
CLEAN_ALONG_P95_PX: float = 3.0
# A z-score standard deviation inside this band means the reported sigma is
# neither optimistic nor padded.
Z_STD_BAND: tuple[float, float] = (0.8, 1.25)
# The provisional no-confident-wrong bound: among results the technique called
# confident, the 99th percentile error must stay inside twice the P95 bound.
CONFIDENT_THRESHOLD: float = 0.5
CONFIDENT_P99_FACTOR: float = 2.0

# Bracket the sigma-scale bisection searches, and the tolerance at which its
# result counts as a root rather than an edge of the bracket.
_SOLVE_BRACKET: tuple[float, float] = (1.0e-3, 1.0e4)
_SOLVE_TOLERANCE: float = 1.0e-6

AXES: tuple[str, ...] = ('cross', 'along')
_PERCENTILES: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)


def load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a rows file, returning ``(manifest, rows)``.

    Parameters:
        path: Path to the JSONL file ``collect.py`` wrote.

    Returns:
        The manifest line and every scene row after it.

    Raises:
        ValueError: If the file carries no manifest line.
    """
    manifest: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get('manifest'):
            manifest = record
            continue
        rows.append(record)
    if manifest is None:
        raise ValueError(f'{path}: no manifest line; regenerate with collect.py')
    return manifest, rows


def committed(rows: list[dict[str, Any]], family: str | None = None) -> list[dict[str, Any]]:
    """Return the rows that produced a committed offset, optionally by family."""
    return [
        r
        for r in rows
        if r.get('outcome') == 'committed' and (family is None or r['family'] == family)
    ]


def _percentiles(values: list[float]) -> dict[str, float]:
    """Return the reported percentiles of ``|values|``, or NaN when empty."""
    if not values:
        return {f'p{int(p)}': float('nan') for p in _PERCENTILES}
    magnitudes = np.abs(np.asarray(values, dtype=float))
    return {f'p{int(p)}': float(np.percentile(magnitudes, p)) for p in _PERCENTILES}


def _fit_sigma(reported_sigma: float, floor_px: float) -> float | None:
    """Recover the pre-floor fit sigma from a reported one.

    The floor is added in quadrature to an isotropic covariance, so it
    contributes exactly ``floor_px**2`` along any unit axis and inverts
    exactly.  Returns None when the reported sigma sits at or below the floor
    (a fit sigma of zero carries no information about the scale).

    Parameters:
        reported_sigma: The technique's reported one-sigma on one axis.
        floor_px: The configured model-error floor.

    Returns:
        The pre-floor fit sigma, or None.
    """
    residual = reported_sigma * reported_sigma - floor_px * floor_px
    if residual <= 1e-12:
        return None
    return math.sqrt(residual)


def _z_std_for_scale(
    errors: list[float], fit_sigmas: list[float], *, floor_px: float, ratio: float
) -> float:
    """Return the z-score standard deviation if the sigma scale were changed.

    ``ratio`` is the candidate scale divided by the scale the rows were
    collected at, so 1.0 reproduces the measured value.  The z-score standard
    deviation is taken about zero (the estimator is meant to be unbiased, and
    a bias must inflate z rather than be subtracted out of it).

    Parameters:
        errors: Per-row recovery errors on one axis.
        fit_sigmas: The matching pre-floor fit sigmas.
        floor_px: The configured model-error floor.
        ratio: Candidate scale relative to the collected one.

    Returns:
        The root-mean-square z-score.
    """
    err = np.asarray(errors, dtype=float)
    sig = np.hypot(ratio * np.asarray(fit_sigmas, dtype=float), floor_px)
    return float(np.sqrt(np.mean((err / sig) ** 2)))


def solve_sigma_scale(
    rows: list[dict[str, Any]],
    axis: str,
    *,
    floor_px: float,
    sigma_floor_px: float,
    current_scale: float,
) -> dict[str, Any]:
    """Solve for the sigma scale that makes one axis's z-scores unit-normal.

    Bisects the monotone relation between the scale and the z-score spread
    (a larger scale can only shrink z), so the answer is exact to a
    thousandth of a scale unit rather than an iterate-and-hope.  The solve
    uses only the rows whose fit sigma is above the configured per-axis
    floor: a clamped row would report that floor at any scale, so it is a
    censored observation of the multiplier under test.

    Parameters:
        rows: Committed rows to solve on.
        axis: ``'cross'`` or ``'along'``.
        floor_px: The isotropic covariance model-error floor.
        sigma_floor_px: The configured per-axis sigma floor the fit clamps to.
        current_scale: The scale the rows were collected at.

    Returns:
        A mapping with the all-row z standard deviation, the uncensored one,
        the clamped fraction, the recommended scale, and the sample sizes.
    """
    errors: list[float] = []
    fit_sigmas: list[float] = []
    free_errors: list[float] = []
    free_sigmas: list[float] = []
    clamp_tolerance = 1.0e-6 * max(sigma_floor_px, 1.0)
    for row in rows:
        sigma = row.get('sigma_px', {}).get(axis)
        error = row.get('error_px', {}).get(axis)
        if sigma is None or error is None:
            continue
        fit_sigma = _fit_sigma(float(sigma), floor_px)
        if fit_sigma is None:
            continue
        errors.append(float(error))
        fit_sigmas.append(fit_sigma)
        if fit_sigma > sigma_floor_px + clamp_tolerance:
            free_errors.append(float(error))
            free_sigmas.append(fit_sigma)
    if not errors:
        return {
            'n': 0,
            'n_free': 0,
            'clamped_fraction': float('nan'),
            'z_std': float('nan'),
            'z_std_free': float('nan'),
            'recommended_scale': current_scale,
            'z_std_at_recommended': float('nan'),
            'bracket_hit': False,
        }
    measured = _z_std_for_scale(errors, fit_sigmas, floor_px=floor_px, ratio=1.0)
    summary: dict[str, Any] = {
        'n': len(errors),
        'n_free': len(free_errors),
        'clamped_fraction': 1.0 - len(free_errors) / len(errors),
        'z_std': measured,
        'z_std_free': float('nan'),
        'recommended_scale': current_scale,
        'z_std_at_recommended': float('nan'),
        'bracket_hit': False,
    }
    if not free_errors:
        return summary
    summary['z_std_free'] = _z_std_for_scale(free_errors, free_sigmas, floor_px=floor_px, ratio=1.0)
    lo, hi = _SOLVE_BRACKET
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if _z_std_for_scale(free_errors, free_sigmas, floor_px=floor_px, ratio=mid) > 1.0:
            lo = mid
        else:
            hi = mid
    ratio = math.sqrt(lo * hi)
    z_at_ratio = _z_std_for_scale(free_errors, free_sigmas, floor_px=floor_px, ratio=ratio)
    # A bisection that ran to either end of its bracket did not find a root;
    # the ratio there is the bracket edge, not an answer, and reporting it as
    # a recommendation would invent one.
    summary['bracket_hit'] = abs(z_at_ratio - 1.0) > _SOLVE_TOLERANCE
    summary['recommended_scale'] = current_scale * ratio
    summary['z_std_at_recommended'] = z_at_ratio
    return summary


def achievable_z_curve(
    rows: list[dict[str, Any]],
    axis: str,
    *,
    floor_px: float,
    sigma_floor_px: float,
    current_scale: float,
    candidates: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01),
) -> list[tuple[float, float]]:
    """Return ``(scale, z_std)`` for candidate scales at or below the current one.

    The clamp makes this exactly computable in that direction and only in that
    direction: a row already sitting on the per-axis floor stays there for any
    SMALLER multiplier, and a free row's sigma scales with the multiplier.
    Raising the scale instead would unclamp rows whose unscaled sigma the
    collected data never recorded, so the curve stops at the collected scale
    rather than extrapolating through censored values.

    ``candidates`` are RELATIVE to the collected scale and are applied to each
    row's collected fit sigma, which is what makes the label and the value
    agree at any collected scale: the ``ratio = 1.0`` row is by construction
    the collected configuration, so its z equals the all-row z reported above
    it.  (Multiplying an unscaled sigma by the same ratio would silently shift
    the whole curve by ``1 / current_scale`` whenever the rows were collected
    at anything but unity.)

    This curve is the evidence behind the setting: when it saturates below the
    acceptance band, no value of the scale can reach the band, and the per-axis
    sigma floor -- not the scale -- is what sets the reported uncertainty.

    Parameters:
        rows: Committed rows.
        axis: ``'cross'`` or ``'along'``.
        floor_px: The isotropic covariance model-error floor.
        sigma_floor_px: The configured per-axis sigma floor.
        current_scale: The scale the rows were collected at.
        candidates: Scale multipliers relative to ``current_scale``; 1.0 must
            be present so the curve is anchored to the collected z.

    Returns:
        ``(absolute scale, z standard deviation)`` pairs, in the given order.

    Raises:
        ValueError: If ``candidates`` omits ``1.0``, which is the anchor that
            makes the curve checkable against the reported all-row z.
    """
    if not any(ratio == 1.0 for ratio in candidates):
        raise ValueError('candidates must include 1.0 so the curve is anchored')
    errors: list[float] = []
    fit_sigmas: list[float] = []
    clamped: list[bool] = []
    for row in rows:
        sigma = row.get('sigma_px', {}).get(axis)
        error = row.get('error_px', {}).get(axis)
        if sigma is None or error is None:
            continue
        fit_sigma = _fit_sigma(float(sigma), floor_px)
        if fit_sigma is None:
            continue
        errors.append(float(error))
        fit_sigmas.append(fit_sigma)
        clamped.append(fit_sigma <= sigma_floor_px * (1.0 + 1.0e-9))
    if not errors:
        return []
    err = np.asarray(errors, dtype=float)
    collected = np.asarray(fit_sigmas, dtype=float)
    was_clamped = np.asarray(clamped, dtype=bool)
    curve: list[tuple[float, float]] = []
    for ratio in candidates:
        scaled = np.where(was_clamped, sigma_floor_px, ratio * collected)
        sigma = np.hypot(np.maximum(scaled, sigma_floor_px), floor_px)
        curve.append((current_scale * ratio, float(np.sqrt(np.mean((err / sigma) ** 2)))))
    return curve


def _outcome_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count rows by outcome, including the errored ones."""
    counts: dict[str, int] = {}
    for row in rows:
        key = 'error' if 'error' in row else str(row.get('outcome', 'unknown'))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _gate_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count the named gate behind each spurious row."""
    counts: dict[str, int] = {}
    for row in rows:
        if row.get('outcome') != 'spurious':
            continue
        key = str(row.get('gate_failed'))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _phase_table(rows: list[dict[str, Any]]) -> list[tuple[str, int, int, float]]:
    """Commit rate by phase bin: ``(label, committed, total, fraction)``.

    The phase dependence is the campaign's headline structural finding, so it
    is reported as its own table rather than folded into the family summary.
    """
    edges = (10.0, 30.0, 50.0, 70.0, 90.0, 110.0, 140.0)
    table: list[tuple[str, int, int, float]] = []
    for lo, hi in itertools.pairwise(edges):
        binned = [r for r in rows if 'phase_deg' in r and lo <= float(r['phase_deg']) < hi]
        good = [r for r in binned if r.get('outcome') == 'committed']
        fraction = len(good) / len(binned) if binned else float('nan')
        table.append((f'{lo:.0f}-{hi:.0f}', len(good), len(binned), fraction))
    return table


def _print_family_accuracy(rows: list[dict[str, Any]], families: list[str]) -> None:
    """Print the per-axis error percentiles and commit rate for every family.

    The commit rate belongs beside the percentiles rather than in its own
    table: a family that refuses most of its frames can post excellent
    percentiles on the few it kept, and the two numbers are only meaningful
    read together.
    """
    print('\nRecovery error by family (px, |error| percentiles over committed rows)')
    header = f'{"family":<18} {"n":>5} {"commit":>7} ' + ' '.join(
        f'{axis + "." + f"p{int(p)}":>11}' for axis in AXES for p in _PERCENTILES
    )
    print(header)
    print('-' * len(header))
    for family in families:
        total = len([r for r in rows if r['family'] == family])
        good = committed(rows, family)
        rate = len(good) / total if total else float('nan')
        cells: list[str] = []
        for axis in AXES:
            stats = _percentiles([float(r['error_px'][axis]) for r in good])
            cells += [f'{stats[f"p{int(p)}"]:11.3f}' for p in _PERCENTILES]
        print(f'{family:<18} {len(good):5d} {rate * 100:6.1f}% ' + ' '.join(cells))


def _print_confident_bound(rows: list[dict[str, Any]]) -> None:
    """Print the confidence-conditioned no-confident-wrong check."""
    confident = [
        r for r in committed(rows) if float(r.get('confidence', 0.0)) >= CONFIDENT_THRESHOLD
    ]
    print(
        f'\nNo-confident-wrong check (provisional, placeholder anchors): '
        f'{len(confident)} of {len(committed(rows))} committed rows carry '
        f'confidence >= {CONFIDENT_THRESHOLD}'
    )
    for axis, bound in (('cross', CLEAN_CROSS_P95_PX), ('along', CLEAN_ALONG_P95_PX)):
        stats = _percentiles([float(r['error_px'][axis]) for r in confident])
        limit = CONFIDENT_P99_FACTOR * bound
        verdict = 'PASS' if stats['p99'] <= limit else 'FAIL'
        print(f'  {axis:<6} P99 = {stats["p99"]:7.3f} px  vs {limit:5.2f} px limit   {verdict}')


def report(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> int:
    """Print the full campaign report; return a shell exit status.

    Parameters:
        manifest: The run manifest.
        rows: Every scene row.

    Returns:
        ``0`` when both clean-scene bounds and both z-score bands hold,
        ``1`` otherwise, so the campaign can gate a script.
    """
    families = list(manifest['families'])
    floor_px = float(manifest['model_error_floor_px'])
    scales = dict(manifest['sigma_scales'])
    print(
        f'Campaign seed {manifest["campaign_seed"]}, {manifest["n_scenes"]} scenes, '
        f'{manifest["per_family"]} per family'
    )
    print(f'Collected at cross_sigma_scale={scales["cross"]}, along_sigma_scale={scales["along"]}')
    print(f'Model-error floor {floor_px} px')
    print(f'\nOutcomes: {_outcome_counts(rows)}')
    gates = _gate_counts(rows)
    if gates:
        print(f'Spurious by gate: {gates}')

    _print_family_accuracy(rows, families)

    print('\nCommit rate by phase bin (deg)')
    for label, good, total, fraction in _phase_table(rows):
        bar = '' if math.isnan(fraction) else f'{fraction * 100:5.1f}%'
        print(f'  {label:>8}: {good:4d}/{total:4d}  {bar}')

    clean = committed(rows, 'clean')
    clean_cross = _percentiles([float(r['error_px']['cross']) for r in clean])
    clean_along = _percentiles([float(r['error_px']['along']) for r in clean])
    ok = True
    print('\nClean-scene acceptance bounds')
    for axis, stats, bound in (
        ('cross', clean_cross, CLEAN_CROSS_P95_PX),
        ('along', clean_along, CLEAN_ALONG_P95_PX),
    ):
        passed = stats['p95'] <= bound
        ok = ok and passed
        print(
            f'  {axis:<6} P95 = {stats["p95"]:7.3f} px  vs {bound:4.1f} px bound   '
            f'{"PASS" if passed else "FAIL"}'
        )

    floors = dict(manifest.get('sigma_floors', {'cross': 0.0, 'along': 0.0}))
    print('\nReported-sigma calibration')
    print(
        '  z(all)  is what the ensemble consumes; z(free) excludes rows whose fit '
        'sigma sat on its configured floor'
    )
    for axis in AXES:
        solved = solve_sigma_scale(
            committed(rows),
            axis,
            floor_px=floor_px,
            sigma_floor_px=float(floors[axis]),
            current_scale=float(scales[axis]),
        )
        in_band = Z_STD_BAND[0] <= solved['z_std'] <= Z_STD_BAND[1]
        ok = ok and in_band
        if solved['n_free'] == 0:
            verdict = 'free-row solve: no unclamped rows'
        elif solved['bracket_hit']:
            verdict = (
                f'free-row solve: bracket hit, no solution in range '
                f'(z would be {solved["z_std_at_recommended"]:.3f})'
            )
        else:
            verdict = (
                f'free-row solve (not a fixed point under the clamp): '
                f'{solved["recommended_scale"]:.3f}'
            )
        print(
            f'  {axis:<6} n={solved["n"]:5d} (free {solved["n_free"]:5d}, '
            f'clamped {solved["clamped_fraction"] * 100:5.1f}%)  '
            f'z(all) = {solved["z_std"]:6.3f} '
            f'[{Z_STD_BAND[0]}-{Z_STD_BAND[1]}: {"PASS" if in_band else "FAIL"}]  '
            f'z(free) = {solved["z_std_free"]:6.3f}  {verdict}'
        )
        curve = achievable_z_curve(
            committed(rows),
            axis,
            floor_px=floor_px,
            sigma_floor_px=float(floors[axis]),
            current_scale=float(scales[axis]),
        )
        if curve:
            # The ratio = 1.0 entry IS the collected configuration, so it must
            # reproduce the all-row z printed above; a mismatch would mean the
            # curve is measuring a different sigma model than the report is.
            anchor = next(z for scale, z in curve if scale == float(scales[axis]))
            assert abs(anchor - solved['z_std']) < 1.0e-9, (
                f'{axis}: z-versus-scale curve anchor {anchor} disagrees with the '
                f'reported all-row z {solved["z_std"]}'
            )
        rendered = '  '.join(f'{scale:.3g}:{z_std:.3f}' for scale, z_std in curve)
        print(f'         z(all) vs scale (exact at and below the collected scale): {rendered}')
        best = max((z for _s, z in curve), default=float('nan'))
        if curve and best < Z_STD_BAND[0]:
            print(
                f'         NOTE: z(all) saturates at {best:.3f}, below the band, so NO scale '
                f'reaches it; the {floors[axis]} px sigma_floor_{axis} is the binding term'
            )

    _print_confident_bound(rows)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """Load a rows file and print its campaign report."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('rows', type=Path)
    args = parser.parse_args(argv)
    manifest, rows = load_rows(args.rows)
    return report(manifest, rows)


if __name__ == '__main__':
    raise SystemExit(main())
