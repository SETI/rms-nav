"""Analyze agreement rows: identifiability map + bias-independence measurements.

Consumes the JSONL rows written by ``collect.py`` and produces a Markdown
report with, per composition cohort:

- the truth-free covariance-components solve (recovered matrices, singular
  spectrum, null space, per-parameter identifiability, bootstrap CIs);
- the truth-based reference (each technique's empirical error covariance
  against the planted offset, in the estimator's own basis frame), which the
  recovered values are checked against;
- the identifiability-map entry: which covariance elements the composition
  determines, and the explicit null-space demonstration where it does not.

When injected-cohort row files are supplied (``--dt-rows`` /
``--noise-rows``), the report adds the bias-independence stage: truth-based
cross-technique error covariances and correlations for the pivotal pairs
under each injection condition, the paired per-scene response of every
technique to the injected shared-layer bias, and the solve-side detection
check (declared pair covariance recovering the injected coupling; the
independence-assumed solve's misattribution).

Run:

    venv/bin/python util/agreement/analyze.py _work/agreement/rows.jsonl \
        --dt-rows _work/agreement/rows_dt.jsonl \
        --noise-rows _work/agreement/rows_noise.jsonl \
        --out _work/agreement/report.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from estimator import (  # noqa: E402
    EstimatorSpec,
    FrameSample,
    SolveResult,
    solve_covariance_components,
)

# Instance-name mapping: technique class -> estimator instance name.
_TECH_TO_INSTANCE = {
    'BodyLimbNav': 'limb',
    'BodyDiscCorrelateNav': 'disc',
    'RingEdgeNav': 'ring',
    'BodyBlobNav': 'blob',
}


@dataclass(frozen=True)
class CohortFrame:
    """One scene's assembled estimator sample plus its planted truth."""

    scene_id: str
    sample: FrameSample
    truth: tuple[float, float]


def load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a rows file, returning (manifest, rows)."""
    with path.open() as f:
        lines = [json.loads(line) for line in f]
    manifest = lines[0] if lines and lines[0].get('manifest') else {}
    rows = [row for row in lines if not row.get('manifest')]
    return manifest, rows


def _instance_offsets(row: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Extract non-spurious per-instance offsets from a row's runs."""
    offsets: dict[str, tuple[float, float]] = {}
    for run_key, techniques in row['runs'].items():
        for tech in techniques:
            if tech['spurious'] or tech['at_edge']:
                continue
            name = _TECH_TO_INSTANCE.get(tech['name'])
            if name is None:
                continue
            if run_key.startswith('body:'):
                instance = f'{name}@{run_key.split(":", 1)[1]}'
            else:
                instance = name
            offsets[instance] = (float(tech['offset_v']), float(tech['offset_u']))
    return offsets


def build_cohort(rows: list[dict[str, Any]], family: str) -> list[CohortFrame]:
    """Assemble estimator frames for one family's rows.

    Ring instances take their per-frame radial axis, and the clipped-limb
    families take the limb's rotating basis angle, from the row geometry
    (derived from the idealized scene parameters, not truth keys).

    Parameters:
        rows: All rows from a collection pass.
        family: Family to select.

    Returns:
        The cohort frames (scenes with an error or fewer than two usable
        instances are dropped).
    """
    frames: list[CohortFrame] = []
    for row in rows:
        if row['family'] != family or 'error' in row:
            continue
        offsets = _instance_offsets(row)
        if len(offsets) < 2:
            continue
        geometry = row['geometry']
        basis: dict[str, float] = {}
        axis: dict[str, float] = {}
        if 'ring_radial_deg' in geometry and 'ring' in offsets:
            axis['ring'] = math.radians(float(geometry['ring_radial_deg']))
        if 'limb_arc_outward_deg' in geometry and 'limb' in offsets:
            basis['limb'] = math.radians(float(geometry['limb_arc_outward_deg']))
        frames.append(
            CohortFrame(
                scene_id=row['scene_id'],
                sample=FrameSample(offsets=offsets, basis_angle_rad=basis, axis_angle_rad=axis),
                truth=(float(row['planted']['dv']), float(row['planted']['du'])),
            )
        )
    return frames


def _rot(theta: float) -> np.ndarray:
    """Rotation matrix with basis vectors at ``theta`` as columns."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def truth_covariance(
    cohort: list[CohortFrame], instance: str, *, basis: str = 'image'
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Empirical error mean and covariance for one instance vs planted truth.

    Parameters:
        cohort: Cohort frames.
        instance: Instance name.
        basis: ``'image'`` or ``'rotating'`` (uses the frame's basis angle).

    Returns:
        ``(mean, covariance, n)`` in the requested basis, or ``None`` when
        the instance appears on fewer than 3 frames.
    """
    errors = []
    for frame in cohort:
        if instance not in frame.sample.offsets:
            continue
        o = frame.sample.offsets[instance]
        e = np.array([o[0] - frame.truth[0], o[1] - frame.truth[1]])
        if basis == 'rotating':
            theta = frame.sample.basis_angle_rad.get(instance, 0.0)
            e = _rot(theta).T @ e
        errors.append(e)
    if len(errors) < 3:
        return None
    arr = np.asarray(errors)
    return arr.mean(axis=0), np.cov(arr.T, ddof=1), len(errors)


def truth_radial_variance(
    cohort: list[CohortFrame], instance: str
) -> tuple[float, float, int] | None:
    """Empirical mean and variance of a rank1 instance's radial error."""
    values = []
    for frame in cohort:
        if instance not in frame.sample.offsets or instance not in frame.sample.axis_angle_rad:
            continue
        alpha = frame.sample.axis_angle_rad[instance]
        a = np.array([math.cos(alpha), math.sin(alpha)])
        o = frame.sample.offsets[instance]
        e = np.array([o[0] - frame.truth[0], o[1] - frame.truth[1]])
        values.append(float(a @ e))
    if len(values) < 3:
        return None
    arr = np.asarray(values)
    return float(arr.mean()), float(arr.var(ddof=1)), len(values)


def truth_cross_covariance(
    cohort: list[CohortFrame], instance_a: str, instance_b: str
) -> tuple[float, float, int] | None:
    """Truth-based scalar cross-covariance and correlation for a pair.

    Both errors are projected onto the ring radial axis when either member
    is the ring (a rank1 instance's tangential component is meaningless);
    otherwise the statistic is the mean of the two per-axis covariances /
    correlations (a single scalar summary of the 2-D coupling).

    Parameters:
        cohort: Cohort frames.
        instance_a: First instance.
        instance_b: Second instance.

    Returns:
        ``(covariance, correlation, n)`` or ``None`` below 3 samples.
    """
    e_a: list[np.ndarray] = []
    e_b: list[np.ndarray] = []
    project = 'ring' in (instance_a, instance_b)
    for frame in cohort:
        if instance_a not in frame.sample.offsets or instance_b not in frame.sample.offsets:
            continue
        o_a = frame.sample.offsets[instance_a]
        o_b = frame.sample.offsets[instance_b]
        ea = np.array([o_a[0] - frame.truth[0], o_a[1] - frame.truth[1]])
        eb = np.array([o_b[0] - frame.truth[0], o_b[1] - frame.truth[1]])
        if project:
            alpha = frame.sample.axis_angle_rad.get(
                'ring', frame.sample.basis_angle_rad.get('ring', 0.0)
            )
            a = np.array([math.cos(alpha), math.sin(alpha)])
            ea = np.array([float(a @ ea)])
            eb = np.array([float(a @ eb)])
        e_a.append(ea)
        e_b.append(eb)
    if len(e_a) < 3:
        return None
    arr_a = np.asarray(e_a)
    arr_b = np.asarray(e_b)
    covs = []
    corrs = []
    for k in range(arr_a.shape[1]):
        cov = float(np.cov(arr_a[:, k], arr_b[:, k], ddof=1)[0, 1])
        sd_a = float(arr_a[:, k].std(ddof=1))
        sd_b = float(arr_b[:, k].std(ddof=1))
        covs.append(cov)
        corrs.append(cov / (sd_a * sd_b) if sd_a > 0 and sd_b > 0 else 0.0)
    return float(np.mean(covs)), float(np.mean(corrs)), len(e_a)


def _fmt_matrix(m: np.ndarray) -> str:
    """Compact one-line 2x2 matrix rendering."""
    return f'[{m[0, 0]:+.4f} {m[0, 1]:+.4f}; {m[1, 0]:+.4f} {m[1, 1]:+.4f}]'


def _null_space_description(result: SolveResult) -> list[str]:
    """Human-readable null-space directions (coefficients on parameters)."""
    lines = []
    for k in range(result.null_space.shape[0]):
        vec = result.null_space[k]
        terms = [
            f'{vec[i]:+.2f}*{name}'
            for i, name in enumerate(result.param_names)
            if abs(vec[i]) > 0.05
        ]
        lines.append('  '.join(terms))
    return lines


def report_solve(
    out: list[str],
    cohort: list[CohortFrame],
    specs: list[EstimatorSpec],
    title: str,
    *,
    pair_covariances: list[tuple[str, str]] | None = None,
    n_bootstrap: int = 200,
) -> SolveResult:
    """Run one solve and append its report section to ``out``.

    Parameters:
        out: Markdown line accumulator.
        cohort: Cohort frames.
        specs: Estimator instances for the solve.
        title: Section title.
        pair_covariances: Optional declared suspect pairs.
        n_bootstrap: Bootstrap replicates.

    Returns:
        The solve result.
    """
    frames = [c.sample for c in cohort]
    result = solve_covariance_components(
        frames,
        specs,
        pair_covariances=pair_covariances or [],
        n_bootstrap=n_bootstrap,
        bootstrap_seed=1,
    )
    out.append(f'### {title}')
    out.append('')
    out.append(
        f'{result.n_frames} frames, {result.n_equations} equations, '
        f'{len(result.param_names)} unknowns; condition number '
        f'{"inf" if math.isinf(result.condition_number) else f"{result.condition_number:.1f}"}; '
        f'null-space dimension {result.null_space.shape[0]}; '
        f'residual RMS {result.residual_rms:.4f} px^2.'
    )
    out.append('')
    out.append('| parameter | recovered | 95% CI | identifiability |')
    out.append('|---|---|---|---|')
    for k, name in enumerate(result.param_names):
        ci = result.bootstrap_ci.get(name)
        ci_str = f'[{ci[0]:+.4f}, {ci[1]:+.4f}]' if ci else 'n/a'
        ident = result.identifiability[name]
        flag = '' if ident > 0.99 else ' (NOT identifiable)' if ident < 0.6 else ' (partial)'
        out.append(f'| {name} | {result.params[k]:+.4f} | {ci_str} | {ident:.3f}{flag} |')
    out.append('')
    null_lines = _null_space_description(result)
    if null_lines:
        out.append('Null-space directions (any multiple can be added to the')
        out.append('solution without changing any observable):')
        out.append('')
        for line in null_lines:
            out.append(f'- `{line}`')
        out.append('')
    return result


def report_truth_reference(
    out: list[str], cohort: list[CohortFrame], specs: list[EstimatorSpec]
) -> None:
    """Append the truth-based per-instance error statistics to ``out``."""
    out.append('Truth reference (empirical error stats vs planted offsets, in')
    out.append('each instance basis frame):')
    out.append('')
    out.append('| instance | n | mean error | covariance |')
    out.append('|---|---|---|---|')
    for spec in specs:
        if spec.kind == 'rank1':
            radial = truth_radial_variance(cohort, spec.name)
            if radial is None:
                continue
            mean, var, n = radial
            out.append(f'| {spec.name} (radial) | {n} | {mean:+.4f} | s2 = {var:.5f} |')
        else:
            stats = truth_covariance(cohort, spec.name, basis=spec.basis)
            if stats is None:
                continue
            mean_vec, cov, n = stats
            out.append(
                f'| {spec.name} | {n} | ({mean_vec[0]:+.4f}, {mean_vec[1]:+.4f}) '
                f'| {_fmt_matrix(cov)} |'
            )
    out.append('')


def report_pair_truth(
    out: list[str], cohort: list[CohortFrame], pairs: list[tuple[str, str]], title: str
) -> None:
    """Append truth-based cross-covariance stats for the given pairs."""
    out.append(f'#### {title}')
    out.append('')
    out.append('| pair | n | cross-covariance (px^2) | correlation |')
    out.append('|---|---|---|---|')
    for a, b in pairs:
        stats = truth_cross_covariance(cohort, a, b)
        if stats is None:
            out.append(f'| {a} vs {b} | <3 | n/a | n/a |')
            continue
        cov, corr, n = stats
        out.append(f'| {a} vs {b} | {n} | {cov:+.5f} | {corr:+.3f} |')
    out.append('')


def report_injection_response(
    out: list[str],
    control: list[dict[str, Any]],
    injected: list[dict[str, Any]],
    family: str,
) -> None:
    """Paired per-scene response of each technique to the injected bias.

    For every technique present (non-spurious) in both the control and the
    injected pass of the same scene, regress the offset delta against the
    injected (bias_v, bias_u); a slope near 1 means the technique tracks
    the shared-layer bias one-for-one, near 0 means it is decoupled.

    Parameters:
        out: Markdown accumulator.
        control: Control rows.
        injected: Injected rows (dt_shift).
        family: Family to pair on.
    """
    ctrl_by_id = {r['scene_id']: r for r in control if r['family'] == family and 'error' not in r}
    deltas: dict[str, list[tuple[float, float, float, float]]] = {}
    for row in injected:
        if row['family'] != family or 'error' in row:
            continue
        ctrl = ctrl_by_id.get(row['scene_id'])
        if ctrl is None or row['injection'].get('kind') != 'dt_shift':
            continue
        bias_v = float(row['injection']['bias_v'])
        bias_u = float(row['injection']['bias_u'])
        inj_off = _instance_offsets(row)
        ctl_off = _instance_offsets(ctrl)
        for name in set(inj_off) & set(ctl_off):
            deltas.setdefault(name, []).append(
                (
                    bias_v,
                    bias_u,
                    inj_off[name][0] - ctl_off[name][0],
                    inj_off[name][1] - ctl_off[name][1],
                )
            )
    out.append('Paired per-scene response to the injected DT-layer shift')
    out.append('(slope of offset delta vs injected bias; 1 = fully coupled,')
    out.append('0 = decoupled from the shared products):')
    out.append('')
    out.append('| instance | n | slope dv/bias_v | slope du/bias_u | residual RMS (px) |')
    out.append('|---|---|---|---|---|')
    for name in sorted(deltas):
        arr = np.asarray(deltas[name])
        if arr.shape[0] < 5:
            continue
        slope_v = float(np.polyfit(arr[:, 0], arr[:, 2], 1)[0])
        slope_u = float(np.polyfit(arr[:, 1], arr[:, 3], 1)[0])
        resid = np.stack([arr[:, 2] - slope_v * arr[:, 0], arr[:, 3] - slope_u * arr[:, 1]], axis=1)
        rms = float(np.sqrt(np.mean(resid**2)))
        out.append(f'| {name} | {arr.shape[0]} | {slope_v:+.3f} | {slope_u:+.3f} | {rms:.3f} |')
    out.append('')


def report_noise_response(
    out: list[str],
    control: list[dict[str, Any]],
    injected: list[dict[str, Any]],
    family: str,
) -> None:
    """Paired per-scene offset deltas under the noise-sigma scaling injection.

    Reports each technique's RMS offset change between the control and the
    noise-scaled pass of the same scene, plus the correlation of the delta
    magnitude with the injected scale factor -- a small RMS with no factor
    dependence means the shared noise-sigma channel does not couple offsets
    at these scales.

    Parameters:
        out: Markdown accumulator.
        control: Control rows.
        injected: Injected rows (noise_scale).
        family: Family to pair on.
    """
    ctrl_by_id = {r['scene_id']: r for r in control if r['family'] == family and 'error' not in r}
    deltas: dict[str, list[tuple[float, float]]] = {}
    for row in injected:
        if row['family'] != family or 'error' in row:
            continue
        ctrl = ctrl_by_id.get(row['scene_id'])
        if ctrl is None or row['injection'].get('kind') != 'noise_scale':
            continue
        factor = float(row['injection']['factor'])
        inj_off = _instance_offsets(row)
        ctl_off = _instance_offsets(ctrl)
        for name in set(inj_off) & set(ctl_off):
            mag = math.hypot(
                inj_off[name][0] - ctl_off[name][0], inj_off[name][1] - ctl_off[name][1]
            )
            deltas.setdefault(name, []).append((factor, mag))
    out.append('Paired per-scene offset change under the noise-sigma scaling')
    out.append('(RMS |delta offset| and its correlation with the scale factor):')
    out.append('')
    out.append('| instance | n | RMS |delta| (px) | corr(|delta|, factor) |')
    out.append('|---|---|---|---|')
    for name in sorted(deltas):
        arr = np.asarray(deltas[name])
        if arr.shape[0] < 5:
            continue
        rms = float(np.sqrt(np.mean(arr[:, 1] ** 2)))
        if arr[:, 1].std() > 0 and arr[:, 0].std() > 0:
            corr = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
        else:
            corr = 0.0
        out.append(f'| {name} | {arr.shape[0]} | {rms:.4f} | {corr:+.3f} |')
    out.append('')


def _specs_for_family(
    family: str,
) -> list[tuple[list[EstimatorSpec], list[tuple[str, str]]]]:
    """The solve configurations reported for each family.

    Returns:
        List of ``(specs, declared_pair_covariances)`` per solve.
    """
    limb_img = EstimatorSpec('limb', 'full')
    limb_rot = EstimatorSpec('limb', 'full', basis='rotating')
    disc = EstimatorSpec('disc', 'full')
    ring = EstimatorSpec('ring', 'rank1')
    blob = EstimatorSpec('blob', 'full')
    no_pairs: list[tuple[str, str]] = []
    if family == 'limb_disc':
        return [([limb_img, disc], no_pairs)]
    if family in ('limb_disc_ring_fixed', 'limb_disc_ring_diverse'):
        return [([limb_img, disc, ring], no_pairs), ([limb_img, disc, ring, blob], no_pairs)]
    if family in ('limb_ring_aniso_fixed', 'limb_ring_aniso_diverse'):
        return [([limb_rot, ring], no_pairs), ([limb_rot, disc, ring], no_pairs)]
    if family == 'multi_body':
        multi = [
            EstimatorSpec('limb@RHEA', 'full', group='limb'),
            EstimatorSpec('limb@DIONE', 'full', group='limb'),
            EstimatorSpec('disc@RHEA', 'full', group='disc'),
            EstimatorSpec('disc@DIONE', 'full', group='disc'),
        ]
        # The naive solve assumes cross-body same-technique independence;
        # the second declares the limb-limb pair covariance instead of
        # assuming it away (the measured coupling is material at ~1.5 px^2,
        # while the disc-disc coupling, though correlated, is ~1e-4 px^2 --
        # immaterial in magnitude, and declaring both within-group pairs
        # would make the system degenerate again).
        declared = [('limb@RHEA', 'limb@DIONE')]
        return [(multi, no_pairs), (multi, declared)]
    raise ValueError(f'unknown family {family!r}')


def main(argv: list[str] | None = None) -> int:
    """Produce the identifiability / bias-independence report."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('rows', type=Path, help='control (no-injection) rows file')
    parser.add_argument('--dt-rows', type=Path, default=None)
    parser.add_argument('--noise-rows', type=Path, default=None)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args(argv)

    manifest, rows = load_rows(args.rows)
    out: list[str] = []
    out.append('# Agreement-estimator validation report')
    out.append('')
    out.append(
        f'Control rows: `{args.rows}` (campaign seed '
        f'{manifest.get("campaign_seed", "?")}, per-family '
        f'{manifest.get("per_family", "?")}).'
    )
    out.append('')

    families = [f for f in manifest.get('families', []) if any(r['family'] == f for r in rows)]
    out.append('## Stage 0a: identifiability per composition')
    out.append('')
    for family in families:
        cohort = build_cohort(rows, family)
        out.append(f'## Composition: {family}')
        out.append('')
        n_rows = sum(1 for r in rows if r['family'] == family)
        n_err = sum(1 for r in rows if r['family'] == family and 'error' in r)
        out.append(f'{n_rows} scenes ({n_err} errors), {len(cohort)} usable frames.')
        out.append('')
        spec_sets = _specs_for_family(family)
        for specs, declared_pairs in spec_sets:
            names = '+'.join(s.name for s in specs)
            if declared_pairs:
                names += ' (declared pair covariances)'
            usable = [c for c in cohort if sum(1 for s in specs if s.name in c.sample.offsets) >= 2]
            report_solve(
                out,
                usable,
                specs,
                f'{family}: solve over {names}',
                pair_covariances=declared_pairs,
            )
            report_truth_reference(out, usable, specs)
        # Intrinsic pair couplings (truth-based): the independence
        # assumption every undeclared pair rides on, measured per cohort.
        all_instances = sorted({n for c in cohort for n in c.sample.offsets})
        pairs = [(a, b) for i, a in enumerate(all_instances) for b in all_instances[i + 1 :]]
        report_pair_truth(out, cohort, pairs, f'{family}: truth-based pair coupling (no injection)')

    if args.dt_rows is not None or args.noise_rows is not None:
        out.append('## Stage 0b: bias independence through the shared layer')
        out.append('')
        family = 'limb_disc_ring_diverse'
        control_cohort = build_cohort(rows, family)
        pivotal = [('limb', 'disc'), ('limb', 'ring'), ('disc', 'ring'), ('limb', 'blob')]
        report_pair_truth(
            out, control_cohort, pivotal, 'Control (no injection): truth-based pair coupling'
        )
        for label, path in (('dt_shift', args.dt_rows), ('noise_scale', args.noise_rows)):
            if path is None:
                continue
            inj_manifest, inj_rows = load_rows(path)
            cohort = build_cohort(inj_rows, family)
            out.append(f'### Injection: {label} (`{path}`)')
            out.append('')
            out.append(
                f'Injection parameters: sigma_px '
                f'{inj_manifest.get("injection_sigma_px", "?")} (dt_shift only).'
            )
            out.append('')
            report_pair_truth(out, cohort, pivotal, f'{label}: truth-based pair coupling')
            if label == 'noise_scale':
                report_noise_response(out, rows, inj_rows, family)
            if label == 'dt_shift':
                report_injection_response(out, rows, inj_rows, family)
                out.append('Solve-side detection (declared limb-ring pair covariance,')
                out.append('blob included for over-determination):')
                out.append('')
                specs = [
                    EstimatorSpec('limb', 'full'),
                    EstimatorSpec('disc', 'full'),
                    EstimatorSpec('ring', 'rank1'),
                    EstimatorSpec('blob', 'full'),
                ]
                usable = [
                    c for c in cohort if sum(1 for s in specs if s.name in c.sample.offsets) >= 2
                ]
                report_solve(
                    out,
                    usable,
                    specs,
                    f'{label}: independence-assumed solve (misattribution check)',
                )
                report_solve(
                    out,
                    usable,
                    specs,
                    f'{label}: solve with declared cov(limb,ring)',
                    pair_covariances=[('limb', 'ring')],
                )
                report_truth_reference(out, usable, specs)

    text = '\n'.join(out) + '\n'
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f'Wrote {args.out}')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
