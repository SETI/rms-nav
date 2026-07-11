"""Generate a deterministic statistics report (Markdown + charts) from the database."""

import argparse
import itertools
import json
import math
import sqlite3
import statistics
from pathlib import Path
from typing import Any

from spindoctor.cli.stats.schema import open_stats_db
from spindoctor.config import MAIN_LOGGER

__all__ = ['build_report', 'main_report']


def _where_clause(
    *,
    instrument: str | None,
    start_date: str | None,
    end_date: str | None,
) -> tuple[str, list[str]]:
    """Build the images-table filter shared by every query."""
    clauses: list[str] = []
    params: list[str] = []
    if instrument:
        clauses.append('instrument = ?')
        params.append(instrument)
    if start_date:
        clauses.append('image_date >= ?')
        params.append(start_date)
    if end_date:
        clauses.append('image_date <= ?')
        params.append(end_date)
    if not clauses:
        return '', []
    return ' WHERE ' + ' AND '.join(clauses), params


def _rows(conn: sqlite3.Connection, sql: str, params: list[str]) -> list[tuple[Any, ...]]:
    return list(conn.execute(sql, params))


def _fmt(value: float | None, digits: int = 3) -> str:
    """Format a float for a Markdown table cell."""
    if value is None:
        return '-'
    return f'{value:.{digits}f}'


def _offset_stats(values: list[float]) -> dict[str, float] | None:
    """Mean / median / stdev / min / max summary of a value list."""
    if not values:
        return None
    return {
        'mean': statistics.fmean(values),
        'median': statistics.median(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values),
    }


def _pairwise_disagreements(
    conn: sqlite3.Connection, where: str, params: list[str]
) -> tuple[dict[tuple[str, str], list[float]], dict[str, list[float]]]:
    """Cross-technique agreement data.

    Returns:
        ``(per_pair, per_image_rank)`` where ``per_pair`` maps a sorted
        technique-name pair to the Euclidean distances between their offsets
        on images where both produced non-spurious results, and
        ``per_image_rank`` maps each image's ``confidence_rank`` to that
        image's maximum pairwise disagreement.
    """
    sql = (
        'SELECT t.image_name, i.confidence_rank, t.technique_name, t.offset_dv, t.offset_du '
        'FROM techniques t JOIN images i ON i.image_name = t.image_name'
        + where.replace('instrument', 'i.instrument').replace('image_date', 'i.image_date')
        + (' AND ' if where else ' WHERE ')
        + 't.spurious = 0 AND t.offset_dv IS NOT NULL AND t.offset_du IS NOT NULL '
        'ORDER BY t.image_name, t.technique_name'
    )
    per_pair: dict[tuple[str, str], list[float]] = {}
    per_image_rank: dict[str, list[float]] = {}
    for _image_name, group in itertools.groupby(_rows(conn, sql, params), key=lambda r: r[0]):
        entries = list(group)
        if len(entries) < 2:
            continue
        rank = str(entries[0][1])
        image_max = 0.0
        for a, b in itertools.combinations(entries, 2):
            delta = math.hypot(a[3] - b[3], a[4] - b[4])
            pair = (min(a[2], b[2]), max(a[2], b[2]))
            per_pair.setdefault(pair, []).append(delta)
            image_max = max(image_max, delta)
        per_image_rank.setdefault(rank, []).append(image_max)
    return per_pair, per_image_rank


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of a non-empty value list."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _write_bar_chart(
    path: Path, labels: list[str], counts: list[int], *, title: str, xlabel: str
) -> None:
    """Write a horizontal bar chart PNG (deterministic, Agg backend)."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, max(2.0, 0.4 * len(labels) + 1.0)))
    positions = range(len(labels))
    ax.barh(list(positions), counts, color='#4878d0')
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _write_offset_hist(path: Path, dv: list[float], du: list[float]) -> None:
    """Write the V/U offset histogram PNG."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, values, label in ((axes[0], dv, 'dV (px)'), (axes[1], du, 'dU (px)')):
        if values:
            ax.hist(values, bins=40, color='#4878d0')
        ax.set_xlabel(label)
        ax.set_ylabel('images')
    fig.suptitle('Fused offset distribution (successful images)')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def build_report(
    conn: sqlite3.Connection,
    output_dir: Path,
    *,
    instrument: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    """Query the statistics database and write ``report.md`` plus charts.

    Parameters:
        conn: Open statistics database connection.
        output_dir: Directory receiving ``report.md`` and the PNG charts
            (created if missing).
        instrument: Optional instrument filter (``coiss`` / ``vgiss`` /
            ``gossi`` / ``nhlorri``).
        start_date: Optional inclusive UTC start date (``YYYY-MM-DD``).
        end_date: Optional inclusive UTC end date (``YYYY-MM-DD``).

    Returns:
        The path of the written ``report.md``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    where, params = _where_clause(instrument=instrument, start_date=start_date, end_date=end_date)
    lines: list[str] = ['# Navigation statistics report', '']
    filters = [
        f'instrument = {instrument}' if instrument else None,
        f'from {start_date}' if start_date else None,
        f'to {end_date}' if end_date else None,
    ]
    active = [f for f in filters if f]
    lines.append(f'Filters: {", ".join(active) if active else "none (full database)"}')
    lines.append('')

    # --- Success / failure counts -----------------------------------------
    status_rows = _rows(
        conn,
        f'SELECT status, COUNT(*) FROM images{where} GROUP BY status ORDER BY status',
        params,
    )
    total = sum(r[1] for r in status_rows)
    lines += ['## Success / failure', '', '| status | images | fraction |', '|---|---|---|']
    for status, count in status_rows:
        lines.append(f'| {status} | {count} | {count / total:.3f} |' if total else '')
    lines += ['', f'Total images: {total}', '']
    _write_bar_chart(
        output_dir / 'status_counts.png',
        [str(r[0]) for r in status_rows],
        [int(r[1]) for r in status_rows],
        title='Navigation status',
        xlabel='images',
    )
    lines += ['![status](status_counts.png)', '']

    reason_rows = _rows(
        conn,
        f'SELECT status_reason, COUNT(*) FROM images{where}'
        + (' AND ' if where else ' WHERE ')
        + "status != 'success' GROUP BY status_reason ORDER BY COUNT(*) DESC, status_reason",
        params,
    )
    if reason_rows:
        lines += ['### Failure reasons', '', '| reason | images |', '|---|---|']
        lines += [f'| {reason or "(none)"} | {count} |' for reason, count in reason_rows]
        lines.append('')
        _write_bar_chart(
            output_dir / 'failure_reasons.png',
            [str(r[0] or '(none)') for r in reason_rows],
            [int(r[1]) for r in reason_rows],
            title='Failure reasons',
            xlabel='images',
        )
        lines += ['![failure reasons](failure_reasons.png)', '']

    # --- Technique usage ----------------------------------------------------
    tech_rows = _rows(
        conn,
        'SELECT t.technique_name, COUNT(*), SUM(1 - t.spurious), AVG(t.confidence) '
        'FROM techniques t JOIN images i ON i.image_name = t.image_name'
        + where.replace('instrument', 'i.instrument').replace('image_date', 'i.image_date')
        + ' GROUP BY t.technique_name ORDER BY COUNT(*) DESC, t.technique_name',
        params,
    )
    lines += [
        '## Technique usage',
        '',
        '| technique | runs | non-spurious | mean confidence |',
        '|---|---|---|---|',
    ]
    for name, runs, good, mean_conf in tech_rows:
        lines.append(f'| {name} | {runs} | {good} | {_fmt(mean_conf)} |')
    lines.append('')
    if tech_rows:
        _write_bar_chart(
            output_dir / 'technique_usage.png',
            [str(r[0]) for r in tech_rows],
            [int(r[1]) for r in tech_rows],
            title='Technique runs',
            xlabel='runs',
        )
        lines += ['![technique usage](technique_usage.png)', '']

    # --- Model / body / ring usage -------------------------------------------
    source_rows = _rows(
        conn,
        'SELECT s.source_model, s.source_name, COUNT(DISTINCT s.image_name), '
        'SUM(s.n_features), SUM(s.n_gated) '
        'FROM feature_sources s JOIN images i ON i.image_name = s.image_name'
        + where.replace('instrument', 'i.instrument').replace('image_date', 'i.image_date')
        + ' GROUP BY s.source_model, s.source_name '
        'ORDER BY s.source_model, COUNT(DISTINCT s.image_name) DESC, s.source_name',
        params,
    )
    lines += [
        '## Model and source usage',
        '',
        '| model | source | images | features | gated |',
        '|---|---|---|---|---|',
    ]
    for model, name, n_images, n_features, n_gated in source_rows:
        lines.append(f'| {model} | {name} | {n_images} | {n_features} | {n_gated} |')
    lines.append('')

    # --- Offset statistics ---------------------------------------------------
    offset_rows = _rows(
        conn,
        f'SELECT offset_dv, offset_du FROM images{where}'
        + (' AND ' if where else ' WHERE ')
        + "status = 'success' AND offset_dv IS NOT NULL",
        params,
    )
    dv = [float(r[0]) for r in offset_rows]
    du = [float(r[1]) for r in offset_rows]
    lines += [
        '## Offset statistics (successful images)',
        '',
        '| axis | n | mean | median | stdev | min | max |',
        '|---|---|---|---|---|---|---|',
    ]
    for axis, values in (('dV', dv), ('dU', du)):
        stats = _offset_stats(values)
        if stats is None:
            lines.append(f'| {axis} | 0 | - | - | - | - | - |')
        else:
            lines.append(
                f'| {axis} | {len(values)} | {_fmt(stats["mean"])} | {_fmt(stats["median"])} '
                f'| {_fmt(stats["stdev"])} | {_fmt(stats["min"])} | {_fmt(stats["max"])} |'
            )
    lines.append('')
    _write_offset_hist(output_dir / 'offsets_hist.png', dv, du)
    lines += ['![offsets](offsets_hist.png)', '']

    # --- Cross-technique agreement -------------------------------------------
    per_pair, per_image_rank = _pairwise_disagreements(conn, where, params)
    lines += [
        '## Cross-technique agreement',
        '',
        'Euclidean distance between per-technique offsets on images where both',
        'techniques produced non-spurious results.',
        '',
        '| technique pair | images | median (px) | p95 (px) |',
        '|---|---|---|---|',
    ]
    for pair in sorted(per_pair):
        deltas = per_pair[pair]
        lines.append(
            f'| {pair[0]} vs {pair[1]} | {len(deltas)} | '
            f'{_fmt(statistics.median(deltas))} | {_fmt(_percentile(deltas, 0.95))} |'
        )
    lines.append('')

    # --- Confidence calibration ------------------------------------------------
    rank_rows = _rows(
        conn,
        f'SELECT confidence_rank, COUNT(*) FROM images{where}'
        + (' AND ' if where else ' WHERE ')
        + 'confidence_rank IS NOT NULL GROUP BY confidence_rank ORDER BY confidence_rank',
        params,
    )
    lines += [
        '## Confidence calibration (agreement as accuracy proxy)',
        '',
        'For each confidence tier: how well the techniques that fed the fused',
        'offset agreed with one another.  Without ground truth, cross-technique',
        'agreement is the standing production check that confidence tiers are',
        'meaningful (the calibrated anchor is the WS-5 sim campaign).',
        '',
        '| tier | images | with >=2 techniques | median max-disagreement (px) | p95 (px) |',
        '|---|---|---|---|---|',
    ]
    for rank, count in rank_rows:
        disagreements = per_image_rank.get(str(rank), [])
        lines.append(
            f'| {rank} | {count} | {len(disagreements)} | '
            f'{_fmt(statistics.median(disagreements)) if disagreements else "-"} | '
            f'{_fmt(_percentile(disagreements, 0.95)) if disagreements else "-"} |'
        )
    lines.append('')
    if per_image_rank:
        ordered_ranks = sorted(per_image_rank)
        _write_bar_chart(
            output_dir / 'agreement_by_tier.png',
            ordered_ranks,
            [len(per_image_rank[r]) for r in ordered_ranks],
            title='Images with cross-technique agreement data, by tier',
            xlabel='images',
        )
        lines += ['![agreement by tier](agreement_by_tier.png)', '']

    # --- Consensus exclusions ---------------------------------------------------
    excluded_rows = _rows(
        conn,
        f'SELECT excluded_from_consensus, COUNT(*) FROM images{where}'
        + (' AND ' if where else ' WHERE ')
        + "excluded_from_consensus != '[]' "
        'GROUP BY excluded_from_consensus ORDER BY COUNT(*) DESC, excluded_from_consensus',
        params,
    )
    if excluded_rows:
        lines += [
            '## Ensemble outlier exclusions',
            '',
            '| excluded techniques | images |',
            '|---|---|',
        ]
        for raw, count in excluded_rows:
            names = ', '.join(json.loads(raw)) or '(none)'
            lines.append(f'| {names} | {count} |')
        lines.append('')

    report_path = output_dir / 'report.md'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


def main_report(cmdline: list[str] | None = None) -> int:
    """Entry point for ``sd_stats_report``.

    Parameters:
        cmdline: Argument list; None uses ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description='Generate a navigation statistics report from an ingested database.'
    )
    parser.add_argument(
        '--db',
        default='nav_stats.sqlite3',
        help='SQLite database path written by sd_stats_ingest (default: %(default)s)',
    )
    parser.add_argument(
        '--output-dir',
        default='nav_stats_report',
        help='Directory receiving report.md and charts (default: %(default)s)',
    )
    parser.add_argument(
        '--instrument',
        choices=['coiss', 'vgiss', 'gossi', 'nhlorri'],
        default=None,
        help='Restrict the report to one instrument',
    )
    parser.add_argument(
        '--start-date', default=None, metavar='YYYY-MM-DD', help='Inclusive UTC start date'
    )
    parser.add_argument(
        '--end-date', default=None, metavar='YYYY-MM-DD', help='Inclusive UTC end date'
    )
    arguments = parser.parse_args(cmdline)

    conn = open_stats_db(arguments.db)
    try:
        report_path = build_report(
            conn,
            Path(arguments.output_dir),
            instrument=arguments.instrument,
            start_date=arguments.start_date,
            end_date=arguments.end_date,
        )
    finally:
        conn.close()
    MAIN_LOGGER.info('Wrote %s', report_path)
    return 0
