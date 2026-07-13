"""Generate a deterministic statistics report (Markdown + charts) from the database."""

import argparse
import itertools
import json
import math
import sqlite3
import statistics
from pathlib import Path

from spindoctor.cli.stats.report_common import (
    ReportContext,
    add_drilldown,
    connector,
    fmt,
    image_number_from_name,
    offset_stats,
    percentile,
    register_image_number_function,
    rows,
    where_clause,
    write_bar_chart,
    write_offset_hist,
)
from spindoctor.cli.stats.report_sections import (
    add_botsim_section,
    add_failure_taxonomy_section,
    add_offset_by_group_section,
    add_runtime_section,
    add_suspect_offset_section,
    write_csv_export,
)
from spindoctor.cli.stats.schema import open_stats_db
from spindoctor.config import MAIN_LOGGER

__all__ = ['build_report', 'main_report']


def _pairwise_disagreements(
    ctx: ReportContext,
) -> tuple[dict[tuple[str, str], list[float]], dict[str, list[float]]]:
    """Cross-technique agreement data.

    Returns:
        ``(per_pair, per_image_rank)`` where ``per_pair`` maps a sorted
        technique-name pair to the Euclidean distances between their offsets
        on images where both produced non-spurious results, and
        ``per_image_rank`` maps each image's ``confidence_rank`` to that
        image's maximum pairwise disagreement.  Images with a NULL
        ``confidence_rank`` contribute to ``per_pair`` but not to
        ``per_image_rank``.
    """
    sql = (
        'SELECT t.image_name, i.confidence_rank, t.technique_name, t.offset_dv, t.offset_du '
        'FROM techniques t JOIN images i ON i.image_name = t.image_name'
        + ctx.where_i
        + connector(ctx.where_i)
        + 't.spurious = 0 AND t.offset_dv IS NOT NULL AND t.offset_du IS NOT NULL '
        'ORDER BY t.image_name, t.technique_name'
    )
    per_pair: dict[tuple[str, str], list[float]] = {}
    per_image_rank: dict[str, list[float]] = {}
    for _image_name, group in itertools.groupby(
        rows(ctx.conn, sql, ctx.params_i), key=lambda r: r[0]
    ):
        entries = list(group)
        if len(entries) < 2:
            continue
        rank = entries[0][1]
        image_max = 0.0
        for a, b in itertools.combinations(entries, 2):
            delta = math.hypot(a[3] - b[3], a[4] - b[4])
            pair = (min(a[2], b[2]), max(a[2], b[2]))
            per_pair.setdefault(pair, []).append(delta)
            image_max = max(image_max, delta)
        if rank is not None:
            per_image_rank.setdefault(str(rank), []).append(image_max)
    return per_pair, per_image_rank


def _add_status_sections(ctx: ReportContext) -> None:
    """Append the success/failure counts and the failure-reason breakdown."""
    status_rows = rows(
        ctx.conn,
        f'SELECT status, COUNT(*) FROM images{ctx.where} GROUP BY status ORDER BY status',
        ctx.params,
    )
    total = sum(r[1] for r in status_rows)
    ctx.lines += ['## Success / failure', '', '| status | images | fraction |', '|---|---|---|']
    for status, count in status_rows:
        ctx.lines.append(f'| {status} | {count} | {count / total:.3f} |' if total > 0 else '')
    ctx.lines += ['', f'Total images: {total}', '']
    write_bar_chart(
        ctx.output_dir / 'status_counts.png',
        [str(r[0]) for r in status_rows],
        [int(r[1]) for r in status_rows],
        title='Navigation status',
        xlabel='images',
    )
    ctx.lines += ['![status](status_counts.png)', '']

    reason_rows = rows(
        ctx.conn,
        f'SELECT status_reason, COUNT(*) FROM images{ctx.where}'
        + connector(ctx.where)
        + "status != 'success' GROUP BY status_reason ORDER BY COUNT(*) DESC, status_reason",
        ctx.params,
    )
    if len(reason_rows) > 0:
        ctx.lines += ['### Failure reasons', '', '| reason | images |', '|---|---|']
        ctx.lines += [f'| {reason or "(none)"} | {count} |' for reason, count in reason_rows]
        ctx.lines.append('')
        name_rows = rows(
            ctx.conn,
            f'SELECT status_reason, image_name FROM images{ctx.where}'
            + connector(ctx.where)
            + "status != 'success' ORDER BY image_name",
            ctx.params,
        )
        names_by_reason: dict[str, list[str]] = {}
        for reason, image_name in name_rows:
            names_by_reason.setdefault(str(reason or '(none)'), []).append(str(image_name))
        add_drilldown(
            ctx,
            [
                (str(reason or '(none)'), names_by_reason.get(str(reason or '(none)'), []))
                for reason, _count in reason_rows
            ],
            label='reason',
            stub_prefix='failure_reason',
        )
        write_bar_chart(
            ctx.output_dir / 'failure_reasons.png',
            [str(r[0] or '(none)') for r in reason_rows],
            [int(r[1]) for r in reason_rows],
            title='Failure reasons',
            xlabel='images',
        )
        ctx.lines += ['![failure reasons](failure_reasons.png)', '']


def _add_technique_usage_section(ctx: ReportContext) -> None:
    """Append per-technique run counts and mean confidence."""
    tech_rows = rows(
        ctx.conn,
        'SELECT t.technique_name, COUNT(*), SUM(1 - t.spurious), AVG(t.confidence) '
        'FROM techniques t JOIN images i ON i.image_name = t.image_name'
        + ctx.where_i
        + ' GROUP BY t.technique_name ORDER BY COUNT(*) DESC, t.technique_name',
        ctx.params_i,
    )
    ctx.lines += [
        '## Technique usage',
        '',
        '| technique | runs | non-spurious | mean confidence |',
        '|---|---|---|---|',
    ]
    for name, runs, good, mean_conf in tech_rows:
        ctx.lines.append(f'| {name} | {runs} | {good} | {fmt(mean_conf)} |')
    ctx.lines.append('')
    if len(tech_rows) > 0:
        write_bar_chart(
            ctx.output_dir / 'technique_usage.png',
            [str(r[0]) for r in tech_rows],
            [int(r[1]) for r in tech_rows],
            title='Technique runs',
            xlabel='runs',
        )
        ctx.lines += ['![technique usage](technique_usage.png)', '']


def _add_source_usage_section(ctx: ReportContext) -> None:
    """Append the per-model / per-source feature-usage table."""
    source_rows = rows(
        ctx.conn,
        'SELECT s.source_model, s.source_name, COUNT(DISTINCT s.image_name), '
        'SUM(s.n_features), SUM(s.n_gated) '
        'FROM feature_sources s JOIN images i ON i.image_name = s.image_name'
        + ctx.where_i
        + ' GROUP BY s.source_model, s.source_name '
        'ORDER BY s.source_model, COUNT(DISTINCT s.image_name) DESC, s.source_name',
        ctx.params_i,
    )
    ctx.lines += [
        '## Model and source usage',
        '',
        '| model | source | images | features | gated |',
        '|---|---|---|---|---|',
    ]
    for model, name, n_images, n_features, n_gated in source_rows:
        ctx.lines.append(f'| {model} | {name} | {n_images} | {n_features} | {n_gated} |')
    ctx.lines.append('')


def _add_offset_section(ctx: ReportContext) -> None:
    """Append the fused-offset statistics table and histogram."""
    offset_rows = rows(
        ctx.conn,
        f'SELECT offset_dv, offset_du FROM images{ctx.where}'
        + connector(ctx.where)
        + "status = 'success' AND offset_dv IS NOT NULL",
        ctx.params,
    )
    dv = [float(r[0]) for r in offset_rows]
    du = [float(r[1]) for r in offset_rows]
    ctx.lines += [
        '## Offset statistics (successful images)',
        '',
        '| axis | n | mean | median | stdev | min | max |',
        '|---|---|---|---|---|---|---|',
    ]
    for axis, values in (('dV', dv), ('dU', du)):
        stats = offset_stats(values)
        if stats is None:
            ctx.lines.append(f'| {axis} | 0 | - | - | - | - | - |')
        else:
            ctx.lines.append(
                f'| {axis} | {len(values)} | {fmt(stats["mean"])} | {fmt(stats["median"])} '
                f'| {fmt(stats["stdev"])} | {fmt(stats["min"])} | {fmt(stats["max"])} |'
            )
    ctx.lines.append('')
    write_offset_hist(ctx.output_dir / 'offsets_hist.png', dv, du)
    ctx.lines += ['![offsets](offsets_hist.png)', '']


def _add_agreement_sections(ctx: ReportContext) -> None:
    """Append the cross-technique agreement and confidence-calibration tables."""
    per_pair, per_image_rank = _pairwise_disagreements(ctx)
    ctx.lines += [
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
        ctx.lines.append(
            f'| {pair[0]} vs {pair[1]} | {len(deltas)} | '
            f'{fmt(statistics.median(deltas))} | {fmt(percentile(deltas, 0.95))} |'
        )
    ctx.lines.append('')

    rank_rows = rows(
        ctx.conn,
        f'SELECT confidence_rank, COUNT(*) FROM images{ctx.where}'
        + connector(ctx.where)
        + 'confidence_rank IS NOT NULL GROUP BY confidence_rank ORDER BY confidence_rank',
        ctx.params,
    )
    ctx.lines += [
        '## Confidence calibration (agreement as accuracy proxy)',
        '',
        'For each confidence tier: how well the techniques that fed the fused',
        'offset agreed with one another.  Without ground truth, cross-technique',
        'agreement is the standing production check that confidence tiers are',
        'meaningful (the calibrated anchor is the simulated-scene campaign).',
        '',
        '| tier | images | with >=2 techniques | median max-disagreement (px) | p95 (px) |',
        '|---|---|---|---|---|',
    ]
    for rank, count in rank_rows:
        disagreements = per_image_rank.get(str(rank), [])
        ctx.lines.append(
            f'| {rank} | {count} | {len(disagreements)} | '
            f'{fmt(statistics.median(disagreements)) if disagreements else "-"} | '
            f'{fmt(percentile(disagreements, 0.95)) if disagreements else "-"} |'
        )
    ctx.lines.append('')
    if len(per_image_rank) > 0:
        ordered_ranks = sorted(per_image_rank)
        write_bar_chart(
            ctx.output_dir / 'agreement_by_tier.png',
            ordered_ranks,
            [len(per_image_rank[r]) for r in ordered_ranks],
            title='Images with cross-technique agreement data, by tier',
            xlabel='images',
        )
        ctx.lines += ['![agreement by tier](agreement_by_tier.png)', '']


def _add_exclusions_section(ctx: ReportContext) -> None:
    """Append the ensemble outlier-exclusion breakdown."""
    excluded_rows = rows(
        ctx.conn,
        f'SELECT excluded_from_consensus, COUNT(*) FROM images{ctx.where}'
        + connector(ctx.where)
        + "excluded_from_consensus != '[]' "
        'GROUP BY excluded_from_consensus ORDER BY COUNT(*) DESC, excluded_from_consensus',
        ctx.params,
    )
    if len(excluded_rows) == 0:
        return
    ctx.lines += [
        '## Ensemble outlier exclusions',
        '',
        '| excluded techniques | images |',
        '|---|---|',
    ]
    for raw, count in excluded_rows:
        names = ', '.join(json.loads(raw)) or '(none)'
        ctx.lines.append(f'| {names} | {count} |')
    ctx.lines.append('')
    name_rows = rows(
        ctx.conn,
        f'SELECT excluded_from_consensus, image_name FROM images{ctx.where}'
        + connector(ctx.where)
        + "excluded_from_consensus != '[]' ORDER BY image_name",
        ctx.params,
    )
    names_by_exclusion: dict[str, list[str]] = {}
    for raw, image_name in name_rows:
        label = ', '.join(json.loads(raw)) or '(none)'
        names_by_exclusion.setdefault(label, []).append(str(image_name))
    ordered_labels = [', '.join(json.loads(raw)) or '(none)' for raw, _count in excluded_rows]
    add_drilldown(
        ctx,
        [(label, names_by_exclusion.get(label, [])) for label in ordered_labels],
        label='exclusion set',
        stub_prefix='excluded',
    )


def _image_bound(value: str | None, *, option: str) -> int | None:
    """Parse a ``--min-image`` / ``--max-image`` value into its numeric bound.

    Parameters:
        value: Image name (``N1454725799``) or bare number, or None.
        option: Option name for the error message.

    Returns:
        The integer bound, or None when ``value`` is None.

    Raises:
        ValueError: If the value contains no digits.
    """
    if value is None:
        return None
    number = image_number_from_name(value)
    if number is None:
        raise ValueError(f'{option} value {value!r} contains no digits')
    return number


def build_report(
    conn: sqlite3.Connection,
    output_dir: Path,
    *,
    instrument: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_image: str | None = None,
    max_image: str | None = None,
    top_n: int = 0,
    filelists: bool = False,
    suspect_fraction: float = 0.9,
    csv_export: bool = False,
) -> Path:
    """Query the statistics database and write ``report.md`` plus charts.

    The report is deterministic: the same database and options always
    produce byte-identical Markdown.  All filters combine and apply to
    every section.

    Parameters:
        conn: Open statistics database connection.
        output_dir: Directory receiving ``report.md``, the PNG charts, and
            (with ``filelists`` / ``csv_export``) the ``filelists/``
            subdirectory and ``images.csv`` (created if missing).
        instrument: Optional instrument filter (``coiss`` / ``vgiss`` /
            ``gossi`` / ``nhlorri``).
        start_date: Optional inclusive UTC start date (``YYYY-MM-DD``).
        end_date: Optional inclusive UTC end date (``YYYY-MM-DD``).
        min_image: Optional inclusive lower bound on the numeric portion
            of the image name; an image name (``N1454725799``) or a bare
            number.
        max_image: Optional inclusive upper bound on the numeric portion
            of the image name.
        top_n: When positive, categorical sections list up to this many
            example image names per category, the suspect-offset and
            worst-BOTSIM-pair tables are capped at this many rows, and the
            slowest images are listed.
        filelists: When True, write one plain-text file per category (one
            image name per line, full list) under ``filelists/``.
        suspect_fraction: Fraction of the per-axis maximum expected
            pointing offset at or beyond which a fused offset is flagged
            as suspect.
        csv_export: When True, write the flattened one-row-per-image
            ``images.csv`` next to ``report.md``.

    Returns:
        The path of the written ``report.md``.

    Raises:
        ValueError: If ``min_image`` or ``max_image`` contains no digits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    register_image_number_function(conn)
    min_image_num = _image_bound(min_image, option='min_image')
    max_image_num = _image_bound(max_image, option='max_image')
    where, params = where_clause(
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        min_image_num=min_image_num,
        max_image_num=max_image_num,
    )
    # Joined queries alias the images table as ``i``; same filter, qualified.
    where_i, params_i = where_clause(
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        min_image_num=min_image_num,
        max_image_num=max_image_num,
        alias='i.',
    )
    ctx = ReportContext(
        conn=conn,
        output_dir=output_dir,
        where=where,
        params=params,
        where_i=where_i,
        params_i=params_i,
        top_n=top_n,
        filelists=filelists,
        suspect_fraction=suspect_fraction,
    )
    ctx.lines += ['# Navigation statistics report', '']
    filters = [
        f'instrument = {instrument}' if instrument is not None else None,
        f'from {start_date}' if start_date is not None else None,
        f'to {end_date}' if end_date is not None else None,
        f'image number >= {min_image_num}' if min_image_num is not None else None,
        f'image number <= {max_image_num}' if max_image_num is not None else None,
    ]
    active = [f for f in filters if f is not None]
    ctx.lines.append(f'Filters: {", ".join(active) if len(active) > 0 else "none (full database)"}')
    ctx.lines.append('')

    _add_status_sections(ctx)
    add_failure_taxonomy_section(ctx)
    _add_technique_usage_section(ctx)
    _add_source_usage_section(ctx)
    _add_offset_section(ctx)
    add_offset_by_group_section(ctx)
    add_suspect_offset_section(ctx)
    add_botsim_section(ctx)
    _add_agreement_sections(ctx)
    _add_exclusions_section(ctx)
    add_runtime_section(ctx)
    if csv_export:
        write_csv_export(ctx)

    report_path = output_dir / 'report.md'
    report_path.write_text('\n'.join(ctx.lines) + '\n', encoding='utf-8')
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
    parser.add_argument(
        '--min-image',
        default=None,
        metavar='NAME',
        help='Inclusive lower bound on the numeric portion of the image name '
        '(an image name like N1454725799 or a bare number)',
    )
    parser.add_argument(
        '--max-image',
        default=None,
        metavar='NAME',
        help='Inclusive upper bound on the numeric portion of the image name',
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=0,
        metavar='N',
        help='List up to N example image names per category in categorical '
        'sections, cap the suspect-offset and worst-BOTSIM-pair tables at N '
        'rows, and list the N slowest images (default: 0 = off)',
    )
    parser.add_argument(
        '--filelists',
        action='store_true',
        default=False,
        help='Write one plain-text file per category (one image name per '
        'line, full list) into the filelists/ subdirectory of the output dir',
    )
    parser.add_argument(
        '--suspect-fraction',
        type=float,
        default=0.9,
        metavar='F',
        help='Flag successful offsets at or beyond this fraction of the '
        'per-axis maximum expected pointing offset (default: %(default)s)',
    )
    parser.add_argument(
        '--csv',
        action='store_true',
        default=False,
        help='Write a flattened one-row-per-image images.csv next to report.md',
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
            min_image=arguments.min_image,
            max_image=arguments.max_image,
            top_n=arguments.top_n,
            filelists=arguments.filelists,
            suspect_fraction=arguments.suspect_fraction,
            csv_export=arguments.csv,
        )
    except ValueError as exc:
        parser.error(str(exc))
    finally:
        conn.close()
    MAIN_LOGGER.info('Wrote %s', report_path)
    return 0
