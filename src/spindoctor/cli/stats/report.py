"""Generate a deterministic statistics report (Markdown + charts) from the index.

The report is the one program for which the results index is not optional: it
has never had a file-reading mode, and it answers every question with a query.
Its output is deterministic -- the same index and the same options always
produce byte-identical Markdown -- so a difference between two runs is a
difference in the data or a defect, never noise.

Every statement runs on either backend.  The reason tables read
``COALESCE(status_reason, status_error)``, because the two are separate columns
holding separate vocabularies and a failure is described by whichever of them
the document carried; the boolean columns are used as booleans rather than as
zeroes and ones; and every join onto ``images`` matches the pair that keys an
image rather than its name, which two volumes may share.
"""

import argparse
import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import sqlalchemy
from filecache import FCPath

from spindoctor.cli.stats.classify import datetime_from_image_et, image_number_from_name
from spindoctor.cli.stats.report_common import (
    IMAGE_JOIN,
    ReportContext,
    add_drilldown,
    add_instrument_count_table,
    connector,
    count_pct,
    fmt,
    image_name_from_filename,
    image_order,
    offset_stats,
    percentile,
    rows,
    safe_filename,
    where_clause,
    write_offset_hist,
    write_stacked_bar_chart,
)
from spindoctor.cli.stats.report_sections import (
    add_botsim_section,
    add_failure_taxonomy_section,
    add_offset_by_group_section,
    add_runtime_section,
    add_suspect_offset_section,
    write_csv_export,
)
from spindoctor.config import DEFAULT_CONFIG, get_results_db_url
from spindoctor.results_index import normalize_root_url, open_index, require_ingested_roots

__all__ = ['build_report', 'main_report']

# Which of the two reason vocabularies describes a non-success outcome depends
# on how the navigation ended: the navigator's own explanation when it ran, and
# the fatal error when it never got that far.  They are separate columns, and a
# report of failure reasons wants whichever one the document carried.
FAILURE_REASON = 'COALESCE({alias}status_reason, {alias}status_error)'

# Confidence tiers, in descending-confidence order, always reported.
_CONFIDENCE_TIERS: tuple[str, ...] = ('high', 'medium', 'low', 'failed', 'conflicted')


def _pairwise_disagreements(
    ctx: ReportContext,
) -> tuple[dict[tuple[str, str, str], list[float]], dict[tuple[str, str], list[float]]]:
    """Cross-technique agreement data.

    Returns:
        ``(per_pair, per_image_rank)`` where ``per_pair`` maps an
        ``(instrument, technique_a, technique_b)`` triple (the technique
        names sorted) to the Euclidean distances between those techniques'
        offsets on images where both produced non-spurious results, and
        ``per_image_rank`` maps an ``(instrument, confidence_rank)`` pair
        to each such image's maximum pairwise disagreement.  Images with a
        NULL ``confidence_rank`` contribute to ``per_pair`` but not to
        ``per_image_rank``.
    """
    sql = (
        'SELECT t.root_url, t.results_path_stub, i.instrument, i.confidence_rank, '
        't.technique_name, t.offset_dv, t.offset_du '
        'FROM techniques t '
        + IMAGE_JOIN.format(alias='t.')
        + ctx.where_i
        + connector(ctx.where_i)
        + 'NOT t.spurious AND t.offset_dv IS NOT NULL AND t.offset_du IS NOT NULL '
        'ORDER BY t.root_url, t.results_path_stub, t.technique_name'
    )
    per_pair: dict[tuple[str, str, str], list[float]] = {}
    per_image_rank: dict[tuple[str, str], list[float]] = {}
    for _image, group in itertools.groupby(
        rows(ctx.connection, sql, ctx.params_i), key=lambda r: (r[0], r[1])
    ):
        entries = list(group)
        if len(entries) < 2:
            continue
        instrument = str(entries[0][2])
        rank = entries[0][3]
        image_max = 0.0
        for a, b in itertools.combinations(entries, 2):
            delta = math.hypot(a[5] - b[5], a[6] - b[6])
            pair = (instrument, min(a[4], b[4]), max(a[4], b[4]))
            per_pair.setdefault(pair, []).append(delta)
            image_max = max(image_max, delta)
        if rank is not None:
            per_image_rank.setdefault((instrument, str(rank)), []).append(image_max)
    return per_pair, per_image_rank


def _extreme_image_name(ctx: ReportContext, instrument: str, *, last: bool) -> str:
    """The lowest- or highest-numbered selected image name of one instrument.

    Parameters:
        ctx: Report context.
        instrument: Instrument name to restrict to.
        last: True for the highest-numbered image, False for the lowest.

    Returns:
        The image name, or ``'-'`` when the instrument has no selected
        image whose name contains a number.
    """
    order = 'DESC' if last else 'ASC'
    found = rows(
        ctx.connection,
        f'SELECT image_name FROM images{ctx.where}'
        + connector(ctx.where)
        + 'instrument = :for_instrument AND image_number IS NOT NULL '
        f'ORDER BY image_number {order}, {image_order()} LIMIT 1',
        {**ctx.params, 'for_instrument': instrument},
    )
    if len(found) == 0:
        return '-'
    return image_name_from_filename(instrument, str(found[0][0]))


def _extreme_times(ctx: ReportContext, instrument: str) -> tuple[str, str]:
    """The earliest and latest epoch among one instrument's selected images.

    Computed over the images that have an epoch, independently of the
    image-name ordering: a single image with no recorded epoch at one end
    of the number range would otherwise hide the whole instrument's time
    span.

    Parameters:
        ctx: Report context.
        instrument: Instrument name to restrict to.

    Returns:
        ``(first, last)`` UTC timestamps to the second, each ``'-'`` when
        no selected image of that instrument has an epoch.
    """
    found = rows(
        ctx.connection,
        f'SELECT MIN(image_et), MAX(image_et) FROM images{ctx.where}'
        + connector(ctx.where)
        + 'instrument = :for_instrument AND image_et IS NOT NULL',
        {**ctx.params, 'for_instrument': instrument},
    )
    if len(found) == 0:
        return '-', '-'
    return (
        datetime_from_image_et(found[0][0]) or '-',
        datetime_from_image_et(found[0][1]) or '-',
    )


def _add_selection_section(ctx: ReportContext) -> None:
    """Append the per-instrument summary of what the filters selected.

    Image numbers are only comparable within one instrument, so the first
    and last image are reported per instrument and never pooled.  The image
    and time bounds are found independently, so the first image is not
    necessarily the one at the first available time.
    """
    ctx.lines += ['## Images selected', '']
    if ctx.total_images == 0:
        ctx.lines += ['No images match the filters.', '']
        return
    ctx.lines += [
        '| instrument | images | first image | last image | first avail. date | last avail. date |',
        '|---|---|---|---|---|---|',
    ]
    for instrument in ctx.instruments:
        first_image = _extreme_image_name(ctx, instrument, last=False)
        last_image = _extreme_image_name(ctx, instrument, last=True)
        first_time, last_time = _extreme_times(ctx, instrument)
        images = count_pct(ctx.images_by_instrument[instrument], ctx.total_images)
        ctx.lines.append(
            f'| {instrument} | {images} | {first_image} | {last_image} '
            f'| {first_time} | {last_time} |'
        )
    ctx.lines += ['', f'Total images: {ctx.total_images}', '']


def _add_status_sections(ctx: ReportContext) -> None:
    """Append the success/failure counts and the failure-reason breakdown."""
    status_rows = rows(
        ctx.connection,
        f'SELECT status, instrument, COUNT(*) FROM images{ctx.where} '
        'GROUP BY status, instrument ORDER BY status, instrument',
        ctx.params,
    )
    by_status: dict[str, dict[str, int]] = {}
    for status, instrument, count in status_rows:
        by_status.setdefault(str(status), {})[str(instrument)] = int(count)
    statuses = sorted(by_status, key=lambda name: (name != 'success', name))
    ctx.lines += ['## Success / failure', '']
    add_instrument_count_table(
        ctx, [([status], by_status[status]) for status in statuses], headers=['status']
    )
    write_stacked_bar_chart(
        ctx.output_dir / 'status_counts.png',
        statuses,
        {
            instrument: [by_status[status].get(instrument, 0) for status in statuses]
            for instrument in ctx.instruments
        },
        ctx.instruments,
        title='Navigation status',
        xlabel='images',
    )
    ctx.lines += ['![status](status_counts.png)', '']

    # Every non-success status is a failure for reporting purposes, so
    # `error` rows (SPICE and otherwise) appear here alongside `failed`.
    reason_column = FAILURE_REASON.format(alias='')
    reason_rows = rows(
        ctx.connection,
        f'SELECT status, {reason_column}, instrument, COUNT(*) FROM images{ctx.where}'
        + connector(ctx.where)
        + f"status != 'success' GROUP BY status, {reason_column}, instrument "
        f'ORDER BY status, {reason_column}, instrument',
        ctx.params,
    )
    if len(reason_rows) == 0:
        return
    by_reason: dict[tuple[str, str], dict[str, int]] = {}
    for status, reason, instrument, count in reason_rows:
        by_reason.setdefault((str(status), str(reason or '(none)')), {})[str(instrument)] = int(
            count
        )
    ordered = sorted(by_reason, key=lambda key: (-sum(by_reason[key].values()), key))
    ctx.lines += ['### Failure reasons', '']
    add_instrument_count_table(
        ctx,
        [([status, reason], by_reason[(status, reason)]) for status, reason in ordered],
        headers=['status', 'reason'],
    )
    name_rows = rows(
        ctx.connection,
        f'SELECT {reason_column}, instrument, image_name FROM images{ctx.where}'
        + connector(ctx.where)
        + f"status != 'success' ORDER BY {image_order()}",
        ctx.params,
    )
    names_by_reason: dict[str, list[tuple[str, str]]] = {}
    for reason, instrument, image_name in name_rows:
        names_by_reason.setdefault(str(reason or '(none)'), []).append(
            (str(instrument), str(image_name))
        )
    add_drilldown(
        ctx,
        [
            (reason, names_by_reason.get(reason, []))
            for reason in dict.fromkeys(r for _s, r in ordered)
        ],
        label='reason',
        stub_prefix='failure_reason',
    )
    labels = [f'{status}/{reason}' for status, reason in ordered]
    write_stacked_bar_chart(
        ctx.output_dir / 'failure_reasons.png',
        labels,
        {
            instrument: [by_reason[key].get(instrument, 0) for key in ordered]
            for instrument in ctx.instruments
        },
        ctx.instruments,
        title='Failure reasons',
        xlabel='images',
    )
    ctx.lines += ['![failure reasons](failure_reasons.png)', '']


def _add_technique_usage_section(ctx: ReportContext) -> None:
    """Append per-technique image counts, spurious shares, and mean confidence."""
    # One row per technique per image is a constraint of the schema, so counting
    # rows counts images, and no count of distinct column pairs is needed.
    tech_rows = rows(
        ctx.connection,
        'SELECT t.technique_name, i.instrument, COUNT(*), '
        'SUM(CASE WHEN t.spurious THEN 0 ELSE 1 END), AVG(t.confidence) '
        'FROM techniques t '
        + IMAGE_JOIN.format(alias='t.')
        + ctx.where_i
        + ' GROUP BY t.technique_name, i.instrument '
        'ORDER BY t.technique_name, i.instrument',
        ctx.params_i,
    )
    if len(tech_rows) == 0:
        return
    images: dict[str, dict[str, int]] = {}
    detail: dict[tuple[str, str], tuple[int, int, float | None]] = {}
    for name, instrument, n_images, n_good, mean_conf in tech_rows:
        images.setdefault(str(name), {})[str(instrument)] = int(n_images)
        detail[(str(name), str(instrument))] = (int(n_images), int(n_good), mean_conf)
    techniques = sorted(images, key=lambda name: (-sum(images[name].values()), name))
    ctx.lines += ['## Technique usage', '', 'Images on which each technique ran.', '']
    add_instrument_count_table(
        ctx, [([name], images[name]) for name in techniques], headers=['technique']
    )
    ctx.lines += [
        '### Per-technique detail',
        '',
        '| technique | instrument | images | non-spurious | mean confidence |',
        '|---|---|---|---|---|',
    ]
    for name in techniques:
        for instrument in ctx.instruments:
            entry = detail.get((name, instrument))
            if entry is None:
                continue
            n_images, n_good, mean_conf = entry
            images_cell = count_pct(n_images, ctx.images_by_instrument[instrument])
            ctx.lines.append(
                f'| {name} | {instrument} | {images_cell} '
                f'| {count_pct(n_good, n_images)} | {fmt(mean_conf)} |'
            )
    ctx.lines.append('')
    write_stacked_bar_chart(
        ctx.output_dir / 'technique_usage.png',
        techniques,
        {
            instrument: [images[name].get(instrument, 0) for name in techniques]
            for instrument in ctx.instruments
        },
        ctx.instruments,
        title='Images per technique',
        xlabel='images',
    )
    ctx.lines += ['![technique usage](technique_usage.png)', '']


def _add_source_usage_section(ctx: ReportContext) -> None:
    """Append the per-model / per-source feature-usage tables."""
    # An image can hold several feature types from one source, so the inner
    # query collapses each image to one row per source before the outer one
    # counts images.  Counting distinct values of the column pair that keys an
    # image would need a spelling no backend shares.
    source_rows = rows(
        ctx.connection,
        'SELECT source_model, source_name, instrument, COUNT(*), '
        'SUM(n_features), SUM(n_gated) FROM ('
        'SELECT s.source_model, s.source_name, i.instrument, s.root_url, s.results_path_stub, '
        'SUM(s.n_features) AS n_features, SUM(s.n_gated) AS n_gated '
        'FROM feature_sources s '
        + IMAGE_JOIN.format(alias='s.')
        + ctx.where_i
        + ' GROUP BY s.source_model, s.source_name, i.instrument, s.root_url, s.results_path_stub'
        ') per_image GROUP BY source_model, source_name, instrument '
        'ORDER BY source_model, source_name, instrument',
        ctx.params_i,
    )
    if len(source_rows) == 0:
        return
    images: dict[tuple[str, str], dict[str, int]] = {}
    features: dict[tuple[str, str, str], tuple[int, int]] = {}
    for model, name, instrument, n_images, n_features, n_gated in source_rows:
        images.setdefault((str(model), str(name)), {})[str(instrument)] = int(n_images)
        features[(str(model), str(name), str(instrument))] = (int(n_features), int(n_gated))
    sources = sorted(images, key=lambda key: (key[0], -sum(images[key].values()), key[1]))
    ctx.lines += ['## Model and source usage', '', 'Images in which each source appears.', '']
    add_instrument_count_table(
        ctx,
        [([model, name], images[(model, name)]) for model, name in sources],
        headers=['model', 'source'],
    )
    ctx.lines += [
        '### Per-source feature counts',
        '',
        '| model | source | instrument | features | gated |',
        '|---|---|---|---|---|',
    ]
    for model, name in sources:
        for instrument in ctx.instruments:
            entry = features.get((model, name, instrument))
            if entry is None:
                continue
            ctx.lines.append(f'| {model} | {name} | {instrument} | {entry[0]} | {entry[1]} |')
    ctx.lines.append('')


def _add_offset_section(ctx: ReportContext) -> None:
    """Append the fused-offset statistics and a histogram per camera.

    Pointing error is a property of the camera, not of the spacecraft, so
    the distributions are grouped by ``(instrument, camera)`` and never
    pooled (a Cassini NAC pixel is a tenth of a WAC pixel, so pooling the
    two would describe neither).
    """
    offset_rows = rows(
        ctx.connection,
        f'SELECT instrument, camera, offset_dv, offset_du FROM images{ctx.where}'
        + connector(ctx.where)
        + "status = 'success' AND offset_dv IS NOT NULL AND offset_du IS NOT NULL "
        'ORDER BY instrument, camera',
        ctx.params,
    )
    by_camera: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for instrument, camera, dv, du in offset_rows:
        entry = by_camera.setdefault((str(instrument), str(camera or '(unknown)')), ([], []))
        entry[0].append(float(dv))
        entry[1].append(float(du))
    ctx.lines += [
        '## Offset statistics (successful images)',
        '',
        'Grouped by camera: pointing errors of different cameras are unrelated',
        'and are never pooled.  Percentages are of the instrument total.',
        '',
        '| instrument | camera | axis | images | mean | median | stdev | min | max |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for instrument, camera in sorted(by_camera):
        dv_values, du_values = by_camera[(instrument, camera)]
        for axis, values in (('dV', dv_values), ('dU', du_values)):
            stats = offset_stats(values)
            images = count_pct(len(values), ctx.images_by_instrument[instrument])
            if stats is None:
                ctx.lines.append(
                    f'| {instrument} | {camera} | {axis} | {images} | - | - | - | - | - |'
                )
            else:
                ctx.lines.append(
                    f'| {instrument} | {camera} | {axis} | {images} | {fmt(stats["mean"])} '
                    f'| {fmt(stats["median"])} | {fmt(stats["stdev"])} '
                    f'| {fmt(stats["min"])} | {fmt(stats["max"])} |'
                )
    ctx.lines.append('')
    for instrument, camera in sorted(by_camera):
        dv_values, du_values = by_camera[(instrument, camera)]
        chart = f'offsets_hist_{safe_filename(f"{instrument}_{camera}")}.png'
        write_offset_hist(
            ctx.output_dir / chart,
            dv_values,
            du_values,
            title=f'Fused offset distribution, {instrument} {camera} (successful images)',
        )
        ctx.lines += [f'![offsets {instrument} {camera}]({chart})', '']


def _add_agreement_sections(ctx: ReportContext) -> None:
    """Append the cross-technique agreement and confidence-calibration tables."""
    per_pair, per_image_rank = _pairwise_disagreements(ctx)
    ctx.lines += [
        '## Cross-technique agreement',
        '',
        'Euclidean distance between per-technique offsets on images where both',
        'techniques produced non-spurious results.',
        '',
        '| instrument | technique pair | images | median (px) | p95 (px) |',
        '|---|---|---|---|---|',
    ]
    for instrument, technique_a, technique_b in sorted(per_pair):
        deltas = per_pair[(instrument, technique_a, technique_b)]
        ctx.lines.append(
            f'| {instrument} | {technique_a} vs {technique_b} '
            f'| {count_pct(len(deltas), ctx.images_by_instrument[instrument])} | '
            f'{fmt(statistics.median(deltas))} | {fmt(percentile(deltas, 0.95))} |'
        )
    ctx.lines.append('')

    rank_rows = rows(
        ctx.connection,
        f'SELECT confidence_rank, instrument, COUNT(*) FROM images{ctx.where}'
        + connector(ctx.where)
        + 'confidence_rank IS NOT NULL GROUP BY confidence_rank, instrument '
        'ORDER BY confidence_rank, instrument',
        ctx.params,
    )
    by_tier: dict[str, dict[str, int]] = {tier: {} for tier in _CONFIDENCE_TIERS}
    for rank, instrument, count in rank_rows:
        by_tier.setdefault(str(rank), {})[str(instrument)] = int(count)
    # The standard tiers always appear, in tier order, so an empty tier reads
    # as a real zero rather than a missing row.  Any unrecognized rank from
    # the database is listed after them rather than dropped.
    tiers = [*_CONFIDENCE_TIERS, *sorted(set(by_tier) - set(_CONFIDENCE_TIERS))]
    ctx.lines += [
        '## Confidence calibration (agreement as accuracy proxy)',
        '',
        'For each confidence tier: how well the techniques that fed the fused',
        'offset agreed with one another.  Without ground truth, cross-technique',
        'agreement is the standing production check that confidence tiers are',
        'meaningful (the calibrated anchor is the simulated-scene campaign).',
        '',
    ]
    add_instrument_count_table(ctx, [([tier], by_tier[tier]) for tier in tiers], headers=['tier'])
    ctx.lines += [
        '| tier | instrument | images | with >=2 techniques | median max-disagreement (px) '
        '| p95 (px) |',
        '|---|---|---|---|---|---|',
    ]
    for tier in tiers:
        for instrument in ctx.instruments:
            count = by_tier[tier].get(instrument, 0)
            disagreements = per_image_rank.get((instrument, tier), [])
            ctx.lines.append(
                f'| {tier} | {instrument} '
                f'| {count_pct(count, ctx.images_by_instrument[instrument])} '
                f'| {count_pct(len(disagreements), count)} | '
                f'{fmt(statistics.median(disagreements)) if disagreements else "-"} | '
                f'{fmt(percentile(disagreements, 0.95)) if disagreements else "-"} |'
            )
    ctx.lines.append('')
    if len(per_image_rank) > 0:
        write_stacked_bar_chart(
            ctx.output_dir / 'agreement_by_tier.png',
            tiers,
            {
                instrument: [len(per_image_rank.get((instrument, tier), [])) for tier in tiers]
                for instrument in ctx.instruments
            },
            ctx.instruments,
            title='Images with cross-technique agreement data, by tier',
            xlabel='images',
        )
        ctx.lines += ['![agreement by tier](agreement_by_tier.png)', '']


def _excluded_techniques(excluded: Any) -> list[str]:
    """The technique names one image's exclusion column holds.

    A textual statement carries no column types, so a JSON column arrives as
    whatever the driver makes of it: decoded by a backend with a native JSON
    type, and as the stored text by one that keeps JSON in a text column.

    Parameters:
        excluded: The value as the driver returned it.

    Returns:
        The names, or an empty list when the column is empty or unreadable.
    """
    value = excluded
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [str(name) for name in value]


def _add_exclusions_section(ctx: ReportContext) -> None:
    """Append the ensemble outlier-exclusion breakdown.

    The exclusion set is a JSON column, and the empty set is filtered out here
    rather than in SQL: comparing a JSON value against a literal is spelled
    differently on every backend, and the rows are already being read.
    """
    excluded_rows = rows(
        ctx.connection,
        f'SELECT excluded_from_consensus, instrument, image_name FROM images{ctx.where}'
        + connector(ctx.where)
        + f'excluded_from_consensus IS NOT NULL ORDER BY {image_order()}',
        ctx.params,
    )
    labelled = [
        (', '.join(names), str(instrument), str(image_name))
        for excluded, instrument, image_name in excluded_rows
        for names in [_excluded_techniques(excluded)]
        if len(names) > 0
    ]
    if len(labelled) == 0:
        return
    by_exclusion: dict[str, dict[str, int]] = {}
    names_by_exclusion: dict[str, list[tuple[str, str]]] = {}
    for label, instrument, image_name in labelled:
        counts = by_exclusion.setdefault(label, {})
        counts[instrument] = counts.get(instrument, 0) + 1
        names_by_exclusion.setdefault(label, []).append((instrument, image_name))
    ordered = sorted(by_exclusion, key=lambda key: (-sum(by_exclusion[key].values()), key))
    ctx.lines += ['## Ensemble outlier exclusions', '']
    add_instrument_count_table(
        ctx,
        [([label], by_exclusion[label]) for label in ordered],
        headers=['excluded techniques'],
    )
    add_drilldown(
        ctx,
        [(label, names_by_exclusion.get(label, [])) for label in ordered],
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
    connection: sqlalchemy.Connection,
    output_dir: str | Path | FCPath,
    *,
    instrument: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_image: str | None = None,
    max_image: str | None = None,
    roots: list[str] | None = None,
    top_n: int = 0,
    filelists: bool = False,
    suspect_fraction: float = 0.9,
    csv_export: bool = False,
) -> FCPath:
    """Query the results index and write ``report.md`` plus charts.

    The report is deterministic: the same index and options always produce
    byte-identical Markdown.  All filters combine and apply to every section.

    Parameters:
        connection: Open connection to the results index.
        output_dir: Directory receiving ``report.md``, the PNG charts, and
            (with ``filelists`` / ``csv_export``) the ``filelists/``
            subdirectory and ``images.csv`` (created if missing).  A local
            directory or any URL the ``filecache`` layer accepts.
        instrument: Optional instrument filter (``coiss`` / ``vgiss`` /
            ``gossi`` / ``nhlorri``).
        start_date: Optional inclusive UTC start date (``YYYY-MM-DD``).
        end_date: Optional inclusive UTC end date (``YYYY-MM-DD``).
        min_image: Optional inclusive lower bound on the numeric portion
            of the image name; an image name (``N1454725799``) or a bare
            number.
        max_image: Optional inclusive upper bound on the numeric portion
            of the image name.
        roots: Optional normalized results-root URLs to restrict to; None
            reports over every root the index holds, which a report may
            legitimately do where a per-image lookup never does.
        top_n: When positive, categorical sections list up to this many
            example image names per category, the suspect-offset and
            worst-BOTSIM-pair tables are capped at this many rows, and the
            slowest images are listed.
        filelists: When True, write one plain-text file per category and
            instrument (one image name per line, full list) under
            ``filelists/``, in the format ``--image-filelist`` reads.
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
    output_path = FCPath(output_dir)
    min_image_num = _image_bound(min_image, option='min_image')
    max_image_num = _image_bound(max_image, option='max_image')
    where, params = where_clause(
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        min_image_num=min_image_num,
        max_image_num=max_image_num,
        roots=roots,
    )
    # Joined queries alias the images table as ``i``; same filter, qualified.
    where_i, params_i = where_clause(
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        min_image_num=min_image_num,
        max_image_num=max_image_num,
        roots=roots,
        alias='i.',
    )
    ctx = ReportContext(
        connection=connection,
        output_dir=output_path,
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
        f'root in {", ".join(roots)}' if roots else None,
    ]
    active = [f for f in filters if f is not None]
    ctx.lines.append(f'Filters: {", ".join(active) if len(active) > 0 else "none (full database)"}')
    ctx.lines.append('')

    _add_selection_section(ctx)
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

    report_path = output_path / 'report.md'
    report_path.write_text('\n'.join(ctx.lines) + '\n', encoding='utf-8')
    return report_path


def _to_stderr(message: str) -> None:
    """Print one diagnostic where this program's other diagnostics go.

    The resolver that names an index reports a value naming none through
    whatever sink its caller supplies, and this program's sink is the stream its
    refusals already print to, so the two arrive in the order they happened.

    Parameters:
        message: The line to print.
    """
    print(message, file=sys.stderr)


def main_report(cmdline: list[str] | None = None) -> int:
    """Entry point for ``sd_stats_report``.

    This program keeps ``print()`` rather than a logger.  Its output *is*
    terminal text for a person reading a report, and a logger would wrap that
    in run-log machinery it has no use for.

    Parameters:
        cmdline: Argument list; None uses ``sys.argv``.

    Returns:
        Process exit code: 0 on success, and 1 when no index was named or the
        one that was cannot be read.

    Raises:
        SystemExit: With status 2, from the argument parser, for a command line
            it will not accept -- an unknown flag, an unparseable bound, a root
            that is not a location that can be read, or a root the index holds no
            completed ingest of, the last of which is a value the index rather
            than the parser rejects but is reported the same way.
    """
    parser = argparse.ArgumentParser(
        description='Generate a navigation statistics report from an ingested results index.'
    )
    parser.add_argument(
        '--results-db',
        default=None,
        metavar='URL',
        help='Connection URL of the results index written by sd_stats_ingest '
        '(a sqlite: URL naming a local path, or a postgresql+psycopg: URL); '
        'overrides the environment.results_db configuration variable and '
        'NAV_RESULTS_DB',
    )
    parser.add_argument(
        '--root',
        action='append',
        default=None,
        metavar='ROOT',
        help='Restrict the report to one ingested navigation-results root; may '
        'be given more than once. Default: every root the index holds',
    )
    parser.add_argument(
        '--output-dir',
        default='nav_stats_report',
        help='Directory (local path or filecache URL) receiving report.md and charts '
        '(default: %(default)s)',
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
        help='Write one plain-text file per category and instrument (one '
        'image name per line, full list, readable by --image-filelist) into '
        'the filelists/ subdirectory of the output dir',
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

    url = get_results_db_url(arguments, DEFAULT_CONFIG, warn=_to_stderr)
    if url is None:
        print(
            'sd_stats_report reads a results index and has no file-reading mode. '
            'Name one with --results-db, the environment.results_db configuration '
            'variable, or NAV_RESULTS_DB, and build it with sd_stats_ingest.',
            file=sys.stderr,
        )
        return 1
    try:
        roots = [normalize_root_url(root) for root in arguments.root or []]
    except ValueError as exc:
        parser.error(f'a --root is not a location that can be read: {exc}')
    try:
        engine = open_index(url)
    except ValueError as exc:
        print(f'Cannot read the results index: {exc}', file=sys.stderr)
        return 1
    try:
        with engine.connect() as connection:
            require_ingested_roots(connection, roots, url=url)
            report_path = build_report(
                connection,
                arguments.output_dir,
                instrument=arguments.instrument,
                start_date=arguments.start_date,
                end_date=arguments.end_date,
                min_image=arguments.min_image,
                max_image=arguments.max_image,
                roots=roots,
                top_n=arguments.top_n,
                filelists=arguments.filelists,
                suspect_fraction=arguments.suspect_fraction,
                csv_export=arguments.csv,
            )
    except ValueError as exc:
        parser.error(str(exc))
    finally:
        engine.dispose()
    print(f'Wrote {report_path}')
    return 0
