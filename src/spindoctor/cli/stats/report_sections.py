"""Report sections: failure taxonomy, suspect offsets, BOTSIM, run time, CSV export."""

import csv
import json
import math
import re
import statistics
from io import StringIO
from typing import Any

from filecache import FCPath

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
    percentile,
    rows,
    write_stacked_value_hist,
)
from spindoctor.config import DEFAULT_CONFIG, Config
from spindoctor.results_index import IMAGES

__all__ = [
    'CSV_LINE_TERMINATOR',
    'IMAGE_COLUMNS',
    'add_botsim_section',
    'add_failure_taxonomy_section',
    'add_offset_by_group_section',
    'add_runtime_section',
    'add_suspect_offset_section',
    'resolve_offset_limit',
    'write_csv_export',
]


# ---------------------------------------------------------------------------
# Suspect-offset screen
# ---------------------------------------------------------------------------

# Config section per instrument holding ``extfov_margin_vu`` (the per-axis
# maximum expected pointing offset the search allows for).  Cassini ISS is
# handled separately because its sections are nested per detector and split
# between raw and CALIB blocks.
_INSTRUMENT_CONFIG_SECTIONS = {
    'gossi': 'galileo_ssi',
    'nhlorri': 'newhorizons_lorri',
    'sim': 'sim',
    'vgiss': 'voyager_iss',
}


def resolve_offset_limit(
    instrument: str,
    image_name: str,
    image_shape_v: int | None,
    *,
    config: Config | None = None,
) -> tuple[float, float] | str:
    """Resolve the per-axis maximum expected pointing offset for one image.

    The limit is the configured ``extfov_margin_vu`` search margin for the
    instrument (for Cassini ISS: per detector, chosen from the image-name
    prefix ``N``/``W`` and the raw-vs-CALIB config block; for margin tables
    keyed by image size, the recorded ``image_shape_v`` selects the entry).

    Parameters:
        instrument: Registered instrument name from the database.
        image_name: Image name (used to pick the Cassini ISS detector and
            config block).
        image_shape_v: Recorded V-axis image size, or None when the
            database row has no shape.
        config: Configuration to read; None uses ``DEFAULT_CONFIG``.

    Returns:
        ``(limit_v, limit_u)`` in pixels, or a human-readable string
        explaining why the limit could not be resolved.
    """
    config = config or DEFAULT_CONFIG
    if instrument == 'coiss':
        section_name = 'cassini_iss_calib' if '_CALIB' in image_name.upper() else 'cassini_iss'
        detector = {'N': 'nac', 'W': 'wac'}.get(image_name.rsplit('/', 1)[-1][:1].upper())
        if detector is None:
            return 'cannot determine NAC/WAC from the image name'
        section = config.category(section_name)
        detector_config = section.get(detector) if section else None
        entry = detector_config.get('extfov_margin_vu') if detector_config else None
        source = f'{section_name}.{detector}'
    else:
        if instrument not in _INSTRUMENT_CONFIG_SECTIONS:
            return f'no configured search limit for instrument {instrument!r}'
        section_name = _INSTRUMENT_CONFIG_SECTIONS[instrument]
        section = config.category(section_name)
        entry = section.get('extfov_margin_vu') if section else None
        source = section_name
    if entry is None:
        return f'config section {source!r} has no extfov_margin_vu'
    if isinstance(entry, dict):
        if image_shape_v is None:
            return 'image shape not recorded in the database'
        if image_shape_v not in entry:
            return f'{source!r} has no extfov_margin_vu entry for image size {image_shape_v}'
        entry = entry[image_shape_v]
    limit_v = float(entry[0])
    limit_u = float(entry[1])
    if limit_v <= 0.0 or limit_u <= 0.0:
        return f'{source!r} extfov_margin_vu is not positive'
    return limit_v, limit_u


def add_suspect_offset_section(ctx: ReportContext) -> None:
    """Flag successful images whose fused offset sits near the search limit.

    An offset at the edge of the search box is as likely to be a
    correlation artifact as a real pointing error, so any image with
    ``|dV|`` or ``|dU|`` at or beyond ``suspect_fraction`` times the
    per-axis limit is listed for operator review.
    """
    image_rows = rows(
        ctx.connection,
        f'SELECT image_name, instrument, offset_dv, offset_du, image_shape_v '
        f'FROM images{ctx.where}'
        + connector(ctx.where)
        + "status = 'success' AND offset_dv IS NOT NULL AND offset_du IS NOT NULL "
        f'ORDER BY {image_order()}',
        ctx.params,
    )
    ctx.lines += [
        '## Suspect offsets (near the search limit)',
        '',
        f'Successful images whose fused offset reaches at least '
        f'{ctx.suspect_fraction:.2f} of the per-axis maximum expected pointing '
        'offset (the configured extfov search margin) on either axis.  These '
        'offsets may be correlation artifacts pinned to the search boundary.',
        '',
    ]
    suspects: list[tuple[float, str, str, float, float, float, str]] = []
    unresolved: dict[str, int] = {}
    screened: dict[str, int] = {}
    suspect_counts: dict[str, int] = {}
    for image_name, instrument, dv, du, shape_v in image_rows:
        limit = resolve_offset_limit(str(instrument), str(image_name), shape_v)
        if isinstance(limit, str):
            reason = f'{instrument}: {limit}'
            unresolved[reason] = unresolved.get(reason, 0) + 1
            continue
        screened[str(instrument)] = screened.get(str(instrument), 0) + 1
        limit_v, limit_u = limit
        ratio = max(abs(float(dv)) / limit_v, abs(float(du)) / limit_u)
        if ratio >= ctx.suspect_fraction:
            suspect_counts[str(instrument)] = suspect_counts.get(str(instrument), 0) + 1
            suspects.append(
                (
                    ratio,
                    str(image_name),
                    str(instrument),
                    float(dv),
                    float(du),
                    math.hypot(float(dv), float(du)),
                    f'({fmt(limit_v, 1)}, {fmt(limit_u, 1)})',
                )
            )
    suspects.sort(key=lambda s: (-s[0], s[1]))
    n_screened = sum(screened.values())
    ctx.lines += [
        f'Suspect images: {count_pct(len(suspects), ctx.total_images)} of {n_screened} screened.',
        '',
    ]
    add_instrument_count_table(ctx, [(['suspect'], suspect_counts)], headers=['category'])
    if len(suspects) > 0:
        shown = suspects[: ctx.top_n] if ctx.top_n > 0 else suspects
        ctx.lines += [
            '| image | instrument | dV | dU | magnitude | limit (v, u) |',
            '|---|---|---|---|---|---|',
        ]
        for _ratio, filename, instrument, dv, du, magnitude, limit_text in shown:
            name = image_name_from_filename(instrument, filename)
            ctx.lines.append(
                f'| {name} | {instrument} | {fmt(dv)} | {fmt(du)} | '
                f'{fmt(magnitude)} | {limit_text} |'
            )
        ctx.lines.append('')
        add_drilldown(
            ctx,
            [('suspect', [(s[2], s[1]) for s in suspects])],
            label='category',
            stub_prefix='suspect_offsets',
        )
    if len(unresolved) > 0:
        ctx.lines.append('Search limit could not be resolved for some images:')
        ctx.lines.append('')
        for reason in sorted(unresolved):
            ctx.lines.append(f'- {reason} ({unresolved[reason]} image(s))')
        ctx.lines.append('')


# ---------------------------------------------------------------------------
# BOTSIM pair consistency (Cassini ISS)
# ---------------------------------------------------------------------------

_BOTSIM_NAME_RE = re.compile(r'^([NW])(\d{10})')


def add_botsim_section(ctx: ReportContext) -> None:
    """Compare NAC and WAC offsets over simultaneously shuttered Cassini pairs.

    BOTSIM observations shutter both cameras at once, so the two frames
    share one spacecraft-clock count and see the same pointing.  One WAC
    pixel is ten NAC pixels; a consistent pair therefore satisfies
    ``NAC offset ~= 10 x WAC offset`` per axis, making the per-axis
    residual ``NAC - 10 x WAC`` an end-to-end accuracy check that needs
    no ground truth.
    """
    image_rows = rows(
        ctx.connection,
        f'SELECT image_name, status, offset_dv, offset_du FROM images{ctx.where}'
        + connector(ctx.where)
        + f"instrument = 'coiss' ORDER BY {image_order()}",
        ctx.params,
    )
    by_clock: dict[str, dict[str, tuple[str, str, Any, Any]]] = {}
    for image_name, status, dv, du in image_rows:
        match = _BOTSIM_NAME_RE.match(str(image_name).rsplit('/', 1)[-1].upper())
        if match is None:
            continue
        camera, clock = match.group(1), match.group(2)
        by_clock.setdefault(clock, {}).setdefault(camera, (str(image_name), str(status), dv, du))
    pairs = {clock: entry for clock, entry in by_clock.items() if len(entry) == 2}
    residuals: list[tuple[float, str, str, str, float, float]] = []
    for clock in sorted(pairs):
        nac = pairs[clock]['N']
        wac = pairs[clock]['W']
        if nac[1] != 'success' or wac[1] != 'success':
            continue
        if None in (nac[2], nac[3], wac[2], wac[3]):
            continue
        residual_dv = float(nac[2]) - 10.0 * float(wac[2])
        residual_du = float(nac[3]) - 10.0 * float(wac[3])
        residuals.append(
            (math.hypot(residual_dv, residual_du), clock, nac[0], wac[0], residual_dv, residual_du)
        )
    ctx.lines += [
        '## BOTSIM pair consistency (Cassini ISS)',
        '',
        'BOTSIM observations shutter the NAC and WAC simultaneously (the image',
        'names share one spacecraft-clock count).  One WAC pixel is ten NAC',
        'pixels, so a consistent pair satisfies NAC offset ~= 10 x WAC offset',
        'per axis.  Residuals below are NAC - 10 x WAC, in NAC pixels.',
        '',
        '| metric | value |',
        '|---|---|',
        f'| pairs identified | {len(pairs)} |',
        f'| pairs with both navigated | {len(residuals)} |',
    ]
    if len(residuals) > 0:
        magnitudes = [r[0] for r in residuals]
        ctx.lines += [
            f'| median residual (px) | {fmt(statistics.median(magnitudes))} |',
            f'| p95 residual (px) | {fmt(percentile(magnitudes, 0.95))} |',
        ]
    ctx.lines.append('')
    if len(residuals) > 0 and ctx.top_n > 0:
        residuals.sort(key=lambda r: (-r[0], r[1]))
        ctx.lines += [
            f'Worst {min(ctx.top_n, len(residuals))} pair(s):',
            '',
            '| clock | NAC image | WAC image | residual dV | residual dU | residual |',
            '|---|---|---|---|---|---|',
        ]
        for magnitude, clock, nac_name, wac_name, residual_dv, residual_du in residuals[
            : ctx.top_n
        ]:
            nac_image = image_name_from_filename('coiss', nac_name)
            wac_image = image_name_from_filename('coiss', wac_name)
            ctx.lines.append(
                f'| {clock} | {nac_image} | {wac_image} | {fmt(residual_dv)} | '
                f'{fmt(residual_du)} | {fmt(magnitude)} |'
            )
        ctx.lines.append('')


# ---------------------------------------------------------------------------
# Failure taxonomy by image content
# ---------------------------------------------------------------------------

_CONTENT_CATEGORIES = (
    'stars-only',
    'single-body',
    'multi-body',
    'rings-only',
    'body+rings',
    'no-features',
)


def _source_kind(source_model: str) -> str:
    """Coarse feature-source kind (``stars`` / ``rings`` / ``body``) of a model name."""
    kind = source_model.split(':', 1)[0].lower()
    if kind in ('star', 'stars'):
        return 'stars'
    if kind.startswith('ring'):
        return 'rings'
    return 'body'


def _content_category(entries: list[tuple[str, str]]) -> str:
    """Classify an image's ``(source_model, source_name)`` inventory."""
    if len(entries) == 0:
        return 'no-features'
    body_names = {name for model, name in entries if _source_kind(model) == 'body'}
    has_rings = any(_source_kind(model) == 'rings' for model, _ in entries)
    if len(body_names) > 0 and has_rings:
        return 'body+rings'
    if has_rings:
        return 'rings-only'
    if len(body_names) == 1:
        return 'single-body'
    if len(body_names) > 1:
        return 'multi-body'
    return 'stars-only'


def add_failure_taxonomy_section(ctx: ReportContext) -> None:
    """Classify failed images by scene content and localize per-body problems.

    Content comes from the ``feature_sources`` inventory recorded per
    image; a body whose failure share is far above its peers points at a
    modeling problem for that body rather than a pipeline-wide issue.
    """
    failed_rows = rows(
        ctx.connection,
        'SELECT root_url, results_path_stub, image_name, instrument, '
        f'COALESCE(status_reason, status_error) FROM images{ctx.where}'
        + connector(ctx.where)
        + f"status != 'success' ORDER BY {image_order()}",
        ctx.params,
    )
    if len(failed_rows) == 0:
        return
    source_rows = rows(
        ctx.connection,
        'SELECT s.root_url, s.results_path_stub, i.image_name, i.instrument, i.status, '
        's.source_model, s.source_name '
        'FROM feature_sources s '
        + IMAGE_JOIN.format(alias='s.')
        + ctx.where_i
        + f' ORDER BY {image_order("i.")}, s.source_model, s.source_name',
        ctx.params_i,
    )
    sources_by_image: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for root_url, stub, _image_name, _instrument, _status, source_model, source_name in source_rows:
        sources_by_image.setdefault((str(root_url), str(stub)), []).append(
            (str(source_model), str(source_name))
        )

    by_category: dict[str, list[tuple[str, str]]] = {
        category: [] for category in _CONTENT_CATEGORIES
    }
    category_counts: dict[str, dict[str, int]] = {category: {} for category in _CONTENT_CATEGORIES}
    reason_counts: dict[tuple[str, str], dict[str, int]] = {}
    for root_url, stub, image_name, instrument, status_reason in failed_rows:
        category = _content_category(sources_by_image.get((str(root_url), str(stub)), []))
        by_category[category].append((str(instrument), str(image_name)))
        counts = category_counts[category]
        counts[str(instrument)] = counts.get(str(instrument), 0) + 1
        key = (category, str(status_reason or '(none)'))
        reason_bucket = reason_counts.setdefault(key, {})
        reason_bucket[str(instrument)] = reason_bucket.get(str(instrument), 0) + 1

    populated = [c for c in _CONTENT_CATEGORIES if len(by_category[c]) > 0]
    ctx.lines += [
        '## Failure taxonomy by image content',
        '',
        'Failed images classified by what the feature inventory says was in',
        'the scene.',
        '',
    ]
    add_instrument_count_table(
        ctx,
        [([category], category_counts[category]) for category in populated],
        headers=['content'],
    )
    ordered_reasons = sorted(
        reason_counts,
        key=lambda key: (
            _CONTENT_CATEGORIES.index(key[0]),
            -sum(reason_counts[key].values()),
            key[1],
        ),
    )
    add_instrument_count_table(
        ctx,
        [
            ([category, reason], reason_counts[(category, reason)])
            for category, reason in ordered_reasons
        ],
        headers=['content', 'reason'],
    )
    add_drilldown(
        ctx,
        [(category, by_category[category]) for category in _CONTENT_CATEGORIES],
        label='content category',
        stub_prefix='failed_content',
    )

    # Per-body failure shares over all images (successful and failed).  An
    # image can contribute several rows for one body, so each bucket holds the
    # key that identifies an image rather than its name, which two volumes may
    # share; the name rides along for the drill-down list.
    body_status: dict[tuple[str, str], dict[str, set[tuple[str, str, str]]]] = {}
    for root_url, stub, image_name, instrument, status, source_model, source_name in source_rows:
        if _source_kind(str(source_model)) != 'body' or str(source_name) == '(none)':
            continue
        buckets = body_status.setdefault(
            (str(source_name), str(instrument)), {'failed': set(), 'success': set()}
        )
        bucket = 'success' if str(status) == 'success' else 'failed'
        buckets[bucket].add((str(image_name), str(root_url), str(stub)))
    if len(body_status) > 0:
        ranked = sorted(
            body_status.items(),
            key=lambda item: (
                -len(item[1]['failed']) / (len(item[1]['failed']) + len(item[1]['success'])),
                -len(item[1]['failed']),
                item[0],
            ),
        )
        ctx.lines += [
            '### Per-body failure shares',
            '',
            'How often each named body appears in failed versus successful',
            'images; a body with a high failure share is a modeling problem.',
            '',
            '| body | instrument | failed images | successful images | failure share |',
            '|---|---|---|---|---|',
        ]
        for (body, instrument), buckets in ranked:
            n_failed = len(buckets['failed'])
            n_success = len(buckets['success'])
            share = n_failed / (n_failed + n_success)
            total = ctx.images_by_instrument[instrument]
            ctx.lines.append(
                f'| {body} | {instrument} | {count_pct(n_failed, total)} '
                f'| {count_pct(n_success, total)} | {share:.3f} |'
            )
        ctx.lines.append('')
        by_body: dict[str, list[tuple[str, str]]] = {}
        for (body, instrument), buckets in ranked:
            by_body.setdefault(body, []).extend(
                (instrument, entry[0]) for entry in sorted(buckets['failed'])
            )
        add_drilldown(
            ctx,
            [(body, entries) for body, entries in by_body.items() if len(entries) > 0],
            label='body',
            stub_prefix='failed_body',
        )


# ---------------------------------------------------------------------------
# Run-time statistics
# ---------------------------------------------------------------------------


def add_runtime_section(ctx: ReportContext) -> None:
    """Summarize per-image run times; skipped when no timing data exists."""
    timing_rows = rows(
        ctx.connection,
        f'SELECT image_name, instrument, elapsed_s FROM images{ctx.where}'
        + connector(ctx.where)
        + f'elapsed_s IS NOT NULL ORDER BY {image_order()}',
        ctx.params,
    )
    if len(timing_rows) == 0:
        return
    elapsed = [float(r[2]) for r in timing_rows]
    by_instrument: dict[str, list[float]] = {}
    for _image_name, instrument, seconds in timing_rows:
        by_instrument.setdefault(str(instrument), []).append(float(seconds))
    ctx.lines += [
        '## Run-time statistics',
        '',
        '| instrument | images | total (s) | min (s) | max (s) | mean (s) | median (s) '
        '| stdev (s) |',
        '|---|---|---|---|---|---|---|---|',
    ]
    series: list[tuple[str, list[float], int]] = [
        (instrument, by_instrument.get(instrument, []), ctx.images_by_instrument[instrument])
        for instrument in ctx.instruments
    ]
    # The pooled row only says something new once more than one instrument
    # contributed to it.
    if len(ctx.instruments) > 1:
        series.append(('(all)', elapsed, ctx.total_images))
    for instrument, values, denominator in series:
        if len(values) == 0:
            continue
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        ctx.lines.append(
            f'| {instrument} | {count_pct(len(values), denominator)} | {fmt(sum(values))} '
            f'| {fmt(min(values))} | {fmt(max(values))} | {fmt(statistics.fmean(values))} '
            f'| {fmt(statistics.median(values))} | {fmt(stdev)} |'
        )
    ctx.lines.append('')
    write_stacked_value_hist(
        ctx.output_dir / 'runtime_hist.png',
        by_instrument,
        ctx.instruments,
        title='Per-image run time',
        xlabel='elapsed (s)',
    )
    ctx.lines += ['![run time](runtime_hist.png)', '']
    if ctx.top_n > 0:
        slowest = sorted(timing_rows, key=lambda r: (-float(r[2]), str(r[0])))[: ctx.top_n]
        ctx.lines += [
            f'Slowest {len(slowest)} image(s):',
            '',
            '| image | instrument | elapsed (s) |',
            '|---|---|---|',
        ]
        for image_name, instrument, seconds in slowest:
            name = image_name_from_filename(str(instrument), str(image_name))
            ctx.lines.append(f'| {name} | {instrument} | {fmt(float(seconds))} |')
        ctx.lines.append('')


# ---------------------------------------------------------------------------
# Per-instrument / per-image-size offset breakdown
# ---------------------------------------------------------------------------


def add_offset_by_group_section(ctx: ReportContext) -> None:
    """Break the fused-offset statistics down by (instrument, camera, image size)."""
    image_rows = rows(
        ctx.connection,
        f'SELECT instrument, camera, image_shape_v, image_shape_u, offset_dv, offset_du '
        f'FROM images{ctx.where}'
        + connector(ctx.where)
        + "status = 'success' AND offset_dv IS NOT NULL AND offset_du IS NOT NULL "
        'ORDER BY instrument, camera, image_shape_v, image_shape_u',
        ctx.params,
    )
    groups: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for instrument, camera, shape_v, shape_u, dv, du in image_rows:
        size = f'{shape_v}x{shape_u}' if shape_v is not None and shape_u is not None else '(none)'
        key = (str(instrument), str(camera or '(unknown)'), size)
        groups.setdefault(key, []).append((float(dv), float(du)))
    if len(groups) == 0:
        return
    ctx.lines += [
        '### By instrument, camera, and image size',
        '',
        '| instrument | camera | size (v x u) | images '
        '| dV mean | dV stdev | dV min | dV max '
        '| dU mean | dU stdev | dU min | dU max |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for instrument, camera, size in sorted(groups):
        offsets = groups[(instrument, camera, size)]
        dv = [o[0] for o in offsets]
        du = [o[1] for o in offsets]
        stdev_dv = statistics.stdev(dv) if len(dv) > 1 else 0.0
        stdev_du = statistics.stdev(du) if len(du) > 1 else 0.0
        images = count_pct(len(offsets), ctx.images_by_instrument[instrument])
        ctx.lines.append(
            f'| {instrument} | {camera} | {size} | {images} '
            f'| {fmt(statistics.fmean(dv))} | {fmt(stdev_dv)} | {fmt(min(dv))} | {fmt(max(dv))} '
            f'| {fmt(statistics.fmean(du))} | {fmt(stdev_du)} | {fmt(min(du))} | {fmt(max(du))} |'
        )
    ctx.lines.append('')


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


IMAGE_COLUMNS: tuple[str, ...] = tuple(IMAGES.columns.keys())
"""Every ``images`` column, in schema order, as the CSV export lists them."""

CSV_LINE_TERMINATOR = '\n'
"""What ends a row of the CSV export, on every platform."""

_CHILD_KEY = 'WHERE {alias}root_url = i.root_url AND {alias}results_path_stub = i.results_path_stub'
"""How a correlated subquery finds one image's child rows."""


def _csv_value(value: Any) -> Any:
    """Render one column value for the CSV.

    A JSON column arrives as the Python value the driver decoded, and a CSV
    carrying a Python container's repr is a CSV nothing else can read back.  It
    goes out as JSON text, which is what the column holds.

    Parameters:
        value: The value as the driver returned it.

    Returns:
        The value to write.
    """
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def write_csv_export(ctx: ReportContext) -> FCPath:
    """Write a flattened one-row-per-image CSV next to ``report.md``.

    Columns are the ``images`` table columns (schema order) plus
    ``n_technique_rows``, ``n_feature_sources``, ``n_features``, and
    ``n_gated`` aggregates, ordered by image name.

    Returns:
        The path of the written ``images.csv``.
    """
    columns = ', '.join(f'i.{column}' for column in IMAGE_COLUMNS)
    technique_key = _CHILD_KEY.format(alias='t.')
    source_key = _CHILD_KEY.format(alias='s.')
    csv_rows = rows(
        ctx.connection,
        f'SELECT {columns}, '
        f'(SELECT COUNT(*) FROM techniques t {technique_key}), '
        f'(SELECT COUNT(*) FROM feature_sources s {source_key}), '
        f'(SELECT COALESCE(SUM(s.n_features), 0) FROM feature_sources s {source_key}), '
        f'(SELECT COALESCE(SUM(s.n_gated), 0) FROM feature_sources s {source_key}) '
        'FROM images i' + ctx.where_i + f' ORDER BY {image_order("i.")}',
        ctx.params_i,
    )
    csv_path = ctx.output_dir / 'images.csv'
    buffer = StringIO()
    # Stated rather than left to the module default, which is CRLF: this file is
    # read back by the regression comparison and by whatever an operator points
    # at it, and a line ending that changes with a library default is a diff
    # nobody asked for.
    writer = csv.writer(buffer, lineterminator=CSV_LINE_TERMINATOR)
    writer.writerow(
        [*IMAGE_COLUMNS, 'n_technique_rows', 'n_feature_sources', 'n_features', 'n_gated']
    )
    writer.writerows([_csv_value(value) for value in row] for row in csv_rows)
    csv_path.write_text(buffer.getvalue(), encoding='utf-8')
    ctx.lines += [
        '## CSV export',
        '',
        f'One row per image: images.csv ({len(csv_rows)} row(s)).',
        '',
    ]
    return csv_path
