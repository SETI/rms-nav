"""Shared context, query, formatting, and chart helpers for the statistics report.

Every query the report issues goes through :func:`rows`, which runs textual SQL
with named bind parameters against a SQLAlchemy Core connection.  Named binds
rather than positional ones because the same filter fragment is spliced into
statements that already carry binds of their own, and a filter that had to know
how many came before it would be a filter that broke whenever a section grew a
condition.

Nothing here is dialect-specific.  The filters compare columns -- including
``image_number``, which is ingested rather than computed by a function
registered on one connection -- so the same statement runs on SQLite and on
PostgreSQL and means the same thing on both.
"""

import math
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import sqlalchemy
from filecache import FCPath

__all__ = [
    'IMAGE_JOIN',
    'ReportContext',
    'add_drilldown',
    'add_instrument_count_table',
    'connector',
    'count_pct',
    'fmt',
    'image_name_from_filename',
    'image_order',
    'instrument_color',
    'offset_stats',
    'percentile',
    'rows',
    'safe_filename',
    'where_clause',
    'write_offset_hist',
    'write_stacked_bar_chart',
    'write_stacked_value_hist',
]


# The database records ``observation.image_name``, which is the source file's
# basename (``N1454725799_1_CALIB.IMG``).  The dataset layer's notion of an
# image name is the shorter token its ``--image-filelist`` selection matches
# against (``N1454725799``), so every name the report prints or writes to a
# filelist goes through the per-instrument rule below.  An unregistered
# instrument falls back to the extension-stripped basename.
_IMAGE_NAME_RULES: dict[str, Callable[[str], str]] = {
    # [NW]dddddddddd[_d[_CALIB]]
    'coiss': lambda stem: stem.split('_', 1)[0],
    # Cddddddd[_GEOMED]
    'vgiss': lambda stem: stem.split('_', 1)[0],
    # Cdddddddddd[RS] (no suffix to strip)
    'gossi': lambda stem: stem,
    # lor_dddddddddd[_0xddd_sci]
    'nhlorri': lambda stem: stem[:14],
}


def image_name_from_filename(instrument: str, filename: str) -> str:
    """The dataset-level image name for a recorded image filename.

    Parameters:
        instrument: Registered instrument name from the database
            (``coiss`` / ``vgiss`` / ``gossi`` / ``nhlorri``); an
            unregistered name only has its extension stripped.
        filename: The recorded ``images.image_name`` value (a basename,
            possibly with a directory prefix).

    Returns:
        The image name in the form ``--image-filelist`` selects on, e.g.
        ``N1454725799_1_CALIB.IMG`` yields ``N1454725799`` and
        ``lor_0003103486_0x630_sci.fit`` yields ``lor_0003103486``.
    """
    stem = filename.rsplit('/', 1)[-1]
    dot = stem.rfind('.')
    if dot > 0:
        stem = stem[:dot]
    rule = _IMAGE_NAME_RULES.get(instrument.lower())
    return rule(stem) if rule is not None else stem


def image_order(alias: str = '') -> str:
    """A total ordering over images, for a query that lists them by name.

    Image name alone is not unique: two volumes may hold images with the same
    basename, and the pair that keys the row breaks the tie.  Without it two
    rows with one name would come back in whatever order the backend liked, and
    the report would not be the same twice.

    Parameters:
        alias: Table alias prefix (e.g. ``'i.'``) qualifying the column names.

    Returns:
        The ``ORDER BY`` column list.
    """
    return f'{alias}image_name, {alias}root_url, {alias}results_path_stub'


IMAGE_JOIN = (
    'JOIN images i ON i.root_url = {alias}root_url '
    'AND i.results_path_stub = {alias}results_path_stub'
)
"""How a child table joins to the image it belongs to.

An image is keyed by the pair, so a child row is matched on the pair.  Joining
on the image name alone would merge two volumes' images of the same name into
one, which is precisely the confusion the pair exists to prevent.
"""


@dataclass
class ReportContext:
    """Mutable state threaded through every report section builder.

    Parameters:
        connection: Open connection to the results index.
        output_dir: Directory receiving ``report.md``, charts, and the
            optional ``filelists/`` subdirectory; a local directory or
            any URL the ``filecache`` layer accepts.
        where: Images-table filter from :func:`where_clause` (empty string
            or a leading-space ``' WHERE ...'`` fragment).
        params: Bind values matching ``where``, by name.
        where_i: The same filter built with ``alias='i.'`` for queries
            that join the ``images`` table as ``i``.
        params_i: Bind values matching ``where_i``, by name.
        lines: Markdown lines accumulated so far.
        top_n: When positive, categorical sections list up to this many
            example image names per category.
        filelists: When True, categorical sections write one plain-text
            file per category and instrument under ``filelists/``.
        suspect_fraction: Fraction of the per-axis search limit at or
            beyond which a fused offset is flagged as suspect.
        images_by_instrument: Selected image count per instrument, in
            instrument-name order; filled in from the database.
        instruments: The selected instrument names, in name order.
        total_images: Total selected images across all instruments.
    """

    connection: sqlalchemy.Connection
    output_dir: FCPath
    where: str
    params: dict[str, Any]
    where_i: str
    params_i: dict[str, Any]
    lines: list[str] = field(default_factory=list)
    top_n: int = 0
    filelists: bool = False
    suspect_fraction: float = 0.9
    images_by_instrument: dict[str, int] = field(default_factory=dict)
    instruments: list[str] = field(default_factory=list)
    total_images: int = 0

    def __post_init__(self) -> None:
        """Load the per-instrument image counts the whole report reports against."""
        counts = rows(
            self.connection,
            f'SELECT instrument, COUNT(*) FROM images{self.where} '
            'GROUP BY instrument ORDER BY instrument',
            self.params,
        )
        self.images_by_instrument = {str(name): int(count) for name, count in counts}
        self.instruments = list(self.images_by_instrument)
        self.total_images = sum(self.images_by_instrument.values())

    def write_filelist(self, stub: str, names: list[str]) -> str:
        """Write one image name per line into ``filelists/<stub>.txt``.

        The written file is directly consumable by the datasets'
        ``--image-filelist`` option: one image name per line, with a
        leading ``#`` comment line naming the category.

        Parameters:
            stub: Category identifier; sanitized into a filesystem-safe
                filename.
            names: Image names to write (written in the given order).

        Returns:
            The path of the written file, relative to ``output_dir``.
        """
        relative = f'filelists/{safe_filename(stub)}.txt'
        body = f'# {stub} ({len(names)} image(s))\n' + ''.join(f'{name}\n' for name in names)
        (self.output_dir / relative).write_text(body, encoding='utf-8')
        return relative


def where_clause(
    *,
    instrument: str | None,
    start_date: str | None,
    end_date: str | None,
    min_image_num: int | None = None,
    max_image_num: int | None = None,
    roots: list[str] | None = None,
    alias: str = '',
) -> tuple[str, dict[str, Any]]:
    """Build the images-table filter shared by every query.

    The binds are named rather than positional because the fragment is spliced
    into statements that carry binds of their own; a positional fragment would
    have to know how many came before it.

    ``image_number`` is a column, so the range filter is an ordinary comparison
    on any backend rather than a call into a function registered on whichever
    connection happened to be open.

    Parameters:
        instrument: Optional instrument filter value.
        start_date: Optional inclusive UTC start date (``YYYY-MM-DD``).
        end_date: Optional inclusive UTC end date (``YYYY-MM-DD``).
        min_image_num: Optional inclusive lower bound on the numeric
            portion of the image name.
        max_image_num: Optional inclusive upper bound on the numeric
            portion of the image name.
        roots: Optional normalized results-root URLs to restrict to; None or
            an empty list reports over every root the index holds.
        alias: Table alias prefix (e.g. ``'i.'``) qualifying the column
            names, for queries that join the ``images`` table.

    Returns:
        ``(where, params)`` where ``where`` is ``''`` or a leading-space
        ``' WHERE ...'`` fragment and ``params`` the bound values by name.
    """
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if instrument is not None:
        clauses.append(f'{alias}instrument = :instrument')
        params['instrument'] = instrument
    if start_date is not None:
        clauses.append(f'{alias}image_date >= :start_date')
        params['start_date'] = start_date
    if end_date is not None:
        clauses.append(f'{alias}image_date <= :end_date')
        params['end_date'] = end_date
    if min_image_num is not None:
        clauses.append(f'{alias}image_number >= :min_image_num')
        params['min_image_num'] = min_image_num
    if max_image_num is not None:
        clauses.append(f'{alias}image_number <= :max_image_num')
        params['max_image_num'] = max_image_num
    if roots:
        names = [f'root_{index}' for index in range(len(roots))]
        placeholders = ', '.join(f':{name}' for name in names)
        clauses.append(f'{alias}root_url IN ({placeholders})')
        params.update(zip(names, roots, strict=True))
    if len(clauses) == 0:
        return '', {}
    return ' WHERE ' + ' AND '.join(clauses), params


def connector(where: str) -> str:
    """The keyword joining an extra condition onto a ``where_clause`` result.

    Parameters:
        where: A filter fragment from :func:`where_clause` (empty string
            or a leading-space ``' WHERE ...'`` fragment).

    Returns:
        ``' AND '`` when ``where`` already has conditions, else
        ``' WHERE '``.
    """
    return ' AND ' if len(where) > 0 else ' WHERE '


def rows(
    connection: sqlalchemy.Connection, sql: str, params: dict[str, Any]
) -> list[tuple[Any, ...]]:
    """Execute a query and return all result rows as a list.

    Parameters:
        connection: Open connection to the results index.
        sql: SQL statement with ``:name`` bind placeholders.
        params: Bind values by name.

    Returns:
        All result rows, in query order, as plain tuples.
    """
    return [tuple(row) for row in connection.execute(sqlalchemy.text(sql), params)]


def fmt(value: float | None, digits: int = 3) -> str:
    """Format a float for a Markdown table cell.

    Parameters:
        value: Value to format, or None.
        digits: Decimal places.

    Returns:
        The fixed-point string, or ``'-'`` for None.
    """
    if value is None:
        return '-'
    return f'{value:.{digits}f}'


def count_pct(count: int, total: int) -> str:
    """Format an image count as ``'5 (3.2%)'``.

    Parameters:
        count: The number of images.
        total: The number of images the percentage is taken against; a
            zero total renders as ``0.0%``.

    Returns:
        The count followed by its percentage of ``total``.
    """
    fraction = count / total if total > 0 else 0.0
    return f'{count} ({fraction * 100:.1f}%)'


def add_instrument_count_table(
    ctx: ReportContext,
    table_rows: list[tuple[list[str], dict[str, int]]],
    *,
    headers: list[str],
) -> None:
    """Append a count table with one column per instrument plus a total.

    Every count cell is rendered by :func:`count_pct`.  An instrument
    column's percentage is of that instrument's selected images; the total
    column's percentage is of all selected images.

    Parameters:
        ctx: Report context.
        table_rows: ``(leading_cells, counts_by_instrument)`` pairs in the
            order they should be listed; instruments absent from a row's
            mapping count zero.
        headers: Column headings for the leading cells (e.g.
            ``['status']``); the instrument and total headings are added
            here.
    """
    columns = [*headers, *ctx.instruments, 'total']
    ctx.lines += [
        '| ' + ' | '.join(columns) + ' |',
        '|' + '---|' * len(columns),
    ]
    for leading, counts in table_rows:
        cells = list(leading)
        for instrument in ctx.instruments:
            cells.append(count_pct(counts.get(instrument, 0), ctx.images_by_instrument[instrument]))
        cells.append(count_pct(sum(counts.values()), ctx.total_images))
        ctx.lines.append('| ' + ' | '.join(cells) + ' |')
    ctx.lines.append('')


def offset_stats(values: list[float]) -> dict[str, float] | None:
    """Mean / median / stdev / min / max summary of a value list.

    Parameters:
        values: Values to summarize; may be empty.

    Returns:
        A dict with ``mean`` / ``median`` / ``stdev`` / ``min`` / ``max``
        keys (``stdev`` is 0.0 for a single value), or None when
        ``values`` is empty.
    """
    if len(values) == 0:
        return None
    return {
        'mean': statistics.fmean(values),
        'median': statistics.median(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values),
    }


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of a non-empty value list.

    Parameters:
        values: Non-empty list of values.
        fraction: Percentile as a fraction in ``[0, 1]`` (e.g. ``0.95``).

    Returns:
        The nearest-rank percentile value.
    """
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def safe_filename(stub: str) -> str:
    """Collapse a category label into a filesystem-safe filename stub.

    Parameters:
        stub: Arbitrary category label (may contain spaces, slashes,
            punctuation).

    Returns:
        The label with every run of unsafe characters replaced by ``_``
        and leading/trailing underscores stripped; ``'unnamed'`` when
        nothing survives.
    """
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', stub).strip('_')
    return cleaned or 'unnamed'


def add_drilldown(
    ctx: ReportContext,
    categories: list[tuple[str, list[tuple[str, str]]]],
    *,
    label: str,
    stub_prefix: str,
) -> None:
    """Append per-category example image names and write per-category filelists.

    Both the inline examples and the filelists are grouped by instrument:
    each category contributes one line (and one file) per instrument that
    has images in it.  Honors ``ctx.top_n`` (examples shown inline,
    alphabetically first N) and ``ctx.filelists`` (full alphabetical lists
    written to ``filelists/``, named ``<stub_prefix>_<category>_<inst>``).
    Does nothing when both are off.

    Parameters:
        ctx: Report context.
        categories: ``(category_label, entries)`` pairs in the order they
            should be listed, where each entry is an
            ``(instrument, image_filename)`` pair; filenames are converted
            to image names by :func:`image_name_from_filename` and sorted
            here.
        label: Human-readable singular noun for the category kind, used
            in the "Examples (up to N per ...)" line.
        stub_prefix: Filelist filename prefix; the category label and
            instrument are appended.
    """
    if ctx.top_n <= 0 and not ctx.filelists:
        return
    # (category, instrument) -> sorted image names, keeping the caller's
    # category order and instrument-name order within each category.
    grouped: list[tuple[str, str, list[str]]] = []
    for category, entries in categories:
        by_instrument: dict[str, list[str]] = {}
        for instrument, filename in entries:
            by_instrument.setdefault(instrument, []).append(
                image_name_from_filename(instrument, filename)
            )
        for instrument in sorted(by_instrument):
            grouped.append((category, instrument, sorted(by_instrument[instrument])))
    if len(grouped) == 0:
        return
    if ctx.top_n > 0:
        ctx.lines += [f'Examples (up to {ctx.top_n} per {label} and instrument):', '']
        for category, instrument, names in grouped:
            shown = ', '.join(names[: ctx.top_n])
            ctx.lines.append(f'- {category} / {instrument}: {shown}')
        ctx.lines.append('')
    if ctx.filelists:
        references = [
            ctx.write_filelist(f'{stub_prefix}_{category}_{instrument}', names)
            for category, instrument, names in grouped
        ]
        ctx.lines += [f'Full lists: {", ".join(references)}', '']


# Fixed per-instrument chart colors so a stacked segment means the same
# instrument in every chart and across runs.
_INSTRUMENT_COLORS: dict[str, str] = {
    'coiss': '#4878d0',
    'vgiss': '#ee854a',
    'gossi': '#6acc64',
    'nhlorri': '#d65f5f',
    'sim': '#956cb4',
}
_FALLBACK_COLORS: tuple[str, ...] = ('#8c8c8c', '#797979', '#b3b3b3', '#5c5c5c')


def instrument_color(instrument: str, index: int) -> str:
    """The chart color for an instrument.

    Parameters:
        instrument: Instrument name.
        index: The instrument's position in the report's instrument list;
            picks a fallback color for unregistered instruments.

    Returns:
        A matplotlib color string.
    """
    known = _INSTRUMENT_COLORS.get(instrument.lower())
    if known is not None:
        return known
    return _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]


def import_pyplot() -> Any:
    """Import matplotlib with the deterministic Agg backend and return pyplot.

    Returns:
        The ``matplotlib.pyplot`` module, with the backend forced to Agg
        so chart output is identical with or without a display.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig: Any, plt: Any, path: FCPath) -> None:
    """Render a figure to PNG bytes and write them through ``FCPath``.

    Rendering to a buffer first means the destination can be a local path
    or any URL the ``filecache`` layer accepts.

    Parameters:
        fig: Matplotlib figure to write.
        plt: The pyplot module (from :func:`import_pyplot`); the figure is
            closed here.
        path: Destination PNG path.
    """
    buffer = BytesIO()
    fig.savefig(buffer, dpi=100, format='png')
    plt.close(fig)
    path.write_bytes(buffer.getvalue())


def write_stacked_bar_chart(
    path: FCPath,
    labels: list[str],
    counts_by_instrument: dict[str, list[int]],
    instruments: list[str],
    *,
    title: str,
    xlabel: str,
) -> None:
    """Write a horizontal stacked bar chart PNG (deterministic, Agg backend).

    Each bar is one category, segmented by instrument.

    Parameters:
        path: Destination PNG path (local or ``filecache`` URL).
        labels: One bar label per category, top to bottom.
        counts_by_instrument: Instrument name to one count per category,
            matching ``labels``; instruments absent here contribute zero.
        instruments: Instrument names, in stacking order (left to right).
        title: Chart title.
        xlabel: X-axis label.
    """
    plt = import_pyplot()
    fig, ax = plt.subplots(figsize=(9, max(2.0, 0.4 * len(labels) + 1.5)))
    positions = list(range(len(labels)))
    left = [0.0] * len(labels)
    for index, instrument in enumerate(instruments):
        counts = counts_by_instrument.get(instrument, [0] * len(labels))
        ax.barh(
            positions,
            counts,
            left=left,
            color=instrument_color(instrument, index),
            label=instrument,
        )
        left = [base + value for base, value in zip(left, counts, strict=True)]
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if len(instruments) > 0:
        ax.legend(title='instrument', loc='lower right')
    fig.tight_layout()
    _save_figure(fig, plt, path)


def write_stacked_value_hist(
    path: FCPath,
    values_by_instrument: dict[str, list[float]],
    instruments: list[str],
    *,
    title: str,
    xlabel: str,
) -> None:
    """Write a single-panel stacked histogram PNG, segmented by instrument.

    Parameters:
        path: Destination PNG path (local or ``filecache`` URL).
        values_by_instrument: Instrument name to the values to histogram;
            all instruments share one set of bins.
        instruments: Instrument names, in stacking order.
        title: Chart title.
        xlabel: X-axis label.
    """
    plt = import_pyplot()
    fig, ax = plt.subplots(figsize=(9, 4))
    series = [values_by_instrument.get(instrument, []) for instrument in instruments]
    if sum(len(values) for values in series) > 0:
        ax.hist(
            series,
            bins=40,
            stacked=True,
            color=[instrument_color(name, index) for index, name in enumerate(instruments)],
            label=instruments,
        )
        ax.legend(title='instrument')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('images')
    ax.set_title(title)
    fig.tight_layout()
    _save_figure(fig, plt, path)


def write_offset_hist(path: FCPath, dv: list[float], du: list[float], *, title: str) -> None:
    """Write a two-panel V/U offset histogram PNG for one instrument.

    Parameters:
        path: Destination PNG path (local or ``filecache`` URL).
        dv: Fused V-axis offsets (pixels) of successful images.
        du: Fused U-axis offsets (pixels) of successful images.
        title: Figure title (the instrument is named by the caller).
    """
    plt = import_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, values, label in ((axes[0], dv, 'dV (px)'), (axes[1], du, 'dU (px)')):
        if len(values) > 0:
            ax.hist(values, bins=40, color='#4878d0')
        ax.set_xlabel(label)
        ax.set_ylabel('images')
    fig.suptitle(title)
    fig.tight_layout()
    _save_figure(fig, plt, path)
