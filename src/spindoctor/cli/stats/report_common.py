"""Shared context, query, formatting, and chart helpers for the statistics report."""

import math
import re
import sqlite3
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    'ReportContext',
    'add_drilldown',
    'connector',
    'fmt',
    'image_number_from_name',
    'offset_stats',
    'percentile',
    'register_image_number_function',
    'rows',
    'safe_filename',
    'where_clause',
    'write_bar_chart',
    'write_offset_hist',
    'write_value_hist',
]


_IMAGE_NUMBER_RE = re.compile(r'\d+')


def image_number_from_name(image_name: str | None) -> int | None:
    """Numeric portion (first digit run) of an image name's basename.

    ``N1454725799_1_CALIB.IMG`` yields ``1454725799``;
    ``lor_0003103486_0x630_sci`` yields ``3103486`` (leading zeros drop in
    the integer).  This is the value the ``--min-image`` / ``--max-image``
    range filter compares.

    Parameters:
        image_name: Image name or path, or None.

    Returns:
        The integer value of the first digit run, or None when the name is
        None or contains no digits.
    """
    if image_name is None:
        return None
    match = _IMAGE_NUMBER_RE.search(image_name.rsplit('/', 1)[-1])
    if match is None:
        return None
    return int(match.group(0))


def register_image_number_function(conn: sqlite3.Connection) -> None:
    """Register the deterministic ``image_number`` SQL function on a connection.

    The image-number range filter clauses produced by :func:`where_clause`
    call this function, so it must be registered before those clauses run.

    Parameters:
        conn: Open statistics database connection.
    """
    conn.create_function('image_number', 1, image_number_from_name, deterministic=True)


@dataclass
class ReportContext:
    """Mutable state threaded through every report section builder.

    Parameters:
        conn: Open statistics database connection (with the
            ``image_number`` SQL function registered).
        output_dir: Directory receiving ``report.md``, charts, and the
            optional ``filelists/`` subdirectory.
        where: Images-table filter from :func:`where_clause` (empty string
            or a leading-space ``' WHERE ...'`` fragment).
        params: Bind values matching ``where``.
        where_i: The same filter built with ``alias='i.'`` for queries
            that join the ``images`` table as ``i``.
        params_i: Bind values matching ``where_i``.
        lines: Markdown lines accumulated so far.
        top_n: When positive, categorical sections list up to this many
            example image names per category.
        filelists: When True, categorical sections write one plain-text
            file per category under ``filelists/``.
        suspect_fraction: Fraction of the per-axis search limit at or
            beyond which a fused offset is flagged as suspect.
    """

    conn: sqlite3.Connection
    output_dir: Path
    where: str
    params: list[Any]
    where_i: str
    params_i: list[Any]
    lines: list[str] = field(default_factory=list)
    top_n: int = 0
    filelists: bool = False
    suspect_fraction: float = 0.9

    def write_filelist(self, stub: str, names: list[str]) -> str:
        """Write one image name per line into ``filelists/<stub>.txt``.

        Parameters:
            stub: Category identifier; sanitized into a filesystem-safe
                filename.
            names: Image names to write (written in the given order).

        Returns:
            The path of the written file, relative to ``output_dir``.
        """
        filelists_dir = self.output_dir / 'filelists'
        filelists_dir.mkdir(parents=True, exist_ok=True)
        relative = f'filelists/{safe_filename(stub)}.txt'
        (self.output_dir / relative).write_text(
            ''.join(f'{name}\n' for name in names), encoding='utf-8'
        )
        return relative


def where_clause(
    *,
    instrument: str | None,
    start_date: str | None,
    end_date: str | None,
    min_image_num: int | None = None,
    max_image_num: int | None = None,
    alias: str = '',
) -> tuple[str, list[Any]]:
    """Build the images-table filter shared by every query.

    Parameters:
        instrument: Optional instrument filter value.
        start_date: Optional inclusive UTC start date (``YYYY-MM-DD``).
        end_date: Optional inclusive UTC end date (``YYYY-MM-DD``).
        min_image_num: Optional inclusive lower bound on the numeric
            portion of the image name (requires
            :func:`register_image_number_function`).
        max_image_num: Optional inclusive upper bound on the numeric
            portion of the image name.
        alias: Table alias prefix (e.g. ``'i.'``) qualifying the column
            names, for queries that join the ``images`` table.

    Returns:
        ``(where, params)`` where ``where`` is ``''`` or a leading-space
        ``' WHERE ...'`` fragment and ``params`` the bound values.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if instrument is not None:
        clauses.append(f'{alias}instrument = ?')
        params.append(instrument)
    if start_date is not None:
        clauses.append(f'{alias}image_date >= ?')
        params.append(start_date)
    if end_date is not None:
        clauses.append(f'{alias}image_date <= ?')
        params.append(end_date)
    if min_image_num is not None:
        clauses.append(f'image_number({alias}image_name) >= ?')
        params.append(min_image_num)
    if max_image_num is not None:
        clauses.append(f'image_number({alias}image_name) <= ?')
        params.append(max_image_num)
    if len(clauses) == 0:
        return '', []
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


def rows(conn: sqlite3.Connection, sql: str, params: list[Any]) -> list[tuple[Any, ...]]:
    """Execute a query and return all result rows as a list.

    Parameters:
        conn: Open statistics database connection.
        sql: SQL statement with ``?`` placeholders.
        params: Bind values matching the placeholders.

    Returns:
        All result rows, in query order.
    """
    return list(conn.execute(sql, params))


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
    categories: list[tuple[str, list[str]]],
    *,
    label: str,
    stub_prefix: str,
) -> None:
    """Append per-category example names and write per-category filelists.

    Honors ``ctx.top_n`` (examples shown inline, alphabetically first N)
    and ``ctx.filelists`` (full alphabetical lists written to
    ``filelists/``).  Does nothing when both are off.

    Parameters:
        ctx: Report context.
        categories: ``(category_label, image_names)`` pairs in the order
            they should be listed; names are sorted here.
        label: Human-readable singular noun for the category kind, used
            in the "Examples (up to N per ...)" line.
        stub_prefix: Filelist filename prefix; the category label is
            appended.
    """
    categories = [(name, sorted(names)) for name, names in categories if len(names) > 0]
    if len(categories) == 0:
        return
    if ctx.top_n > 0:
        ctx.lines.append(f'Examples (up to {ctx.top_n} per {label}):')
        ctx.lines.append('')
        for name, names in categories:
            ctx.lines.append(f'- {name}: {", ".join(names[: ctx.top_n])}')
        ctx.lines.append('')
    if ctx.filelists:
        references = [
            ctx.write_filelist(f'{stub_prefix}_{name}', names) for name, names in categories
        ]
        ctx.lines.append(f'Full lists: {", ".join(references)}')
        ctx.lines.append('')


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


def write_bar_chart(
    path: Path, labels: list[str], counts: list[int], *, title: str, xlabel: str
) -> None:
    """Write a horizontal bar chart PNG (deterministic, Agg backend).

    Parameters:
        path: Destination PNG path.
        labels: One bar label per row, top to bottom.
        counts: Bar lengths matching ``labels``.
        title: Chart title.
        xlabel: X-axis label.
    """
    plt = import_pyplot()
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


def write_offset_hist(path: Path, dv: list[float], du: list[float]) -> None:
    """Write the V/U offset histogram PNG.

    Parameters:
        path: Destination PNG path.
        dv: Fused V-axis offsets (pixels) of successful images.
        du: Fused U-axis offsets (pixels) of successful images.
    """
    plt = import_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, values, label in ((axes[0], dv, 'dV (px)'), (axes[1], du, 'dU (px)')):
        if len(values) > 0:
            ax.hist(values, bins=40, color='#4878d0')
        ax.set_xlabel(label)
        ax.set_ylabel('images')
    fig.suptitle('Fused offset distribution (successful images)')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_value_hist(path: Path, values: list[float], *, title: str, xlabel: str) -> None:
    """Write a single-panel histogram PNG for a value list.

    Parameters:
        path: Destination PNG path.
        values: Values to histogram; an empty list produces empty axes.
        title: Chart title.
        xlabel: X-axis label.
    """
    plt = import_pyplot()
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(values) > 0:
        ax.hist(values, bins=40, color='#4878d0')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('images')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
