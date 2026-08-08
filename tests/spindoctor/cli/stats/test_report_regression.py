"""The report must still say what it said before its queries moved.

Every query the statistics user guide documents changed in the move onto the
results index: the reason tables read two columns through a ``COALESCE``, the
joins match a column pair instead of an image name, the boolean columns are used
as booleans, and the image-number filter compares a column instead of calling a
Python function registered on the connection.  A change in *how* a number is
computed must not change the number.

``data/golden`` holds ``report.md`` and ``images.csv`` as the statistics code
produced them from ``data/results_tree`` before any of that moved.  The tree
covers every section: two volumes and a bare-basename stub, successes, a
failure, a fatal error with no navigation result at all, a BOTSIM pair, a
suspect offset, an ensemble exclusion, a spurious technique, gated features, and
an image whose search limit cannot be resolved.  A diff here is a defect, not
drift.

Two documented differences are the point of the change rather than a regression
in it, and each is asserted for rather than waved past:

- ``images.csv`` gains the columns the schema gained, in schema order, and its
  ``status_reason`` column no longer carries a fatal error -- ``status_error``
  is its own column now, and merging the two vocabularies is what the report
  stopped doing.
- The two feature-count aggregates come back as integers.  They are counts, and
  the aggregate that produced a floating-point zero for an image with no
  features was a SQLite spelling.

One byte-level difference is disclosed rather than compared away.  The frozen
``images.csv`` files carry LF line endings; the implementation that produced
them left ``csv.writer`` at its CRLF default, and the committed blobs were
normalized when they were frozen.  The export now states its line terminator
instead of inheriting one, and states LF, so what it writes matches the frozen
bytes.  The comparisons here read fields rather than lines and would not have
noticed either way, so the terminator is pinned by a test of its own.
"""

import csv
import uuid
from pathlib import Path
from typing import Any

import pdslogger
import pytest

from .conftest import GOLDEN_DIR, RESULTS_TREE, index_url, report_from_tree

_TREE_PLACEHOLDER = '{results_tree}'
"""What the frozen CSV holds in place of the tree's absolute path."""

_VARIANTS: dict[str, dict[str, Any]] = {
    'full': {'top_n': 5, 'filelists': True, 'csv_export': True},
    'filtered': {
        'instrument': 'coiss',
        'min_image': '1294561202',
        'max_image': '1294563000',
        'top_n': 3,
        'csv_export': True,
    },
}
"""The two report invocations the frozen output was produced by."""

_MERGED_REASON_COLUMN = 'status_reason'
"""The column the frozen CSV merged both reason vocabularies into."""

_COUNT_COLUMNS = ('n_features', 'n_gated')
"""Aggregates the frozen CSV rendered as floating-point counts."""


@pytest.fixture(scope='module', params=sorted(_VARIANTS))
def report_variant(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, Path]:
    """Build one report variant once for the whole module.

    Six comparisons are made against each variant and every one of them only
    reads, so building the variant per test ingests the same tree twelve times
    to produce two outputs.

    Parameters:
        request: The fixture request, carrying the variant name.
        tmp_path_factory: Factory the index and the output live under.

    Returns:
        The variant name and the directory its report was written into.
    """
    variant = str(request.param)
    logger = pdslogger.PdsLogger(f'stats_regression_{uuid.uuid4().hex}')
    logger.set_level('ERROR')
    root = tmp_path_factory.mktemp('regression')
    out = report_from_tree(
        index_url(root / 'index.sqlite3'), root / variant, logger=logger, **_VARIANTS[variant]
    )
    return variant, out


def _csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV export into one mapping per row.

    Parameters:
        path: The CSV file.

    Returns:
        One mapping per data row, keyed by column name.
    """
    return list(csv.DictReader(path.read_text(encoding='utf-8').splitlines()))


def _frozen_csv_rows(variant: str) -> list[dict[str, str]]:
    """Read a frozen CSV export, restoring the tree's absolute path.

    Parameters:
        variant: Which entry of :data:`_VARIANTS` to read.

    Returns:
        One mapping per data row, keyed by column name.
    """
    text = (GOLDEN_DIR / variant / 'images.csv').read_text(encoding='utf-8')
    text = text.replace(_TREE_PLACEHOLDER, RESULTS_TREE.as_posix())
    return list(csv.DictReader(text.splitlines()))


def test_the_report_is_byte_identical(report_variant: tuple[str, Path]) -> None:
    """Every section, every number, every ordering, unchanged.

    Parameters:
        report_variant: The variant name and the report it wrote.
    """
    variant, out = report_variant
    frozen = (GOLDEN_DIR / variant / 'report.md').read_text(encoding='utf-8')
    assert (out / 'report.md').read_text(encoding='utf-8') == frozen


def test_the_csv_holds_the_same_images(report_variant: tuple[str, Path]) -> None:
    """The export covers the same images in the same order.

    Parameters:
        report_variant: The variant name and the report it wrote.
    """
    variant, out = report_variant
    produced = [row['image_name'] for row in _csv_rows(out / 'images.csv')]
    frozen = [row['image_name'] for row in _frozen_csv_rows(variant)]
    assert produced == frozen


def test_every_carried_over_csv_column_is_unchanged(report_variant: tuple[str, Path]) -> None:
    """Every column the export already had holds what it held.

    The two documented exceptions are checked by the tests below rather than
    skipped: the merged reason column, and the two count aggregates.

    Parameters:
        report_variant: The variant name and the report it wrote.
    """
    variant, out = report_variant
    produced = _csv_rows(out / 'images.csv')
    frozen = _frozen_csv_rows(variant)
    differences = [
        f'{row["image_name"]}.{column}: {value!r} became {row[column]!r}'
        for row, frozen_row in zip(produced, frozen, strict=True)
        for column, value in frozen_row.items()
        if column not in (_MERGED_REASON_COLUMN, *_COUNT_COLUMNS)
        if row[column] != value
    ]
    assert differences == []


def test_the_merged_reason_column_is_reproduced_by_the_pair(
    report_variant: tuple[str, Path],
) -> None:
    """The export split one column into two, and the pair says what one said.

    Parameters:
        report_variant: The variant name and the report it wrote.
    """
    variant, out = report_variant
    produced = _csv_rows(out / 'images.csv')
    frozen = _frozen_csv_rows(variant)
    differences = [
        f'{row["image_name"]}: {frozen_row[_MERGED_REASON_COLUMN]!r} became '
        f'{row["status_reason"]!r} / {row["status_error"]!r}'
        for row, frozen_row in zip(produced, frozen, strict=True)
        if (row['status_reason'] or row['status_error']) != frozen_row[_MERGED_REASON_COLUMN]
    ]
    assert differences == []


def test_the_count_aggregates_are_the_same_numbers(report_variant: tuple[str, Path]) -> None:
    """They are counts now rather than sums that could not be integers.

    Parameters:
        report_variant: The variant name and the report it wrote.
    """
    variant, out = report_variant
    produced = _csv_rows(out / 'images.csv')
    frozen = _frozen_csv_rows(variant)
    differences = [
        f'{row["image_name"]}.{column}: {frozen_row[column]!r} became {row[column]!r}'
        for row, frozen_row in zip(produced, frozen, strict=True)
        for column in _COUNT_COLUMNS
        if float(row[column]) != float(frozen_row[column])
    ]
    assert differences == []


def test_the_csv_gained_the_key_columns(report_variant: tuple[str, Path]) -> None:
    """The export now says which root and which stub each row came from.

    Parameters:
        report_variant: The variant name and the report it wrote.
    """
    _variant, out = report_variant
    header = (out / 'images.csv').read_text(encoding='utf-8').splitlines()[0].split(',')
    assert header[:2] == ['root_url', 'results_path_stub']


def test_every_filelist_is_unchanged(report_variant: tuple[str, Path]) -> None:
    """The drill-down lists feed back into re-runs, so their contents are a contract.

    Parameters:
        report_variant: The variant name and the report it wrote.
    """
    variant, out = report_variant
    frozen_lists = sorted((GOLDEN_DIR / variant).glob('filelists/*.txt'))
    differences = [
        path.name
        for path in frozen_lists
        if not (out / 'filelists' / path.name).exists()
        or (out / 'filelists' / path.name).read_text(encoding='utf-8')
        != path.read_text(encoding='utf-8')
    ]
    assert differences == []


def test_the_frozen_output_is_not_empty() -> None:
    """A comparison against nothing passes for the wrong reason."""
    assert len(sorted((GOLDEN_DIR / 'full').glob('filelists/*.txt'))) > 0


def test_the_frozen_report_covers_every_section() -> None:
    """A fixture tree that populated half the report would prove half of it."""
    text = (GOLDEN_DIR / 'full' / 'report.md').read_text(encoding='utf-8')
    missing = [
        heading
        for heading in (
            '## Images selected',
            '## Success / failure',
            '### Failure reasons',
            '## Failure taxonomy by image content',
            '### Per-body failure shares',
            '## Technique usage',
            '## Model and source usage',
            '## Offset statistics (successful images)',
            '### By instrument, camera, and image size',
            '## Suspect offsets (near the search limit)',
            '## BOTSIM pair consistency (Cassini ISS)',
            '## Cross-technique agreement',
            '## Confidence calibration (agreement as accuracy proxy)',
            '## Ensemble outlier exclusions',
            '## Run-time statistics',
            '## CSV export',
        )
        if heading not in text
    ]
    assert missing == []


def test_the_frozen_report_exercises_the_merged_reason_column() -> None:
    """Without a fatal error in the tree, the COALESCE rewrite proves nothing."""
    text = (GOLDEN_DIR / 'full' / 'report.md').read_text(encoding='utf-8')
    assert '| error | missing_spice_data |' in text
