"""The vendored Titan validation cohort: flags, holdings paths, frame metadata.

``titan_images.csv`` carries the cohort itself -- one row per image with the
flags this repo reads off the legacy annotation.  Everything else a campaign
needs (where the frame lives in the holdings tree, when it was taken, which
filters it used) comes from the PDS3 volume indexes and is resolved here so
the collector and the analyzer share one definition.

Resolution is by scan of the per-volume ``*_index.tab`` files under
``$PDS3_HOLDINGS_DIR/metadata``.  The cumulative ``*999`` indexes are skipped:
they carry every image but name a volume that holds no data.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
COHORT_CSV = HERE / 'titan_images.csv'

VOLSETS = ('COISS_1xxx', 'COISS_2xxx')

FLAGS = (
    'rings_occluding',
    'moon_occluding',
    'high_phase',
    'near_edge',
    'off_edge',
    'known_bad',
    'clean',
)

# Flags that make a frame a negative control: the plan's acceptance line
# requires each of these to gate out or fail a named technique gate rather
# than lock confidently on the wrong answer.
ADVERSE_FLAGS = frozenset({'rings_occluding', 'moon_occluding', 'off_edge', 'known_bad'})

_IMG_RE = re.compile(r'([NW]\d{10}_\d)\.IMG')
_FILESPEC_RE = re.compile(r'data/[^",\s]+\.IMG')


@dataclass(frozen=True)
class CohortFrame:
    """One cohort frame with its flags and (once resolved) its holdings path.

    Parameters:
        image_id: PDS3 product id stem, e.g. ``'W1822132529_1'``.
        flags: The flags read off the legacy annotation.
        notes: The legacy annotation, verbatim (empty when unannotated).
        volset: Volume set the frame lives in, or None when unresolved.
        volume: Volume the frame lives in, or None when unresolved.
        rel_path: Holdings-relative path of the ``.IMG``, or None.
        image_time: PDS3 ``IMAGE_TIME`` (UTC ISO), or None.
        filter1: First filter-wheel name, or None.
        filter2: Second filter-wheel name, or None.
        target: PDS3 ``TARGET_NAME``, or None.
        exposure_sec: PDS3 ``EXPOSURE_DURATION`` in seconds, or None.
    """

    image_id: str
    flags: tuple[str, ...]
    notes: str
    volset: str | None = None
    volume: str | None = None
    rel_path: str | None = None
    image_time: str | None = None
    filter1: str | None = None
    filter2: str | None = None
    target: str | None = None
    exposure_sec: float | None = None

    @property
    def camera(self) -> str:
        """``'NAC'`` or ``'WAC'``, from the product id's leading letter."""
        return 'NAC' if self.image_id.startswith('N') else 'WAC'

    @property
    def is_clean(self) -> bool:
        """True when the annotation named no defect (``clean`` flag present)."""
        return 'clean' in self.flags

    @property
    def is_adverse(self) -> bool:
        """True when any :data:`ADVERSE_FLAGS` flag is set."""
        return bool(ADVERSE_FLAGS & set(self.flags))

    @property
    def filters(self) -> str:
        """Both filter names joined, e.g. ``'CL1+RED'``; ``'?'`` when unknown."""
        if self.filter1 is None or self.filter2 is None:
            return '?'
        return f'{self.filter1}+{self.filter2}'


def load_cohort() -> list[CohortFrame]:
    """Read ``titan_images.csv`` (comment lines skipped) into frames.

    Returns:
        One :class:`CohortFrame` per row, in file order, unresolved.
    """
    with COHORT_CSV.open() as fh:
        lines = [line for line in fh if not line.startswith('#')]
    frames = []
    for row in csv.DictReader(lines):
        flags = tuple(f for f in row['flags'].split(';') if f)
        unknown = set(flags) - set(FLAGS)
        if unknown:
            raise ValueError(f'{row["image_id"]}: unknown flags {sorted(unknown)}')
        frames.append(CohortFrame(image_id=row['image_id'], flags=flags, notes=row['notes']))
    return frames


def holdings_root() -> Path:
    """Return ``$PDS3_HOLDINGS_DIR`` as a path.

    Raises:
        RuntimeError: when the variable is unset (``source setup.sh`` first).
    """
    root = os.environ.get('PDS3_HOLDINGS_DIR')
    if not root:
        raise RuntimeError('PDS3_HOLDINGS_DIR is not set; source /seti/newnav/setup.sh')
    return Path(root)


def _index_columns(lbl_path: Path) -> dict[str, int]:
    """Return ``{COLUMN_NAME: 0-based field index}`` for one index label.

    Vector columns occupy ``ITEMS`` consecutive comma-separated fields, so
    field indices accumulate rather than tracking ``COLUMN_NUMBER``.

    Parameters:
        lbl_path: The ``*_index.lbl`` label file.

    Returns:
        Mapping from column name to its first field index.
    """
    cols: dict[str, int] = {}
    field_index = 0
    in_column = False
    name: str | None = None
    items = 1
    for raw in lbl_path.read_text(errors='replace').splitlines():
        line = raw.strip()
        if re.match(r'^OBJECT\s*=\s*COLUMN\s*$', line):
            in_column, name, items = True, None, 1
        elif line.startswith('END_OBJECT') and in_column:
            if name is not None:
                cols[name] = field_index
            field_index += items
            in_column = False
        elif in_column:
            match = re.match(r'^\s*NAME\s*=\s*"?([A-Za-z0-9_]+)"?', line)
            if match and name is None:
                name = match.group(1)
            match = re.match(r'^\s*ITEMS\s*=\s*(\d+)', line)
            if match:
                items = int(match.group(1))
    return cols


def resolve(frames: list[CohortFrame]) -> list[CohortFrame]:
    """Attach holdings path, epoch, filters, and target to every frame.

    Parameters:
        frames: The unresolved cohort.

    Returns:
        A new list in the same order; frames not found in any index keep
        their unresolved fields as None.
    """
    wanted = {frame.image_id for frame in frames}
    resolved: dict[str, dict[str, str]] = {}
    root = holdings_root()
    for volset in VOLSETS:
        volset_dir = root / 'metadata' / volset
        if not volset_dir.is_dir():
            continue
        for volume_dir in sorted(volset_dir.iterdir()):
            if volume_dir.name.endswith('999'):
                continue
            tab = volume_dir / f'{volume_dir.name}_index.tab'
            lbl = volume_dir / f'{volume_dir.name}_index.lbl'
            if not tab.is_file() or not lbl.is_file():
                continue
            cols = _index_columns(lbl)
            for row in csv.reader(tab.read_text(errors='replace').splitlines()):
                joined = ','.join(row)
                match = _IMG_RE.search(joined)
                if match is None or match.group(1) not in wanted:
                    continue
                filespec = _FILESPEC_RE.search(joined)
                if filespec is None:
                    continue

                def field(
                    name: str,
                    offset: int = 0,
                    row: list[str] = row,
                    cols: dict[str, int] = cols,
                ) -> str:
                    index = cols.get(name)
                    if index is None or index + offset >= len(row):
                        return ''
                    return row[index + offset].strip().strip('"')

                # FILTER_NAME is a two-item vector column (one name per
                # filter wheel), so the second wheel is the next CSV field.
                resolved[match.group(1)] = {
                    'volset': volset,
                    'volume': volume_dir.name,
                    'rel_path': f'volumes/{volset}/{volume_dir.name}/{filespec.group(0)}',
                    'image_time': field('IMAGE_TIME'),
                    'filter1': field('FILTER_NAME'),
                    'filter2': field('FILTER_NAME', 1),
                    'target': field('TARGET_NAME'),
                    'exposure_sec': field('EXPOSURE_DURATION'),
                }
    out = []
    for frame in frames:
        extra = resolved.get(frame.image_id)
        if extra is None:
            out.append(frame)
            continue
        out.append(
            CohortFrame(
                image_id=frame.image_id,
                flags=frame.flags,
                notes=frame.notes,
                volset=extra['volset'],
                volume=extra['volume'],
                rel_path=extra['rel_path'],
                image_time=extra['image_time'] or None,
                filter1=extra['filter1'] or None,
                filter2=extra['filter2'] or None,
                target=extra['target'] or None,
                exposure_sec=(
                    float(extra['exposure_sec']) / 1000.0 if extra['exposure_sec'] else None
                ),
            )
        )
    return out


def resolved_cohort() -> list[CohortFrame]:
    """Load and resolve the cohort in one call."""
    return resolve(load_cohort())


def main() -> int:
    """Print the resolved cohort as a table (a self-check on the resolution)."""
    frames = resolved_cohort()
    root = holdings_root()
    missing = [f.image_id for f in frames if f.rel_path is None]
    absent = [
        f.image_id for f in frames if f.rel_path is not None and not (root / f.rel_path).is_file()
    ]
    print(f'{len(frames)} cohort frames; {len(missing)} unresolved, {len(absent)} absent on disk')
    for frame in frames:
        print(
            f'{frame.image_id}  {frame.camera}  {frame.filters:<10} '
            f'{frame.image_time or "?":<24} {";".join(frame.flags)}'
        )
    if missing:
        print(f'UNRESOLVED: {missing}')
    if absent:
        print(f'ABSENT: {absent}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
