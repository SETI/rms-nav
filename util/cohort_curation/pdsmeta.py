"""Minimal PDS3 metadata-table access for cohort curation (Stage A).

Parses .lbl column definitions (authoritative per volume set, since layouts
differ between missions) and reads the comma-separated .tab geometry tables
and inventory .csv files under $PDS3_HOLDINGS_DIR/metadata/<VOLSET>/<VOLUME>/.

Not part of the spindoctor package; workflow tooling for
plans/COHORT_CURATION_PLAN.md Stage A.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

METADATA_ROOT = Path('/mnt/ganymede/PDS/holdings/metadata')

_NAME_RE = re.compile(r'^\s*NAME\s*=\s*"?([A-Za-z0-9_]+)"?')
_ITEMS_RE = re.compile(r'^\s*ITEMS\s*=\s*(\d+)')


def parse_label_columns(lbl_path: Path) -> dict[str, int]:
    """Return {COLUMN_NAME: 0-based field index} from a PDS3 table label.

    Columns are laid out in file order; a column with ITEMS = n (vector
    columns, e.g. SC_*_POSITION_VECTOR in the Cassini index) occupies n
    consecutive CSV fields, so field indices are cumulative -- do NOT use
    COLUMN_NUMBER, which counts columns, not fields.
    """
    cols: dict[str, int] = {}
    field = 0
    in_col = False
    name: str | None = None
    items = 1
    for raw in lbl_path.read_text(errors='replace').splitlines():
        line = raw.strip()
        if re.match(r'^OBJECT\s*=\s*COLUMN\s*$', line):
            in_col, name, items = True, None, 1
        elif line.startswith('END_OBJECT') and in_col:
            if name is not None:
                cols[name] = field
            field += items
            in_col = False
        elif in_col:
            m = _NAME_RE.match(line)
            if m and name is None:
                name = m.group(1)
            m = _ITEMS_RE.match(line)
            if m:
                items = int(m.group(1))
    return cols


class SummaryTable:
    """One <VOLUME>_<kind>.tab with named-column access."""

    def __init__(self, tab_path: Path, lbl_path: Path) -> None:
        self.path = tab_path
        self.cols = parse_label_columns(lbl_path)
        with open(tab_path, errors='replace', newline='') as f:
            self.rows = [
                [field.strip() for field in row]
                for row in csv.reader(f)
                if row
            ]

    def get(self, row: list[str], name: str, *alt_names: str) -> str | None:
        for n in (name, *alt_names):
            idx = self.cols.get(n)
            if idx is not None and idx < len(row):
                return row[idx]
        return None

    def num(self, row: list[str], name: str, *alt_names: str) -> float | None:
        """Numeric value or None when missing or a -999-family sentinel."""
        val = self.get(row, name, *alt_names)
        if val is None:
            return None
        try:
            x = float(val)
        except ValueError:
            return None
        # Sentinels: -999 family (COISS/GO) and -99.9999 (VGISS exposure).
        if x <= -99.0:
            return None
        return x


def load_table(volset: str, volume: str, kind: str) -> SummaryTable | None:
    """Load e.g. ('COISS_2xxx', 'COISS_2001', 'moon_summary'); None if absent."""
    d = METADATA_ROOT / volset / volume
    tab = d / f'{volume}_{kind}.tab'
    lbl = d / f'{volume}_{kind}.lbl'
    if not tab.exists() or not lbl.exists():
        return None
    return SummaryTable(tab, lbl)


def load_inventory(volset: str, volume: str) -> dict[str, list[str]]:
    """Return {FILE_SPECIFICATION_NAME: [body, ...]} from <VOLUME>_inventory.csv."""
    path = METADATA_ROOT / volset / volume / f'{volume}_inventory.csv'
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    with open(path, errors='replace', newline='') as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            filespec = row[1].strip()
            # Layouts differ: COISS has an OPUS_ID third field, Galileo does not.
            rest = [x.strip() for x in row[2:]]
            if rest and ('-' in rest[0] and rest[0].islower()):
                rest = rest[1:]
            out[filespec] = [b for b in rest if b]
    return out


def volumes(volset: str) -> list[str]:
    """Volume dirs, excluding the cumulative *_?999 pseudo-volumes whose
    tables duplicate every real volume's rows."""
    root = METADATA_ROOT / volset
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and not p.name.startswith('.')
                  and not p.name.endswith('999'))
