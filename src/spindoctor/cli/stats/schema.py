"""SQLite schema and upsert helpers for the navigation statistics database."""

import sqlite3
from pathlib import Path
from typing import Any

__all__ = ['IMAGE_COLUMNS', 'open_stats_db', 'upsert_image']

# One row per image (keyed by image name), with normalized child tables for
# per-technique results and per-source feature usage.  Re-ingesting an image
# replaces its row and its children, so ingestion is idempotent.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    image_name TEXT PRIMARY KEY,
    instrument TEXT NOT NULL,
    image_path TEXT,
    image_et REAL,
    image_date TEXT,
    status TEXT NOT NULL,
    status_reason TEXT,
    offset_dv REAL,
    offset_du REAL,
    sigma_dv REAL,
    sigma_du REAL,
    confidence REAL,
    confidence_rank TEXT,
    n_techniques INTEGER NOT NULL,
    excluded_from_consensus TEXT,
    image_class TEXT,
    noise_sigma REAL,
    image_shape_v INTEGER,
    image_shape_u INTEGER,
    run_start TEXT,
    run_end TEXT,
    elapsed_s REAL,
    config_hash TEXT,
    git_sha TEXT,
    pipeline_run TEXT,
    source_file TEXT
);
CREATE TABLE IF NOT EXISTS techniques (
    image_name TEXT NOT NULL REFERENCES images(image_name) ON DELETE CASCADE,
    technique_name TEXT NOT NULL,
    offset_dv REAL,
    offset_du REAL,
    sigma_dv REAL,
    sigma_du REAL,
    confidence REAL,
    spurious INTEGER NOT NULL,
    at_edge INTEGER NOT NULL,
    source_names TEXT,
    diagnostics TEXT
);
CREATE TABLE IF NOT EXISTS feature_sources (
    image_name TEXT NOT NULL REFERENCES images(image_name) ON DELETE CASCADE,
    feature_type TEXT NOT NULL,
    source_model TEXT NOT NULL,
    source_name TEXT NOT NULL,
    n_features INTEGER NOT NULL,
    n_gated INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_date ON images(image_date);
CREATE INDEX IF NOT EXISTS idx_images_instrument ON images(instrument);
CREATE INDEX IF NOT EXISTS idx_techniques_image ON techniques(image_name);
CREATE INDEX IF NOT EXISTS idx_sources_image ON feature_sources(image_name);
"""

# Column order of the ``images`` table, kept in sync with ``_SCHEMA``.  The
# CSV export uses this so its column order is stable and documented.
IMAGE_COLUMNS: tuple[str, ...] = (
    'image_name',
    'instrument',
    'image_path',
    'image_et',
    'image_date',
    'status',
    'status_reason',
    'offset_dv',
    'offset_du',
    'sigma_dv',
    'sigma_du',
    'confidence',
    'confidence_rank',
    'n_techniques',
    'excluded_from_consensus',
    'image_class',
    'noise_sigma',
    'image_shape_v',
    'image_shape_u',
    'run_start',
    'run_end',
    'elapsed_s',
    'config_hash',
    'git_sha',
    'pipeline_run',
    'source_file',
)

_IMAGES_COLUMNS = frozenset(IMAGE_COLUMNS)
_TECHNIQUES_COLUMNS = frozenset(
    {
        'image_name',
        'technique_name',
        'offset_dv',
        'offset_du',
        'sigma_dv',
        'sigma_du',
        'confidence',
        'spurious',
        'at_edge',
        'source_names',
        'diagnostics',
    }
)
_SOURCES_COLUMNS = frozenset(
    {
        'image_name',
        'feature_type',
        'source_model',
        'source_name',
        'n_features',
        'n_gated',
    }
)


def open_stats_db(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if necessary) the statistics database.

    There is no schema migration: a database whose ``images`` table does
    not match the current column set is rejected with instructions to
    delete the file and re-ingest (ingestion is cheap and idempotent, so
    the database is always disposable).

    Parameters:
        db_path: Filesystem path of the SQLite database.

    Returns:
        An open connection with foreign keys enabled and the schema applied.

    Raises:
        ValueError: If the database exists with a different ``images``
            column set.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute('PRAGMA foreign_keys = ON')
    # Check any pre-existing images table BEFORE applying the schema
    # script -- the script's index statements fail confusingly against a
    # table with a different column set.
    found = {row[1] for row in conn.execute('PRAGMA table_info(images)')}
    if len(found) > 0 and found != _IMAGES_COLUMNS:
        missing = ', '.join(sorted(_IMAGES_COLUMNS - found)) or '(none)'
        extra = ', '.join(sorted(found - _IMAGES_COLUMNS)) or '(none)'
        conn.close()
        raise ValueError(
            f'{db_path}: images table does not match the current schema '
            f'(missing: {missing}; unexpected: {extra}). Delete the database '
            f'file and re-run sd_stats_ingest.'
        )
    conn.executescript(_SCHEMA)
    return conn


def _insert(
    conn: sqlite3.Connection,
    table: str,
    row: dict[str, Any],
    *,
    allowed_columns: frozenset[str],
) -> None:
    """Insert one row, validating column names against the table's schema.

    Values are always bound as SQL parameters; column names are interpolated
    into the statement text and therefore must come from the fixed schema
    allowlist.

    Parameters:
        conn: Open statistics database connection.
        table: Target table name.
        row: Column-to-value mapping for the new row.
        allowed_columns: The table's schema columns; any other key raises.

    Raises:
        ValueError: If ``row`` contains a key that is not a schema column.
    """
    unknown = sorted(set(row) - allowed_columns)
    if len(unknown) > 0:
        raise ValueError(f'unknown {table} column(s): {", ".join(unknown)}')
    columns = sorted(row)
    conn.execute(
        f'INSERT INTO {table} ({", ".join(columns)}) VALUES ({", ".join("?" for _ in columns)})',
        [row[c] for c in columns],
    )


def upsert_image(
    conn: sqlite3.Connection,
    image_row: dict[str, Any],
    *,
    technique_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> None:
    """Insert or replace one image and its child rows.

    Transaction management belongs to the caller (``ingest_metadata_files``
    wraps a whole scan in one transaction); this function only issues the
    DELETE and INSERT statements.

    Parameters:
        conn: Open statistics database connection.
        image_row: Column mapping for the ``images`` table; must contain
            ``image_name``.
        technique_rows: Column mappings for the ``techniques`` table
            (``image_name`` is filled in here).
        source_rows: Column mappings for the ``feature_sources`` table
            (``image_name`` is filled in here).

    Raises:
        ValueError: If any row contains a key that is not a schema column.
    """
    image_name = image_row['image_name']
    conn.execute('DELETE FROM images WHERE image_name = ?', (image_name,))
    _insert(conn, 'images', image_row, allowed_columns=_IMAGES_COLUMNS)
    for row in technique_rows:
        _insert(
            conn,
            'techniques',
            {**row, 'image_name': image_name},
            allowed_columns=_TECHNIQUES_COLUMNS,
        )
    for row in source_rows:
        _insert(
            conn,
            'feature_sources',
            {**row, 'image_name': image_name},
            allowed_columns=_SOURCES_COLUMNS,
        )
