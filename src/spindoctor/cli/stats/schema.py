"""SQLite schema and upsert helpers for the navigation statistics database."""

import sqlite3
from pathlib import Path
from typing import Any

__all__ = ['open_stats_db', 'upsert_image']

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


def open_stats_db(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if necessary) the statistics database.

    Parameters:
        db_path: Filesystem path of the SQLite database.

    Returns:
        An open connection with foreign keys enabled and the schema applied.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute('PRAGMA foreign_keys = ON')
    conn.executescript(_SCHEMA)
    return conn


def upsert_image(
    conn: sqlite3.Connection,
    image_row: dict[str, Any],
    technique_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> None:
    """Insert or replace one image and its child rows.

    Parameters:
        conn: Open statistics database connection.
        image_row: Column mapping for the ``images`` table; must contain
            ``image_name``.
        technique_rows: Column mappings for the ``techniques`` table
            (``image_name`` is filled in here).
        source_rows: Column mappings for the ``feature_sources`` table
            (``image_name`` is filled in here).
    """
    image_name = image_row['image_name']
    with conn:
        conn.execute('DELETE FROM images WHERE image_name = ?', (image_name,))
        columns = sorted(image_row)
        conn.execute(
            f'INSERT INTO images ({", ".join(columns)}) VALUES ({", ".join("?" for _ in columns)})',
            [image_row[c] for c in columns],
        )
        for row in technique_rows:
            row = {**row, 'image_name': image_name}
            columns = sorted(row)
            conn.execute(
                f'INSERT INTO techniques ({", ".join(columns)}) '
                f'VALUES ({", ".join("?" for _ in columns)})',
                [row[c] for c in columns],
            )
        for row in source_rows:
            row = {**row, 'image_name': image_name}
            columns = sorted(row)
            conn.execute(
                f'INSERT INTO feature_sources ({", ".join(columns)}) '
                f'VALUES ({", ".join("?" for _ in columns)})',
                [row[c] for c in columns],
            )
