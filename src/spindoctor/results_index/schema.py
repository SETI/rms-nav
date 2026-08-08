"""SQLAlchemy Core schema for the navigation results index.

The index is a database whose rows are derived from the navigation results tree
by a separate ingest step.  It is not authoritative: the per-image
``_metadata.json`` documents are, and the index can be deleted and rebuilt from
them at any time.  Consumers that only need a few fields per image read one row
instead of one document.

Keying
------

The primary key of ``images`` is the pair ``(root_url, results_path_stub)``.
``results_path_stub`` is the volume-and-filespec fragment every consumer already
uses to address a result (for example
``COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB``); it is the only
identifier unique across volumes, because two volumes may hold images with the
same basename.  ``root_url`` is the ingested results root in normalized form, so
one database can serve several roots without a consumer seeing another root's
rows.  The child tables ``techniques`` and ``feature_sources`` key on the same
pair and cascade on delete, so re-ingesting an image replaces its children.  Each
of them also declares the tuple that identifies one of its rows -- the technique
name, or the feature type and its source -- unique within an image, so a repeated
or retried insert is refused rather than silently doubling every count read from
it.

``results_path_stub`` carries a non-unique index of its own, for lookups that
already know the root, and ``image_date`` and ``instrument`` carry one each,
because the report groups and filters on both across a whole root.

Types
-----

Every pixel, ET, covariance and sigma column is ``Double`` rather than ``Float``,
because a dialect is free to map ``Float`` to single precision and the stored
offset must round-trip the document's value bit for bit.  Booleans are
``Boolean``, never an integer flag, so that a PostgreSQL backend rejects integer
arithmetic on them.  Structured values are ``JSON``, which is TEXT on SQLite and
``jsonb`` on PostgreSQL.

Versioning
----------

``schema_meta`` holds a single row carrying :data:`SCHEMA_VERSION`.  There are no
migrations: ingest is cheap relative to navigation and entirely reproducible from
the tree, so any change to the column set increments the version and the operator
deletes the database and re-ingests.
"""

from typing import Any

import sqlalchemy
from sqlalchemy.dialects import postgresql

__all__ = [
    'FEATURE_SOURCES',
    'IMAGES',
    'INGEST_RUNS',
    'METADATA',
    'SCHEMA_META',
    'SCHEMA_VERSION',
    'TECHNIQUES',
]

SCHEMA_VERSION = 2
"""Column-set version of the index.

Incremented by any change to the column set of any table, and by any change to
the constraints over it.  A database stamped with a different version is refused
rather than migrated.
"""

# TEXT on SQLite, jsonb on PostgreSQL.  jsonb is the type PostgreSQL's array and
# object accessors operate on, so a direct-SQL query against the index can reach
# inside these values without a cast.
_JSON = sqlalchemy.JSON().with_variant(postgresql.JSONB(), 'postgresql')

METADATA = sqlalchemy.MetaData()
"""Container for every table of the index; the source of all DDL."""


def _image_key_columns() -> tuple[sqlalchemy.Column[Any], sqlalchemy.Column[Any]]:
    """Return a fresh pair of columns naming the image a child row belongs to.

    A ``Column`` object binds to exactly one table, so each child table needs its
    own pair rather than a shared constant.

    Returns:
        The ``root_url`` and ``results_path_stub`` columns, both NOT NULL.
    """
    return (
        sqlalchemy.Column('root_url', sqlalchemy.Text, nullable=False),
        sqlalchemy.Column('results_path_stub', sqlalchemy.Text, nullable=False),
    )


def _image_foreign_key() -> sqlalchemy.ForeignKeyConstraint:
    """Return a fresh composite foreign key from a child table to ``images``.

    Deleting an image row deletes its children, which is what makes the
    delete-then-insert upsert of one image atomic and complete.

    Returns:
        The constraint, which the caller passes to its ``Table``.
    """
    return sqlalchemy.ForeignKeyConstraint(
        ['root_url', 'results_path_stub'],
        ['images.root_url', 'images.results_path_stub'],
        ondelete='CASCADE',
    )


IMAGES = sqlalchemy.Table(
    'images',
    METADATA,
    # Identity.  Derived from the file's own location under the ingest root, not
    # from the document, so it is exact by construction.
    sqlalchemy.Column('root_url', sqlalchemy.Text, primary_key=True),
    sqlalchemy.Column('results_path_stub', sqlalchemy.Text, primary_key=True),
    # First path segment of the stub; NULL when the stub has no separator, which
    # is what the simulated dataset's bare scene basenames produce.
    sqlalchemy.Column('volume', sqlalchemy.Text),
    # Observation.
    sqlalchemy.Column('image_name', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('instrument', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('camera', sqlalchemy.Text),
    sqlalchemy.Column('image_path', sqlalchemy.Text),
    sqlalchemy.Column('image_et', sqlalchemy.Double),
    sqlalchemy.Column('image_date', sqlalchemy.Text),
    # Outcome.  status_error and status_reason are different vocabularies:
    # status_error is what the SPICE-error selection filter matches verbatim,
    # status_reason is the navigator's explanation of a non-success outcome.
    sqlalchemy.Column('status', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('status_error', sqlalchemy.Text),
    sqlalchemy.Column('status_reason', sqlalchemy.Text),
    # The authoritative offset every consumer applies, stored unrounded.
    sqlalchemy.Column('offset_dv', sqlalchemy.Double),
    sqlalchemy.Column('offset_du', sqlalchemy.Double),
    sqlalchemy.Column('sigma_dv', sqlalchemy.Double),
    sqlalchemy.Column('sigma_du', sqlalchemy.Double),
    # The 2x2 offset block only.  For a twist-fitted result the rotation row and
    # column of the 3x3 matrix are deliberately not indexed; sigma_rotation_deg
    # is the only twist uncertainty the index carries.
    sqlalchemy.Column('covariance_vv', sqlalchemy.Double),
    sqlalchemy.Column('covariance_vu', sqlalchemy.Double),
    sqlalchemy.Column('covariance_uu', sqlalchemy.Double),
    sqlalchemy.Column('sigma_along_unobservable_px', sqlalchemy.Double),
    sqlalchemy.Column('rotation_deg', sqlalchemy.Double),
    sqlalchemy.Column('sigma_rotation_deg', sqlalchemy.Double),
    # Quality.
    sqlalchemy.Column('confidence', sqlalchemy.Double),
    sqlalchemy.Column('confidence_rank', sqlalchemy.Text),
    sqlalchemy.Column('n_techniques', sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column('excluded_from_consensus', _JSON),
    sqlalchemy.Column('image_class', sqlalchemy.Text),
    sqlalchemy.Column('noise_sigma', sqlalchemy.Double),
    sqlalchemy.Column('image_shape_v', sqlalchemy.Integer),
    sqlalchemy.Column('image_shape_u', sqlalchemy.Integer),
    # Run provenance.
    sqlalchemy.Column('run_start', sqlalchemy.Text),
    sqlalchemy.Column('run_end', sqlalchemy.Text),
    sqlalchemy.Column('elapsed_s', sqlalchemy.Double),
    sqlalchemy.Column('config_hash', sqlalchemy.Text),
    sqlalchemy.Column('git_sha', sqlalchemy.Text),
    sqlalchemy.Column('pipeline_run', sqlalchemy.Text),
    # The numeric portion of the image name, so a range filter compares against a
    # column instead of calling a function.  BigInteger because an instrument
    # naming scheme is free to run past the 32-bit range a dialect may impose.
    sqlalchemy.Column('image_number', sqlalchemy.BigInteger),
    # Whether the ingest walk saw a summary PNG beside the metadata file.
    sqlalchemy.Column('has_summary_png', sqlalchemy.Boolean),
    # Corrected-pointing fields.  Declared with the names and shapes their
    # producer specifies and NULL until it lands, because a column-set change
    # costs every operator a rebuild.
    sqlalchemy.Column('start_et', sqlalchemy.Double),
    sqlalchemy.Column('stop_et', sqlalchemy.Double),
    sqlalchemy.Column('exposure_s', sqlalchemy.Double),
    sqlalchemy.Column('sclk_start', sqlalchemy.Text),
    sqlalchemy.Column('sclk_midtime', sqlalchemy.Text),
    sqlalchemy.Column('sclk_stop', sqlalchemy.Text),
    sqlalchemy.Column('camera_frame_id', sqlalchemy.Integer),
    sqlalchemy.Column('ck_frame_id', sqlalchemy.Integer),
    sqlalchemy.Column('cmatrix', _JSON),
    sqlalchemy.Column('cmatrix_original', _JSON),
    # File provenance.  mtime_ns and size_bytes drive the incremental skip, and
    # both need 64 bits: a nanosecond epoch alone is far past the 32-bit range.
    sqlalchemy.Column('source_file', sqlalchemy.Text),
    sqlalchemy.Column('mtime_ns', sqlalchemy.BigInteger),
    sqlalchemy.Column('size_bytes', sqlalchemy.BigInteger),
    sqlalchemy.Index('ix_images_results_path_stub', 'results_path_stub'),
    sqlalchemy.Index('ix_images_image_date', 'image_date'),
    sqlalchemy.Index('ix_images_instrument', 'instrument'),
)
"""One row per navigated image, keyed by ``(root_url, results_path_stub)``."""

TECHNIQUES = sqlalchemy.Table(
    'techniques',
    METADATA,
    *_image_key_columns(),
    sqlalchemy.Column('technique_name', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('offset_dv', sqlalchemy.Double),
    sqlalchemy.Column('offset_du', sqlalchemy.Double),
    sqlalchemy.Column('sigma_dv', sqlalchemy.Double),
    sqlalchemy.Column('sigma_du', sqlalchemy.Double),
    sqlalchemy.Column('confidence', sqlalchemy.Double),
    sqlalchemy.Column('spurious', sqlalchemy.Boolean, nullable=False),
    sqlalchemy.Column('at_edge', sqlalchemy.Boolean, nullable=False),
    sqlalchemy.Column('source_names', _JSON),
    sqlalchemy.Column('diagnostics', _JSON),
    _image_foreign_key(),
    # One row per technique per image, enforced rather than assumed: a technique
    # reports once for an image, and a retried or duplicated ingest that inserted
    # a second row would change every count and average read from this table
    # without replacing anything.  The constraint's index also serves the lookup
    # by image, whose columns are its leading pair.
    sqlalchemy.UniqueConstraint(
        'root_url', 'results_path_stub', 'technique_name', name='uq_techniques_image_technique'
    ),
)
"""One row per technique that produced a result for an image."""

FEATURE_SOURCES = sqlalchemy.Table(
    'feature_sources',
    METADATA,
    *_image_key_columns(),
    sqlalchemy.Column('feature_type', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('source_model', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('source_name', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('n_features', sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column('n_gated', sqlalchemy.Integer, nullable=False),
    _image_foreign_key(),
    # The inventory is aggregated by exactly this tuple, so a second row for one
    # of them is a contradiction rather than more detail.  As on techniques, the
    # constraint's index leads with the image key and serves the lookup by image.
    sqlalchemy.UniqueConstraint(
        'root_url',
        'results_path_stub',
        'feature_type',
        'source_model',
        'source_name',
        name='uq_feature_sources_image_source',
    ),
)
"""Feature inventory of an image, aggregated per feature type and source."""

SCHEMA_META = sqlalchemy.Table(
    'schema_meta',
    METADATA,
    # A constant primary key with a check constraint: the table describes the
    # database, so a second row would describe a contradiction.
    sqlalchemy.Column('singleton', sqlalchemy.Integer, primary_key=True, autoincrement=False),
    sqlalchemy.Column('schema_version', sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column('created_utc', sqlalchemy.Text, nullable=False),
    sqlalchemy.CheckConstraint('singleton = 1', name='ck_schema_meta_singleton'),
)
"""Single row stamping the database with the column-set version that wrote it."""

INGEST_RUNS = sqlalchemy.Table(
    'ingest_runs',
    METADATA,
    sqlalchemy.Column('run_id', sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column('root_url', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('started_utc', sqlalchemy.Text, nullable=False),
    # NULL while the run is in flight.  A root whose newest row has no finish
    # time, or has no row at all, has not been ingested, and a consumer must say
    # so rather than read absence as "nothing was navigated".
    sqlalchemy.Column('finished_utc', sqlalchemy.Text),
    sqlalchemy.Column('files_seen', sqlalchemy.Integer),
    sqlalchemy.Column('files_ingested', sqlalchemy.Integer),
    sqlalchemy.Column('files_skipped', sqlalchemy.Integer),
    sqlalchemy.Column('files_failed', sqlalchemy.Integer),
    sqlalchemy.Column('schema_version', sqlalchemy.Integer, nullable=False),
    sqlalchemy.Index('ix_ingest_runs_root_url', 'root_url'),
)
"""One row per ingest pass over one root, recording what it covered."""
