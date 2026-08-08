"""spindoctor.results_index -- an optional, rebuildable database over the results tree.

The navigation pipeline writes one ``_metadata.json`` document per image.  Programs
downstream of navigation need only a few fields from each, and reading a whole
document per image costs a round trip per image on a cloud root.  This package
defines a database whose rows carry those fields, built by one pass over the tree,
so a consumer answers its questions with a query instead of a walk.

The index is derived and disposable.  The documents remain the authoritative
record, no program requires an index, and deleting one costs nothing but the time
to rebuild it.  It is also a snapshot: it reflects the tree as of the last ingest,
with no staleness detection and no automatic refresh.

Public API:

    SCHEMA_VERSION   -- column-set version a database is stamped with
    METADATA         -- SQLAlchemy MetaData holding every table
    IMAGES           -- one row per image, keyed by (root_url, results_path_stub)
    TECHNIQUES       -- per-technique results for an image
    FEATURE_SOURCES  -- aggregated feature inventory for an image
    SCHEMA_META      -- the single row stamping the schema version
    INGEST_RUNS      -- one row per ingest pass over one root
    open_index       -- the only opener, with the version gate
"""

from spindoctor.results_index.engine import open_index
from spindoctor.results_index.schema import (
    FEATURE_SOURCES,
    IMAGES,
    INGEST_RUNS,
    METADATA,
    SCHEMA_META,
    SCHEMA_VERSION,
    TECHNIQUES,
)

__all__ = [
    'FEATURE_SOURCES',
    'IMAGES',
    'INGEST_RUNS',
    'METADATA',
    'SCHEMA_META',
    'SCHEMA_VERSION',
    'TECHNIQUES',
    'open_index',
]
