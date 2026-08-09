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
    FAILED_FILES     -- files that are not current-schema navigation documents
    SCHEMA_META      -- the single row stamping the schema version
    INGEST_RUNS      -- one row per ingest pass over one root
    open_index       -- the only opener, with the version gate
    masked_url       -- a connection URL with any password in it hidden
    normalize_root_url    -- the one spelling of a results root
    ingested_roots        -- the roots a completed ingest covered
    require_ingested_roots -- refuse to read absence from a root nobody ingested
    directories_missed    -- how much of a root the newest pass over it did not list
    FATAL_STATUS          -- the status the error selection filters match
    SPICE_STATUS_ERROR    -- the status_error the SPICE selection filters match
    ResultStubs           -- what a root holds, as a selection filter asks it
    read_result_stubs     -- one query answering an enumeration's selection filters
"""

from spindoctor.results_index.engine import masked_url, open_index
from spindoctor.results_index.roots import (
    directories_missed,
    ingested_roots,
    normalize_root_url,
    require_ingested_roots,
)
from spindoctor.results_index.schema import (
    FAILED_FILES,
    FEATURE_SOURCES,
    IMAGES,
    INGEST_RUNS,
    METADATA,
    SCHEMA_META,
    SCHEMA_VERSION,
    TECHNIQUES,
)
from spindoctor.results_index.selection import (
    FATAL_STATUS,
    SPICE_STATUS_ERROR,
    ResultStubs,
    read_result_stubs,
)

__all__ = [
    'FAILED_FILES',
    'FATAL_STATUS',
    'FEATURE_SOURCES',
    'IMAGES',
    'INGEST_RUNS',
    'METADATA',
    'SCHEMA_META',
    'SCHEMA_VERSION',
    'SPICE_STATUS_ERROR',
    'TECHNIQUES',
    'ResultStubs',
    'directories_missed',
    'ingested_roots',
    'masked_url',
    'normalize_root_url',
    'open_index',
    'read_result_stubs',
    'require_ingested_roots',
]
